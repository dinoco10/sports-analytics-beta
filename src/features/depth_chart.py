"""
Depth Chart Allocation Module (Layer 2)
=======================================

This module answers: "How many PA/IP does each player actually get?"

It sits BETWEEN Marcel (Layer 1, rate projections) and the season simulator
(Layer 3, Monte Carlo). Marcel tells us how good a player is per PA/IP.
The depth chart tells us how many PA/IP they'll accumulate based on their
roster position.

Architecture:
  player_projections.py → depth_chart.py → project_season_wins.py
  (rate stats only)       (PA/IP + WAR)    (season simulation)

The depth chart:
  1. Loads Marcel rate projections (wOBA, FIP) from CSVs
  2. Ranks players by quality within each position slot
  3. Assigns PA/IP by tier (starter, platoon, bench, emergency)
  4. Enforces team-level caps (5,700 PA hitters / 1,450 IP pitchers)
  5. Computes WAR using the assigned PA/IP (not Marcel's playing time)
  6. Outputs depth_chart_hitters_2026.csv and depth_chart_pitchers_2026.csv

This separation follows how professional systems work:
  - FanGraphs uses RosterResource humans for playing time
  - ZiPS uses Monte Carlo with injury distributions
  - We use automated position hierarchy with PA/IP tiers
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "features"

# Team-level budgets — these are the physical constraints of a 162-game season
TEAM_PA_CAP = 5700    # ~6,200 actual PA per team, but not all go to position players
TEAM_IP_CAP = 1450    # ~1,458 IP per team (162 games × 9 innings)

# Individual caps by tier — prevents Marcel from giving a backup 500 PA
HITTER_PA_CAP = 600   # No hitter exceeds 600 PA in projection
SP_IP_CAP = 185       # No SP exceeds 185 IP

# ── Hitter PA tiers ──────────────────────────────────────────
# Assigned by rank within each position slot.
# Most positions: starter gets full-time, backup gets bench role.
# Catcher: defaults to 65/35 split (no team runs a full-time catcher anymore).
HITTER_TIERS = {
    "starter":  560,   # Full-time regular
    "platoon":  280,   # Platoon partner or catcher backup
    "bench":    150,   # Utility / pinch hitter
    "emergency":  0,   # 40-man filler, won't accumulate PA
}

# Catcher-specific: forced 65/35 split between top two
CATCHER_STARTER_PA = 390   # 65% of ~600 total catcher PA
CATCHER_BACKUP_PA = 210    # 35% of ~600 total catcher PA

# ── Pitcher IP tiers ─────────────────────────────────────────
# Modern rotation: ~790 IP from starters, rest from bullpen.
SP_TIERS = {
    "SP1": 175,    # Ace, durable
    "SP2": 160,    # #2 starter
    "SP3": 160,    # #3 starter
    "SP4": 150,    # #4 starter
    "SP5": 145,    # #5 starter
    "SP6":  75,    # Swingman / spot starter
}

RP_TIERS = {
    "closer":    65,   # 9th inning
    "setup":     65,   # 7th-8th
    "middle_1":  50,   # Middle relief
    "middle_2":  50,
    "middle_3":  50,
    "middle_4":  50,
    "low_lev_1": 35,   # Low-leverage / mop-up
    "low_lev_2": 35,
    "low_lev_3": 35,
    "emergency": 0,    # Won't pitch
}

# ── Position adjustment runs (per 162 games) ─────────────────
# Standard fWAR positional adjustments from FanGraphs.
# Prorated by PA/600 for actual assigned playing time.
POSITION_ADJUSTMENT_RUNS = {
    "C":   +12.5,
    "SS":  +7.5,
    "2B":  +2.5,
    "3B":  +2.5,
    "CF":  +2.5,
    "LF":  -7.5,
    "RF":  -7.5,
    "1B":  -12.5,
    "DH":  -17.5,
}

# WAR constants (same as config/settings.py — duplicated here intentionally
# so depth_chart.py is self-contained and doesn't inherit from Marcel)
LEAGUE_AVG_WOBA = 0.310
LEAGUE_AVG_WOBA_SCALE = 1.15
RUNS_PER_WIN = 10
LEAGUE_FIP = 4.20

# ── Position slot mapping ────────────────────────────────────
# Maps MLB API position abbreviations to our 9 lineup slots.
# Some positions map to the same slot (e.g., all OF corners → LF/RF).
POSITION_SLOTS = {
    "C": "C", "1B": "1B", "2B": "2B", "SS": "SS", "3B": "3B",
    "LF": "LF", "CF": "CF", "RF": "RF", "DH": "DH",
    # Aliases from MLB API
    "OF": "LF",     # Generic OF → treat as corner OF
    "IF": "2B",     # Generic IF → treat as middle IF
    "P": None,      # Pitchers handled separately
    "SP": None,
    "RP": None,
    "TWP": None,    # Two-way player
    "PH": "DH",     # Pinch hitter → DH slot
    "PR": None,     # Pinch runner → skip
}

# The 9 hitter slots every team must fill
LINEUP_SLOTS = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH"]


# ═══════════════════════════════════════════════════════════════
# HITTER DEPTH CHART
# ═══════════════════════════════════════════════════════════════

def compute_wRAA_per_pa(woba: float) -> float:
    """
    Convert wOBA to wRAA per PA (runs above average per plate appearance).
    This is the pure rate stat used to rank hitters within position slots.
    wRAA/PA = (wOBA - lgWOBA) / wOBA_scale
    """
    return (woba - LEAGUE_AVG_WOBA) / LEAGUE_AVG_WOBA_SCALE


def assign_hitter_pa(hitters: pd.DataFrame) -> pd.DataFrame:
    """
    Assign PA to each hitter based on position hierarchy.

    For each team:
      1. Map each hitter to their position slot
      2. Rank hitters by wRAA/PA within each slot
      3. Assign PA by tier (starter, platoon, bench)
      4. Catcher uses 65/35 split instead of full starter
      5. Cap individual PA at 600
      6. Enforce team total PA cap at 5,700

    Returns: DataFrame with 'depth_pa' and 'tier' columns added.
    """
    df = hitters.copy()

    # Compute ranking metric: wRAA per PA (pure rate, no volume)
    df["wraa_per_pa"] = df["statcast_adjusted_woba"].apply(compute_wRAA_per_pa)

    # Map to position slot
    df["pos_slot"] = df["position"].map(POSITION_SLOTS)

    # Drop players with unmappable positions (pitchers listed as hitters, etc.)
    unmapped = df["pos_slot"].isna()
    if unmapped.any():
        log.info(f"  Dropping {unmapped.sum()} hitters with unmappable positions")
    df = df[~unmapped].copy()

    # Assign PA per team
    results = []
    for team, team_df in df.groupby("current_team"):
        team_assignments = _assign_team_hitter_pa(team, team_df)
        results.append(team_assignments)

    result = pd.concat(results, ignore_index=True)
    return result


def _assign_team_hitter_pa(team: str, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign PA for one team's hitters using position hierarchy.

    Logic:
      - For each of the 9 lineup slots, find the best available hitter
      - Best = highest wRAA/PA among unassigned hitters at that position
      - Starter gets full PA, second-best gets platoon/bench PA
      - Catcher is special: 65/35 split between top two
      - Any leftover hitters get bench or emergency PA
      - Cap individuals at 600 PA, then scale team to 5,700 cap
    """
    df = team_df.copy()
    df["depth_pa"] = 0.0
    df["tier"] = "emergency"

    assigned_ids = set()

    # Pass 1: Fill each position slot with the best available player
    for slot in LINEUP_SLOTS:
        slot_players = df[
            (df["pos_slot"] == slot) &
            (~df["mlb_player_id"].isin(assigned_ids))
        ].sort_values("wraa_per_pa", ascending=False)

        if slot_players.empty:
            continue

        if slot == "C":
            # Catcher: 65/35 split between top two
            starter = slot_players.iloc[0]
            df.loc[df["mlb_player_id"] == starter["mlb_player_id"], "depth_pa"] = CATCHER_STARTER_PA
            df.loc[df["mlb_player_id"] == starter["mlb_player_id"], "tier"] = "starter"
            assigned_ids.add(starter["mlb_player_id"])

            if len(slot_players) > 1:
                backup = slot_players.iloc[1]
                df.loc[df["mlb_player_id"] == backup["mlb_player_id"], "depth_pa"] = CATCHER_BACKUP_PA
                df.loc[df["mlb_player_id"] == backup["mlb_player_id"], "tier"] = "platoon"
                assigned_ids.add(backup["mlb_player_id"])
        else:
            # All other positions: starter gets full PA
            starter = slot_players.iloc[0]
            df.loc[df["mlb_player_id"] == starter["mlb_player_id"], "depth_pa"] = HITTER_TIERS["starter"]
            df.loc[df["mlb_player_id"] == starter["mlb_player_id"], "tier"] = "starter"
            assigned_ids.add(starter["mlb_player_id"])

            # Second-best gets bench PA (not platoon — platoon is for catcher)
            if len(slot_players) > 1:
                backup = slot_players.iloc[1]
                df.loc[df["mlb_player_id"] == backup["mlb_player_id"], "depth_pa"] = HITTER_TIERS["bench"]
                df.loc[df["mlb_player_id"] == backup["mlb_player_id"], "tier"] = "bench"
                assigned_ids.add(backup["mlb_player_id"])

    # Pass 2: Unassigned hitters with decent rates get utility/bench PA
    unassigned = df[
        (~df["mlb_player_id"].isin(assigned_ids)) &
        (df["wraa_per_pa"] > -0.030)  # Better than replacement level
    ].sort_values("wraa_per_pa", ascending=False)

    for _, player in unassigned.iterrows():
        df.loc[df["mlb_player_id"] == player["mlb_player_id"], "depth_pa"] = HITTER_TIERS["bench"]
        df.loc[df["mlb_player_id"] == player["mlb_player_id"], "tier"] = "bench"
        assigned_ids.add(player["mlb_player_id"])

    # Cap individual PA at 600
    df["depth_pa"] = df["depth_pa"].clip(upper=HITTER_PA_CAP)

    # Enforce team PA cap — scale proportionally if over budget
    team_total_pa = df["depth_pa"].sum()
    if team_total_pa > TEAM_PA_CAP:
        scale = TEAM_PA_CAP / team_total_pa
        df["depth_pa"] = (df["depth_pa"] * scale).round(0)

    return df


# ═══════════════════════════════════════════════════════════════
# PITCHER DEPTH CHART
# ═══════════════════════════════════════════════════════════════

def assign_pitcher_ip(pitchers: pd.DataFrame) -> pd.DataFrame:
    """
    Assign IP to each pitcher based on role hierarchy.

    For each team:
      1. Separate starters (SP) from relievers (RP)
      2. Rank SPs by FIP (lower = better), assign SP1-SP6 tiers
      3. Rank RPs by FIP, assign closer → setup → middle → low-leverage
      4. Cap individual SP at 185 IP
      5. Enforce team total IP cap at 1,450

    Returns: DataFrame with 'depth_ip' and 'tier' columns added.
    """
    df = pitchers.copy()

    results = []
    for team, team_df in df.groupby("current_team"):
        team_assignments = _assign_team_pitcher_ip(team, team_df)
        results.append(team_assignments)

    result = pd.concat(results, ignore_index=True)
    return result


def _assign_team_pitcher_ip(team: str, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign IP for one team's pitchers using role hierarchy.

    SPs ranked by FIP (lower = better), assigned SP1 through SP6.
    RPs ranked by FIP, assigned closer → setup → middle → low-leverage.
    """
    df = team_df.copy()
    df["depth_ip"] = 0.0
    df["tier"] = "emergency"

    # Split by role
    sps = df[df["role"] == "SP"].sort_values("proj_fip", ascending=True)
    rps = df[df["role"] == "RP"].sort_values("proj_fip", ascending=True)

    # Assign SP tiers (SP1 through SP6)
    sp_tier_names = list(SP_TIERS.keys())
    for i, (idx, row) in enumerate(sps.iterrows()):
        if i < len(sp_tier_names):
            tier = sp_tier_names[i]
            ip = SP_TIERS[tier]
        else:
            # Extra SPs beyond SP6 get emergency (0 IP)
            tier = "emergency"
            ip = 0
        df.loc[idx, "depth_ip"] = ip
        df.loc[idx, "tier"] = tier

    # Assign RP tiers (closer → setup → middle → low-leverage)
    rp_tier_names = list(RP_TIERS.keys())
    for i, (idx, row) in enumerate(rps.iterrows()):
        if i < len(rp_tier_names):
            tier = rp_tier_names[i]
            ip = RP_TIERS[tier]
        else:
            tier = "emergency"
            ip = 0
        df.loc[idx, "depth_ip"] = ip
        df.loc[idx, "tier"] = tier

    # Cap individual SP at 185 IP
    df["depth_ip"] = df["depth_ip"].clip(upper=SP_IP_CAP)

    # Enforce team IP cap — scale proportionally if over budget
    team_total_ip = df["depth_ip"].sum()
    if team_total_ip > TEAM_IP_CAP:
        scale = TEAM_IP_CAP / team_total_ip
        df["depth_ip"] = (df["depth_ip"] * scale).round(0)

    return df


# ═══════════════════════════════════════════════════════════════
# WAR CALCULATION (using depth chart PA/IP, not Marcel's)
# ═══════════════════════════════════════════════════════════════

def compute_hitter_war(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fWAR for each hitter using depth chart PA (not Marcel PA).

    Formula:
      batting_runs = wRAA/PA × depth_pa
      pos_adj_runs = POSITION_ADJUSTMENT_RUNS[pos] × (depth_pa / 600)
      fWAR = (batting_runs + pos_adj_runs) / RUNS_PER_WIN

    This is the same formula as player_projections.py, but PA comes from
    the depth chart tier system instead of Marcel's playing time estimate.
    """
    out = df.copy()

    # Batting runs: rate × volume
    out["batting_runs"] = out["wraa_per_pa"] * out["depth_pa"]

    # Position adjustment: prorated by PA/600
    out["pos_adj_runs"] = out["position"].map(POSITION_ADJUSTMENT_RUNS).fillna(0)
    out["pos_adj_runs"] = out["pos_adj_runs"] * (out["depth_pa"] / 600)

    # fWAR
    out["depth_war"] = (out["batting_runs"] + out["pos_adj_runs"]) / RUNS_PER_WIN

    # WAR floor: -1.5 per player (same as roster filter logic)
    out["depth_war"] = out["depth_war"].clip(lower=-1.5)

    return out


def compute_pitcher_war(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WAR for each pitcher using depth chart IP (not Marcel IP).

    Formula:
      WAR = (lgFIP - projFIP) / RUNS_PER_WIN × (depth_ip / 9)

    Pitchers with below-league FIP get positive WAR (they're helping),
    pitchers with above-league FIP get negative WAR (they're hurting).
    """
    out = df.copy()

    out["depth_war"] = (
        (LEAGUE_FIP - out["proj_fip"]) / RUNS_PER_WIN * (out["depth_ip"] / 9)
    )

    # WAR floor: -1.5 per player
    out["depth_war"] = out["depth_war"].clip(lower=-1.5)

    return out


# ═══════════════════════════════════════════════════════════════
# MAIN: BUILD DEPTH CHARTS
# ═══════════════════════════════════════════════════════════════

def build_depth_charts(team_filter: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load Marcel rates → assign PA/IP → compute WAR.

    Returns:
      (hitters_df, pitchers_df) with depth_pa/depth_ip, tier, and depth_war.
    """
    log.info("=" * 70)
    log.info("DEPTH CHART ALLOCATION (Layer 2)")
    log.info("=" * 70)

    # ── Load Marcel rate projections ──────────────────────────
    log.info("\nLoading Marcel rate projections...")
    hitters = pd.read_csv(FEATURES_DIR / "hitter_projections_2026.csv")
    pitchers = pd.read_csv(FEATURES_DIR / "pitcher_projections_2026.csv")

    # Filter out free agents
    hitters = hitters[hitters["current_team"] != "Free Agent"].copy()
    pitchers = pitchers[pitchers["current_team"] != "Free Agent"].copy()
    log.info(f"  Rostered hitters:  {len(hitters)}")
    log.info(f"  Rostered pitchers: {len(pitchers)}")

    if team_filter:
        hitters = hitters[hitters["current_team"].str.contains(team_filter, case=False)].copy()
        pitchers = pitchers[pitchers["current_team"].str.contains(team_filter, case=False)].copy()
        log.info(f"  Filtered to: {team_filter}")

    # ── Assign PA/IP by depth chart ──────────────────────────
    log.info("\nAssigning playing time by position hierarchy...")
    hitters = assign_hitter_pa(hitters)
    pitchers = assign_pitcher_ip(pitchers)

    # ── Compute WAR with depth chart PA/IP ───────────────────
    log.info("\nComputing WAR with depth chart playing time...")
    hitters = compute_hitter_war(hitters)
    pitchers = compute_pitcher_war(pitchers)

    # ── Summary stats ────────────────────────────────────────
    active_h = hitters[hitters["depth_pa"] > 0]
    active_p = pitchers[pitchers["depth_ip"] > 0]
    log.info(f"\n  Active hitters:  {len(active_h)} (of {len(hitters)} rostered)")
    log.info(f"  Active pitchers: {len(active_p)} (of {len(pitchers)} rostered)")

    # Team-level summary
    h_team = hitters.groupby("current_team").agg(
        total_pa=("depth_pa", "sum"),
        hit_war=("depth_war", "sum"),
        n_active=("depth_pa", lambda x: (x > 0).sum()),
    ).reset_index()

    p_team = pitchers.groupby("current_team").agg(
        total_ip=("depth_ip", "sum"),
        pitch_war=("depth_war", "sum"),
        n_active=("depth_ip", lambda x: (x > 0).sum()),
    ).reset_index()

    team_summary = h_team.merge(p_team, on="current_team", suffixes=("_h", "_p"))
    team_summary["total_war"] = team_summary["hit_war"] + team_summary["pitch_war"]
    team_summary = team_summary.sort_values("total_war", ascending=False)

    log.info(f"\n{'Team':<28} {'PA':>6} {'IP':>6} {'hWAR':>6} {'pWAR':>6} {'WAR':>6}")
    log.info("-" * 64)
    for _, r in team_summary.iterrows():
        log.info(
            f"  {r.current_team:<28} {r.total_pa:>5.0f} {r.total_ip:>5.0f} "
            f"{r.hit_war:>+6.1f} {r.pitch_war:>+6.1f} {r.total_war:>+6.1f}"
        )

    # ── Save depth chart CSVs ────────────────────────────────
    h_out = FEATURES_DIR / "depth_chart_hitters_2026.csv"
    p_out = FEATURES_DIR / "depth_chart_pitchers_2026.csv"
    hitters.to_csv(h_out, index=False)
    pitchers.to_csv(p_out, index=False)
    log.info(f"\nSaved: {h_out}")
    log.info(f"Saved: {p_out}")

    return hitters, pitchers


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build depth chart allocations")
    parser.add_argument("--team", type=str, help="Filter to one team")
    args = parser.parse_args()

    build_depth_charts(team_filter=args.team)
