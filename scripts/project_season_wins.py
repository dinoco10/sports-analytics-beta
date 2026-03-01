"""
Floor 3: Season Win Total Projections.

Three-phase pipeline:
  Phase A — Calibrated WAR baseline (regress our WAR → actual wins on 2024-2025)
  Phase B — Monte Carlo simulation (Log5 per-game probs, 10K sims)
  Phase C — Vegas benchmark (compare to sportsbook lines, flag value bets)

Usage:
  python scripts/project_season_wins.py              # Full pipeline
  python scripts/project_season_wins.py --phase A    # Calibrated baseline only
  python scripts/project_season_wins.py --sims 50000 # More simulations
  python scripts/project_season_wins.py --team "NYY"  # Single team detail
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import func, case

# ─── Project imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    DATA_DIR, FEATURES_DIR, EXTERNAL_DIR,
    REPLACEMENT_LEVEL_WINS, RUNS_PER_WIN,
)
from src.storage.database import get_session
from src.storage.models import Game, Team
from src.ingestion.mlb_api import MLBApiClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────
CALIBRATION_SEASONS = [2024, 2025]
PROJECTION_SEASON = 2026
NUM_SIMS_DEFAULT = 10_000
REGULAR_SEASON_CUTOFF_DAY = 1   # October 1 — games on/after are postseason
GAMES_PER_SEASON = 162

# 2026 regular season dates (Opening Day March 26, last day Sept 27)
SEASON_2026_START = date(2026, 3, 26)
SEASON_2026_END = date(2026, 9, 27)

# Playoff format: 3 division winners + 3 wild cards per league = 12 total
PLAYOFF_SPOTS_PER_LEAGUE = 6  # 3 div + 3 WC

# Vegas CSV path
VEGAS_CSV = EXTERNAL_DIR / "vegas_win_totals_2026.csv"

# Output paths
OUTPUT_CSV = FEATURES_DIR / "season_projections_2026.csv"


# ═══════════════════════════════════════════════════════════════
# PHASE A — CALIBRATED WAR BASELINE
# ═══════════════════════════════════════════════════════════════

def get_historical_records(session) -> pd.DataFrame:
    """
    Query actual team W-L records for calibration seasons (2024-2025).
    Filters out postseason by excluding October+ games.
    Returns: DataFrame with columns [team, season, wins, losses, games].
    """
    log.info("Querying historical records from DB...")

    # Build a query that counts wins per team per season
    # Each game appears twice (once as home, once as away), so we union them
    records = []

    teams = session.query(Team).filter(Team.active == True).all()
    team_map = {t.id: t.name for t in teams}

    for season in CALIBRATION_SEASONS:
        for team in teams:
            # Count games where this team was home or away (regular season only)
            games_as_home = (
                session.query(Game)
                .filter(
                    Game.season == season,
                    Game.home_team_id == team.id,
                    Game.home_score.isnot(None),
                    # Filter out postseason: before October
                    func.extract("month", Game.date) < 10,
                )
                .all()
            )
            games_as_away = (
                session.query(Game)
                .filter(
                    Game.season == season,
                    Game.away_team_id == team.id,
                    Game.home_score.isnot(None),
                    func.extract("month", Game.date) < 10,
                )
                .all()
            )

            wins = sum(1 for g in games_as_home if g.winner_id == team.id)
            wins += sum(1 for g in games_as_away if g.winner_id == team.id)
            total_games = len(games_as_home) + len(games_as_away)

            records.append({
                "team": team.name,
                "season": season,
                "wins": wins,
                "losses": total_games - wins,
                "games": total_games,
            })

    df = pd.DataFrame(records)
    log.info(f"  Got {len(df)} team-seasons ({df['games'].mean():.0f} avg games)")
    return df


def fit_calibration(records_df: pd.DataFrame, proj_war_series: pd.Series) -> tuple:
    """
    Calibrate projected WAR → expected wins using z-score mapping.

    Our projected WAR (from Marcel) is on a compressed scale (std ~4-5)
    compared to actual win spreads (std ~11-13). We can't directly regress
    because we don't have historical projected WAR — only 2026 projections.

    Instead, we use z-score mapping:
      1. Compute historical win distribution (mean, std) from 2024-2025
      2. Dampen std by 0.75 for preseason (projections are tighter than results)
      3. Scale: calibrated = 81 + (proj_war - mean_war) * (damped_win_std / war_std)

    This is the standard approach when projection and outcome scales differ.
    Returns: (league_mean_wins, scale_factor).
    """
    log.info("Fitting WAR → wins calibration (z-score mapping)...")

    # Historical win distribution
    hist_mean = records_df["wins"].mean()
    hist_std = records_df["wins"].std()
    log.info(f"  Historical wins: mean={hist_mean:.1f}, std={hist_std:.1f}")

    # Projected WAR distribution
    war_mean = proj_war_series.mean()
    war_std = proj_war_series.std()
    log.info(f"  Projected WAR:   mean={war_mean:.1f}, std={war_std:.1f}")

    # Dampen for preseason: projections should be tighter than actual results
    # Typical ratio: preseason projection std / actual std ≈ 0.70-0.80
    PRESEASON_DAMPEN = 0.75
    damped_std = hist_std * PRESEASON_DAMPEN
    log.info(f"  Dampened win std: {damped_std:.1f} "
             f"(actual {hist_std:.1f} x {PRESEASON_DAMPEN})")

    # Scale factor: wins per unit of projected WAR
    scale_factor = damped_std / war_std
    log.info(f"  Scale factor: {scale_factor:.2f} wins per WAR unit")

    # Verify on extremes
    top_war = proj_war_series.max()
    bot_war = proj_war_series.min()
    top_wins = hist_mean + (top_war - war_mean) * scale_factor
    bot_wins = hist_mean + (bot_war - war_mean) * scale_factor
    log.info(f"  Projected range: {bot_wins:.0f}W (worst) to {top_wins:.0f}W (best)")

    return hist_mean, war_mean, scale_factor


def load_2026_projections() -> pd.DataFrame:
    """
    Load depth chart projections and aggregate by team.

    Uses depth_chart_hitters/pitchers_2026.csv (Layer 2 output) instead of
    raw Marcel projections. The depth chart module handles:
      - Free agent filtering
      - PA/IP allocation by position hierarchy
      - Team PA/IP caps (5,700 / 1,450)
      - WAR calculation with depth chart playing time
      - Per-player WAR floor (-1.5)

    Returns: DataFrame with team-level WAR and regression signals.
    """
    log.info("Loading 2026 depth chart projections...")

    # Load depth chart outputs (Layer 2) — these have depth_pa/depth_ip/depth_war
    h_path = FEATURES_DIR / "depth_chart_hitters_2026.csv"
    p_path = FEATURES_DIR / "depth_chart_pitchers_2026.csv"

    if not h_path.exists() or not p_path.exists():
        log.warning("  Depth chart CSVs not found — run: python -m src.features.depth_chart")
        log.warning("  Falling back to raw Marcel projections...")
        return _load_2026_projections_fallback()

    hitters = pd.read_csv(h_path)
    pitchers = pd.read_csv(p_path)

    # Only count active players (those assigned playing time by depth chart)
    active_h = hitters[hitters["depth_pa"] > 0]
    active_p = pitchers[pitchers["depth_ip"] > 0]
    log.info(f"  Active hitters:  {len(active_h)} (of {len(hitters)} rostered)")
    log.info(f"  Active pitchers: {len(active_p)} (of {len(pitchers)} rostered)")

    # Aggregate hitter stats by team — using depth_war (not Marcel's proj_war)
    hit_agg = active_h.groupby("current_team").agg(
        hit_war=("depth_war", "sum"),
        avg_bounce_back=("bounce_back_score", "mean"),
        avg_regression_risk_hit=("regression_risk_score", "mean"),
        n_hitters=("mlb_player_id", "count"),
    ).reset_index()

    # Aggregate pitcher stats by team
    pitch_agg = active_p.groupby("current_team").agg(
        pitch_war=("depth_war", "sum"),
        avg_sustainability=("sustainability_score", "mean"),
        avg_regression_risk_pitch=("regression_risk_score", "mean"),
        avg_breakout=("breakout_score", "mean"),
        n_pitchers=("mlb_player_id", "count"),
    ).reset_index()

    # Merge hitter and pitcher aggregates
    team_proj = hit_agg.merge(pitch_agg, on="current_team", how="outer")
    team_proj = team_proj.fillna(0)
    team_proj["total_war"] = team_proj["hit_war"] + team_proj["pitch_war"]

    # Rename for consistency
    team_proj = team_proj.rename(columns={"current_team": "team"})

    log.info(f"  {len(team_proj)} teams with projections")
    log.info(f"  WAR range: {team_proj['total_war'].min():.1f} "
             f"to {team_proj['total_war'].max():.1f}")

    return team_proj


def _load_2026_projections_fallback() -> pd.DataFrame:
    """Fallback: load raw Marcel projections if depth chart CSVs don't exist."""
    hitters = pd.read_csv(FEATURES_DIR / "hitter_projections_2026.csv")
    pitchers = pd.read_csv(FEATURES_DIR / "pitcher_projections_2026.csv")
    hitters = hitters[hitters["current_team"] != "Free Agent"].copy()
    pitchers = pitchers[pitchers["current_team"] != "Free Agent"].copy()
    hitters = hitters[hitters["proj_pa"] >= 100].copy()
    pitchers = pitchers[pitchers["proj_ip"] >= 30].copy()
    hitters["proj_war"] = hitters["proj_war"].clip(lower=-1.5)
    pitchers["proj_war"] = pitchers["proj_war"].clip(lower=-1.5)

    hit_agg = hitters.groupby("current_team").agg(
        hit_war=("proj_war", "sum"),
        avg_bounce_back=("bounce_back_score", "mean"),
        avg_regression_risk_hit=("regression_risk_score", "mean"),
        n_hitters=("mlb_player_id", "count"),
    ).reset_index()
    pitch_agg = pitchers.groupby("current_team").agg(
        pitch_war=("proj_war", "sum"),
        avg_sustainability=("sustainability_score", "mean"),
        avg_regression_risk_pitch=("regression_risk_score", "mean"),
        avg_breakout=("breakout_score", "mean"),
        n_pitchers=("mlb_player_id", "count"),
    ).reset_index()

    team_proj = hit_agg.merge(pitch_agg, on="current_team", how="outer").fillna(0)
    team_proj["total_war"] = team_proj["hit_war"] + team_proj["pitch_war"]
    team_proj = team_proj.rename(columns={"current_team": "team"})
    log.info(f"  {len(team_proj)} teams (fallback mode)")
    return team_proj


def compute_regression_adjustment(team_proj: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up regression signals to a net win adjustment per team.
    Positive = team should improve, negative = team likely to regress.

    Factors:
      - Bounce-back score (hitters): >60 suggests improvement
      - Regression risk (hitters+pitchers): >60 suggests decline
      - Sustainability (pitchers): <40 suggests decline
      - Breakout (pitchers): >60 suggests improvement

    Net adjustment is capped at ±3 wins.
    """
    log.info("Computing regression adjustments...")

    # Each signal: convert 0-100 scale to a directional adjustment
    # Neutral = 50, so (score - 50) / 50 gives -1 to +1 range
    adj = pd.DataFrame()
    adj["team"] = team_proj["team"]

    # Bounce-back: higher = more upside (positive adjustment)
    adj["bb_adj"] = (team_proj["avg_bounce_back"] - 50) / 50

    # Regression risk: higher = more likely to decline (negative adjustment)
    adj["reg_risk_adj"] = -(team_proj["avg_regression_risk_hit"] - 50) / 50
    adj["reg_risk_pitch_adj"] = -(team_proj["avg_regression_risk_pitch"] - 50) / 50

    # Sustainability: lower = more likely to decline
    adj["sust_adj"] = (team_proj["avg_sustainability"] - 50) / 50

    # Breakout: higher = more upside
    adj["breakout_adj"] = (team_proj["avg_breakout"] - 50) / 50

    # Weighted combination (equal weights, scale to ±3 wins max)
    adj["raw_adj"] = (
        adj["bb_adj"] * 0.25
        + adj["reg_risk_adj"] * 0.20
        + adj["reg_risk_pitch_adj"] * 0.20
        + adj["sust_adj"] * 0.15
        + adj["breakout_adj"] * 0.20
    )

    # Scale: raw is roughly -1 to +1, multiply by 3 for ±3 win range
    adj["regression_adj"] = (adj["raw_adj"] * 3).clip(-3, 3)

    team_proj = team_proj.merge(adj[["team", "regression_adj"]], on="team")
    log.info(f"  Adjustment range: {team_proj['regression_adj'].min():.1f} "
             f"to {team_proj['regression_adj'].max():.1f} wins")

    return team_proj


def compute_sos_adjustment(team_proj: pd.DataFrame, session) -> pd.DataFrame:
    """
    Strength-of-schedule adjustment based on division opponents.
    Teams in strong divisions face tougher schedules (~76 divisional games).

    SOS adj = -(division_avg_war - league_avg_war) * 0.15
    (negative because facing stronger teams reduces your wins)
    """
    log.info("Computing strength-of-schedule adjustments...")

    # Get division info from DB
    teams_db = session.query(Team).filter(Team.active == True).all()
    team_info = pd.DataFrame([{
        "team": t.name,
        "league": t.league,
        "division": f"{t.league} {t.division}",
    } for t in teams_db])

    team_proj = team_proj.merge(team_info, on="team", how="left")

    # Compute division average WAR (excluding the team itself)
    league_avg_war = team_proj["total_war"].mean()

    sos_records = []
    for div in team_proj["division"].dropna().unique():
        div_teams = team_proj[team_proj["division"] == div]
        for _, row in div_teams.iterrows():
            # Average WAR of division opponents (excluding self)
            opponents_war = div_teams[div_teams["team"] != row["team"]]["total_war"].mean()
            # Adjustment: facing stronger opponents costs wins
            sos_adj = -(opponents_war - league_avg_war) * 0.15
            sos_records.append({"team": row["team"], "sos_adj": round(sos_adj, 2)})

    sos_df = pd.DataFrame(sos_records)
    team_proj = team_proj.merge(sos_df, on="team", how="left")
    team_proj["sos_adj"] = team_proj["sos_adj"].fillna(0)

    log.info(f"  SOS range: {team_proj['sos_adj'].min():.1f} "
             f"to {team_proj['sos_adj'].max():.1f} wins")

    return team_proj


def run_phase_a(session) -> pd.DataFrame:
    """
    Phase A: Calibrate WAR → wins using 2024-2025 data, then project 2026.
    Returns: DataFrame with calibrated win projections per team.
    """
    log.info("\n" + "=" * 60)
    log.info("PHASE A — CALIBRATED WAR BASELINE")
    log.info("=" * 60)

    # Step 1: Historical records (for win distribution stats)
    records_df = get_historical_records(session)

    # Step 4: Load 2026 projections (need WAR before calibration)
    team_proj = load_2026_projections()

    # Step 3: Fit calibration using z-score mapping
    hist_mean, war_mean, scale_factor = fit_calibration(
        records_df, team_proj["total_war"]
    )

    # Step 5: Regression adjustment
    team_proj = compute_regression_adjustment(team_proj)

    # Step 6: SOS adjustment
    team_proj = compute_sos_adjustment(team_proj, session)

    # Step 7: Calibrated wins via z-score mapping
    # calibrated = league_avg + (proj_war - avg_war) * scale + adjustments
    team_proj["calibrated_wins"] = (
        hist_mean
        + (team_proj["total_war"] - war_mean) * scale_factor
        + team_proj["regression_adj"]
        + team_proj["sos_adj"]
    ).round(1)

    # Sanity clamp: no team below 40 or above 120
    team_proj["calibrated_wins"] = team_proj["calibrated_wins"].clip(40, 120)

    # Sort by projected wins
    team_proj = team_proj.sort_values("calibrated_wins", ascending=False).reset_index(drop=True)

    log.info(f"\n  Phase A complete. Projection range: "
             f"{team_proj['calibrated_wins'].min():.0f} to "
             f"{team_proj['calibrated_wins'].max():.0f} wins")

    # Print top 10 / bottom 5
    log.info("\n  Top 10:")
    for _, row in team_proj.head(10).iterrows():
        log.info(f"    {row['team']:<28s} {row['calibrated_wins']:5.1f}W "
                 f"(WAR={row['total_war']:+5.1f}, "
                 f"reg={row['regression_adj']:+4.1f}, "
                 f"sos={row['sos_adj']:+4.1f})")

    log.info("\n  Bottom 5:")
    for _, row in team_proj.tail(5).iterrows():
        log.info(f"    {row['team']:<28s} {row['calibrated_wins']:5.1f}W "
                 f"(WAR={row['total_war']:+5.1f}, "
                 f"reg={row['regression_adj']:+4.1f}, "
                 f"sos={row['sos_adj']:+4.1f})")

    return team_proj


# ═══════════════════════════════════════════════════════════════
# PHASE B — MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════

def fetch_schedule_2026() -> pd.DataFrame:
    """
    Fetch the 2026 regular-season schedule from MLB API.
    Caches to CSV to avoid re-fetching on subsequent runs.
    """
    cache_path = EXTERNAL_DIR / "schedule_2026.csv"

    if cache_path.exists():
        log.info(f"Loading cached schedule from {cache_path}")
        schedule = pd.read_csv(cache_path)
        log.info(f"  {len(schedule)} games loaded from cache")
        return schedule

    log.info("Fetching 2026 schedule from MLB API...")
    client = MLBApiClient()

    # Fetch month-by-month to stay within API limits
    all_games = []
    current = SEASON_2026_START
    while current <= SEASON_2026_END:
        # End of month or end of season
        month_end = date(
            current.year,
            current.month + 1 if current.month < 12 else 12,
            1,
        ) - pd.Timedelta(days=1)
        end = min(month_end.date() if hasattr(month_end, 'date') else month_end,
                  SEASON_2026_END)

        log.info(f"  Fetching {current} to {end}...")
        chunk = client.get_schedule(current, end)
        if len(chunk) > 0:
            all_games.append(chunk)

        # Move to next month
        if current.month < 12:
            current = date(current.year, current.month + 1, 1)
        else:
            break

    if not all_games:
        log.warning("  No schedule data returned — API may not have 2026 schedule yet")
        return pd.DataFrame()

    schedule = pd.concat(all_games, ignore_index=True)

    # Filter to regular season games only (exclude All-Star, etc.)
    # Keep only "Scheduled" or "Final" status games between real teams
    schedule = schedule[
        schedule["home_team"].notna() & schedule["away_team"].notna()
    ].copy()

    # Cache to CSV
    schedule.to_csv(cache_path, index=False)
    log.info(f"  Cached {len(schedule)} games to {cache_path}")

    return schedule


def build_synthetic_schedule(team_names: list) -> pd.DataFrame:
    """
    If the real 2026 schedule isn't available yet, build a balanced
    synthetic schedule: each team plays 162 games (13-14 vs each opponent).

    This is a round-robin approximation — not perfect but good enough
    for preseason Monte Carlo when the real schedule isn't published.
    """
    log.info("Building synthetic 162-game schedule...")

    games = []
    n_teams = len(team_names)

    # Each pair plays ~5.6 games. We'll do 5 or 6 to get close to 162 each.
    # With 29 opponents: 29 * 5 = 145, need 17 more → add extras for division
    # Simplified: create ~2430 total games (162 * 30 / 2)

    # Round-robin: each pair plays 5 games, then fill remaining
    game_id = 0
    pair_games = {}
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            pair_games[(i, j)] = 0

    # First pass: 5 games per pair = 145 games per team
    for (i, j) in pair_games:
        for g in range(5):
            home = i if g % 2 == 0 else j
            away = j if g % 2 == 0 else i
            games.append({
                "home_team": team_names[home],
                "away_team": team_names[away],
            })
            pair_games[(i, j)] += 1
            game_id += 1

    # Second pass: add 17 more games per team to reach 162
    # Count current games per team
    team_game_counts = {t: 0 for t in team_names}
    for g in games:
        team_game_counts[g["home_team"]] += 1
        team_game_counts[g["away_team"]] += 1

    # Add games for teams that need more (prioritize within-division would be
    # ideal, but for Monte Carlo purposes, random opponents work fine)
    rng = np.random.default_rng(42)
    while any(c < GAMES_PER_SEASON for c in team_game_counts.values()):
        # Find teams needing more games
        needy = [t for t, c in team_game_counts.items() if c < GAMES_PER_SEASON]
        if len(needy) < 2:
            break
        rng.shuffle(needy)
        for k in range(0, len(needy) - 1, 2):
            if (team_game_counts[needy[k]] < GAMES_PER_SEASON and
                    team_game_counts[needy[k + 1]] < GAMES_PER_SEASON):
                games.append({
                    "home_team": needy[k],
                    "away_team": needy[k + 1],
                })
                team_game_counts[needy[k]] += 1
                team_game_counts[needy[k + 1]] += 1

    schedule = pd.DataFrame(games)
    log.info(f"  Created {len(schedule)} games "
             f"(avg {sum(team_game_counts.values()) / n_teams:.0f} per team)")
    return schedule


def log5(p_home: float, p_away: float) -> float:
    """
    Log5 formula: probability that team A beats team B.

    Given each team's "true" win probability (against an average opponent),
    Log5 computes the expected probability of A beating B:

      P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)

    This is the standard method for preseason simulations.
    """
    numerator = p_home * (1 - p_away)
    denominator = p_home * (1 - p_away) + p_away * (1 - p_home)
    if denominator == 0:
        return 0.5
    return numerator / denominator


def run_phase_b(team_proj: pd.DataFrame, n_sims: int = NUM_SIMS_DEFAULT,
                session=None) -> pd.DataFrame:
    """
    Phase B: Monte Carlo simulation of the 2026 season.

    For each game on the schedule, compute Log5 probability and simulate
    outcomes. Repeat n_sims times. Track wins, division standings, and
    playoff probabilities.
    """
    log.info("\n" + "=" * 60)
    log.info(f"PHASE B — MONTE CARLO SIMULATION ({n_sims:,} sims)")
    log.info("=" * 60)

    # Step 8: Fetch or build schedule
    schedule = fetch_schedule_2026()

    # If schedule is empty (API doesn't have 2026 yet), use synthetic
    if schedule.empty:
        log.info("  Real schedule not available — using synthetic schedule")
        schedule = build_synthetic_schedule(team_proj["team"].tolist())

    # Step 9: Team strength ratings
    team_wpct = {}
    for _, row in team_proj.iterrows():
        wpct = row["calibrated_wins"] / GAMES_PER_SEASON
        # Clamp to reasonable range
        team_wpct[row["team"]] = np.clip(wpct, 0.300, 0.700)

    # Step 10: Home-field advantage from DB
    if session:
        total = (session.query(func.count(Game.id))
                 .filter(Game.home_score.isnot(None),
                         Game.season.in_(CALIBRATION_SEASONS))
                 .scalar())
        home_wins = (session.query(func.count(Game.id))
                     .filter(Game.home_score > Game.away_score,
                             Game.season.in_(CALIBRATION_SEASONS))
                     .scalar())
        hfa = home_wins / total if total > 0 else 0.535
    else:
        hfa = 0.535

    # HFA boost: how much being home adds to win probability
    # If league home W% is 53.5%, the HFA boost is +3.5% on top of neutral
    hfa_boost = hfa - 0.500
    log.info(f"  Home-field advantage: {hfa:.3f} ({hfa_boost:+.3f})")

    # Step 11: Compute per-game Log5 probabilities
    # Map schedule team names to our projection team names
    proj_teams = set(team_proj["team"].tolist())

    # Filter schedule to games where both teams are in our projections
    valid_games = schedule[
        schedule["home_team"].isin(proj_teams) &
        schedule["away_team"].isin(proj_teams)
    ].copy()

    if len(valid_games) == 0:
        # Try to match by fuzzy name (e.g., "Athletics" vs "Oakland Athletics")
        log.warning("  No games matched — attempting name reconciliation...")
        schedule_teams = set(schedule["home_team"].tolist() +
                             schedule["away_team"].tolist())
        log.info(f"  Schedule teams: {sorted(schedule_teams)[:5]}...")
        log.info(f"  Projection teams: {sorted(proj_teams)[:5]}...")
        return team_proj

    log.info(f"  {len(valid_games)} games in schedule")

    # Precompute probabilities for each game
    home_teams = valid_games["home_team"].values
    away_teams = valid_games["away_team"].values

    game_probs = np.array([
        log5(team_wpct[h] + hfa_boost, team_wpct[a])
        for h, a in zip(home_teams, away_teams)
    ])

    # Step 12: Simulate seasons (vectorized)
    log.info(f"  Simulating {n_sims:,} seasons...")
    rng = np.random.default_rng(2026)

    n_games = len(valid_games)
    # Random matrix: (n_sims, n_games) — each cell is a uniform [0,1)
    random_draws = rng.random((n_sims, n_games))

    # Home team wins if random draw < game probability
    home_wins_matrix = random_draws < game_probs  # shape: (n_sims, n_games)

    # Count wins per team per simulation
    team_list = sorted(proj_teams)
    team_idx = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)

    # Build index arrays for fast accumulation
    home_idx = np.array([team_idx[t] for t in home_teams])
    away_idx = np.array([team_idx[t] for t in away_teams])

    # Accumulate wins per team per sim
    sim_wins = np.zeros((n_sims, n_teams), dtype=np.int32)

    for g in range(n_games):
        # For each game, add 1 to winner's column across all sims
        home_won = home_wins_matrix[:, g]  # bool array of shape (n_sims,)
        sim_wins[:, home_idx[g]] += home_won.astype(np.int32)
        sim_wins[:, away_idx[g]] += (~home_won).astype(np.int32)

    # Step 13: Compute summary statistics per team
    log.info("  Computing summary statistics...")
    results = []
    for i, team in enumerate(team_list):
        wins = sim_wins[:, i]
        results.append({
            "team": team,
            "mean_wins": round(wins.mean(), 1),
            "median_wins": int(np.median(wins)),
            "std_wins": round(wins.std(), 1),
            "p10_wins": int(np.percentile(wins, 10)),
            "p90_wins": int(np.percentile(wins, 90)),
            "p5_wins": int(np.percentile(wins, 5)),
            "p95_wins": int(np.percentile(wins, 95)),
        })

    sim_results = pd.DataFrame(results)

    # Zero-sum check: total wins per sim should be ~half of total games
    total_wins_per_sim = sim_wins.sum(axis=1)
    expected_total = len(valid_games)
    log.info(f"  Zero-sum check: avg total wins = {total_wins_per_sim.mean():.0f} "
             f"(expected: {expected_total})")

    # Step 14: Division standings + playoff probabilities
    log.info("  Computing playoff probabilities...")

    # Get division/league info
    div_info = team_proj[["team", "league", "division"]].copy()
    team_division = dict(zip(div_info["team"], div_info["division"]))
    team_league = dict(zip(div_info["team"], div_info["league"]))

    # Group teams by division
    divisions = {}
    for team in team_list:
        div = team_division.get(team, "Unknown")
        if div not in divisions:
            divisions[div] = []
        divisions[div].append(team_idx[team])

    # Group teams by league
    leagues = {}
    for team in team_list:
        lg = team_league.get(team, "Unknown")
        if lg not in leagues:
            leagues[lg] = []
        leagues[lg].append((team, team_idx[team]))

    # Count division titles and playoff appearances
    div_titles = np.zeros(n_teams, dtype=np.int32)
    playoff_apps = np.zeros(n_teams, dtype=np.int32)

    for sim in range(n_sims):
        wins_this_sim = sim_wins[sim, :]

        # Division winners (best record in each division)
        div_winner_indices = []
        for div, team_indices in divisions.items():
            div_wins = [(idx, wins_this_sim[idx]) for idx in team_indices]
            # Sort by wins (desc), break ties randomly
            div_wins.sort(key=lambda x: (-x[1], rng.random()))
            winner_idx = div_wins[0][0]
            div_titles[winner_idx] += 1
            div_winner_indices.append(winner_idx)
            playoff_apps[winner_idx] += 1

        # Wild cards: best non-division-winners per league (3 per league)
        for lg, lg_teams in leagues.items():
            non_winners = [(t, idx, wins_this_sim[idx])
                           for t, idx in lg_teams
                           if idx not in div_winner_indices]
            non_winners.sort(key=lambda x: (-x[2], rng.random()))
            for t, idx, w in non_winners[:3]:
                playoff_apps[idx] += 1

    # Add probabilities to results
    for i, team in enumerate(team_list):
        idx = sim_results[sim_results["team"] == team].index[0]
        sim_results.loc[idx, "division_pct"] = round(
            div_titles[i] / n_sims * 100, 1
        )
        sim_results.loc[idx, "playoff_pct"] = round(
            playoff_apps[i] / n_sims * 100, 1
        )

    # Merge simulation results into team_proj
    team_proj = team_proj.merge(sim_results, on="team", how="left")

    log.info("  Phase B complete.")
    return team_proj


# ═══════════════════════════════════════════════════════════════
# PHASE C — VEGAS BENCHMARK
# ═══════════════════════════════════════════════════════════════

def create_vegas_csv():
    """
    Create a template Vegas CSV if it doesn't exist.
    User can manually fill in sportsbook lines.
    """
    if VEGAS_CSV.exists():
        return

    log.info(f"Creating Vegas template at {VEGAS_CSV}")
    log.info("  Fill in 'vegas_wins' column with sportsbook over/under lines")

    # Preseason consensus win totals (approximate, from major sportsbooks)
    # These are placeholder estimates — update with actual lines when available
    vegas_data = {
        "team": [
            "Los Angeles Dodgers", "New York Yankees", "Atlanta Braves",
            "Baltimore Orioles", "Philadelphia Phillies", "Houston Astros",
            "New York Mets", "Milwaukee Brewers", "San Diego Padres",
            "Cleveland Guardians", "Seattle Mariners", "Minnesota Twins",
            "Detroit Tigers", "Toronto Blue Jays", "Texas Rangers",
            "Tampa Bay Rays", "Boston Red Sox", "San Francisco Giants",
            "Arizona Diamondbacks", "Chicago Cubs", "Cincinnati Reds",
            "St. Louis Cardinals", "Kansas City Royals", "Pittsburgh Pirates",
            "Los Angeles Angels", "Miami Marlins", "Washington Nationals",
            "Athletics", "Colorado Rockies", "Chicago White Sox",
        ],
        "vegas_wins": [
            # Placeholder — replace with actual sportsbook lines
            97.5, 90.5, 88.5,
            87.5, 87.5, 85.5,
            84.5, 84.5, 83.5,
            82.5, 82.5, 80.5,
            79.5, 79.5, 79.5,
            78.5, 78.5, 77.5,
            77.5, 76.5, 76.5,
            75.5, 74.5, 73.5,
            72.5, 68.5, 67.5,
            65.5, 61.5, 58.5,
        ],
    }

    df = pd.DataFrame(vegas_data)
    df.to_csv(VEGAS_CSV, index=False)
    log.info(f"  Saved {len(df)} teams to {VEGAS_CSV}")


def run_phase_c(team_proj: pd.DataFrame) -> pd.DataFrame:
    """
    Phase C: Compare projections to Vegas lines.
    Flags value bets where |diff| > 2 wins.
    """
    log.info("\n" + "=" * 60)
    log.info("PHASE C — VEGAS BENCHMARK")
    log.info("=" * 60)

    # Create template if needed
    create_vegas_csv()

    if not VEGAS_CSV.exists():
        log.warning("  No Vegas data available — skipping Phase C")
        return team_proj

    vegas = pd.read_csv(VEGAS_CSV)
    log.info(f"  Loaded {len(vegas)} Vegas lines")

    # Merge Vegas lines with projections
    team_proj = team_proj.merge(vegas, on="team", how="left")

    # Use mean_wins if available (Monte Carlo), otherwise calibrated_wins
    proj_col = "mean_wins" if "mean_wins" in team_proj.columns else "calibrated_wins"
    team_proj["vs_vegas_diff"] = (
        team_proj[proj_col] - team_proj["vegas_wins"]
    ).round(1)

    # Flag value bets: |diff| > 2 wins
    team_proj["value_flag"] = team_proj["vs_vegas_diff"].apply(
        lambda d: "OVER" if d > 2 else ("UNDER" if d < -2 else "")
    )

    # Calculate MAE
    valid = team_proj[team_proj["vegas_wins"].notna()]
    mae = valid["vs_vegas_diff"].abs().mean()
    log.info(f"  MAE vs Vegas: {mae:.1f} wins")

    # Count value bets
    n_over = (team_proj["value_flag"] == "OVER").sum()
    n_under = (team_proj["value_flag"] == "UNDER").sum()
    log.info(f"  Value bets: {n_over} OVER, {n_under} UNDER")

    return team_proj


# ═══════════════════════════════════════════════════════════════
# OUTPUT & DISPLAY
# ═══════════════════════════════════════════════════════════════

def display_results(team_proj: pd.DataFrame, team_filter: str = None):
    """
    Print a formatted results table to the terminal.
    """
    log.info("\n" + "=" * 60)
    log.info("SEASON PROJECTIONS 2026")
    log.info("=" * 60)

    # Determine which columns are available
    has_sim = "mean_wins" in team_proj.columns
    has_vegas = "vegas_wins" in team_proj.columns

    # Sort by projected wins
    sort_col = "mean_wins" if has_sim else "calibrated_wins"
    df = team_proj.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Filter to single team if requested
    if team_filter:
        # Match by name or abbreviation substring
        mask = (
            df["team"].str.contains(team_filter, case=False, na=False)
        )
        if mask.sum() == 0:
            log.warning(f"  No team matching '{team_filter}' found")
            return
        df = df[mask]

    # Build header
    if has_sim and has_vegas:
        header = (f"{'#':>2} {'Team':<28s} {'Proj':>5s} {'80% CI':>10s} "
                  f"{'Div%':>5s} {'PLF%':>5s} {'Vegas':>5s} {'Diff':>5s} {'Flag':>6s}")
        log.info(header)
        log.info("-" * len(header))

        for i, (_, row) in enumerate(df.iterrows(), 1):
            ci = f"{row.get('p10_wins', ''):>3}-{row.get('p90_wins', ''):<3}"
            vegas = f"{row.get('vegas_wins', ''):>5.1f}" if pd.notna(row.get("vegas_wins")) else "  N/A"
            diff = f"{row.get('vs_vegas_diff', ''):>+5.1f}" if pd.notna(row.get("vs_vegas_diff")) else "  N/A"
            flag = row.get("value_flag", "")
            flag_str = f"**{flag}**" if flag else ""

            log.info(
                f"{i:>2} {row['team']:<28s} {row['mean_wins']:>5.1f} {ci:>10s} "
                f"{row.get('division_pct', 0):>5.1f} {row.get('playoff_pct', 0):>5.1f} "
                f"{vegas} {diff} {flag_str:>6s}"
            )

    elif has_sim:
        header = (f"{'#':>2} {'Team':<28s} {'Proj':>5s} {'80% CI':>10s} "
                  f"{'Div%':>5s} {'PLF%':>5s}")
        log.info(header)
        log.info("-" * len(header))

        for i, (_, row) in enumerate(df.iterrows(), 1):
            ci = f"{row.get('p10_wins', ''):>3}-{row.get('p90_wins', ''):<3}"
            log.info(
                f"{i:>2} {row['team']:<28s} {row['mean_wins']:>5.1f} {ci:>10s} "
                f"{row.get('division_pct', 0):>5.1f} {row.get('playoff_pct', 0):>5.1f}"
            )

    else:
        header = f"{'#':>2} {'Team':<28s} {'Proj':>5s} {'WAR':>5s} {'Reg':>5s} {'SOS':>5s}"
        log.info(header)
        log.info("-" * len(header))

        for i, (_, row) in enumerate(df.iterrows(), 1):
            log.info(
                f"{i:>2} {row['team']:<28s} {row['calibrated_wins']:>5.1f} "
                f"{row['total_war']:>+5.1f} {row['regression_adj']:>+5.1f} "
                f"{row['sos_adj']:>+5.1f}"
            )

    # Single team detail
    if team_filter and has_sim:
        row = df.iloc[0]
        log.info(f"\n{'=' * 40}")
        log.info(f"DETAIL: {row['team']}")
        log.info(f"{'=' * 40}")
        log.info(f"  Total WAR:        {row['total_war']:+.1f} "
                 f"(Hit: {row['hit_war']:+.1f}, Pitch: {row['pitch_war']:+.1f})")
        log.info(f"  Calibrated wins:  {row['calibrated_wins']:.1f}")
        log.info(f"  Regression adj:   {row['regression_adj']:+.1f}")
        log.info(f"  SOS adj:          {row['sos_adj']:+.1f}")
        log.info(f"  Monte Carlo mean: {row['mean_wins']:.1f} (std: {row['std_wins']:.1f})")
        log.info(f"  80% CI:           {row['p10_wins']} - {row['p90_wins']}")
        log.info(f"  90% CI:           {row['p5_wins']} - {row['p95_wins']}")
        log.info(f"  Division title:   {row.get('division_pct', 0):.1f}%")
        log.info(f"  Playoff:          {row.get('playoff_pct', 0):.1f}%")
        if has_vegas and pd.notna(row.get("vegas_wins")):
            log.info(f"  Vegas O/U:        {row['vegas_wins']:.1f}")
            log.info(f"  vs Vegas:         {row['vs_vegas_diff']:+.1f} "
                     f"{'** ' + row['value_flag'] + ' **' if row.get('value_flag') else ''}")


def save_results(team_proj: pd.DataFrame):
    """Save final projections to CSV."""
    # Select output columns (only include what exists)
    base_cols = [
        "team", "division", "league", "total_war", "hit_war", "pitch_war",
        "calibrated_wins", "regression_adj", "sos_adj",
    ]
    sim_cols = [
        "mean_wins", "median_wins", "std_wins",
        "p10_wins", "p90_wins", "p5_wins", "p95_wins",
        "division_pct", "playoff_pct",
    ]
    vegas_cols = ["vegas_wins", "vs_vegas_diff", "value_flag"]

    all_cols = base_cols + sim_cols + vegas_cols
    out_cols = [c for c in all_cols if c in team_proj.columns]

    sort_col = "mean_wins" if "mean_wins" in team_proj.columns else "calibrated_wins"
    output = team_proj[out_cols].sort_values(sort_col, ascending=False)

    output.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nSaved to {OUTPUT_CSV}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Floor 3: Season Win Total Projections"
    )
    parser.add_argument(
        "--phase", choices=["A", "B", "C", "AB", "ABC"],
        default="ABC",
        help="Which phases to run (default: ABC = full pipeline)"
    )
    parser.add_argument(
        "--sims", type=int, default=NUM_SIMS_DEFAULT,
        help=f"Number of Monte Carlo simulations (default: {NUM_SIMS_DEFAULT:,})"
    )
    parser.add_argument(
        "--team", type=str, default=None,
        help="Filter output to a single team (partial name match)"
    )
    args = parser.parse_args()

    phases = args.phase.upper()

    session = get_session()

    try:
        # Phase A is always needed (provides calibrated baseline)
        team_proj = run_phase_a(session)

        # Phase B: Monte Carlo simulation
        if "B" in phases:
            team_proj = run_phase_b(team_proj, n_sims=args.sims, session=session)

        # Phase C: Vegas benchmark
        if "C" in phases:
            team_proj = run_phase_c(team_proj)

        # Display and save
        display_results(team_proj, team_filter=args.team)
        save_results(team_proj)

    finally:
        session.close()


if __name__ == "__main__":
    main()
