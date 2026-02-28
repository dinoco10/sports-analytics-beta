"""
build_features.py — Feature Engineering Pipeline
=================================================
Takes raw game data from SQLite and produces a feature matrix CSV.
Each row = one game. Columns = features for home and away teams.

Based on Numeristical's proven approach (10 video series):
  Layer 1: Team hitting (OBP, SLG) — rolling windows
  Layer 2: Starting pitcher (K%, BB%, WHIP, FIP, mod SLG) — rolling starts
  Layer 3: Bullpen (ERA, K%, BB%, WHIP) — rolling team games
  Layer 4: Lineup-specific (avg OBP + SLG of starting 9) — rolling per-batter

Usage:
  python scripts/build_features.py                    # All seasons in DB
  python scripts/build_features.py --season 2025      # Single season output
  python scripts/build_features.py --summary          # Show what's available
  python scripts/build_features.py --skip-lineup       # Skip lineup (faster)

Author: Loko
"""

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DB_PATH = Path(__file__).parent.parent / "data" / "mlb_analytics.db"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "features"

# Rolling windows (proven optimal by Numeristical)
TEAM_WINDOWS = [14, 30]
SP_WINDOWS = [5, 10]
BP_WINDOWS = [10, 35]
BATTER_WINDOW = 30

FIP_CONSTANT = 3.15

# ═══════════════════════════════════════════════════════════════
# DEFAULTS — Position-aware regression for small samples
# ═══════════════════════════════════════════════════════════════

# {position: (BA, OBP, SLG)}
BATTER_DEFAULTS = {
    "P":   (0.100, 0.150, 0.120),
    "C":   (0.220, 0.290, 0.350),
    "SS":  (0.205, 0.285, 0.330),
    "2B":  (0.240, 0.310, 0.380),
    "3B":  (0.240, 0.310, 0.380),
    "DEFAULT": (0.255, 0.325, 0.420),
}

BATTER_MIN_PA = 50

SP_DEFAULT_ERA = 5.00
SP_DEFAULT_FIP = 4.80
SP_DEFAULT_K_PCT = 0.180
SP_DEFAULT_BB_PCT = 0.090
SP_DEFAULT_WHIP = 1.45
SP_MIN_IP = 20.0


# ═══════════════════════════════════════════════════════════════
# DATA LOADING — Matched to YOUR actual schema
# ═══════════════════════════════════════════════════════════════
#
# games: id, mlb_game_id, date, season, home_team_id, away_team_id,
#         ballpark_id, home_starter_id, away_starter_id, umpire_id,
#         home_score, away_score, winner_id, innings, temperature_f,
#         wind_speed_mph, wind_direction, is_dome, day_night,
#         home_rest_days, away_rest_days
#
# pitching_game_stats: id, game_id, player_id, team_id, date, ip,
#         hits, runs, earned_runs, walks, strikeouts, home_runs,
#         pitches, strikes, avg_exit_velo, barrel_pct, hard_hit_pct,
#         gb_pct, whiff_pct, chase_pct, swstr_pct
#
# hitting_game_stats: id, game_id, player_id, team_id, date,
#         plate_appearances, at_bats, hits, doubles, triples,
#         home_runs, rbi, runs, walks, strikeouts, stolen_bases,
#         avg_exit_velo, barrel_pct, hard_hit_pct, launch_angle, sprint_speed
#
# players: id, mlb_id, name, birth_date, bats, throws,
#          primary_position, current_team_id, active


def get_db_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}\n"
            f"Run backfill_games.py first to populate the database."
        )
    return sqlite3.connect(str(DB_PATH))


def load_games(conn):
    query = """
    SELECT
        id AS game_id,
        mlb_game_id,
        date,
        season,
        home_team_id,
        away_team_id,
        home_starter_id,
        away_starter_id,
        home_score,
        away_score,
        day_night,
        home_rest_days,
        away_rest_days,
        temperature_f,
        wind_speed_mph,
        wind_direction,
        is_dome,
        ballpark_id
    FROM games
    WHERE home_score IS NOT NULL
    ORDER BY date, id
    """
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_runs"] = df["home_score"] + df["away_score"]
    return df


def load_ballparks(conn):
    """Load ballpark data for park factor features."""
    query = """
    SELECT id, name, team_id, park_factor, park_factor_hr,
           park_factor_lhb, park_factor_rhb, elevation_ft, is_dome
    FROM ballparks
    """
    return pd.read_sql(query, conn)


def load_pitching_stats(conn):
    query = """
    SELECT 
        pgs.game_id,
        pgs.player_id,
        pgs.team_id,
        pgs.date,
        pgs.ip,
        pgs.hits,
        pgs.runs,
        pgs.earned_runs,
        pgs.walks,
        pgs.strikeouts,
        pgs.home_runs,
        pgs.pitches,
        g.season,
        g.home_team_id,
        g.away_team_id,
        g.home_starter_id,
        g.away_starter_id
    FROM pitching_game_stats pgs
    JOIN games g ON pgs.game_id = g.id
    WHERE g.home_score IS NOT NULL
    ORDER BY pgs.date, pgs.game_id
    """
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    
    # Convert IP to decimal (5.1 → 5.333)
    df["ip_decimal"] = df["ip"].apply(lambda x: _ip_to_decimal(x) if pd.notna(x) else 0.0)
    
    # Mark starters using games.home_starter_id / away_starter_id
    df["is_starter"] = (
        ((df["player_id"] == df["home_starter_id"]) & (df["team_id"] == df["home_team_id"])) |
        ((df["player_id"] == df["away_starter_id"]) & (df["team_id"] == df["away_team_id"]))
    )
    
    return df


def load_hitting_stats(conn):
    query = """
    SELECT 
        hgs.game_id,
        hgs.player_id,
        hgs.team_id,
        hgs.date,
        hgs.plate_appearances,
        hgs.at_bats,
        hgs.hits,
        hgs.doubles,
        hgs.triples,
        hgs.home_runs,
        hgs.walks,
        hgs.strikeouts,
        hgs.stolen_bases,
        g.season,
        g.home_team_id,
        g.away_team_id
    FROM hitting_game_stats hgs
    JOIN games g ON hgs.game_id = g.id
    WHERE g.home_score IS NOT NULL
    ORDER BY hgs.date, hgs.game_id
    """
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_player_positions(conn):
    query = "SELECT id, primary_position FROM players WHERE primary_position IS NOT NULL"
    try:
        return pd.read_sql(query, conn).set_index("id")["primary_position"].to_dict()
    except Exception:
        return {}


def _ip_to_decimal(ip_val):
    if pd.isna(ip_val):
        return 0.0
    ip_val = float(ip_val)
    whole = int(ip_val)
    fraction = ip_val - whole
    return whole + (fraction * 10 / 3)


# ═══════════════════════════════════════════════════════════════
# LAYER 1: TEAM-LEVEL ROLLING STATS
# ═══════════════════════════════════════════════════════════════

def compute_team_rolling(games_df, pitching_df, windows=TEAM_WINDOWS):
    """
    Rolling team stats: runs scored/allowed, Pythagorean W%, ERA, WHIP, K%.
    All features use shift(1) — only data BEFORE the game.
    """
    print("  Building team game logs...")
    
    # Aggregate pitching per team per game
    team_pitching = (
        pitching_df.groupby(["game_id", "team_id"])
        .agg(
            team_ip=("ip_decimal", "sum"),
            team_h_allowed=("hits", "sum"),
            team_bb_allowed=("walks", "sum"),
            team_so_pitching=("strikeouts", "sum"),
            team_hr_allowed=("home_runs", "sum"),
            team_er=("earned_runs", "sum"),
        )
        .reset_index()
    )
    
    team_games = []
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        date = game["date"]
        
        for side in ["home", "away"]:
            team = game["home_team_id"] if side == "home" else game["away_team_id"]
            runs_scored = game["home_score"] if side == "home" else game["away_score"]
            runs_allowed = game["away_score"] if side == "home" else game["home_score"]
            
            tp = team_pitching[
                (team_pitching["game_id"] == game_id) & 
                (team_pitching["team_id"] == team)
            ]
            
            team_games.append({
                "game_id": game_id,
                "date": date,
                "team_id": team,
                "side": side,
                "runs_scored": runs_scored,
                "runs_allowed": runs_allowed,
                "team_ip": tp["team_ip"].sum() if len(tp) > 0 else 9.0,
                "team_h_allowed": tp["team_h_allowed"].sum() if len(tp) > 0 else 0,
                "team_bb_allowed": tp["team_bb_allowed"].sum() if len(tp) > 0 else 0,
                "team_so_pitching": tp["team_so_pitching"].sum() if len(tp) > 0 else 0,
                "team_hr_allowed": tp["team_hr_allowed"].sum() if len(tp) > 0 else 0,
                "team_er": tp["team_er"].sum() if len(tp) > 0 else 0,
            })
    
    tg = pd.DataFrame(team_games).sort_values(["team_id", "date", "game_id"])
    
    print(f"  Computing rolling stats for {tg['team_id'].nunique()} teams...")
    
    all_features = []
    for team, group in tg.groupby("team_id"):
        group = group.reset_index(drop=True)
        
        for window in windows:
            sfx = f"_t{window}"
            
            group[f"rs{sfx}"] = (
                group["runs_scored"].shift(1)
                .rolling(window=window, min_periods=1).mean()
            )
            group[f"ra{sfx}"] = (
                group["runs_allowed"].shift(1)
                .rolling(window=window, min_periods=1).mean()
            )
            
            rs_sum = group["runs_scored"].shift(1).rolling(window=window, min_periods=1).sum()
            ra_sum = group["runs_allowed"].shift(1).rolling(window=window, min_periods=1).sum()
            group[f"pyth{sfx}"] = rs_sum**1.83 / (rs_sum**1.83 + ra_sum**1.83 + 1e-10)
            
            er_sum = group["team_er"].shift(1).rolling(window=window, min_periods=1).sum()
            ip_sum = group["team_ip"].shift(1).rolling(window=window, min_periods=1).sum()
            group[f"team_era{sfx}"] = (er_sum / (ip_sum + 1e-10)) * 9
            
            h_sum = group["team_h_allowed"].shift(1).rolling(window=window, min_periods=1).sum()
            bb_sum = group["team_bb_allowed"].shift(1).rolling(window=window, min_periods=1).sum()
            group[f"team_whip{sfx}"] = (h_sum + bb_sum) / (ip_sum + 1e-10)
            
            so_sum = group["team_so_pitching"].shift(1).rolling(window=window, min_periods=1).sum()
            bf_approx = ip_sum * 3 + h_sum + bb_sum
            group[f"team_k_pct{sfx}"] = so_sum / (bf_approx + 1e-10)
        
        all_features.append(group)
    
    result = pd.concat(all_features, ignore_index=True)
    
    feature_cols = ["game_id", "team_id", "side"] + [
        c for c in result.columns 
        if any(c.startswith(p) for p in ["rs_", "ra_", "pyth", "team_era", "team_whip", "team_k_pct"])
    ]
    return result[feature_cols]


# ═══════════════════════════════════════════════════════════════
# LAYER 2: STARTING PITCHER ROLLING STATS
# ═══════════════════════════════════════════════════════════════

def compute_pitcher_rolling(pitching_df, windows=SP_WINDOWS):
    """Rolling stats for each starting pitcher with regression for small samples."""
    starters = pitching_df[pitching_df["is_starter"]].copy()
    starters = starters.sort_values(["player_id", "date", "game_id"])
    
    print(f"  Found {len(starters)} starter appearances by {starters['player_id'].nunique()} pitchers")
    
    all_sp = []
    
    for player_id, group in starters.groupby("player_id"):
        group = group.reset_index(drop=True)
        
        for window in windows:
            sfx = f"_sp{window}"
            
            ip_roll = group["ip_decimal"].shift(1).rolling(window=window, min_periods=1).sum()
            er_roll = group["earned_runs"].shift(1).rolling(window=window, min_periods=1).sum()
            h_roll = group["hits"].shift(1).rolling(window=window, min_periods=1).sum()
            bb_roll = group["walks"].shift(1).rolling(window=window, min_periods=1).sum()
            so_roll = group["strikeouts"].shift(1).rolling(window=window, min_periods=1).sum()
            hr_roll = group["home_runs"].shift(1).rolling(window=window, min_periods=1).sum()
            
            bf_roll = ip_roll * 3 + h_roll + bb_roll
            
            # ERA (regressed)
            def _reg_era(ip, er):
                if pd.isna(ip) or ip < SP_MIN_IP:
                    aip = ip if pd.notna(ip) else 0
                    aer = er if pd.notna(er) else 0
                    der = (SP_DEFAULT_ERA / 9) * (SP_MIN_IP - aip)
                    return ((aer + der) / SP_MIN_IP) * 9
                return (er / ip) * 9
            
            group[f"era{sfx}"] = [_reg_era(ip, er) for ip, er in zip(ip_roll, er_roll)]
            
            # Rate stat regression helper
            def _reg_rate(num, denom, default, min_d=BATTER_MIN_PA):
                if pd.isna(denom) or denom < min_d:
                    ad = denom if pd.notna(denom) else 0
                    an = num if pd.notna(num) else 0
                    dn = default * (min_d - ad)
                    return (an + dn) / min_d
                return num / denom
            
            group[f"k_pct{sfx}"] = [_reg_rate(so, bf, SP_DEFAULT_K_PCT) for so, bf in zip(so_roll, bf_roll)]
            group[f"bb_pct{sfx}"] = [_reg_rate(bb, bf, SP_DEFAULT_BB_PCT) for bb, bf in zip(bb_roll, bf_roll)]
            
            # WHIP (regressed)
            def _reg_whip(h, bb, ip):
                if pd.isna(ip) or ip < SP_MIN_IP:
                    aip = ip if pd.notna(ip) else 0
                    ah = h if pd.notna(h) else 0
                    abb = bb if pd.notna(bb) else 0
                    dhbb = SP_DEFAULT_WHIP * (SP_MIN_IP - aip)
                    return (ah + abb + dhbb) / SP_MIN_IP
                return (h + bb) / ip
            
            group[f"whip{sfx}"] = [_reg_whip(h, bb, ip) for h, bb, ip in zip(h_roll, bb_roll, ip_roll)]
            
            # FIP (regressed)
            def _reg_fip(hr, bb, so, ip):
                if pd.isna(ip) or ip < SP_MIN_IP:
                    return SP_DEFAULT_FIP
                hr = hr if pd.notna(hr) else 0
                bb = bb if pd.notna(bb) else 0
                so = so if pd.notna(so) else 0
                return (13 * hr + 3 * bb - 2 * so) / ip + FIP_CONSTANT
            
            group[f"fip{sfx}"] = [_reg_fip(hr, bb, so, ip) for hr, bb, so, ip in zip(hr_roll, bb_roll, so_roll, ip_roll)]
            
            # IP per start
            n_starts = group["ip_decimal"].shift(1).expanding().count()
            group[f"ip_per_start{sfx}"] = ip_roll / (n_starts.clip(lower=1))
        
        all_sp.append(group)
    
    result = pd.concat(all_sp, ignore_index=True)
    
    feature_cols = ["game_id", "player_id", "team_id", "is_starter"] + [
        c for c in result.columns 
        if any(c.startswith(p) for p in ["era_sp", "k_pct_sp", "bb_pct_sp", "whip_sp", "fip_sp", "ip_per_start"])
    ]
    return result[feature_cols]


# ═══════════════════════════════════════════════════════════════
# LAYER 2b: STARTING PITCHER ROLLING GAME SCORE (538 rGS)
# ═══════════════════════════════════════════════════════════════
#
# FiveThirtyEight's single largest pregame signal. Game Score formula
# from Tangotiger (modified): captures SP quality in one number.
# We compute per-pitcher rolling avg game score, then team avg,
# and feed both into game features as home_sp_rgs, away_sp_rgs, diff_sp_rgs.

def game_score(ip, k, bb, h, r, hr):
    """538-style Game Score for a single start.

    Formula: 47.4 + (outs * 1.5) + K - (BB * 2) - (H * 2) - (R * 3) - (HR * 4)
    Average is ~50; elite aces sustain 60-65+.
    """
    # ip is decimal (6.1 = 6 and 1/3 innings)
    full_innings = int(ip) if pd.notna(ip) else 0
    partial = round((ip % 1) * 10) if pd.notna(ip) else 0
    outs = full_innings * 3 + partial

    k = k if pd.notna(k) else 0
    bb = bb if pd.notna(bb) else 0
    h = h if pd.notna(h) else 0
    r = r if pd.notna(r) else 0
    hr = hr if pd.notna(hr) else 0

    return 47.4 + (outs * 1.5) + k - (bb * 2) - (h * 2) - (r * 3) - (hr * 4)


SP_RGS_DEFAULT = 50.0   # League-average game score
SP_RGS_MIN_STARTS = 3   # Minimum starts before trusting raw rGS
SP_RGS_WINDOW = 10       # Rolling window (starts, not days)


def compute_sp_game_score(pitching_df, games_df):
    """Compute rolling Game Score (rGS) per starting pitcher, plus team avg rGS.

    Returns DataFrame with columns: game_id, home_sp_rgs, away_sp_rgs, diff_sp_rgs,
    home_team_rgs, away_team_rgs (team averages for pitcher_adj calculation).

    The 538 pitcher adjustment is: 4.7 * (pitcher_rGS - team_rGS).
    An ace 10 GS above team avg adds 47 Elo points (~7% win prob shift).
    """
    starters = pitching_df[pitching_df["is_starter"]].copy()
    starters = starters.sort_values(["player_id", "date", "game_id"])

    print(f"  Computing Game Score for {starters['player_id'].nunique()} starters...")

    # Step 1: Compute raw game score for each start
    starters["gs_raw"] = [
        game_score(ip, k, bb, h, r, hr)
        for ip, k, bb, h, r, hr in zip(
            starters["ip_decimal"], starters["strikeouts"], starters["walks"],
            starters["hits"], starters["runs"], starters["home_runs"]
        )
    ]

    # Step 2: Rolling average per pitcher (shifted so it's pre-game, not post-game)
    all_sp = []
    for player_id, group in starters.groupby("player_id"):
        group = group.reset_index(drop=True)

        # shift(1) makes it purely pre-game (uses only prior starts)
        rolled = group["gs_raw"].shift(1).rolling(
            window=SP_RGS_WINDOW, min_periods=1
        ).mean()

        # Count prior starts for regression
        n_prior = group["gs_raw"].shift(1).expanding().count()

        # Regress toward league average when few starts available
        # Weight: n_prior / (n_prior + SP_RGS_MIN_STARTS)
        weight = n_prior / (n_prior + SP_RGS_MIN_STARTS)
        group["sp_rgs"] = weight * rolled + (1 - weight) * SP_RGS_DEFAULT

        all_sp.append(group[["game_id", "player_id", "team_id", "date", "sp_rgs", "gs_raw"]])

    sp_rgs = pd.concat(all_sp, ignore_index=True)

    # Step 3: Team average rGS (rolling avg of all starters on the team)
    # This is needed for the pitcher_adj formula: 4.7 * (pitcher_rGS - team_rGS)
    sp_rgs = sp_rgs.sort_values(["team_id", "date", "game_id"])

    team_rgs_parts = []
    for team_id, tgroup in sp_rgs.groupby("team_id"):
        tgroup = tgroup.reset_index(drop=True)
        # Team avg rGS from last N starts across all SPs (shifted pre-game)
        team_rolled = tgroup["gs_raw"].shift(1).rolling(
            window=SP_RGS_WINDOW * 5, min_periods=3  # ~50 team starts = full rotation
        ).mean()

        n_team = tgroup["gs_raw"].shift(1).expanding().count()
        tw = n_team / (n_team + 10)  # Regress more heavily for team avg
        tgroup["team_rgs"] = tw * team_rolled + (1 - tw) * SP_RGS_DEFAULT

        team_rgs_parts.append(tgroup[["game_id", "player_id", "team_id", "sp_rgs", "team_rgs"]])

    sp_rgs_full = pd.concat(team_rgs_parts, ignore_index=True)

    # Step 4: Map to game-level features (home/away)
    sp_rgs_full = sp_rgs_full.merge(
        games_df[["game_id", "home_team_id", "away_team_id"]],
        on="game_id", how="left"
    )

    # Home SP rGS
    home_sp = sp_rgs_full[sp_rgs_full["team_id"] == sp_rgs_full["home_team_id"]].copy()
    home_sp = home_sp.drop_duplicates(subset=["game_id"], keep="first")
    home_sp = home_sp.rename(columns={
        "sp_rgs": "home_sp_rgs",
        "team_rgs": "home_team_rgs"
    })

    # Away SP rGS
    away_sp = sp_rgs_full[sp_rgs_full["team_id"] == sp_rgs_full["away_team_id"]].copy()
    away_sp = away_sp.drop_duplicates(subset=["game_id"], keep="first")
    away_sp = away_sp.rename(columns={
        "sp_rgs": "away_sp_rgs",
        "team_rgs": "away_team_rgs"
    })

    # Merge home + away into game-level DF
    result = games_df[["game_id"]].drop_duplicates().merge(
        home_sp[["game_id", "home_sp_rgs", "home_team_rgs"]],
        on="game_id", how="left"
    ).merge(
        away_sp[["game_id", "away_sp_rgs", "away_team_rgs"]],
        on="game_id", how="left"
    )

    # Fill missing (openers, unknown starters) with league avg
    for col in ["home_sp_rgs", "away_sp_rgs", "home_team_rgs", "away_team_rgs"]:
        result[col] = result[col].fillna(SP_RGS_DEFAULT)

    # Diff features
    result["diff_sp_rgs"] = result["home_sp_rgs"] - result["away_sp_rgs"]

    # 538 pitcher adjustment: how much this SP lifts/drags team rating
    # pitcher_adj = 4.7 * (pitcher_rGS - team_rGS)
    result["home_sp_rgs_adj"] = 4.7 * (result["home_sp_rgs"] - result["home_team_rgs"])
    result["away_sp_rgs_adj"] = 4.7 * (result["away_sp_rgs"] - result["away_team_rgs"])
    result["diff_sp_rgs_adj"] = result["home_sp_rgs_adj"] - result["away_sp_rgs_adj"]

    coverage = result["home_sp_rgs"].notna().sum()
    print(f"  Game Score features: {coverage}/{len(result)} games covered")
    print(f"  Home SP rGS: mean={result['home_sp_rgs'].mean():.1f}, std={result['home_sp_rgs'].std():.1f}")
    print(f"  Away SP rGS: mean={result['away_sp_rgs'].mean():.1f}, std={result['away_sp_rgs'].std():.1f}")
    print(f"  SP adj range: [{result['home_sp_rgs_adj'].min():.1f}, {result['home_sp_rgs_adj'].max():.1f}] Elo points")

    return result


# ═══════════════════════════════════════════════════════════════
# LAYER 3: BULLPEN (team total minus starter)
# ═══════════════════════════════════════════════════════════════

def compute_bullpen_rolling(pitching_df, games_df, windows=BP_WINDOWS):
    """Bullpen = everything left over after the starter."""
    
    team_total = (
        pitching_df.groupby(["game_id", "team_id"])
        .agg(
            total_ip=("ip_decimal", "sum"),
            total_er=("earned_runs", "sum"),
            total_h=("hits", "sum"),
            total_bb=("walks", "sum"),
            total_so=("strikeouts", "sum"),
        )
        .reset_index()
    )
    
    starters = pitching_df[pitching_df["is_starter"]].copy()
    starter_stats = (
        starters.groupby(["game_id", "team_id"])
        .agg(
            sp_ip=("ip_decimal", "sum"),
            sp_er=("earned_runs", "sum"),
            sp_h=("hits", "sum"),
            sp_bb=("walks", "sum"),
            sp_so=("strikeouts", "sum"),
        )
        .reset_index()
    )
    
    bp = team_total.merge(starter_stats, on=["game_id", "team_id"], how="left")
    bp["bp_ip"] = bp["total_ip"] - bp["sp_ip"].fillna(0)
    bp["bp_er"] = bp["total_er"] - bp["sp_er"].fillna(0)
    bp["bp_h"] = bp["total_h"] - bp["sp_h"].fillna(0)
    bp["bp_bb"] = bp["total_bb"] - bp["sp_bb"].fillna(0)
    bp["bp_so"] = bp["total_so"] - bp["sp_so"].fillna(0)
    
    bp = bp.merge(games_df[["game_id", "date"]], on="game_id", how="left")
    bp = bp.sort_values(["team_id", "date", "game_id"])
    
    print(f"  Computing bullpen rolling for {bp['team_id'].nunique()} teams...")
    
    all_bp = []
    for team, group in bp.groupby("team_id"):
        group = group.reset_index(drop=True)
        
        for window in windows:
            sfx = f"_bp{window}"
            
            ip_roll = group["bp_ip"].shift(1).rolling(window=window, min_periods=1).sum()
            er_roll = group["bp_er"].shift(1).rolling(window=window, min_periods=1).sum()
            h_roll = group["bp_h"].shift(1).rolling(window=window, min_periods=1).sum()
            bb_roll = group["bp_bb"].shift(1).rolling(window=window, min_periods=1).sum()
            so_roll = group["bp_so"].shift(1).rolling(window=window, min_periods=1).sum()
            
            group[f"bp_era{sfx}"] = (er_roll / (ip_roll + 1e-10)) * 9
            group[f"bp_whip{sfx}"] = (h_roll + bb_roll) / (ip_roll + 1e-10)
            
            bf_approx = ip_roll * 3 + h_roll + bb_roll
            group[f"bp_k_pct{sfx}"] = so_roll / (bf_approx + 1e-10)
            group[f"bp_bb_pct{sfx}"] = bb_roll / (bf_approx + 1e-10)
        
        all_bp.append(group)
    
    result = pd.concat(all_bp, ignore_index=True)
    
    feature_cols = ["game_id", "team_id"] + [
        c for c in result.columns 
        if any(c.startswith(p) for p in ["bp_era", "bp_whip", "bp_k_pct", "bp_bb_pct"])
    ]
    return result[feature_cols]


# ═══════════════════════════════════════════════════════════════
# LAYER 3b: BULLPEN AVAILABILITY (fatigue / workload)
# ═══════════════════════════════════════════════════════════════

def compute_bullpen_availability(pitching_df, games_df):
    """
    Compute bullpen workload/availability features.

    For each team before each game, compute:
    - bp_ip_last1/2/3: total reliever IP in last 1/2/3 team games
    - bp_pitches_last1/2/3: total reliever pitches in last 1/2/3 team games
    - bp_high_lev_unavail: number of top relievers (by IP) who pitched back-to-back

    These capture whether a team's bullpen is fresh or gassed —
    information that no projection or rolling average captures.
    """
    # Identify relievers: anyone who pitched but wasn't the starter
    relievers = pitching_df[~pitching_df["is_starter"]].copy()
    if relievers.empty:
        return pd.DataFrame()

    # Aggregate reliever workload per team per game
    bp_game = (
        relievers.groupby(["game_id", "team_id"])
        .agg(
            bp_ip=("ip_decimal", "sum"),
            bp_pitches=("pitches", "sum"),
            n_relievers=("player_id", "nunique"),
        )
        .reset_index()
    )

    # Merge dates
    bp_game = bp_game.merge(
        games_df[["game_id", "date", "season"]],
        on="game_id", how="left"
    )
    bp_game = bp_game.sort_values(["team_id", "date", "game_id"])

    # Compute rolling workload over last 1/2/3 games (shift(1) = pre-game)
    results = []
    for team_id, group in bp_game.groupby("team_id"):
        group = group.copy()

        for window in [1, 2, 3]:
            group[f"bp_ip_last{window}"] = (
                group["bp_ip"].shift(1).rolling(window, min_periods=1).sum()
            )
            group[f"bp_pitches_last{window}"] = (
                group["bp_pitches"].shift(1).rolling(window, min_periods=1).sum()
            )
            group[f"bp_relievers_last{window}"] = (
                group["n_relievers"].shift(1).rolling(window, min_periods=1).sum()
            )

        results.append(group)

    bp_avail = pd.concat(results, ignore_index=True)

    # Now build game-level features (home and away)
    game_results = []
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        row = {"game_id": game_id}

        for side in ["home", "away"]:
            team_id = game[f"{side}_team_id"]
            team_bp = bp_avail[
                (bp_avail["game_id"] == game_id) & (bp_avail["team_id"] == team_id)
            ]

            if len(team_bp) > 0:
                t = team_bp.iloc[0]
                for window in [1, 2, 3]:
                    row[f"{side}_bp_ip_last{window}"] = t.get(f"bp_ip_last{window}", np.nan)
                    row[f"{side}_bp_pitches_last{window}"] = t.get(f"bp_pitches_last{window}", np.nan)
            else:
                for window in [1, 2, 3]:
                    row[f"{side}_bp_ip_last{window}"] = np.nan
                    row[f"{side}_bp_pitches_last{window}"] = np.nan

        # Diff features (higher = home bullpen more fatigued)
        for window in [1, 2, 3]:
            h_ip = row.get(f"home_bp_ip_last{window}", 0) or 0
            a_ip = row.get(f"away_bp_ip_last{window}", 0) or 0
            row[f"diff_bp_fatigue_last{window}"] = h_ip - a_ip

        game_results.append(row)

    result_df = pd.DataFrame(game_results)

    # Stats
    non_null = result_df["home_bp_ip_last1"].notna().sum()
    print(f"  Bullpen availability for {non_null}/{len(result_df)} games")

    return result_df


# ═══════════════════════════════════════════════════════════════
# LAYER 4: LINEUP FEATURES (avg OBP + SLG of hitters that game)
# ═══════════════════════════════════════════════════════════════

def compute_lineup_features(hitting_df, games_df, player_positions, window=BATTER_WINDOW):
    """
    Average OBP and SLG of the hitters who appeared in each game.
    
    Note: schema doesn't have batting_order, so we use ALL hitters
    who got plate appearances. This is close enough — Numeristical
    proved simple average > weighted > individual anyway.
    """
    hitters = hitting_df.copy()
    hitters = hitters.sort_values(["player_id", "date", "game_id"])
    
    print(f"  Processing {hitters['player_id'].nunique()} batters...")
    
    # --- Step 1: Compute each batter's trailing stats per game ---
    batter_features = {}
    
    for player_id, group in hitters.groupby("player_id"):
        group = group.reset_index(drop=True)
        
        pos = player_positions.get(player_id, "DEFAULT")
        if pos not in BATTER_DEFAULTS:
            pos = "DEFAULT"
        default_ba, default_obp, default_slg = BATTER_DEFAULTS[pos]
        
        ab_roll = group["at_bats"].shift(1).rolling(window=window, min_periods=1).sum()
        h_roll = group["hits"].shift(1).rolling(window=window, min_periods=1).sum()
        bb_roll = group["walks"].shift(1).rolling(window=window, min_periods=1).sum()
        hr_roll = group["home_runs"].shift(1).rolling(window=window, min_periods=1).sum()
        doubles_roll = group["doubles"].shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
        triples_roll = group["triples"].shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
        pa_roll = group["plate_appearances"].shift(1).rolling(window=window, min_periods=1).sum()
        
        singles_roll = h_roll - doubles_roll - triples_roll - hr_roll
        tb_roll = singles_roll + 2 * doubles_roll + 3 * triples_roll + 4 * hr_roll
        
        for i in range(len(group)):
            game_id = group.iloc[i]["game_id"]
            pa = pa_roll.iloc[i] if i < len(pa_roll) and pd.notna(pa_roll.iloc[i]) else 0
            ab = ab_roll.iloc[i] if i < len(ab_roll) and pd.notna(ab_roll.iloc[i]) else 0
            
            if pa == 0:
                obp = default_obp
                slg = default_slg
            else:
                if pa < BATTER_MIN_PA:
                    h_val = h_roll.iloc[i] if pd.notna(h_roll.iloc[i]) else 0
                    bb_val = bb_roll.iloc[i] if pd.notna(bb_roll.iloc[i]) else 0
                    actual_obh = h_val + bb_val
                    default_obh = default_obp * (BATTER_MIN_PA - pa)
                    obp = (actual_obh + default_obh) / BATTER_MIN_PA
                else:
                    h_val = h_roll.iloc[i] if pd.notna(h_roll.iloc[i]) else 0
                    bb_val = bb_roll.iloc[i] if pd.notna(bb_roll.iloc[i]) else 0
                    obp = (h_val + bb_val) / pa
                
                if ab < BATTER_MIN_PA:
                    actual_tb = tb_roll.iloc[i] if pd.notna(tb_roll.iloc[i]) else 0
                    default_tb = default_slg * (BATTER_MIN_PA - ab)
                    slg = (actual_tb + default_tb) / BATTER_MIN_PA
                else:
                    tb = tb_roll.iloc[i] if pd.notna(tb_roll.iloc[i]) else 0
                    slg = tb / ab if ab > 0 else default_slg
            
            batter_features[(player_id, game_id)] = {"obp": obp, "slg": slg}
    
    print(f"  Computed trailing stats for {len(batter_features)} batter-game pairs")
    
    # --- Step 2: Average across each game's hitters ---
    lineup_results = []
    
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        
        for side in ["home", "away"]:
            team = game["home_team_id"] if side == "home" else game["away_team_id"]
            
            game_hitters = hitting_df[
                (hitting_df["game_id"] == game_id) & 
                (hitting_df["team_id"] == team)
            ]
            
            if len(game_hitters) == 0:
                lineup_results.append({
                    "game_id": game_id, "team_id": team, "side": side,
                    "lineup_obp": np.nan, "lineup_slg": np.nan,
                })
                continue
            
            obps = []
            slgs = []
            for _, hitter in game_hitters.iterrows():
                key = (hitter["player_id"], game_id)
                if key in batter_features:
                    obps.append(batter_features[key]["obp"])
                    slgs.append(batter_features[key]["slg"])
            
            lineup_results.append({
                "game_id": game_id,
                "team_id": team,
                "side": side,
                "lineup_obp": np.mean(obps) if obps else np.nan,
                "lineup_slg": np.mean(slgs) if slgs else np.nan,
            })
    
    return pd.DataFrame(lineup_results)


# ═══════════════════════════════════════════════════════════════
# LAYER 6c: HOME/AWAY SPLITS (venue-specific rolling stats)
# ═══════════════════════════════════════════════════════════════
#
# Current rolling stats mix home and away games.  Some teams play
# significantly differently at home vs on the road (park effects,
# comfort, travel fatigue).  This layer computes venue-specific
# rolling win%, RS, and RA using only home-only or away-only games.

SPLIT_WINDOW = 30  # last 30 venue-specific games (~half a season)

def compute_home_away_splits(games_df):
    """
    For each team, compute rolling stats using ONLY their home games
    or ONLY their away games. This captures venue-specific performance.

    Output columns (per game):
    - home_venue_wpct: home team's win% in their recent home games
    - away_venue_wpct: away team's win% in their recent away games
    - home_venue_rs / home_venue_ra: runs at home
    - away_venue_rs / away_venue_ra: runs on road
    - diff_venue_wpct: home_venue_wpct - away_venue_wpct
    """
    # Build team game logs split by venue
    rows = []
    for _, game in games_df.iterrows():
        for side in ["home", "away"]:
            team_id = game[f"{side}_team_id"]
            won = (game["home_win"] == 1) if side == "home" else (game["home_win"] == 0)
            rs = game["home_score"] if side == "home" else game["away_score"]
            ra = game["away_score"] if side == "home" else game["home_score"]
            rows.append({
                "game_id": game["game_id"],
                "date": game["date"],
                "season": game["season"],
                "team_id": team_id,
                "side": side,
                "won": int(won),
                "rs": rs,
                "ra": ra,
            })

    tg = pd.DataFrame(rows)

    results = []
    for side in ["home", "away"]:
        # Filter to only games where this team played in this venue role
        venue_games = tg[tg["side"] == side].copy()
        venue_games = venue_games.sort_values(["team_id", "season", "date", "game_id"])

        all_venue = []
        for team_id, group in venue_games.groupby("team_id"):
            group = group.reset_index(drop=True)

            # Rolling stats using only venue-specific games, shifted by 1
            group[f"{side}_venue_wpct"] = (
                group["won"].shift(1)
                .rolling(window=SPLIT_WINDOW, min_periods=3).mean()
            )
            group[f"{side}_venue_rs"] = (
                group["rs"].shift(1)
                .rolling(window=SPLIT_WINDOW, min_periods=3).mean()
            )
            group[f"{side}_venue_ra"] = (
                group["ra"].shift(1)
                .rolling(window=SPLIT_WINDOW, min_periods=3).mean()
            )

            all_venue.append(group[["game_id", f"{side}_venue_wpct",
                                     f"{side}_venue_rs", f"{side}_venue_ra"]])

        results.append(pd.concat(all_venue, ignore_index=True))

    # Merge home-side and away-side results
    result = results[0].merge(results[1], on="game_id", how="outer")

    # Diff: home team's home form vs away team's road form
    result["diff_venue_wpct"] = result["home_venue_wpct"] - result["away_venue_wpct"]
    result["diff_venue_rs"] = result["home_venue_rs"] - result["away_venue_rs"]

    n_computed = result["home_venue_wpct"].notna().sum()
    print(f"  Computed venue splits for {n_computed}/{len(result)} games "
          f"(min 3 venue-specific games required)")

    return result


# ═══════════════════════════════════════════════════════════════
# LAYER 8: FEATURE GAPS (defensive efficiency, barrel rate)
# ═══════════════════════════════════════════════════════════════
#
# These features fill known gaps identified by competitor model research:
# 1. Team defensive efficiency (1 - BABIP allowed) — only major category missing
# 2. Team rolling barrel rate — contact quality distinct from SLG

DEFENSE_WINDOW = 30  # 30 games rolling

def compute_defensive_efficiency(pitching_df, games_df, hitting_df):
    """
    Compute team defensive efficiency: 1 - BABIP allowed.

    BABIP = (H - HR) / (BF - K - HR - SF)
    Defensive efficiency = 1 - BABIP = fraction of BIP converted to outs.

    This is the only major feature category completely missing from our model.
    Better defenses convert more balls in play to outs → fewer runs.
    """
    # Aggregate team pitching per game
    team_pitch = (
        pitching_df.groupby(["game_id", "team_id"])
        .agg(
            team_h=("hits", "sum"),
            team_hr=("home_runs", "sum"),
            team_so=("strikeouts", "sum"),
            team_ip=("ip_decimal", "sum"),
            team_bb=("walks", "sum"),
        )
        .reset_index()
    )

    # Approximate BF from IP and components
    team_pitch["bf_approx"] = team_pitch["team_ip"] * 3 + team_pitch["team_h"] + team_pitch["team_bb"]

    # BABIP = (H - HR) / (BF - K - HR)
    denom = team_pitch["bf_approx"] - team_pitch["team_so"] - team_pitch["team_hr"]
    team_pitch["babip"] = (team_pitch["team_h"] - team_pitch["team_hr"]) / denom.clip(lower=1)
    team_pitch["babip"] = team_pitch["babip"].clip(0, 1)

    # Merge dates
    team_pitch = team_pitch.merge(
        games_df[["game_id", "date"]], on="game_id", how="left"
    )
    team_pitch = team_pitch.sort_values(["team_id", "date", "game_id"])

    # Rolling BABIP → defensive efficiency
    results = []
    for team_id, group in team_pitch.groupby("team_id"):
        group = group.reset_index(drop=True)

        rolling_babip = (
            group["babip"].shift(1)
            .rolling(window=DEFENSE_WINDOW, min_periods=5).mean()
        )
        group["def_eff"] = 1.0 - rolling_babip

        results.append(group[["game_id", "team_id", "def_eff"]])

    def_df = pd.concat(results, ignore_index=True)

    # Pivot to game-level (home/away)
    game_results = []
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        row = {"game_id": game_id}

        for side in ["home", "away"]:
            team_id = game[f"{side}_team_id"]
            team_def = def_df[(def_df["game_id"] == game_id) & (def_df["team_id"] == team_id)]

            if len(team_def) > 0:
                row[f"{side}_def_eff"] = team_def.iloc[0]["def_eff"]
            else:
                row[f"{side}_def_eff"] = np.nan

        row["diff_def_eff"] = (row.get("home_def_eff") or 0) - (row.get("away_def_eff") or 0)
        game_results.append(row)

    result_df = pd.DataFrame(game_results)
    n_valid = result_df["home_def_eff"].notna().sum()
    print(f"  Defensive efficiency for {n_valid}/{len(result_df)} games")
    return result_df


def compute_team_barrel_rate(hitting_df, games_df):
    """
    Compute team rolling barrel rate as a contact quality signal.

    Barrel rate captures hard contact quality distinct from SLG.
    Source: hitting_game_stats.barrel_pct (where available from Statcast).

    Note: barrel_pct is per-player per-game. We need team averages
    weighted by plate appearances.
    """
    # Check if barrel_pct exists and is populated
    if "barrel_pct" not in hitting_df.columns:
        print("  barrel_pct not in hitting stats — skipping")
        return pd.DataFrame()

    # Filter to rows with barrel data
    with_barrel = hitting_df[hitting_df.get("barrel_pct", pd.Series()).notna()].copy()
    if len(with_barrel) < 100:
        print(f"  Only {len(with_barrel)} games with barrel data — skipping")
        return pd.DataFrame()

    # Team average barrel rate per game (PA-weighted)
    team_barrel = (
        with_barrel.groupby(["game_id", "team_id"])
        .apply(lambda g: np.average(g["barrel_pct"], weights=g.get("plate_appearances", 1))
               if len(g) > 0 else np.nan, include_groups=False)
        .reset_index(name="team_barrel_pct")
    )

    team_barrel = team_barrel.merge(
        games_df[["game_id", "date"]], on="game_id", how="left"
    )
    team_barrel = team_barrel.sort_values(["team_id", "date", "game_id"])

    results = []
    for team_id, group in team_barrel.groupby("team_id"):
        group = group.reset_index(drop=True)

        rolling = (
            group["team_barrel_pct"].shift(1)
            .rolling(window=14, min_periods=3).mean()
        )
        group["team_barrel_t14"] = rolling
        results.append(group[["game_id", "team_id", "team_barrel_t14"]])

    barrel_df = pd.concat(results, ignore_index=True)

    # Pivot to game-level
    game_results = []
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        row = {"game_id": game_id}

        for side in ["home", "away"]:
            team_id = game[f"{side}_team_id"]
            tb = barrel_df[(barrel_df["game_id"] == game_id) & (barrel_df["team_id"] == team_id)]

            if len(tb) > 0:
                row[f"{side}_team_barrel_t14"] = tb.iloc[0]["team_barrel_t14"]
            else:
                row[f"{side}_team_barrel_t14"] = np.nan

        hb = row.get("home_team_barrel_t14")
        ab = row.get("away_team_barrel_t14")
        row["diff_team_barrel_t14"] = (hb if hb else 0) - (ab if ab else 0)
        game_results.append(row)

    result_df = pd.DataFrame(game_results)
    n_valid = result_df["home_team_barrel_t14"].notna().sum()
    print(f"  Team barrel rate for {n_valid}/{len(result_df)} games")
    return result_df


# ═══════════════════════════════════════════════════════════════
# LAYER 6a: REST DAYS (computed from game dates)
# ═══════════════════════════════════════════════════════════════
#
# games.home_rest_days / away_rest_days are always NULL in the DB,
# so we compute them here from the actual game schedule.
# For each team, rest_days = (game_date - previous_game_date).days - 1
# Doubleheaders (same date) = 0 rest.  Season openers = NaN (median fill).
# Clipped to [0, 7] to cap long breaks (ASB, postponements).

def compute_rest_days(games_df):
    """
    Compute rest days for each team in each game from the schedule.

    Strategy:
    1. Explode games into 2 rows per game (one for each team)
    2. Sort by (team_id, season, date) chronologically
    3. Compute days between consecutive games per team per season
    4. Pivot back to game-level (home_rest_days, away_rest_days, diff)
    """
    # Build team schedule: 2 rows per game
    rows = []
    for _, game in games_df.iterrows():
        for side in ["home", "away"]:
            rows.append({
                "game_id": game["game_id"],
                "date": game["date"],
                "season": game["season"],
                "team_id": game[f"{side}_team_id"],
                "side": side,
            })

    schedule = pd.DataFrame(rows)
    schedule = schedule.sort_values(["team_id", "season", "date", "game_id"])

    # Compute rest days per team per season
    schedule["prev_date"] = schedule.groupby(["team_id", "season"])["date"].shift(1)
    schedule["rest_days"] = (schedule["date"] - schedule["prev_date"]).dt.days - 1

    # Season openers have no prev_date → NaN (will be median-filled later)
    # Doubleheaders: same date → -1 days, clip to 0
    schedule["rest_days"] = schedule["rest_days"].clip(lower=0, upper=7)

    # Pivot back to game-level
    home_rest = schedule[schedule["side"] == "home"][["game_id", "rest_days"]].rename(
        columns={"rest_days": "home_rest_days"}
    )
    away_rest = schedule[schedule["side"] == "away"][["game_id", "rest_days"]].rename(
        columns={"rest_days": "away_rest_days"}
    )

    result = home_rest.merge(away_rest, on="game_id", how="outer")
    result["diff_rest_days"] = result["home_rest_days"] - result["away_rest_days"]

    n_computed = result["home_rest_days"].notna().sum()
    print(f"  Computed rest days for {n_computed}/{len(result)} games "
          f"(season openers = NaN, median-filled later)")

    return result


# ═══════════════════════════════════════════════════════════════
# LAYER 6b: HANDEDNESS MATCHUPS (platoon advantage)
# ═══════════════════════════════════════════════════════════════
#
# Platoon advantage: batters who have the "opposite hand" vs the SP.
# SP throws R → LHB and Switch hitters have advantage.
# SP throws L → RHB and Switch hitters have advantage.
# Output is a 0.0-1.0 scale = fraction of lineup with platoon advantage.

def compute_handedness_features(games_df, hitting_df, conn):
    """
    For each game, compute what fraction of each team's lineup
    has platoon advantage against the opposing starting pitcher.

    Platoon advantage rules:
    - SP throws R → advantage for L and S batters
    - SP throws L → advantage for R and S batters
    - SP throws unknown → default 0.50
    """
    # Load player bats/throws from DB
    player_info = pd.read_sql(
        "SELECT id, bats, throws FROM players", conn
    )
    bats_map = dict(zip(player_info["id"], player_info["bats"]))    # player_id → 'R'/'L'/'S'/None
    throws_map = dict(zip(player_info["id"], player_info["throws"]))  # player_id → 'R'/'L'/None

    # Build game_id → {side: [player_ids]} from hitting_df
    game_lineups = {}
    for _, row in hitting_df.iterrows():
        gid = row["game_id"]
        tid = row["team_id"]
        pid = row["player_id"]
        if gid not in game_lineups:
            game_lineups[gid] = {}
        if tid not in game_lineups[gid]:
            game_lineups[gid][tid] = []
        game_lineups[gid][tid].append(pid)

    results = []
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        row = {"game_id": game_id}

        for side, opp_side in [("home", "away"), ("away", "home")]:
            team_id = game[f"{side}_team_id"]
            opp_starter_id = game.get(f"{opp_side}_starter_id")

            # Get opposing SP's throwing hand
            sp_throws = None
            if pd.notna(opp_starter_id):
                sp_throws = throws_map.get(int(opp_starter_id))

            if sp_throws is None:
                # Unknown SP hand → default 0.50
                row[f"{side}_platoon_adv"] = 0.50
                continue

            # Count hitters with platoon advantage
            hitter_pids = game_lineups.get(game_id, {}).get(team_id, [])

            if not hitter_pids:
                row[f"{side}_platoon_adv"] = 0.50
                continue

            adv_count = 0
            total_count = 0
            for pid in hitter_pids:
                bat_hand = bats_map.get(pid)
                if bat_hand is None:
                    continue
                total_count += 1
                # Switch hitters always have advantage
                if bat_hand == "S":
                    adv_count += 1
                # LHB vs RHP or RHB vs LHP
                elif (sp_throws == "R" and bat_hand == "L") or \
                     (sp_throws == "L" and bat_hand == "R"):
                    adv_count += 1

            row[f"{side}_platoon_adv"] = adv_count / total_count if total_count > 0 else 0.50

        row["diff_platoon_adv"] = row.get("home_platoon_adv", 0.5) - row.get("away_platoon_adv", 0.5)
        results.append(row)

    result_df = pd.DataFrame(results)

    avg_adv = result_df["home_platoon_adv"].mean()
    print(f"  Computed platoon advantage for {len(result_df)} games "
          f"(avg home platoon adv: {avg_adv:.3f})")

    return result_df


# ═══════════════════════════════════════════════════════════════
# LAYER 5: PROJECTION-BASED FEATURES
# ═══════════════════════════════════════════════════════════════
#
# These features use static preseason projections (Marcel + Statcast)
# as a player quality signal. The model learns: "games with better-projected
# pitchers and lineups tend to be won more often."
#
# This is standard practice — PECOTA, Steamer, ZiPS all use preseason
# projections as game-level features. The projection captures long-term
# player quality that rolling stats miss (especially early in the season).

def load_projection_maps():
    """
    Load Marcel projection snapshots for each season into lookup dicts.

    Uses frozen snapshots from data/features/snapshots/ to prevent lookahead
    bias: 2023 games get 2023 projections (built from 2021-2022 data only),
    2024 games get 2024 projections, etc.

    Falls back to the production 2026 CSVs for any season without a snapshot.

    Returns:
        pitcher_maps: {season: {mlb_player_id: {...}}}
        hitter_maps:  {season: {mlb_player_id: {...}}}
    """
    snapshot_dir = OUTPUT_DIR / "snapshots"
    proj_dir = OUTPUT_DIR

    pitcher_maps = {}
    hitter_maps = {}

    # Snapshot seasons: each game year maps to its projection year
    # Games in 2023 use the 2023 Marcel snapshot (built from 2021-2022 data)
    snapshot_years = [2023, 2024, 2025, 2026]

    for year in snapshot_years:
        pitcher_map = {}
        hitter_map = {}

        # Try snapshot first, fall back to production CSVs for 2026
        p_path = snapshot_dir / f"marcel_pitchers_{year}.csv"
        if not p_path.exists() and year == 2026:
            p_path = proj_dir / "pitcher_projections_2026.csv"
        h_path = snapshot_dir / f"marcel_hitters_{year}.csv"
        if not h_path.exists() and year == 2026:
            h_path = proj_dir / "hitter_projections_2026.csv"

        if p_path.exists():
            p_df = pd.read_csv(p_path)
            for _, row in p_df.iterrows():
                pitcher_map[row['mlb_player_id']] = {
                    'proj_war': row.get('proj_war', 0),
                    'proj_fip': row.get('proj_fip', 4.20),
                    'proj_era': row.get('proj_era', 4.20),
                    'statcast_adjusted_era': row.get('statcast_adjusted_era',
                                                     row.get('proj_era', 4.20)),
                    'sustainability_score': row.get('sustainability_score', 50),
                    'breakout_score': row.get('breakout_score', 50),
                    'proj_k_bb_pct': row.get('proj_k_bb_pct',
                                             row.get('proj_k_pct', 20) - row.get('proj_bb_pct', 10)
                                             if 'proj_k_pct' in row.index else 10.0),
                    'proj_arsenal_depth': row.get('proj_arsenal_depth', 1.5),
                    'proj_fb_velocity': row.get('proj_fb_velocity', 93.9),
                    'proj_fb_ivb': row.get('proj_fb_ivb', 12.8),
                }

        if h_path.exists():
            h_df = pd.read_csv(h_path)
            for _, row in h_df.iterrows():
                hitter_map[row['mlb_player_id']] = {
                    'sc_woba': row.get('statcast_adjusted_woba', 0.310),
                    'wrc_plus': row.get('wrc_plus', 100),
                    'bounce_back': row.get('bounce_back_score', 50),
                    'proj_war': row.get('proj_war', 0),
                }

        pitcher_maps[year] = pitcher_map
        hitter_maps[year] = hitter_map
        print(f"    {year} snapshot: {len(pitcher_map)} pitchers, {len(hitter_map)} hitters")

    return pitcher_maps, hitter_maps


def compute_projection_features(games_df, hitting_df, pitcher_maps, hitter_maps):
    """
    For each game, look up the correct season's Marcel snapshot and extract:
    1. Starting pitcher's projected fWAR and FIP
    2. Lineup's average statcast_adjusted_woba and bounce-back score

    Uses season-keyed maps to prevent lookahead bias: 2023 games get
    2023 projections (built from 2021-2022 data), not 2026 projections.

    Returns a DataFrame with one row per game, columns:
    - home_proj_sp_war, away_proj_sp_war (starter quality)
    - home_proj_sp_fip, away_proj_sp_fip (starter quality alt metric)
    - home_proj_lineup_woba, away_proj_lineup_woba (lineup quality)
    - home_proj_lineup_bb_score, away_proj_lineup_bb_score (upside signal)
    - diff_proj_sp_war, diff_proj_lineup_woba (differentials)
    """
    results = []

    # Pre-compute: for each game, which hitters appeared for each team?
    game_hitters = {}
    for _, row in hitting_df.iterrows():
        gid = row['game_id']
        tid = row['team_id']
        pid = row['player_id']
        if gid not in game_hitters:
            game_hitters[gid] = {}
        if tid not in game_hitters[gid]:
            game_hitters[gid][tid] = []
        game_hitters[gid][tid].append(pid)

    # Need to map internal player_id -> mlb_id for hitter lookups
    conn = sqlite3.connect(str(DB_PATH))
    pid_to_mlb = pd.read_sql("SELECT id, mlb_id FROM players", conn)
    conn.close()
    pid_map = dict(zip(pid_to_mlb['id'], pid_to_mlb['mlb_id']))

    default_sp = {'proj_war': 0, 'proj_fip': 4.50, 'proj_era': 4.50,
                  'statcast_adjusted_era': 4.50, 'sustainability_score': 50,
                  'breakout_score': 50, 'proj_k_bb_pct': 10.0,
                  'proj_arsenal_depth': 1.5, 'proj_fb_velocity': 93.9,
                  'proj_fb_ivb': 12.8}
    default_hitter = {'sc_woba': 0.310, 'wrc_plus': 100, 'bounce_back': 50, 'proj_war': 0}

    for _, game in games_df.iterrows():
        game_id = game['game_id']
        game_season = game['season']
        row = {'game_id': game_id}

        # Pick the correct season's snapshot (2023 games → 2023 projections)
        pitcher_map = pitcher_maps.get(game_season, pitcher_maps.get(2026, {}))
        hitter_map = hitter_maps.get(game_season, hitter_maps.get(2026, {}))

        for side in ['home', 'away']:
            team_id = game[f'{side}_team_id']

            # --- Starter projection ---
            starter_id = game.get(f'{side}_starter_id')
            if pd.notna(starter_id):
                starter_mlb = pid_map.get(int(starter_id))
                sp_proj = pitcher_map.get(starter_mlb, default_sp)
            else:
                sp_proj = default_sp

            row[f'{side}_proj_sp_war'] = sp_proj['proj_war']
            row[f'{side}_proj_sp_fip'] = sp_proj['proj_fip']
            row[f'{side}_proj_sp_sc_era'] = sp_proj['statcast_adjusted_era']
            row[f'{side}_proj_sp_sust'] = sp_proj['sustainability_score']
            row[f'{side}_proj_sp_breakout'] = sp_proj['breakout_score']
            row[f'{side}_proj_sp_k_bb'] = sp_proj['proj_k_bb_pct']
            row[f'{side}_proj_sp_arsenal'] = sp_proj['proj_arsenal_depth']
            row[f'{side}_proj_sp_velo'] = sp_proj['proj_fb_velocity']
            row[f'{side}_proj_sp_ivb'] = sp_proj['proj_fb_ivb']

            # --- Lineup projection ---
            hitter_pids = game_hitters.get(game_id, {}).get(team_id, [])
            wobas = []
            bb_scores = []

            for pid in hitter_pids:
                mlb_id = pid_map.get(pid)
                if mlb_id and mlb_id in hitter_map:
                    h_proj = hitter_map[mlb_id]
                    wobas.append(h_proj['sc_woba'])
                    bb_scores.append(h_proj['bounce_back'])

            row[f'{side}_proj_lineup_woba'] = np.mean(wobas) if wobas else 0.310
            row[f'{side}_proj_lineup_bb_score'] = np.mean(bb_scores) if bb_scores else 50

        # Differentials
        row['diff_proj_sp_war'] = row['home_proj_sp_war'] - row['away_proj_sp_war']
        row['diff_proj_sp_fip'] = row['away_proj_sp_fip'] - row['home_proj_sp_fip']  # Lower FIP = better
        row['diff_proj_sp_sc_era'] = row['away_proj_sp_sc_era'] - row['home_proj_sp_sc_era']  # Lower ERA = better
        row['diff_proj_sp_sust'] = row['home_proj_sp_sust'] - row['away_proj_sp_sust']
        row['diff_proj_sp_k_bb'] = row['home_proj_sp_k_bb'] - row['away_proj_sp_k_bb']
        row['diff_proj_sp_arsenal'] = row['home_proj_sp_arsenal'] - row['away_proj_sp_arsenal']
        row['diff_proj_sp_velo'] = row['home_proj_sp_velo'] - row['away_proj_sp_velo']
        row['diff_proj_sp_ivb'] = row['home_proj_sp_ivb'] - row['away_proj_sp_ivb']
        row['diff_proj_lineup_woba'] = row['home_proj_lineup_woba'] - row['away_proj_lineup_woba']

        results.append(row)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# ELO RATINGS
# ═══════════════════════════════════════════════════════════════

# Elo constants — tuned for MLB
ELO_K = 4                # Update magnitude per game (MLB is low-variance)
ELO_HOME_ADV = 24        # ~54% implied home win rate
ELO_MEAN = 1500          # League average
ELO_SEASON_REVERT = 0.33 # Regress 1/3 toward mean between seasons
ELO_MOV_MULTIPLIER = True  # Scale K by margin of victory


def _elo_expected(rating_a, rating_b):
    """Expected win probability for team A given ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_mov_mult(mov, elo_diff):
    """
    Margin-of-victory multiplier (FiveThirtyEight formula).
    Rewards blowouts slightly more, but diminishing returns.
    Accounts for autocorrelation: winning team with higher Elo
    gets less credit for running up the score.
    """
    if mov == 0:
        return 1.0
    return np.log(abs(mov) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))


def compute_elo_ratings(games_df):
    """
    Compute running Elo ratings for all teams across all games.

    Returns a DataFrame with game_id, home_elo, away_elo, diff_elo.
    Elo values are PRE-GAME ratings (before the game result updates them).
    Season boundaries trigger a regression toward the mean.

    The Elo system captures team quality trajectory — something rolling
    stats and static projections can't. It's orthogonal to both.
    """
    # Sort chronologically
    games = games_df.sort_values(["date", "game_id"]).copy()

    # Initialize all teams at 1500
    all_teams = set(games["home_team_id"].unique()) | set(games["away_team_id"].unique())
    elo = {team: ELO_MEAN for team in all_teams}

    results = []
    prev_season = None

    for _, game in games.iterrows():
        season = game["season"]
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        # Season boundary: regress toward mean
        if prev_season is not None and season != prev_season:
            for team in elo:
                elo[team] = elo[team] + ELO_SEASON_REVERT * (ELO_MEAN - elo[team])

        prev_season = season

        # Pre-game ratings (what the model sees)
        home_rating = elo.get(home_id, ELO_MEAN)
        away_rating = elo.get(away_id, ELO_MEAN)

        results.append({
            "game_id": game["game_id"],
            "home_elo": round(home_rating, 1),
            "away_elo": round(away_rating, 1),
            "diff_elo": round(home_rating - away_rating, 1),
        })

        # Update ratings if game has a result
        home_score = game.get("home_score")
        away_score = game.get("away_score")

        if pd.notna(home_score) and pd.notna(away_score):
            # Include home advantage in expected score
            home_expected = _elo_expected(
                home_rating + ELO_HOME_ADV, away_rating
            )
            home_actual = 1.0 if home_score > away_score else 0.0

            k = ELO_K
            if ELO_MOV_MULTIPLIER:
                mov = abs(home_score - away_score)
                elo_diff = abs(home_rating - away_rating)
                k = k * _elo_mov_mult(mov, elo_diff)

            # Update both teams
            update = k * (home_actual - home_expected)
            elo[home_id] = home_rating + update
            elo[away_id] = away_rating - update

    elo_df = pd.DataFrame(results)

    # Summary stats
    final_elos = sorted(elo.items(), key=lambda x: -x[1])
    top3 = [f"team {t}: {r:.0f}" for t, r in final_elos[:3]]
    bot3 = [f"team {t}: {r:.0f}" for t, r in final_elos[-3:]]
    print(f"  Elo ratings computed for {len(results)} games")
    print(f"    Top 3: {', '.join(top3)}")
    print(f"    Bot 3: {', '.join(bot3)}")
    print(f"    Elo spread: {final_elos[0][1]:.0f} to {final_elos[-1][1]:.0f}")

    return elo_df


def compute_team_projection_features(games_df):
    """
    Add team-level projected WAR from Marcel snapshots.

    For each game, looks up the team's total projected hitting WAR,
    pitching WAR, and combined WAR from the correct season's snapshot.
    These are static preseason features that don't change game-to-game.

    New features:
    - home_team_proj_hit_war / away_team_proj_hit_war
    - home_team_proj_pitch_war / away_team_proj_pitch_war
    - home_team_proj_war / away_team_proj_war (total)
    - diff_team_proj_war (home advantage in roster quality)
    """
    snapshot_dir = OUTPUT_DIR / "snapshots"

    # Build team WAR lookup per season
    # Key: (season, team_name) -> {hit_war, pitch_war, total_war}
    team_war = {}
    for year in [2023, 2024, 2025, 2026]:
        h_path = snapshot_dir / f"marcel_hitters_{year}.csv"
        p_path = snapshot_dir / f"marcel_pitchers_{year}.csv"
        if not h_path.exists() or not p_path.exists():
            continue

        h_df = pd.read_csv(h_path)
        p_df = pd.read_csv(p_path)

        # Filter out free agents and apply WAR floor
        h_df = h_df[h_df['current_team'] != 'Free Agent'].copy()
        p_df = p_df[p_df['current_team'] != 'Free Agent'].copy()
        h_df = h_df[h_df['proj_pa'] >= 100].copy()
        p_df = p_df[p_df['proj_ip'] >= 30].copy()
        h_df['proj_war'] = h_df['proj_war'].clip(lower=-1.5)
        p_df['proj_war'] = p_df['proj_war'].clip(lower=-1.5)

        hit_agg = h_df.groupby('current_team')['proj_war'].sum()
        pitch_agg = p_df.groupby('current_team')['proj_war'].sum()

        for team_name in set(hit_agg.index) | set(pitch_agg.index):
            hw = hit_agg.get(team_name, 0)
            pw = pitch_agg.get(team_name, 0)
            team_war[(year, team_name)] = {
                'hit_war': hw, 'pitch_war': pw, 'total_war': hw + pw,
            }

    # Need team_id -> team_name mapping
    conn = sqlite3.connect(str(DB_PATH))
    teams = pd.read_sql("SELECT id, name FROM teams", conn)
    conn.close()
    tid_to_name = dict(zip(teams['id'], teams['name']))

    results = []
    default_war = {'hit_war': 0, 'pitch_war': 0, 'total_war': 0}

    for _, game in games_df.iterrows():
        game_id = game['game_id']
        season = game['season']
        row = {'game_id': game_id}

        for side in ['home', 'away']:
            team_id = game[f'{side}_team_id']
            team_name = tid_to_name.get(team_id, '')
            tw = team_war.get((season, team_name), default_war)

            row[f'{side}_team_proj_hit_war'] = round(tw['hit_war'], 1)
            row[f'{side}_team_proj_pitch_war'] = round(tw['pitch_war'], 1)
            row[f'{side}_team_proj_war'] = round(tw['total_war'], 1)

        row['diff_team_proj_war'] = row['home_team_proj_war'] - row['away_team_proj_war']
        row['diff_team_proj_hit_war'] = row['home_team_proj_hit_war'] - row['away_team_proj_hit_war']
        row['diff_team_proj_pitch_war'] = row['home_team_proj_pitch_war'] - row['away_team_proj_pitch_war']

        results.append(row)

    print(f"    Team WAR features: {len(results)} games, "
          f"{len(team_war)} team-season entries")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════

def assemble_game_features(games_df, team_features, sp_features, bp_features,
                           lineup_features=None, projection_features=None,
                           rest_days_df=None, handedness_df=None,
                           venue_splits_df=None, team_proj_features=None,
                           elo_df=None, bp_avail_features=None,
                           sp_rgs_features=None, weather_features=None,
                           def_eff_features=None, barrel_features=None):
    """Merge all feature layers into a single game-level matrix."""

    result = games_df[["game_id", "mlb_game_id", "date", "season",
                        "home_team_id", "away_team_id",
                        "home_score", "away_score", "home_win", "total_runs"]].copy()

    for col in ["day_night"]:
        if col in games_df.columns:
            result[col] = games_df[col]
    
    # --- Team features ---
    for side in ["home", "away"]:
        side_feat = team_features[team_features["side"] == side].copy()
        
        rename = {}
        for col in side_feat.columns:
            if col not in ["game_id", "team_id", "side"]:
                rename[col] = f"{side}_{col}"
        side_feat = side_feat.rename(columns=rename)
        
        result = result.merge(
            side_feat.drop(columns=["team_id", "side"], errors="ignore"),
            on="game_id", how="left"
        )
    
    # --- Starting pitcher features ---
    for side in ["home", "away"]:
        side_sp = sp_features[sp_features["is_starter"]].copy()
        side_sp = side_sp.merge(
            games_df[["game_id", "home_team_id", "away_team_id"]],
            on="game_id", how="left"
        )
        
        if side == "home":
            side_sp = side_sp[side_sp["team_id"] == side_sp["home_team_id"]]
        else:
            side_sp = side_sp[side_sp["team_id"] == side_sp["away_team_id"]]
        
        rename = {}
        for col in side_sp.columns:
            if col not in ["game_id", "player_id", "team_id", "is_starter", 
                          "home_team_id", "away_team_id"]:
                rename[col] = f"{side}_{col}"
        side_sp = side_sp.rename(columns=rename)
        side_sp[f"{side}_sp_id"] = side_sp["player_id"]
        
        merge_cols = ["game_id"] + [c for c in side_sp.columns if c.startswith(side)]
        result = result.merge(
            side_sp[merge_cols].drop_duplicates(subset=["game_id"]),
            on="game_id", how="left"
        )
    
    # --- Bullpen features ---
    for side in ["home", "away"]:
        side_bp = bp_features.merge(
            games_df[["game_id", "home_team_id", "away_team_id"]],
            on="game_id", how="left"
        )
        
        if side == "home":
            side_bp = side_bp[side_bp["team_id"] == side_bp["home_team_id"]]
        else:
            side_bp = side_bp[side_bp["team_id"] == side_bp["away_team_id"]]
        
        rename = {}
        for col in side_bp.columns:
            if col not in ["game_id", "team_id", "home_team_id", "away_team_id"]:
                rename[col] = f"{side}_{col}"
        side_bp = side_bp.rename(columns=rename)
        
        merge_cols = ["game_id"] + [c for c in side_bp.columns if c.startswith(side)]
        result = result.merge(
            side_bp[merge_cols].drop_duplicates(subset=["game_id"]),
            on="game_id", how="left"
        )
    
    # --- Lineup features ---
    if lineup_features is not None and len(lineup_features) > 0:
        for side in ["home", "away"]:
            side_lu = lineup_features[lineup_features["side"] == side].copy()
            
            rename = {}
            for col in side_lu.columns:
                if col not in ["game_id", "team_id", "side"]:
                    rename[col] = f"{side}_{col}"
            side_lu = side_lu.rename(columns=rename)
            
            merge_cols = ["game_id"] + [c for c in side_lu.columns if c.startswith(side)]
            result = result.merge(
                side_lu[merge_cols],
                on="game_id", how="left"
            )
    
    # --- Projection features (Layer 5) ---
    if projection_features is not None and len(projection_features) > 0:
        proj_cols = [c for c in projection_features.columns if c != 'game_id']
        result = result.merge(
            projection_features[['game_id'] + proj_cols],
            on='game_id', how='left'
        )

    # --- Team projection features (Layer 5b) ---
    if team_proj_features is not None and len(team_proj_features) > 0:
        tp_cols = [c for c in team_proj_features.columns if c != 'game_id']
        result = result.merge(
            team_proj_features[['game_id'] + tp_cols],
            on='game_id', how='left'
        )

    # --- Elo ratings ---
    if elo_df is not None and len(elo_df) > 0:
        result = result.merge(
            elo_df[["game_id", "home_elo", "away_elo", "diff_elo"]],
            on="game_id", how="left"
        )

    # --- SP Game Score (538 rGS) ---
    if sp_rgs_features is not None and len(sp_rgs_features) > 0:
        rgs_cols = [c for c in sp_rgs_features.columns if c != 'game_id']
        result = result.merge(
            sp_rgs_features[['game_id'] + rgs_cols],
            on='game_id', how='left'
        )

    # --- Weather & park factor features ---
    if weather_features is not None and len(weather_features) > 0:
        wx_cols = [c for c in weather_features.columns if c != 'game_id']
        result = result.merge(
            weather_features[['game_id'] + wx_cols],
            on='game_id', how='left'
        )

    # --- Defensive efficiency ---
    if def_eff_features is not None and len(def_eff_features) > 0:
        de_cols = [c for c in def_eff_features.columns if c != 'game_id']
        result = result.merge(
            def_eff_features[['game_id'] + de_cols],
            on='game_id', how='left'
        )

    # --- Team barrel rate ---
    if barrel_features is not None and len(barrel_features) > 0:
        br_cols = [c for c in barrel_features.columns if c != 'game_id']
        result = result.merge(
            barrel_features[['game_id'] + br_cols],
            on='game_id', how='left'
        )

    # --- Bullpen availability ---
    if bp_avail_features is not None and len(bp_avail_features) > 0:
        bp_cols = [c for c in bp_avail_features.columns if c != 'game_id']
        result = result.merge(
            bp_avail_features[['game_id'] + bp_cols],
            on='game_id', how='left'
        )

    # --- Rest days (Layer 6a) ---
    if rest_days_df is not None and len(rest_days_df) > 0:
        result = result.merge(
            rest_days_df[["game_id", "home_rest_days", "away_rest_days", "diff_rest_days"]],
            on="game_id", how="left"
        )

    # --- Venue splits (Layer 6c) ---
    if venue_splits_df is not None and len(venue_splits_df) > 0:
        split_cols = [c for c in venue_splits_df.columns if c != "game_id"]
        result = result.merge(
            venue_splits_df[["game_id"] + split_cols],
            on="game_id", how="left"
        )

    # --- Handedness matchups (Layer 6b) ---
    if handedness_df is not None and len(handedness_df) > 0:
        result = result.merge(
            handedness_df[["game_id", "home_platoon_adv", "away_platoon_adv", "diff_platoon_adv"]],
            on="game_id", how="left"
        )

    # --- Arsenal x Handedness interactions (Layer 6c) ---
    # sp_same_hand_pct = fraction of lineup that does NOT have platoon advantage
    # i.e., 1.0 - opposing_platoon_adv (higher = SP faces more same-hand batters)
    # Interaction: velocity/IVB matter more when SP faces same-hand batters
    if handedness_df is not None and len(handedness_df) > 0:
        for side, opp in [("home", "away"), ("away", "home")]:
            # Same-hand pct for this side's SP = 1 - opposing lineup's platoon adv
            same_col = f"{side}_sp_same_hand_pct"
            result[same_col] = 1.0 - result.get(f"{opp}_platoon_adv", 0.5)

            velo_col = f"{side}_proj_sp_velo"
            ivb_col = f"{side}_proj_sp_ivb"

            if velo_col in result.columns:
                result[f"{side}_velo_x_same_hand"] = result[velo_col] * result[same_col]
            if ivb_col in result.columns:
                result[f"{side}_ivb_x_same_hand"] = result[ivb_col] * result[same_col]

        # Interaction diffs
        for feat in ["velo_x_same_hand", "ivb_x_same_hand"]:
            hc = f"home_{feat}"
            ac = f"away_{feat}"
            if hc in result.columns and ac in result.columns:
                result[f"diff_{feat}"] = result[hc] - result[ac]

    # --- Derived difference features ---
    for window in TEAM_WINDOWS:
        sfx = f"_t{window}"
        if f"home_rs{sfx}" in result.columns and f"away_rs{sfx}" in result.columns:
            result[f"diff_rs{sfx}"] = result[f"home_rs{sfx}"] - result[f"away_rs{sfx}"]
            result[f"diff_pyth{sfx}"] = result[f"home_pyth{sfx}"] - result[f"away_pyth{sfx}"]

    # Bullpen diff features (away - home, bp35 only — bp10 too noisy,
    # raw home/away columns redundant with team-level stats at r=0.75+)
    for stat in ["era", "whip", "k_pct", "bb_pct"]:
        home_col = f"home_bp_{stat}_bp35"
        away_col = f"away_bp_{stat}_bp35"
        if home_col in result.columns and away_col in result.columns:
            result[f"diff_bp_{stat}_bp35"] = result[away_col] - result[home_col]

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build game-level feature matrix")
    parser.add_argument("--season", type=int, help="Filter output to single season")
    parser.add_argument("--summary", action="store_true", help="Show DB summary only")
    parser.add_argument("--skip-lineup", action="store_true", help="Skip lineup features (faster)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    conn = get_db_connection()
    
    if args.summary:
        games = load_games(conn)
        print(f"\nGames in database: {len(games)}")
        print(f"Seasons: {games['season'].value_counts().sort_index().to_dict()}")
        conn.close()
        return
    
    # ---- Load ----
    print("\n[1/7] Loading games...")
    games = load_games(conn)
    print(f"  Loaded {len(games)} games ({games['season'].min()}-{games['season'].max()})")
    
    print("\n[2/7] Loading pitcher stats...")
    pitchers = load_pitching_stats(conn)
    n_starters = pitchers["is_starter"].sum()
    print(f"  Loaded {len(pitchers)} pitcher lines, {n_starters} identified as starters")
    
    # ---- Compute ----
    print("\n[3/7] Computing team rolling stats...")
    team_features = compute_team_rolling(games, pitchers, windows=TEAM_WINDOWS)
    print(f"  Generated {len(team_features)} team feature rows")
    
    print("\n[4/7] Computing starting pitcher rolling stats...")
    sp_features = compute_pitcher_rolling(pitchers, windows=SP_WINDOWS)
    print(f"  Generated {len(sp_features)} SP feature rows")
    
    # Layer 2b: SP rolling Game Score (538's biggest signal)
    sp_rgs_features = None
    print("\n[4b/14] Computing SP Game Score (538 rGS)...")
    try:
        sp_rgs_features = compute_sp_game_score(pitchers, games)
        if sp_rgs_features is not None and len(sp_rgs_features) > 0:
            print(f"  Generated {len(sp_rgs_features)} SP Game Score rows")
    except Exception as e:
        print(f"  SP Game Score skipped: {e}")

    print("\n[5/7] Computing bullpen rolling stats...")
    bp_features = compute_bullpen_rolling(pitchers, games, windows=BP_WINDOWS)
    print(f"  Generated {len(bp_features)} bullpen feature rows")

    # Layer 3b: Bullpen availability/fatigue
    bp_avail_features = None
    print("\n[5b/14] Computing bullpen availability (fatigue)...")
    try:
        bp_avail_features = compute_bullpen_availability(pitchers, games)
        if bp_avail_features is not None and len(bp_avail_features) > 0:
            print(f"  Generated {len(bp_avail_features)} bullpen availability rows")
    except Exception as e:
        print(f"  Bullpen availability skipped: {e}")

    lineup_features = None
    hitters = None
    if not args.skip_lineup:
        print("\n[6/11] Computing lineup features...")
        try:
            hitters = load_hitting_stats(conn)
            print(f"  Loaded {len(hitters)} hitter game lines")
            player_positions = load_player_positions(conn)
            print(f"  Loaded positions for {len(player_positions)} players")
            lineup_features = compute_lineup_features(hitters, games, player_positions, window=BATTER_WINDOW)
            print(f"  Generated {len(lineup_features)} lineup feature rows")
        except Exception as e:
            print(f"  Lineup features skipped: {e}")
    else:
        print("\n[6/11] Skipping lineup features (--skip-lineup)")

    # Layer 5: Projection-based features
    projection_features = None
    print("\n[7/11] Computing projection features (Layer 5)...")
    try:
        pitcher_maps, hitter_maps = load_projection_maps()
        if any(pitcher_maps.values()) and any(hitter_maps.values()):
            if hitters is None:
                hitters = load_hitting_stats(conn)
            projection_features = compute_projection_features(
                games, hitters, pitcher_maps, hitter_maps
            )
            print(f"  Generated {len(projection_features)} projection feature rows")
        else:
            print("  Skipped — no projection snapshots available")
    except Exception as e:
        print(f"  Projection features skipped: {e}")

    # Layer 5b: Elo ratings
    elo_df = None
    print("\n[8/13] Computing Elo ratings...")
    try:
        elo_df = compute_elo_ratings(games)
    except Exception as e:
        print(f"  Elo ratings skipped: {e}")

    # Layer 5c: Team-level projected WAR
    team_proj_features = None
    print("\n[9/13] Computing team projection features (Layer 5c)...")
    try:
        team_proj_features = compute_team_projection_features(games)
        if team_proj_features is not None and len(team_proj_features) > 0:
            print(f"  Generated {len(team_proj_features)} team projection rows")
        else:
            print("  Skipped — no team projection data")
    except Exception as e:
        print(f"  Team projection features skipped: {e}")

    # Layer 6a: Rest days (computed from game dates)
    rest_days_df = None
    print("\n[10/13] Computing rest days (Layer 6a)...")
    try:
        rest_days_df = compute_rest_days(games)
    except Exception as e:
        print(f"  Rest days skipped: {e}")

    # Layer 6b: Handedness matchups (platoon advantage)
    handedness_df = None
    print("\n[11/13] Computing handedness features (Layer 6b)...")
    try:
        if hitters is None:
            hitters = load_hitting_stats(conn)
        handedness_df = compute_handedness_features(games, hitters, conn)
    except Exception as e:
        print(f"  Handedness features skipped: {e}")

    # Layer 8: Feature gaps (defensive efficiency, barrel rate)
    def_eff_features = None
    barrel_features = None
    print("\n[12/16] Computing defensive efficiency (Layer 8a)...")
    try:
        if hitters is None:
            hitters = load_hitting_stats(conn)
        def_eff_features = compute_defensive_efficiency(pitchers, games, hitters)
    except Exception as e:
        print(f"  Defensive efficiency skipped: {e}")

    print("\n[13/16] Computing team barrel rate (Layer 8b)...")
    try:
        if hitters is None:
            hitters = load_hitting_stats(conn)
        barrel_features = compute_team_barrel_rate(hitters, games)
    except Exception as e:
        print(f"  Team barrel rate skipped: {e}")

    # Layer 7: Weather & park factor features
    weather_features = None
    print("\n[14/16] Computing weather & park factor features (Layer 7)...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.features.weather import compute_weather_features
        ballparks_df = load_ballparks(conn)
        if len(ballparks_df) > 0:
            weather_features = compute_weather_features(games, ballparks_df)
            print(f"  Generated {len(weather_features)} weather feature rows")
        else:
            print("  Skipped — no ballpark data (run populate_ballparks.py)")
    except Exception as e:
        print(f"  Weather features skipped: {e}")
        import traceback
        traceback.print_exc()

    # Layer 6c: Home/away venue splits
    venue_splits_df = None
    print("\n[15/16] Computing home/away venue splits (Layer 6c)...")
    try:
        venue_splits_df = compute_home_away_splits(games)
    except Exception as e:
        print(f"  Venue splits skipped: {e}")

    # ---- Assemble ----
    print("\n[16/16] Assembling game feature matrix...")
    game_features = assemble_game_features(
        games, team_features, sp_features, bp_features,
        lineup_features, projection_features,
        rest_days_df, handedness_df, venue_splits_df,
        team_proj_features, elo_df, bp_avail_features,
        sp_rgs_features, weather_features,
        def_eff_features, barrel_features
    )
    
    if args.season:
        game_features = game_features[game_features["season"] == args.season]
    
    # ---- Report ----
    feature_cols = [c for c in game_features.columns 
                    if c.startswith(("home_", "away_", "diff_")) 
                    and not c.endswith(("_team_id", "_score", "_sp_id"))]
    numeric_features = [c for c in feature_cols if game_features[c].dtype in ["float64", "int64"]]
    
    print(f"\n{'=' * 70}")
    print(f"FEATURE MATRIX COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Games:    {len(game_features)}")
    print(f"  Features: {len(numeric_features)} numeric columns")
    print(f"  Missing:  {game_features[numeric_features].isna().sum().sum():.0f} total NaN values")
    print(f"  Seasons:  {game_features['season'].value_counts().sort_index().to_dict()}")
    
    print(f"\n  Feature columns ({len(numeric_features)}):")
    for i, col in enumerate(sorted(numeric_features)):
        non_null = game_features[col].notna().sum()
        mean_val = game_features[col].mean()
        print(f"    {col:45s} | {non_null:5d} values | mean={mean_val:.4f}")
        if i >= 35:
            remaining = len(numeric_features) - i - 1
            if remaining > 0:
                print(f"    ... ({remaining} more)")
            break
    
    # ---- Save ----
    output_path = OUTPUT_DIR / "game_features.csv"
    game_features.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")
    
    conn.close()


if __name__ == "__main__":
    main()