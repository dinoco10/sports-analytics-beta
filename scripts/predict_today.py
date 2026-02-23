"""
predict_today.py — Daily Game Win Probability Predictions
=========================================================
Fetches today's MLB schedule, computes features from historical DB data
and Marcel snapshots, runs the saved LightGBM model, and outputs
win probabilities.

Usage:
  python scripts/predict_today.py                    # Today's games
  python scripts/predict_today.py --date 2026-04-15  # Specific date
  python scripts/predict_today.py --save             # Write to DB + CSV
  python scripts/predict_today.py --backtest 2025-09-28  # Backtest a past date

Author: Loko
"""

import argparse
import json
import sqlite3
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

# Fix Windows cp1252 encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path so we can import build_features functions
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
from src.ingestion.mlb_api import MLBApiClient

# Import feature computation functions from build_features
from build_features import (
    get_db_connection,
    load_games,
    load_pitching_stats,
    load_hitting_stats,
    load_player_positions,
    load_projection_maps,
    compute_team_rolling,
    compute_pitcher_rolling,
    compute_bullpen_rolling,
    compute_lineup_features,
    compute_projection_features,
    compute_team_projection_features,
    compute_elo_ratings,
    compute_rest_days,
    compute_handedness_features,
    compute_home_away_splits,
    assemble_game_features,
    TEAM_WINDOWS,
    SP_WINDOWS,
    BP_WINDOWS,
    BATTER_WINDOW,
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DB_PATH = PROJECT_ROOT / "data" / "mlb_analytics.db"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

# Statuses indicating a game hasn't been played yet
PREGAME_STATUSES = {"Preview", "Pre-Game", "Warmup", "Scheduled"}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_model():
    """Load the saved LightGBM model, feature list, and medians."""
    model_path = MODELS_DIR / "win_probability_lgbm.txt"
    meta_path = MODELS_DIR / "win_probability_meta.json"
    medians_path = MODELS_DIR / "feature_medians.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Run: python scripts/train_win_model.py --pruned --save"
        )

    booster = lgb.Booster(model_file=str(model_path))

    with open(meta_path) as f:
        meta = json.load(f)
    features = meta["features"]

    medians = {}
    if medians_path.exists():
        with open(medians_path) as f:
            medians = json.load(f)

    return booster, features, medians


def load_team_mappings(conn):
    """Build team ID lookup tables: mlb_id <-> db_id, name <-> db_id."""
    teams = pd.read_sql("SELECT id, mlb_id, name FROM teams", conn)
    mlb_to_db = dict(zip(teams["mlb_id"], teams["id"]))
    db_to_name = dict(zip(teams["id"], teams["name"]))
    name_to_db = dict(zip(teams["name"], teams["id"]))
    return mlb_to_db, db_to_name, name_to_db


def load_player_mappings(conn):
    """Build player mlb_id -> db_id lookup."""
    players = pd.read_sql("SELECT id, mlb_id, name FROM players", conn)
    mlb_to_db = dict(zip(players["mlb_id"], players["id"]))
    db_to_name = dict(zip(players["id"], players["name"]))
    return mlb_to_db, db_to_name


def ensure_player_exists(conn, mlb_id, name):
    """Insert a player if not in DB. Returns the DB id."""
    cursor = conn.execute(
        "SELECT id FROM players WHERE mlb_id = ?", (mlb_id,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Insert minimal record
    conn.execute(
        "INSERT INTO players (mlb_id, name, active) VALUES (?, ?, 1)",
        (mlb_id, name),
    )
    conn.commit()
    cursor = conn.execute(
        "SELECT id FROM players WHERE mlb_id = ?", (mlb_id,)
    )
    return cursor.fetchone()[0]


def fetch_schedule(target_date, backtest=False):
    """
    Fetch games for target_date.

    For live predictions: filters to pre-game only.
    For backtest: returns all games (including completed).
    """
    client = MLBApiClient()
    schedule = client.get_schedule(target_date)

    if schedule.empty:
        return schedule

    if backtest:
        # For backtesting, we want completed games so we can compare
        return schedule[schedule["status"] == "Final"]
    else:
        # For live predictions, only pre-game
        return schedule[schedule["status"].isin(PREGAME_STATUSES)]


def build_placeholder_games(schedule_df, team_map, player_map, conn):
    """
    Convert API schedule into placeholder game rows matching load_games() format.
    Uses negative IDs to avoid collision with real game_ids.
    """
    rows = []
    for i, game in schedule_df.iterrows():
        home_db_id = team_map.get(game["home_team_id"])
        away_db_id = team_map.get(game["away_team_id"])

        if home_db_id is None or away_db_id is None:
            print(f"  WARNING: Unknown team in game {game['mlb_game_id']}, skipping")
            continue

        # Map pitcher MLB IDs to DB IDs (create if needed)
        home_sp_db = None
        away_sp_db = None

        if pd.notna(game.get("home_starter_id")):
            home_sp_mlb = int(game["home_starter_id"])
            if home_sp_mlb not in player_map:
                db_id = ensure_player_exists(conn, home_sp_mlb, game.get("home_starter", "Unknown"))
                player_map[home_sp_mlb] = db_id
            home_sp_db = player_map[home_sp_mlb]

        if pd.notna(game.get("away_starter_id")):
            away_sp_mlb = int(game["away_starter_id"])
            if away_sp_mlb not in player_map:
                db_id = ensure_player_exists(conn, away_sp_mlb, game.get("away_starter", "Unknown"))
                player_map[away_sp_mlb] = db_id
            away_sp_db = player_map[away_sp_mlb]

        rows.append({
            "game_id": -(i + 1),  # negative to avoid collision
            "mlb_game_id": game["mlb_game_id"],
            "date": pd.to_datetime(game["date"]),
            "season": pd.to_datetime(game["date"]).year,
            "home_team_id": home_db_id,
            "away_team_id": away_db_id,
            "home_starter_id": home_sp_db,
            "away_starter_id": away_sp_db,
            "home_score": np.nan,
            "away_score": np.nan,
            "day_night": None,
            "home_rest_days": None,
            "away_rest_days": None,
            "home_win": np.nan,
            "total_runs": np.nan,
        })

    return pd.DataFrame(rows)


def confidence_tier(prob):
    """Classify prediction confidence."""
    edge = abs(prob - 0.5)
    if edge >= 0.15:
        return "STRONG"
    elif edge >= 0.08:
        return "LEAN"
    else:
        return "TOSS-UP"


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def predict_games(target_date, save=False, backtest=False):
    """
    Full prediction pipeline for a given date.

    1. Fetch schedule from MLB API
    2. Load historical data from DB
    3. Compute features (rolling + projections)
    4. Run LightGBM model
    5. Output predictions
    """
    print("=" * 70)
    print(f"MLB WIN PROBABILITY PREDICTIONS — {target_date}")
    print("=" * 70)

    # --- Step 1: Fetch schedule ---
    mode = "BACKTEST" if backtest else "LIVE"
    print(f"\n[1/6] Fetching schedule ({mode})...")
    schedule = fetch_schedule(target_date, backtest=backtest)

    if schedule.empty:
        print("  No games found for this date.")
        return None

    print(f"  Found {len(schedule)} games")

    # --- Step 2: Load DB data ---
    print("\n[2/6] Loading historical data from DB...")
    conn = get_db_connection()
    team_map, db_to_team, name_to_db = load_team_mappings(conn)
    player_map, db_to_player = load_player_mappings(conn)

    historical_games = load_games(conn)
    pitchers = load_pitching_stats(conn)
    hitters = load_hitting_stats(conn)
    player_positions = load_player_positions(conn)
    print(f"  {len(historical_games)} historical games, "
          f"{len(pitchers)} pitcher lines, {len(hitters)} hitter lines")

    # --- Step 3: Build placeholders ---
    print("\n[3/6] Building game placeholders...")
    placeholders = build_placeholder_games(schedule, team_map, player_map, conn)

    if placeholders.empty:
        print("  No valid games to predict.")
        conn.close()
        return None

    # Show the matchups
    for _, g in placeholders.iterrows():
        home = db_to_team.get(g["home_team_id"], "?")
        away = db_to_team.get(g["away_team_id"], "?")
        home_sp = db_to_player.get(g["home_starter_id"], "TBD")
        away_sp = db_to_player.get(g["away_starter_id"], "TBD")
        print(f"  {away} ({away_sp}) @ {home} ({home_sp})")

    # Combine historical + today's placeholders
    all_games = pd.concat([historical_games, placeholders], ignore_index=True)
    all_games["date"] = pd.to_datetime(all_games["date"])
    today_ids = set(placeholders["game_id"].values)

    # --- Step 4: Compute features ---
    print("\n[4/6] Computing features...")

    print("  Team rolling stats...")
    team_features = compute_team_rolling(all_games, pitchers, windows=TEAM_WINDOWS)

    print("  SP rolling stats...")
    sp_features = compute_pitcher_rolling(pitchers, windows=SP_WINDOWS)

    print("  Bullpen rolling stats...")
    bp_features = compute_bullpen_rolling(pitchers, all_games, windows=BP_WINDOWS)

    print("  Lineup features...")
    try:
        lineup_features = compute_lineup_features(
            hitters, all_games, player_positions, window=BATTER_WINDOW
        )
    except Exception as e:
        print(f"    Lineup skipped: {e}")
        lineup_features = None

    print("  Projection features...")
    try:
        pitcher_maps, hitter_maps = load_projection_maps()
        projection_features = compute_projection_features(
            all_games, hitters, pitcher_maps, hitter_maps
        )
    except Exception as e:
        print(f"    Projections skipped: {e}")
        projection_features = None

    print("  Elo ratings...")
    try:
        elo_df = compute_elo_ratings(all_games)
    except Exception as e:
        print(f"    Elo skipped: {e}")
        elo_df = None

    print("  Team projection features...")
    try:
        team_proj_features = compute_team_projection_features(all_games)
    except Exception as e:
        print(f"    Team projections skipped: {e}")
        team_proj_features = None

    print("  Rest days...")
    try:
        rest_days_df = compute_rest_days(all_games)
    except Exception as e:
        print(f"    Rest days skipped: {e}")
        rest_days_df = None

    print("  Handedness matchups...")
    try:
        handedness_df = compute_handedness_features(all_games, hitters, conn)
    except Exception as e:
        print(f"    Handedness skipped: {e}")
        handedness_df = None

    print("  Venue splits...")
    try:
        venue_splits_df = compute_home_away_splits(all_games)
    except Exception as e:
        print(f"    Venue splits skipped: {e}")
        venue_splits_df = None

    # Assemble
    print("  Assembling features...")
    game_features = assemble_game_features(
        all_games, team_features, sp_features, bp_features,
        lineup_features, projection_features,
        rest_days_df, handedness_df, venue_splits_df,
        team_proj_features, elo_df,
    )

    # Filter to today's games only
    today_features = game_features[game_features["game_id"].isin(today_ids)].copy()
    print(f"  {len(today_features)} games with features")

    if today_features.empty:
        print("  No features computed for today's games.")
        conn.close()
        return None

    # --- Step 5: Load model and predict ---
    print("\n[5/6] Running model...")
    booster, feature_list, medians = load_model()

    # Prepare feature matrix
    X = today_features[feature_list].copy()
    for col in feature_list:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
        else:
            X[col] = X[col].fillna(0.0)

    probs = booster.predict(X)

    # --- Step 6: Output ---
    print("\n[6/6] Generating predictions...")

    results = []
    for idx, (_, row) in enumerate(today_features.iterrows()):
        home_prob = probs[idx]
        away_prob = 1.0 - home_prob

        # Use schedule data for display (reliable team/pitcher names)
        sched_row = schedule[schedule["mlb_game_id"] == row["mlb_game_id"]]
        if len(sched_row) > 0:
            s = sched_row.iloc[0]
            home_team = s["home_team"]
            away_team = s["away_team"]
            home_sp = s.get("home_starter", "TBD")
            away_sp = s.get("away_starter", "TBD")
        else:
            home_team = db_to_team.get(row.get("home_team_id"), "?")
            away_team = db_to_team.get(row.get("away_team_id"), "?")
            home_sp = "TBD"
            away_sp = "TBD"

        result = {
            "date": target_date.strftime("%Y-%m-%d"),
            "mlb_game_id": row["mlb_game_id"],
            "home_team": home_team,
            "away_team": away_team,
            "home_sp": home_sp,
            "away_sp": away_sp,
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(away_prob, 4),
            "confidence": confidence_tier(home_prob),
        }

        # For backtest: include actual result
        if backtest and len(sched_row) > 0:
            h_score = sched_row["home_score"].values[0]
            a_score = sched_row["away_score"].values[0]
            if pd.notna(h_score) and pd.notna(a_score):
                actual_winner = "HOME" if h_score > a_score else "AWAY"
                predicted_winner = "HOME" if home_prob > 0.5 else "AWAY"
                correct = actual_winner == predicted_winner
                result["actual_winner"] = actual_winner
                result["correct"] = correct
                result["home_score"] = int(h_score)
                result["away_score"] = int(a_score)

        results.append(result)

    results_df = pd.DataFrame(results)

    # Print formatted output
    print(f"\n{'=' * 80}")
    print(f"  PREDICTIONS FOR {target_date}")
    print(f"{'=' * 80}")
    print(f"  {'Away':>22s}   {'Home':<22s}   {'P(Home)':>7s}  {'Conf':>8s}", end="")
    if backtest:
        print(f"  {'Result':>8s}  {'Correct':>7s}", end="")
    print()
    print(f"  {'-' * 22}   {'-' * 22}   {'-' * 7}  {'-' * 8}", end="")
    if backtest:
        print(f"  {'-' * 8}  {'-' * 7}", end="")
    print()

    for _, r in results_df.iterrows():
        prob_str = f"{r['home_win_prob']:.1%}"
        line = f"  {r['away_team']:>22s}   {r['home_team']:<22s}   {prob_str:>7s}  {r['confidence']:>8s}"
        if backtest and "actual_winner" in r:
            score = f"{r['away_score']}-{r['home_score']}"
            marker = "Y" if r["correct"] else "N"
            line += f"  {score:>8s}  {marker:>7s}"
        print(line)

    # Pitchers detail
    print(f"\n  Pitching matchups:")
    for _, r in results_df.iterrows():
        print(f"    {r['away_sp']:>20s}  vs  {r['home_sp']:<20s}")

    # Summary stats for backtest
    if backtest and "correct" in results_df.columns:
        n = len(results_df)
        correct = results_df["correct"].sum()
        print(f"\n  Backtest: {correct}/{n} correct ({correct/n:.1%})")

    # Save outputs
    if save:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = PREDICTIONS_DIR / f"predictions_{target_date.strftime('%Y-%m-%d')}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Saved to: {csv_path}")

        # Write to DB
        try:
            model_version = "lgbm_v1_pruned43"
            for _, r in results_df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO model_predictions
                       (model_version, prediction_date, home_win_prob, away_win_prob, confidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (model_version, r["date"], r["home_win_prob"],
                     r["away_win_prob"], r["home_win_prob"]),
                )
            conn.commit()
            print(f"  Written {len(results_df)} predictions to DB")
        except Exception as e:
            print(f"  DB write skipped: {e}")

    conn.close()
    return results_df


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Daily MLB win probability predictions")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--backtest", type=str, help="Backtest a past date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save to CSV and DB")
    args = parser.parse_args()

    if args.backtest:
        target_date = datetime.strptime(args.backtest, "%Y-%m-%d").date()
        predict_games(target_date, save=args.save, backtest=True)
    elif args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        predict_games(target_date, save=args.save, backtest=False)
    else:
        target_date = date.today()
        predict_games(target_date, save=args.save, backtest=False)


if __name__ == "__main__":
    main()
