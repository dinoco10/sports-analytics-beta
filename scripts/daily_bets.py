"""
daily_bets.py — Daily Betting Recommendations
==============================================
Combines model win probabilities with sportsbook odds to find
value bets. Outputs recommendations with EV, Kelly sizing, and
confidence tiers. Tracks P/L over time.

Usage:
  # Run predictions + enter odds interactively
  python scripts/daily_bets.py

  # Use existing predictions CSV + odds CSV
  python scripts/daily_bets.py --predictions data/predictions/predictions_2026-04-01.csv \
                               --odds data/odds/daily/2026-04-01.csv

  # Specify date (runs predict_today internally)
  python scripts/daily_bets.py --date 2026-04-01

  # Settle yesterday's bets (mark results)
  python scripts/daily_bets.py --settle 2026-03-31

  # Show P/L summary
  python scripts/daily_bets.py --summary

  # Backtest mode: use historical closing lines from DB
  python scripts/daily_bets.py --backtest 2025-09-28
"""

import argparse
import csv
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Fix Windows cp1252 encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "mlb_analytics.db"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
BETS_DIR = ROOT / "data" / "bets"
DAILY_ODDS_DIR = ROOT / "data" / "odds" / "daily"

# Betting parameters
MIN_EDGE = 0.03          # 3% minimum edge to recommend
KELLY_FRACTION = 0.25    # Quarter-Kelly
BANKROLL = 10_000.0      # Starting/current bankroll for Kelly sizing
FLAT_STAKE = 100.0       # Flat bet amount


# ═══════════════════════════════════════════════════════════════
# ODDS HELPERS
# ═══════════════════════════════════════════════════════════════

def american_to_decimal(ml):
    """Convert American moneyline to decimal odds."""
    if ml < 0:
        return 1 + 100 / abs(ml)
    else:
        return 1 + ml / 100


def american_to_implied_prob(ml):
    """Convert American moneyline to raw implied probability (includes vig)."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def remove_vig(home_ml, away_ml):
    """Remove vig from moneyline pair, return fair probabilities."""
    home_raw = american_to_implied_prob(home_ml)
    away_raw = american_to_implied_prob(away_ml)
    total = home_raw + away_raw
    return home_raw / total, away_raw / total


def ml_payout(stake, ml, won):
    """Calculate profit from a moneyline bet."""
    if not won:
        return -stake
    if ml < 0:
        return stake * (100 / abs(ml))
    else:
        return stake * (ml / 100)


def kelly_size(edge, decimal_odds, bankroll, fraction=KELLY_FRACTION):
    """Calculate Kelly criterion bet size.

    Kelly formula: f* = edge / (decimal_odds - 1)
    We use quarter-Kelly for conservative sizing.
    """
    denom = decimal_odds - 1
    if denom <= 0 or edge <= 0:
        return 0.0
    kelly_f = (edge / denom) * fraction
    kelly_f = max(kelly_f, 0.0)
    return bankroll * kelly_f


# ═══════════════════════════════════════════════════════════════
# LOAD PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def load_predictions(predictions_path=None, target_date=None):
    """Load predictions from CSV or run predict_today.py inline."""

    if predictions_path and Path(predictions_path).exists():
        df = pd.read_csv(predictions_path)
        print(f"  Loaded {len(df)} predictions from {predictions_path}")
        return df

    # Try default path for the date
    if target_date:
        default_path = PREDICTIONS_DIR / f"predictions_{target_date}.csv"
        if default_path.exists():
            df = pd.read_csv(default_path)
            print(f"  Loaded {len(df)} predictions from {default_path}")
            return df

    # Run predict_today.py inline
    print("  No predictions file found. Running predict_today.py...")
    sys.path.insert(0, str(ROOT / "scripts"))
    from predict_today import predict_games

    d = datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else date.today()
    results_df = predict_games(d, save=True, backtest=False)
    return results_df


# ═══════════════════════════════════════════════════════════════
# LOAD ODDS
# ═══════════════════════════════════════════════════════════════

def load_daily_odds(target_date, odds_path=None):
    """Load odds from CSV file.

    Expected CSV format:
      home_team,away_team,home_ml,away_ml[,sportsbook]

    Example:
      New York Yankees,Boston Red Sox,-150,+130,bet365
      Los Angeles Dodgers,San Francisco Giants,-180,+155,fanduel
    """
    if odds_path and Path(odds_path).exists():
        df = pd.read_csv(odds_path)
        print(f"  Loaded {len(df)} odds lines from {odds_path}")
        return df

    # Try default path
    default_path = DAILY_ODDS_DIR / f"odds_{target_date}.csv"
    if default_path.exists():
        df = pd.read_csv(default_path)
        print(f"  Loaded {len(df)} odds lines from {default_path}")
        return df

    return None


def load_historical_odds(target_date):
    """Load historical odds from DB for backtesting."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_ml, away_ml,
               vegas_home_prob, vegas_away_prob, sportsbook
        FROM odds
        WHERE date = ?
    """
    df = pd.read_sql_query(query, conn, params=(target_date,))
    conn.close()

    if len(df) > 0:
        print(f"  Loaded {len(df)} historical odds from DB")
    return df


def enter_odds_manually(predictions_df):
    """Prompt user to enter odds for each game."""
    print("\n  Enter moneyline odds for each game (or 'skip' to skip):")
    print("  Format: home_ml away_ml (e.g., '-150 +130')")
    print()

    odds_rows = []
    for _, row in predictions_df.iterrows():
        matchup = f"  {row['away_team']:>22s} @ {row['home_team']:<22s}"
        prob_str = f"(model: {row['home_win_prob']:.1%} home)"
        user_input = input(f"{matchup} {prob_str}: ").strip()

        if user_input.lower() in ("skip", "s", ""):
            continue

        try:
            parts = user_input.replace("+", "").split()
            home_ml = float(parts[0])
            away_ml = float(parts[1])
            odds_rows.append({
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_ml": home_ml,
                "away_ml": away_ml,
                "sportsbook": "manual",
            })
        except (ValueError, IndexError):
            print("    Invalid format, skipping...")

    if odds_rows:
        return pd.DataFrame(odds_rows)
    return None


# ═══════════════════════════════════════════════════════════════
# BET RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def find_value_bets(predictions_df, odds_df, min_edge=MIN_EDGE):
    """Compare model probabilities to odds and find value bets.

    Returns a DataFrame with columns:
      date, home_team, away_team, side, model_prob, vegas_prob,
      edge, ev_per_dollar, ml, kelly_stake, flat_stake, confidence
    """
    bets = []

    for _, pred in predictions_df.iterrows():
        # Match prediction to odds by team names
        odds_match = odds_df[
            (odds_df["home_team"].str.contains(pred["home_team"], case=False, na=False)) |
            (odds_df["home_team"] == pred["home_team"])
        ]

        if len(odds_match) == 0:
            # Try matching on away team
            odds_match = odds_df[
                (odds_df["away_team"].str.contains(pred["away_team"], case=False, na=False)) |
                (odds_df["away_team"] == pred["away_team"])
            ]

        if len(odds_match) == 0:
            continue

        odds_row = odds_match.iloc[0]
        home_ml = odds_row["home_ml"]
        away_ml = odds_row["away_ml"]
        sportsbook = odds_row.get("sportsbook", "unknown")

        # Fair probabilities (vig removed)
        vegas_home_prob, vegas_away_prob = remove_vig(home_ml, away_ml)

        model_home_prob = pred["home_win_prob"]
        model_away_prob = 1 - model_home_prob

        # Check home side
        home_edge = model_home_prob - vegas_home_prob
        if home_edge >= min_edge:
            dec_odds = american_to_decimal(home_ml)
            ev = model_home_prob * (dec_odds - 1) - (1 - model_home_prob)
            kelly = kelly_size(home_edge, dec_odds, BANKROLL)
            bets.append({
                "date": pred.get("date", ""),
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "side": "HOME",
                "team_bet": pred["home_team"],
                "model_prob": model_home_prob,
                "vegas_prob": vegas_home_prob,
                "edge": home_edge,
                "ev_per_dollar": ev,
                "ml": home_ml,
                "kelly_stake": kelly,
                "flat_stake": FLAT_STAKE,
                "sportsbook": sportsbook,
                "home_sp": pred.get("home_sp", ""),
                "away_sp": pred.get("away_sp", ""),
            })

        # Check away side
        away_edge = model_away_prob - vegas_away_prob
        if away_edge >= min_edge:
            dec_odds = american_to_decimal(away_ml)
            ev = model_away_prob * (dec_odds - 1) - (1 - model_away_prob)
            kelly = kelly_size(away_edge, dec_odds, BANKROLL)
            bets.append({
                "date": pred.get("date", ""),
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "side": "AWAY",
                "team_bet": pred["away_team"],
                "model_prob": model_away_prob,
                "vegas_prob": vegas_away_prob,
                "edge": away_edge,
                "ev_per_dollar": ev,
                "ml": away_ml,
                "kelly_stake": kelly,
                "flat_stake": FLAT_STAKE,
                "sportsbook": sportsbook,
                "home_sp": pred.get("home_sp", ""),
                "away_sp": pred.get("away_sp", ""),
            })

    if not bets:
        return pd.DataFrame()

    df = pd.DataFrame(bets)
    df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════

def display_recommendations(bets_df, target_date):
    """Print formatted betting recommendations."""

    print(f"\n{'=' * 72}")
    print(f"  BETTING RECOMMENDATIONS — {target_date}")
    print(f"{'=' * 72}")

    if len(bets_df) == 0:
        print("  No value bets found today.")
        print(f"  (minimum edge threshold: {MIN_EDGE:.0%})")
        return

    print(f"  Found {len(bets_df)} value bet(s) (edge >= {MIN_EDGE:.0%})\n")

    # Tier the bets
    strong = bets_df[bets_df["edge"] >= 0.10]
    lean = bets_df[(bets_df["edge"] >= 0.05) & (bets_df["edge"] < 0.10)]
    mild = bets_df[bets_df["edge"] < 0.05]

    for tier_name, tier_df in [("STRONG", strong), ("LEAN", lean), ("MILD", mild)]:
        if len(tier_df) == 0:
            continue

        print(f"  --- {tier_name} ({len(tier_df)}) ---")
        for _, b in tier_df.iterrows():
            matchup = f"{b['away_team']} @ {b['home_team']}"
            side_str = f"Bet {b['side']}: {b['team_bet']}"

            print(f"\n    {matchup}")
            print(f"    {side_str} ({b['ml']:+.0f})")
            print(f"    Model: {b['model_prob']:.1%}  |  Vegas: {b['vegas_prob']:.1%}  |  "
                  f"Edge: {b['edge']:.1%}  |  EV: ${b['ev_per_dollar']*100:+.1f}/100")
            print(f"    Kelly: ${b['kelly_stake']:.0f}  |  Flat: ${b['flat_stake']:.0f}")
            if b.get("home_sp") and b.get("away_sp"):
                print(f"    Pitchers: {b['away_sp']} vs {b['home_sp']}")
        print()

    # Summary
    total_kelly = bets_df["kelly_stake"].sum()
    total_flat = len(bets_df) * FLAT_STAKE
    avg_edge = bets_df["edge"].mean()
    avg_ev = bets_df["ev_per_dollar"].mean()

    print(f"  {'=' * 50}")
    print(f"  Total bets:   {len(bets_df)}")
    print(f"  Avg edge:     {avg_edge:.1%}")
    print(f"  Avg EV/$100:  ${avg_ev*100:+.1f}")
    print(f"  Total stake:  ${total_kelly:.0f} (Kelly) / ${total_flat:.0f} (flat)")


# ═══════════════════════════════════════════════════════════════
# BET TRACKING (CSV-based)
# ═══════════════════════════════════════════════════════════════

def save_bets(bets_df, target_date):
    """Save bet recommendations to CSV for tracking."""
    BETS_DIR.mkdir(parents=True, exist_ok=True)
    path = BETS_DIR / f"bets_{target_date}.csv"

    # Add tracking columns
    bets_df = bets_df.copy()
    bets_df["result"] = "pending"
    bets_df["profit_loss"] = None
    bets_df["settled"] = False

    bets_df.to_csv(path, index=False)
    print(f"\n  Bets saved to: {path}")
    return path


def settle_bets(settle_date):
    """Settle bets for a given date using actual game results from DB.

    Reads bets CSV, looks up game results in DB, calculates P/L.
    """
    bets_path = BETS_DIR / f"bets_{settle_date}.csv"
    if not bets_path.exists():
        print(f"  No bets file for {settle_date}")
        return

    bets_df = pd.read_csv(bets_path)
    if len(bets_df) == 0:
        print(f"  No bets to settle for {settle_date}")
        return

    # Look up results from DB
    conn = sqlite3.connect(str(DB_PATH))
    games = pd.read_sql_query(
        """SELECT g.date, ht.name as home_team, at.name as away_team,
                  g.home_score, g.away_score
           FROM games g
           JOIN teams ht ON g.home_team_id = ht.id
           JOIN teams at ON g.away_team_id = at.id
           WHERE g.date = ?""",
        conn, params=(settle_date,)
    )
    conn.close()

    if len(games) == 0:
        print(f"  No game results found for {settle_date}")
        return

    settled = 0
    total_pl_flat = 0.0
    total_pl_kelly = 0.0

    for idx, bet in bets_df.iterrows():
        # Match to game result
        game = games[
            (games["home_team"].str.contains(bet["home_team"], case=False, na=False))
        ]
        if len(game) == 0:
            continue

        game = game.iloc[0]
        home_won = game["home_score"] > game["away_score"]

        if bet["side"] == "HOME":
            won = home_won
        else:
            won = not home_won

        # Calculate P/L
        flat_pl = ml_payout(FLAT_STAKE, bet["ml"], won)
        kelly_pl = ml_payout(bet["kelly_stake"], bet["ml"], won)

        bets_df.at[idx, "result"] = "win" if won else "loss"
        bets_df.at[idx, "profit_loss"] = flat_pl
        bets_df.at[idx, "settled"] = True

        total_pl_flat += flat_pl
        total_pl_kelly += kelly_pl
        settled += 1

    # Save updated CSV
    bets_df.to_csv(bets_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"  SETTLEMENT — {settle_date}")
    print(f"{'=' * 50}")
    print(f"  Settled: {settled}/{len(bets_df)} bets")

    if settled > 0:
        wins = (bets_df["result"] == "win").sum()
        losses = (bets_df["result"] == "loss").sum()
        print(f"  Record:  {wins}W-{losses}L ({wins/settled:.0%})")
        print(f"  P/L (flat $100): ${total_pl_flat:+,.0f}")
        print(f"  P/L (Kelly):     ${total_pl_kelly:+,.0f}")


def show_summary():
    """Show overall P/L summary across all tracked bet dates."""
    if not BETS_DIR.exists():
        print("  No bet tracking data found.")
        return

    all_files = sorted(BETS_DIR.glob("bets_*.csv"))
    if not all_files:
        print("  No bet tracking data found.")
        return

    all_bets = []
    for f in all_files:
        df = pd.read_csv(f)
        all_bets.append(df)

    combined = pd.concat(all_bets, ignore_index=True)
    settled = combined[combined["settled"] == True]
    pending = combined[combined["settled"] != True]

    print(f"\n{'=' * 60}")
    print(f"  BETTING P/L SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total bets tracked: {len(combined)}")
    print(f"  Settled: {len(settled)}  |  Pending: {len(pending)}")

    if len(settled) > 0:
        wins = (settled["result"] == "win").sum()
        losses = (settled["result"] == "loss").sum()
        total_pl = settled["profit_loss"].sum()
        total_staked = len(settled) * FLAT_STAKE
        roi = total_pl / total_staked * 100 if total_staked > 0 else 0

        print(f"\n  Record:  {wins}W-{losses}L ({wins/len(settled):.1%})")
        print(f"  Total P/L:  ${total_pl:+,.0f}")
        print(f"  Total staked: ${total_staked:,.0f}")
        print(f"  ROI: {roi:+.1f}%")

        # By-date breakdown
        print(f"\n  {'Date':>12s}  {'Bets':>5s}  {'W-L':>6s}  {'P/L':>10s}  {'ROI':>8s}")
        print(f"  {'-'*45}")
        for f in all_files:
            df = pd.read_csv(f)
            s = df[df["settled"] == True]
            if len(s) == 0:
                continue
            d = f.stem.replace("bets_", "")
            w = (s["result"] == "win").sum()
            l = (s["result"] == "loss").sum()
            pl = s["profit_loss"].sum()
            r = pl / (len(s) * FLAT_STAKE) * 100
            print(f"  {d:>12s}  {len(s):>5d}  {w}-{l:>3d}  ${pl:>+9,.0f}  {r:>+7.1f}%")

    # Pending bets
    if len(pending) > 0:
        print(f"\n  Pending bets ({len(pending)}):")
        for _, b in pending.iterrows():
            print(f"    {b.get('date', '?'):>12s}  {b['side']:>4s} {b['team_bet']:<22s}  "
                  f"{b['ml']:+.0f}  edge={b['edge']:.1%}")


# ═══════════════════════════════════════════════════════════════
# BACKTEST MODE
# ═══════════════════════════════════════════════════════════════

def backtest(target_date):
    """Backtest betting recommendations against historical odds and results."""
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST — {target_date}")
    print(f"{'=' * 60}")

    # Load historical odds from DB
    odds_df = load_historical_odds(target_date)
    if len(odds_df) == 0:
        print("  No historical odds found for this date.")
        return

    # Load model predictions (run backtest)
    sys.path.insert(0, str(ROOT / "scripts"))
    from predict_today import predict_games

    d = datetime.strptime(target_date, "%Y-%m-%d").date()
    preds_df = predict_games(d, save=False, backtest=True)

    if preds_df is None or len(preds_df) == 0:
        print("  No predictions generated.")
        return

    # Find value bets
    bets_df = find_value_bets(preds_df, odds_df)
    display_recommendations(bets_df, target_date)

    if len(bets_df) == 0:
        return

    # Settle using actual results (already in preds_df from backtest)
    total_pl_flat = 0.0
    wins = 0
    for _, bet in bets_df.iterrows():
        game = preds_df[preds_df["home_team"] == bet["home_team"]]
        if len(game) == 0:
            continue
        game = game.iloc[0]
        if "actual_winner" not in game or pd.isna(game.get("home_score")):
            continue

        home_won = game["home_score"] > game["away_score"]
        won = home_won if bet["side"] == "HOME" else not home_won
        pl = ml_payout(FLAT_STAKE, bet["ml"], won)
        total_pl_flat += pl
        if won:
            wins += 1

    n = len(bets_df)
    print(f"\n  Backtest results: {wins}/{n} ({wins/n:.0%})")
    print(f"  P/L (flat $100): ${total_pl_flat:+,.0f}")
    print(f"  ROI: {total_pl_flat/(n*FLAT_STAKE)*100:+.1f}%")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Daily MLB betting recommendations"
    )
    parser.add_argument("--date", type=str,
                        help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--predictions", type=str,
                        help="Path to predictions CSV")
    parser.add_argument("--odds", type=str,
                        help="Path to odds CSV (home_team,away_team,home_ml,away_ml)")
    parser.add_argument("--settle", type=str,
                        help="Settle bets for date (YYYY-MM-DD)")
    parser.add_argument("--summary", action="store_true",
                        help="Show P/L summary")
    parser.add_argument("--backtest", type=str,
                        help="Backtest against historical odds (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true",
                        help="Save bet recommendations to CSV")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE,
                        help=f"Minimum edge threshold (default: {MIN_EDGE})")
    args = parser.parse_args()

    # Summary mode
    if args.summary:
        show_summary()
        return

    # Settle mode
    if args.settle:
        settle_bets(args.settle)
        return

    # Backtest mode
    if args.backtest:
        backtest(args.backtest)
        return

    # Normal mode: predictions + odds → recommendations
    target_date = args.date or date.today().strftime("%Y-%m-%d")
    print(f"\n  Daily Bets — {target_date}")
    print(f"  Min edge: {args.min_edge:.0%}")

    # Load predictions
    preds_df = load_predictions(args.predictions, target_date)
    if preds_df is None or len(preds_df) == 0:
        print("  No predictions available.")
        return

    # Load odds
    odds_df = load_daily_odds(target_date, args.odds)

    if odds_df is None or len(odds_df) == 0:
        print("\n  No odds file found. Enter odds manually:")
        odds_df = enter_odds_manually(preds_df)

    if odds_df is None or len(odds_df) == 0:
        print("  No odds provided. Exiting.")
        return

    # Find value bets
    bets_df = find_value_bets(preds_df, odds_df, min_edge=args.min_edge)

    # Display
    display_recommendations(bets_df, target_date)

    # Save
    if args.save and len(bets_df) > 0:
        save_bets(bets_df, target_date)


if __name__ == "__main__":
    main()
