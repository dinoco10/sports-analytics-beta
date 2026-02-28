"""
track_clv.py — Closing Line Value (CLV) Tracker
=================================================
Measures whether our model predictions beat the closing line —
the gold standard benchmark for sports betting model quality.

CLV = model_prob - close_implied_prob (devigged)
Positive CLV over hundreds of bets = confirmed edge.

Modes:
  --historical           Backfill CLV from existing JSON odds (2021-2025)
  --morning              Log today's predictions + opening lines
  --closing              Fetch closing lines, calculate CLV
  --settle               Settle yesterday's games
  --report [--last N]    Show CLV statistics

Uses Bet365 as primary book (sharpest in our dataset).
Pinnacle support planned for future.

Author: Loko
"""

import argparse
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "mlb_analytics.db"
ODDS_JSON = ROOT / "data" / "odds" / "mlb_odds_dataset.json"
CLV_DB_PATH = ROOT / "data" / "clv_tracking.db"

# Primary sportsbook for CLV measurement
PRIMARY_BOOK = "bet365"
FALLBACK_BOOKS = ["draftkings", "fanduel", "caesars", "betmgm"]

# Minimum edge for "bet-worthy" CLV analysis
MIN_EDGE = 0.03


# ═══════════════════════════════════════════════════════════════
# ODDS MATH
# ═══════════════════════════════════════════════════════════════

def american_to_implied(ml):
    """American moneyline -> raw implied probability (includes vig)."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def devig_power(home_ml, away_ml):
    """Remove vig using the power method (more accurate than multiplicative).

    The power method solves for exponent n such that:
        implied_home^n + implied_away^n = 1

    Falls back to multiplicative if power method fails to converge.
    """
    home_raw = american_to_implied(home_ml)
    away_raw = american_to_implied(away_ml)
    total = home_raw + away_raw

    if total <= 1.0 or total > 1.20:
        # No vig or extreme vig — use multiplicative
        return home_raw / total, away_raw / total

    # Binary search for exponent n
    lo, hi = 0.5, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2
        val = home_raw ** mid + away_raw ** mid
        if val > 1.0:
            lo = mid
        else:
            hi = mid

    n = (lo + hi) / 2
    fair_home = home_raw ** n
    fair_away = away_raw ** n
    s = fair_home + fair_away
    return fair_home / s, fair_away / s


def american_to_decimal(ml):
    """American moneyline -> decimal odds."""
    if ml < 0:
        return 1 + 100 / abs(ml)
    else:
        return 1 + ml / 100


def ml_payout(stake, ml, won):
    """Calculate profit from a moneyline bet."""
    if not won:
        return -stake
    if ml < 0:
        return stake * (100 / abs(ml))
    else:
        return stake * (ml / 100)


# ═══════════════════════════════════════════════════════════════
# CLV DATABASE
# ═══════════════════════════════════════════════════════════════

def init_clv_db():
    """Create CLV tracking table if it doesn't exist."""
    conn = sqlite3.connect(str(CLV_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clv_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            -- Model predictions
            model_home_prob REAL,
            model_away_prob REAL,
            -- Opening line (from primary book)
            open_home_ml REAL,
            open_away_ml REAL,
            open_home_implied REAL,
            open_away_implied REAL,
            -- Closing line (from primary book)
            close_home_ml REAL,
            close_away_ml REAL,
            close_home_implied REAL,
            close_away_implied REAL,
            -- CLV metrics
            clv_home REAL,       -- model_home_prob - close_home_implied
            clv_away REAL,       -- model_away_prob - close_away_implied
            -- Bet info
            bet_side TEXT,       -- 'home', 'away', or 'none'
            bet_edge REAL,       -- model edge at open (model_prob - open_implied)
            bet_ml REAL,         -- moneyline odds bet was placed at
            -- Results
            home_score INTEGER,
            away_score INTEGER,
            actual_winner TEXT,   -- 'home' or 'away'
            bet_result TEXT,      -- 'win', 'loss', or NULL
            pnl REAL,            -- dollar P/L on $100 flat bet
            -- Metadata
            sportsbook TEXT DEFAULT 'bet365',
            UNIQUE(date, home_team, away_team, bet_side)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_clv_date ON clv_tracking(date)
    """)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════════════
# LOAD ODDS FROM JSON
# ═══════════════════════════════════════════════════════════════

def load_odds_json():
    """Load the full JSON odds dataset."""
    if not ODDS_JSON.exists():
        raise FileNotFoundError(f"Odds JSON not found: {ODDS_JSON}")

    with open(ODDS_JSON) as f:
        return json.load(f)


def get_game_odds(game_data, book=PRIMARY_BOOK):
    """Extract moneyline odds for a specific sportsbook from a game entry.

    Returns (open_home_ml, open_away_ml, close_home_ml, close_away_ml) or None.
    """
    mls = game_data.get("odds", {}).get("moneyline", [])
    for ml in mls:
        if ml.get("sportsbook") == book:
            opening = ml.get("openingLine", {})
            closing = ml.get("currentLine", {})
            return (
                opening.get("homeOdds"),
                opening.get("awayOdds"),
                closing.get("homeOdds"),
                closing.get("awayOdds"),
            )

    # Fallback to other books
    for fallback in FALLBACK_BOOKS:
        for ml in mls:
            if ml.get("sportsbook") == fallback:
                opening = ml.get("openingLine", {})
                closing = ml.get("currentLine", {})
                return (
                    opening.get("homeOdds"),
                    opening.get("awayOdds"),
                    closing.get("homeOdds"),
                    closing.get("awayOdds"),
                )
    return None


def get_game_info(game_data):
    """Extract game metadata from JSON entry."""
    gv = game_data.get("gameView", {})
    home = gv.get("homeTeam", {})
    away = gv.get("awayTeam", {})
    return {
        "home_team": home.get("fullName", home.get("name", "")),
        "home_short": home.get("shortName", ""),
        "away_team": away.get("fullName", away.get("name", "")),
        "away_short": away.get("shortName", ""),
        "home_score": gv.get("homeTeamScore"),
        "away_score": gv.get("awayTeamScore"),
        "status": gv.get("gameStatusText", ""),
        "venue": gv.get("venueName", ""),
    }


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def load_model_predictions(target_date):
    """Load model predictions from game_features.csv + saved model.

    For historical backfill, we use the saved features + model.
    """
    features_path = ROOT / "data" / "features" / "game_features.csv"
    if not features_path.exists():
        return None

    import lightgbm as lgb

    model_path = ROOT / "models" / "win_probability_lgbm.txt"
    meta_path = ROOT / "models" / "win_probability_meta.json"
    medians_path = ROOT / "models" / "feature_medians.json"

    if not model_path.exists():
        print("  Model not found. Run train_win_model.py first.")
        return None

    booster = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta["features"]

    medians = {}
    if medians_path.exists():
        with open(medians_path) as f:
            medians = json.load(f)

    # Load features
    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    day_df = df[df["date"] == target_date].copy()

    if len(day_df) == 0:
        return None

    # Load team names
    conn = sqlite3.connect(str(DB_PATH))
    teams = pd.read_sql("SELECT id, name FROM teams", conn)
    conn.close()
    id_to_name = dict(zip(teams["id"], teams["name"]))

    # Predict
    X = day_df[features].copy()
    for col in features:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
        else:
            X[col] = X[col].fillna(0.0)

    probs = booster.predict(X)

    results = []
    for idx, (_, row) in enumerate(day_df.iterrows()):
        results.append({
            "home_team": id_to_name.get(row["home_team_id"], f"team_{row['home_team_id']}"),
            "away_team": id_to_name.get(row["away_team_id"], f"team_{row['away_team_id']}"),
            "model_home_prob": float(probs[idx]),
            "model_away_prob": float(1.0 - probs[idx]),
            "home_score": row.get("home_score"),
            "away_score": row.get("away_score"),
        })

    return results


# ═══════════════════════════════════════════════════════════════
# TEAM NAME MATCHING
# ═══════════════════════════════════════════════════════════════

TEAM_ALIASES = {
    "Arizona": "Diamondbacks", "Arizona Diamondbacks": "Diamondbacks",
    "Atlanta": "Braves", "Atlanta Braves": "Braves",
    "Baltimore": "Orioles", "Baltimore Orioles": "Orioles",
    "Boston": "Red Sox", "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs", "Chi Cubs": "Cubs",
    "Chicago White Sox": "White Sox", "Chi White Sox": "White Sox",
    "Cincinnati": "Reds", "Cincinnati Reds": "Reds",
    "Cleveland": "Guardians", "Cleveland Guardians": "Guardians",
    "Colorado": "Rockies", "Colorado Rockies": "Rockies",
    "Detroit": "Tigers", "Detroit Tigers": "Tigers",
    "Houston": "Astros", "Houston Astros": "Astros",
    "Kansas City": "Royals", "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels", "LA Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers", "LA Dodgers": "Dodgers",
    "Miami": "Marlins", "Miami Marlins": "Marlins",
    "Milwaukee": "Brewers", "Milwaukee Brewers": "Brewers",
    "Minnesota": "Twins", "Minnesota Twins": "Twins",
    "New York Mets": "Mets", "NY Mets": "Mets",
    "New York Yankees": "Yankees", "NY Yankees": "Yankees",
    "Oakland": "Athletics", "Oakland Athletics": "Athletics",
    "Philadelphia": "Phillies", "Philadelphia Phillies": "Phillies",
    "Pittsburgh": "Pirates", "Pittsburgh Pirates": "Pirates",
    "San Diego": "Padres", "San Diego Padres": "Padres",
    "San Francisco": "Giants", "San Francisco Giants": "Giants",
    "Seattle": "Mariners", "Seattle Mariners": "Mariners",
    "St. Louis": "Cardinals", "St. Louis Cardinals": "Cardinals",
    "Tampa Bay": "Rays", "Tampa Bay Rays": "Rays",
    "Texas": "Rangers", "Texas Rangers": "Rangers",
    "Toronto": "Blue Jays", "Toronto Blue Jays": "Blue Jays",
    "Washington": "Nationals", "Washington Nationals": "Nationals",
}


def normalize_team(name):
    """Normalize a team name to our DB format."""
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    for alias, db_name in TEAM_ALIASES.items():
        if alias.lower() in name.lower() or name.lower() in alias.lower():
            return db_name
    return name


def match_prediction_to_game(pred, game_info):
    """Check if a prediction matches a game from the odds JSON."""
    pred_home = normalize_team(pred["home_team"])
    pred_away = normalize_team(pred["away_team"])
    game_home = normalize_team(game_info["home_team"])
    game_away = normalize_team(game_info["away_team"])
    return pred_home == game_home and pred_away == game_away


# ═══════════════════════════════════════════════════════════════
# HISTORICAL BACKFILL
# ═══════════════════════════════════════════════════════════════

def backfill_historical(start_date=None, end_date=None):
    """Backfill CLV data from JSON odds + model predictions.

    Computes CLV for every game where we have both model predictions
    and Bet365 closing lines.
    """
    print("=" * 70)
    print("  CLV HISTORICAL BACKFILL")
    print("=" * 70)

    odds_data = load_odds_json()
    all_dates = sorted(odds_data.keys())

    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    print(f"  Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} dates)")

    conn = init_clv_db()
    total_inserted = 0
    total_matched = 0
    total_dates = 0

    for d in all_dates:
        preds = load_model_predictions(d)
        if preds is None:
            continue

        games = odds_data[d]
        day_inserted = 0

        for game_data in games:
            game_info = get_game_info(game_data)
            odds = get_game_odds(game_data)

            if odds is None:
                continue

            open_h_ml, open_a_ml, close_h_ml, close_a_ml = odds

            if not all([open_h_ml, open_a_ml, close_h_ml, close_a_ml]):
                continue

            # Match to prediction
            matched_pred = None
            for pred in preds:
                if match_prediction_to_game(pred, game_info):
                    matched_pred = pred
                    break

            if matched_pred is None:
                continue

            total_matched += 1

            # Devig
            open_h_fair, open_a_fair = devig_power(open_h_ml, open_a_ml)
            close_h_fair, close_a_fair = devig_power(close_h_ml, close_a_ml)

            model_h = matched_pred["model_home_prob"]
            model_a = matched_pred["model_away_prob"]

            # CLV: model vs closing line
            clv_home = model_h - close_h_fair
            clv_away = model_a - close_a_fair

            # Determine bet side (which side had edge at open)
            home_edge = model_h - open_h_fair
            away_edge = model_a - open_a_fair

            bet_side = "none"
            bet_edge = 0.0
            bet_ml = None

            if home_edge >= MIN_EDGE and home_edge >= away_edge:
                bet_side = "home"
                bet_edge = home_edge
                bet_ml = open_h_ml
            elif away_edge >= MIN_EDGE:
                bet_side = "away"
                bet_edge = away_edge
                bet_ml = open_a_ml

            # Game result
            h_score = game_info.get("home_score")
            a_score = game_info.get("away_score")
            actual_winner = None
            bet_result = None
            pnl = None

            if h_score is not None and a_score is not None:
                try:
                    h_score = int(h_score)
                    a_score = int(a_score)
                    actual_winner = "home" if h_score > a_score else "away"

                    if bet_side != "none" and bet_ml is not None:
                        won = (bet_side == actual_winner)
                        bet_result = "win" if won else "loss"
                        pnl = ml_payout(100.0, bet_ml, won)
                except (ValueError, TypeError):
                    pass

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO clv_tracking (
                        date, home_team, away_team,
                        model_home_prob, model_away_prob,
                        open_home_ml, open_away_ml,
                        open_home_implied, open_away_implied,
                        close_home_ml, close_away_ml,
                        close_home_implied, close_away_implied,
                        clv_home, clv_away,
                        bet_side, bet_edge, bet_ml,
                        home_score, away_score, actual_winner,
                        bet_result, pnl, sportsbook
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    d, normalize_team(game_info["home_team"]),
                    normalize_team(game_info["away_team"]),
                    model_h, model_a,
                    open_h_ml, open_a_ml, open_h_fair, open_a_fair,
                    close_h_ml, close_a_ml, close_h_fair, close_a_fair,
                    clv_home, clv_away,
                    bet_side, bet_edge, bet_ml,
                    h_score, a_score, actual_winner,
                    bet_result, pnl, PRIMARY_BOOK,
                ))
                day_inserted += 1
            except Exception:
                pass

        if day_inserted > 0:
            conn.commit()
            total_inserted += day_inserted
            total_dates += 1

        if total_dates % 100 == 0 and total_dates > 0:
            print(f"  Processed {total_dates} dates, {total_inserted} games...")

    conn.commit()
    conn.close()

    print(f"\n  Done!")
    print(f"  Dates processed: {total_dates}")
    print(f"  Games matched:   {total_matched}")
    print(f"  Games inserted:  {total_inserted}")


# ═══════════════════════════════════════════════════════════════
# CLV REPORT
# ═══════════════════════════════════════════════════════════════

def report(last_n_days=None, season=None):
    """Generate CLV statistics report."""
    conn = init_clv_db()

    query = "SELECT * FROM clv_tracking WHERE actual_winner IS NOT NULL"
    params = []

    if last_n_days:
        cutoff = (date.today() - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
        query += " AND date >= ?"
        params.append(cutoff)

    if season:
        query += " AND date LIKE ?"
        params.append(f"{season}-%")

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if len(df) == 0:
        print("  No CLV data found. Run --historical first.")
        return

    print("=" * 70)
    print("  CLOSING LINE VALUE (CLV) REPORT — Bet365")
    if last_n_days:
        print(f"  Period: last {last_n_days} days")
    if season:
        print(f"  Season: {season}")
    print("=" * 70)

    # --- Overall CLV Stats ---
    print(f"\n  ALL GAMES ({len(df)} total)")
    print(f"  {'-' * 55}")

    # CLV on the side the model favored
    df["favored_side"] = np.where(df["model_home_prob"] >= 0.5, "home", "away")
    df["model_best_prob"] = np.where(
        df["favored_side"] == "home", df["model_home_prob"], df["model_away_prob"]
    )
    df["close_best_implied"] = np.where(
        df["favored_side"] == "home", df["close_home_implied"], df["close_away_implied"]
    )
    df["clv_best"] = df["model_best_prob"] - df["close_best_implied"]

    avg_clv = df["clv_best"].mean()
    pct_positive = (df["clv_best"] > 0).mean()
    median_clv = df["clv_best"].median()

    print(f"  Average CLV:         {avg_clv:+.4f} ({avg_clv*100:+.2f}%)")
    print(f"  Median CLV:          {median_clv:+.4f} ({median_clv*100:+.2f}%)")
    print(f"  % Positive CLV:      {pct_positive:.1%}")
    print(f"  CLV std dev:         {df['clv_best'].std():.4f}")

    # --- Bets Only ---
    bets = df[df["bet_side"] != "none"].copy()

    if len(bets) > 0:
        print(f"\n  VALUE BETS ({len(bets)} bets, {MIN_EDGE:.0%}+ edge at open)")
        print(f"  {'-' * 55}")

        bets["clv_bet_side"] = np.where(
            bets["bet_side"] == "home", bets["clv_home"], bets["clv_away"]
        )

        avg_bet_clv = bets["clv_bet_side"].mean()
        pct_pos_clv = (bets["clv_bet_side"] > 0).mean()

        wins = (bets["bet_result"] == "win").sum()
        losses = (bets["bet_result"] == "loss").sum()
        total_pnl = bets["pnl"].sum()
        total_staked = len(bets) * 100
        roi = total_pnl / total_staked * 100 if total_staked > 0 else 0

        print(f"  Avg CLV (bet side):  {avg_bet_clv:+.4f} ({avg_bet_clv*100:+.2f}%)")
        print(f"  % Positive CLV:      {pct_pos_clv:.1%}")
        print(f"  Avg edge at open:    {bets['bet_edge'].mean():.1%}")
        print(f"  Record:              {wins}W-{losses}L ({wins/(wins+losses):.1%})")
        print(f"  P/L (flat $100):     ${total_pnl:+,.0f}")
        print(f"  ROI:                 {roi:+.1f}%")

        # CLV+ vs CLV- bet performance
        clv_pos = bets[bets["clv_bet_side"] > 0]
        clv_neg = bets[bets["clv_bet_side"] <= 0]

        if len(clv_pos) > 0 and len(clv_neg) > 0:
            print(f"\n  CLV+ BETS vs CLV- BETS")
            print(f"  {'-' * 55}")
            for label, subset in [("CLV+", clv_pos), ("CLV-", clv_neg)]:
                w = (subset["bet_result"] == "win").sum()
                l = (subset["bet_result"] == "loss").sum()
                pl = subset["pnl"].sum()
                r = pl / (len(subset) * 100) * 100
                print(f"  {label}: {len(subset)} bets, {w}W-{l}L ({w/(w+l):.1%}), "
                      f"P/L ${pl:+,.0f}, ROI {r:+.1f}%")

    # --- By Season ---
    df["season"] = df["date"].str[:4]
    seasons = sorted(df["season"].unique())

    if len(seasons) > 1:
        print(f"\n  BY SEASON")
        print(f"  {'-' * 55}")
        print(f"  {'Season':>6s}  {'Games':>6s}  {'Avg CLV':>9s}  {'CLV+%':>6s}  "
              f"{'Bets':>5s}  {'W-L':>7s}  {'P/L':>10s}  {'ROI':>7s}")

        for s in seasons:
            sdf = df[df["season"] == s]
            sbets = sdf[sdf["bet_side"] != "none"]

            avg_c = sdf["clv_best"].mean()
            pct_p = (sdf["clv_best"] > 0).mean()

            if len(sbets) > 0:
                w = (sbets["bet_result"] == "win").sum()
                l = (sbets["bet_result"] == "loss").sum()
                pl = sbets["pnl"].sum()
                r = pl / (len(sbets) * 100) * 100
                wl = f"{w}-{l}"
                pl_str = f"${pl:+,.0f}"
                roi_str = f"{r:+.1f}%"
            else:
                wl = "0-0"
                pl_str = "$0"
                roi_str = "N/A"

            print(f"  {s:>6s}  {len(sdf):>6d}  {avg_c:>+9.4f}  {pct_p:>5.1%}  "
                  f"{len(sbets):>5d}  {wl:>7s}  {pl_str:>10s}  {roi_str:>7s}")

    # --- By Month ---
    target_season = seasons[-1] if not season else str(season)
    mdf = df[df["season"] == target_season].copy()
    mdf["month"] = mdf["date"].str[:7]
    months = sorted(mdf["month"].unique())

    if len(months) > 1:
        print(f"\n  BY MONTH ({target_season})")
        print(f"  {'-' * 55}")
        print(f"  {'Month':>7s}  {'Games':>6s}  {'Avg CLV':>9s}  {'CLV+%':>6s}  "
              f"{'Bets':>5s}  {'P/L':>10s}  {'ROI':>7s}")

        for m in months:
            mmdf = mdf[mdf["month"] == m]
            mbets = mmdf[mmdf["bet_side"] != "none"]

            avg_c = mmdf["clv_best"].mean()
            pct_p = (mmdf["clv_best"] > 0).mean()

            if len(mbets) > 0:
                pl = mbets["pnl"].sum()
                r = pl / (len(mbets) * 100) * 100
                pl_str = f"${pl:+,.0f}"
                roi_str = f"{r:+.1f}%"
            else:
                pl_str = "$0"
                roi_str = "N/A"

            print(f"  {m:>7s}  {len(mmdf):>6d}  {avg_c:>+9.4f}  {pct_p:>5.1%}  "
                  f"{len(mbets):>5d}  {pl_str:>10s}  {roi_str:>7s}")

    # --- CLV Calibration ---
    print(f"\n  CLV CALIBRATION (does positive CLV predict wins?)")
    print(f"  {'-' * 55}")

    all_with_bets = df[df["bet_side"] != "none"].copy()
    if len(all_with_bets) > 10:
        all_with_bets["clv_bet_side"] = np.where(
            all_with_bets["bet_side"] == "home",
            all_with_bets["clv_home"],
            all_with_bets["clv_away"]
        )

        bins = [(-1, -0.10), (-0.10, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.10), (0.10, 1)]
        labels = ["<-10%", "-10/-5%", "-5/0%", "0/+5%", "+5/+10%", ">+10%"]

        print(f"  {'CLV Bucket':>10s}  {'N':>5s}  {'Win%':>6s}  {'P/L':>10s}  {'ROI':>7s}")
        for (lo, hi), label in zip(bins, labels):
            bucket = all_with_bets[
                (all_with_bets["clv_bet_side"] > lo) &
                (all_with_bets["clv_bet_side"] <= hi)
            ]
            if len(bucket) == 0:
                continue
            w = (bucket["bet_result"] == "win").sum()
            t = len(bucket)
            pl = bucket["pnl"].sum()
            r = pl / (t * 100) * 100
            print(f"  {label:>10s}  {t:>5d}  {w/t:>5.1%}  ${pl:>+9,.0f}  {r:>+6.1f}%")


# ═══════════════════════════════════════════════════════════════
# LIVE TRACKING (2026 Season stubs)
# ═══════════════════════════════════════════════════════════════

def morning_log(target_date=None):
    """Log today's predictions + opening lines (2026 season)."""
    target = target_date or date.today().strftime("%Y-%m-%d")
    print(f"\n  Morning log for {target}")
    print(f"  Will run predict_today + fetch opening lines from odds API.")
    print(f"  Not yet implemented — use --historical for backfill.")


def closing_log(target_date=None):
    """Fetch closing lines and calculate CLV (2026 season)."""
    target = target_date or date.today().strftime("%Y-%m-%d")
    print(f"\n  Closing log for {target}")
    print(f"  Will fetch closing lines ~30min before first pitch.")
    print(f"  Not yet implemented — use --historical for backfill.")


def settle(target_date=None):
    """Settle yesterday's games."""
    target = target_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n  Settlement for {target}")
    print(f"  Will look up game results and calculate P/L.")
    print(f"  Not yet implemented — use --historical for backfill.")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLV (Closing Line Value) tracker for MLB predictions"
    )
    parser.add_argument("--historical", action="store_true",
                        help="Backfill CLV from JSON odds + model predictions")
    parser.add_argument("--start", type=str,
                        help="Start date for historical backfill (YYYY-MM-DD)")
    parser.add_argument("--end", type=str,
                        help="End date for historical backfill (YYYY-MM-DD)")
    parser.add_argument("--morning", action="store_true",
                        help="Log morning predictions + opening lines")
    parser.add_argument("--closing", action="store_true",
                        help="Fetch closing lines and calculate CLV")
    parser.add_argument("--settle", action="store_true",
                        help="Settle yesterday's games")
    parser.add_argument("--report", action="store_true",
                        help="Show CLV statistics report")
    parser.add_argument("--last", type=int,
                        help="Limit report to last N days")
    parser.add_argument("--season", type=int,
                        help="Limit report to specific season")
    parser.add_argument("--date", type=str,
                        help="Target date for morning/closing/settle")
    args = parser.parse_args()

    if args.historical:
        backfill_historical(args.start, args.end)
    elif args.morning:
        morning_log(args.date)
    elif args.closing:
        closing_log(args.date)
    elif args.settle:
        settle(args.date)
    elif args.report:
        report(last_n_days=args.last, season=args.season)
    else:
        parser.print_help()
        print("\n  Quick start:")
        print("    python scripts/track_clv.py --historical          # Backfill 2021-2025")
        print("    python scripts/track_clv.py --historical --start 2025-04-01  # Just 2025")
        print("    python scripts/track_clv.py --report              # Show CLV stats")
        print("    python scripts/track_clv.py --report --season 2025  # 2025 only")


if __name__ == "__main__":
    main()
