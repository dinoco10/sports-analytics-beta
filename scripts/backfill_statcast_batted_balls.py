"""
backfill_statcast_batted_balls.py — Download Statcast pitch-level data and compute batted ball metrics
======================================================================================================
Uses pybaseball to download pitch-level Statcast data from Baseball Savant,
filters to balls in play, computes barrel/pulled/air ball flags using the
Statcast sliding-scale barrel definition, then aggregates to per-batter-per-game
level and stores in SQLite.

Downloads one week at a time with resume support — safe to interrupt and restart.

Usage:
  python scripts/backfill_statcast_batted_balls.py                          # Full backfill 2021-2025
  python scripts/backfill_statcast_batted_balls.py --start-date 2024-04-01  # Start from 2024
  python scripts/backfill_statcast_batted_balls.py --batch-days 3           # Smaller batches
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Windows encoding fix — avoids UnicodeEncodeError when printing emoji/special chars
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# pybaseball is the main dependency — downloads from Baseball Savant
from pybaseball import statcast

# ─── Paths ────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "mlb_analytics.db"


# ─── Table Creation ───────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS statcast_batted_balls (
    game_pk          INTEGER NOT NULL,
    batter_id        INTEGER NOT NULL,
    game_date        DATE    NOT NULL,
    team             TEXT    NOT NULL,
    opponent         TEXT    NOT NULL,
    stand            TEXT    NOT NULL,
    bip              INTEGER NOT NULL,
    barrels          INTEGER NOT NULL DEFAULT 0,
    pulled           INTEGER NOT NULL DEFAULT 0,
    air_balls        INTEGER NOT NULL DEFAULT 0,
    pulled_air       INTEGER NOT NULL DEFAULT 0,
    pulled_barrels   INTEGER NOT NULL DEFAULT 0,
    barrel_pct       REAL,
    pulled_air_pct   REAL,
    pulled_barrel_pct REAL,
    avg_exit_velo    REAL,
    avg_launch_angle REAL,
    UNIQUE(game_pk, batter_id)
);
"""

# Index for fast lookups by date range (used in feature engineering)
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_statcast_bb_date
    ON statcast_batted_balls(game_date);
"""


def ensure_table(conn):
    """Create the statcast_batted_balls table if it doesn't exist yet."""
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_INDEX_SQL)
    conn.commit()


def get_existing_dates(conn):
    """
    Return a set of dates (as strings 'YYYY-MM-DD') that already have data in the DB.
    We use this for resume support — if a date already has rows, we skip it.
    """
    cursor = conn.execute(
        "SELECT DISTINCT game_date FROM statcast_batted_balls"
    )
    return {row[0] for row in cursor.fetchall()}


def date_range_covered(existing_dates, start_dt, end_dt):
    """
    Check if ALL dates in [start_dt, end_dt] are already in the DB.
    Returns True if the entire batch can be skipped.

    We check every date in the range. If even one date is missing,
    we re-download the whole batch (pybaseball downloads in date ranges).
    """
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        if date_str not in existing_dates:
            return False
        current += timedelta(days=1)
    return True


def compute_is_barrel(ev, la):
    """
    Statcast barrel definition (sliding scale):
    - Minimum 98 mph exit velocity
    - At 98 mph: launch angle must be between 26 and 30 degrees
    - For every 1 mph above 98, the LA range widens by 2 degrees on each side
    - The acceptable LA range is clamped to [8, 50] degrees

    Example at 100 mph (2 mph above 98):
      lower = 26 - (2 * 2) = 22
      upper = 30 + (2 * 2) = 34
      But clamped: lower = max(22, 8) = 22, upper = min(34, 50) = 34

    Example at 110 mph (12 mph above 98):
      lower = 26 - (12 * 2) = 2 → clamped to 8
      upper = 30 + (12 * 2) = 54 → clamped to 50
      So LA must be between 8 and 50

    Returns a boolean Series.
    """
    # How many mph above the 98 threshold
    ev_above = ev - 98.0

    # Calculate the acceptable launch angle window
    la_lower = 26.0 - (ev_above * 2.0)
    la_upper = 30.0 + (ev_above * 2.0)

    # Clamp the window to [8, 50]
    la_lower = la_lower.clip(lower=8.0)
    la_upper = la_upper.clip(upper=50.0)

    # A barrel requires: EV >= 98 AND LA within the sliding window
    return (ev >= 98.0) & (la >= la_lower) & (la <= la_upper)


def compute_is_pulled(hc_x, stand):
    """
    Determine if a ball was pulled based on spray chart coordinates.

    Statcast hc_x is the horizontal hit coordinate:
    - Lower values = toward 3rd base side (left field for RHB)
    - Higher values = toward 1st base side (right field for LHB)

    For a right-handed batter, "pulled" means hitting to left field (hc_x < 100).
    For a left-handed batter, "pulled" means hitting to right field (hc_x > 150).

    These thresholds are approximate — Statcast's coordinate system puts
    home plate around hc_x=125, so <100 is pull side for RHB and >150
    is pull side for LHB.
    """
    return (
        ((stand == "R") & (hc_x < 100.0)) |
        ((stand == "L") & (hc_x > 150.0))
    )


def process_batch(df):
    """
    Take a raw Statcast pitch-level DataFrame, filter to balls in play,
    compute per-pitch flags, then aggregate to batter-game level.

    Returns a DataFrame ready for INSERT into the database, or None if
    no balls in play were found in this batch.
    """
    # ── Step 1: Filter to balls in play ──────────────────────
    # The 'description' column tells us what happened on each pitch.
    # 'hit_into_play' means the batter made contact and put the ball in play
    # (as opposed to swinging strikes, called strikes, foul balls, etc.)
    bip = df[df["description"] == "hit_into_play"].copy()

    if bip.empty:
        return None

    # ── Step 2: Clean numeric columns ────────────────────────
    # These columns can have NaN values (e.g., if tracking system failed).
    # pd.to_numeric with errors='coerce' turns non-numeric values into NaN.
    bip["launch_speed"] = pd.to_numeric(bip["launch_speed"], errors="coerce")
    bip["launch_angle"] = pd.to_numeric(bip["launch_angle"], errors="coerce")
    bip["hc_x"] = pd.to_numeric(bip["hc_x"], errors="coerce")

    # ── Step 3: Compute per-pitch flags ──────────────────────

    # Barrel: uses the Statcast sliding-scale definition
    # We need both EV and LA to be non-null to compute this
    has_ev_la = bip["launch_speed"].notna() & bip["launch_angle"].notna()
    bip["is_barrel"] = False
    bip.loc[has_ev_la, "is_barrel"] = compute_is_barrel(
        bip.loc[has_ev_la, "launch_speed"],
        bip.loc[has_ev_la, "launch_angle"],
    )

    # Pulled: depends on batter handedness and hc_x spray coordinate
    has_hc = bip["hc_x"].notna()
    bip["is_pulled"] = False
    bip.loc[has_hc, "is_pulled"] = compute_is_pulled(
        bip.loc[has_hc, "hc_x"],
        bip.loc[has_hc, "stand"],
    )

    # Air ball: launch angle > 10 degrees (line drives + fly balls)
    # Ground balls are typically < 10 degrees
    has_la = bip["launch_angle"].notna()
    bip["is_air_ball"] = False
    bip.loc[has_la, "is_air_ball"] = bip.loc[has_la, "launch_angle"] > 10.0

    # Pulled air ball: pulled AND in the air — the "home run swing" profile
    bip["is_pulled_air"] = bip["is_pulled"] & bip["is_air_ball"]

    # Pulled barrel: pulled AND a barrel — the elite quality contact
    bip["is_pulled_barrel"] = bip["is_pulled"] & bip["is_barrel"]

    # ── Step 4: Determine batter's team ──────────────────────
    # Statcast has 'home_team' and 'away_team' columns, plus 'inning_topbot'
    # which tells us if the batter's team is home (Bot = bottom) or away (Top = top).
    # If it's the bottom of the inning, the home team is batting.
    bip["team"] = np.where(
        bip["inning_topbot"] == "Bot",
        bip["home_team"],
        bip["away_team"],
    )
    bip["opponent"] = np.where(
        bip["inning_topbot"] == "Bot",
        bip["away_team"],
        bip["home_team"],
    )

    # ── Step 5: Aggregate to batter-game level ───────────────
    # Group by game + batter + date + teams + handedness
    group_cols = ["game_pk", "batter", "game_date", "team", "opponent", "stand"]

    agg = bip.groupby(group_cols, as_index=False).agg(
        bip=("is_barrel", "count"),              # Total balls in play
        barrels=("is_barrel", "sum"),             # Count of barrels
        pulled=("is_pulled", "sum"),              # Count of pulled balls
        air_balls=("is_air_ball", "sum"),         # Count of air balls
        pulled_air=("is_pulled_air", "sum"),      # Count of pulled + air
        pulled_barrels=("is_pulled_barrel", "sum"),  # Count of pulled + barrel
        avg_exit_velo=("launch_speed", "mean"),   # Average exit velocity
        avg_launch_angle=("launch_angle", "mean"),  # Average launch angle
    )

    # ── Step 6: Compute rate stats ───────────────────────────
    # Rates are more useful than counts for modeling because they normalize
    # for the number of balls in play (a guy with 5 BIP and 2 barrels is
    # more impressive than a guy with 1 BIP and 1 barrel in context).
    agg["barrel_pct"] = (agg["barrels"] / agg["bip"]).round(4)
    agg["pulled_air_pct"] = (agg["pulled_air"] / agg["bip"]).round(4)
    agg["pulled_barrel_pct"] = (agg["pulled_barrels"] / agg["bip"]).round(4)

    # Convert integer columns from float (sum of booleans) to int
    int_cols = ["bip", "barrels", "pulled", "air_balls", "pulled_air", "pulled_barrels"]
    agg[int_cols] = agg[int_cols].astype(int)

    # Rename 'batter' to 'batter_id' for the DB schema
    agg = agg.rename(columns={"batter": "batter_id"})

    # Ensure game_date is a string in YYYY-MM-DD format for SQLite
    agg["game_date"] = pd.to_datetime(agg["game_date"]).dt.strftime("%Y-%m-%d")

    # Convert pandas NA/NaN to Python None for SQLite compatibility
    # SQLite doesn't understand pandas NAType — it needs Python None for NULL
    agg = agg.where(agg.notna(), None)

    return agg


def insert_batch(conn, agg_df):
    """
    Insert aggregated batter-game rows into the database.
    Uses INSERT OR REPLACE so re-running on the same data is safe
    (the UNIQUE constraint on game_pk + batter_id handles deduplication).
    """
    insert_sql = """
    INSERT OR REPLACE INTO statcast_batted_balls (
        game_pk, batter_id, game_date, team, opponent, stand,
        bip, barrels, pulled, air_balls, pulled_air, pulled_barrels,
        barrel_pct, pulled_air_pct, pulled_barrel_pct,
        avg_exit_velo, avg_launch_angle
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # Convert DataFrame to list of tuples, replacing pd.NA/np.nan with None
    # SQLite can't handle pandas NAType — needs Python None for NULL
    cols = [
        "game_pk", "batter_id", "game_date", "team", "opponent", "stand",
        "bip", "barrels", "pulled", "air_balls", "pulled_air", "pulled_barrels",
        "barrel_pct", "pulled_air_pct", "pulled_barrel_pct",
        "avg_exit_velo", "avg_launch_angle",
    ]
    rows = []
    for _, row in agg_df[cols].iterrows():
        rows.append(tuple(
            None if pd.isna(v) else v for v in row
        ))

    conn.executemany(insert_sql, rows)
    conn.commit()

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Statcast batted ball data from Baseball Savant"
    )
    parser.add_argument(
        "--start-date", default="2021-04-01",
        help="First date to download (YYYY-MM-DD). Default: 2021-04-01"
    )
    parser.add_argument(
        "--end-date", default="2025-11-02",
        help="Last date to download (YYYY-MM-DD). Default: 2025-11-02"
    )
    parser.add_argument(
        "--batch-days", type=int, default=7,
        help="Number of days per download batch. Default: 7"
    )
    args = parser.parse_args()

    # Parse the date range
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    batch_days = args.batch_days

    print(f"Statcast Batted Ball Backfill")
    print(f"  Range: {start.date()} to {end.date()}")
    print(f"  Batch size: {batch_days} days")
    print(f"  DB: {DB_PATH}")
    print()

    # ── Connect to SQLite and ensure the table exists ────────
    conn = sqlite3.connect(str(DB_PATH))
    ensure_table(conn)

    # ── Load existing dates for resume support ───────────────
    existing_dates = get_existing_dates(conn)
    print(f"  Existing dates in DB: {len(existing_dates)}")
    print()

    # ── Build list of batches ────────────────────────────────
    # Each batch is a (start_date, end_date) tuple covering batch_days days.
    batches = []
    current = start
    while current <= end:
        batch_end = min(current + timedelta(days=batch_days - 1), end)

        # Skip the 2020 COVID season entirely — it adds noise to models
        if current.year == 2020 or batch_end.year == 2020:
            current = batch_end + timedelta(days=1)
            continue

        batches.append((current, batch_end))
        current = batch_end + timedelta(days=1)

    total_batches = len(batches)
    total_rows = 0
    skipped = 0
    errors = 0

    print(f"  Total batches to process: {total_batches}")
    print("=" * 60)

    for i, (batch_start, batch_end) in enumerate(batches, 1):
        start_str = batch_start.strftime("%Y-%m-%d")
        end_str = batch_end.strftime("%Y-%m-%d")

        # ── Resume support: skip if all dates in this batch are already done ──
        if date_range_covered(existing_dates, batch_start, batch_end):
            skipped += 1
            print(f"  [{i}/{total_batches}] {start_str} to {end_str} — SKIP (already in DB)")
            continue

        print(f"  [{i}/{total_batches}] {start_str} to {end_str} — downloading...", end=" ")

        try:
            # ── Download pitch-level data from Baseball Savant ──
            # pybaseball.statcast() scrapes the Statcast search CSV endpoint.
            # It returns a DataFrame with ~118 columns per pitch.
            raw = statcast(start_dt=start_str, end_dt=end_str)

            if raw is None or raw.empty:
                print("no data")
                # Mark these dates as done so we don't retry them
                for d in range((batch_end - batch_start).days + 1):
                    dt = batch_start + timedelta(days=d)
                    existing_dates.add(dt.strftime("%Y-%m-%d"))
                continue

            # ── Process: filter BIP, compute flags, aggregate ──
            agg = process_batch(raw)

            if agg is None or agg.empty:
                print("no BIP found")
                continue

            # ── Insert into SQLite ──
            n_rows = insert_batch(conn, agg)
            total_rows += n_rows

            # Update the existing dates set for future skip checks
            for d in range((batch_end - batch_start).days + 1):
                dt = batch_start + timedelta(days=d)
                existing_dates.add(dt.strftime("%Y-%m-%d"))

            print(f"{n_rows} batter-games inserted (total: {total_rows})")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")

        # Be nice to Baseball Savant — wait 1 second between requests
        time.sleep(1)

    # ── Summary ──────────────────────────────────────────────
    conn.close()
    print()
    print("=" * 60)
    print(f"Backfill complete!")
    print(f"  Total batter-game rows inserted: {total_rows:,}")
    print(f"  Batches skipped (already in DB): {skipped}")
    print(f"  Batches with errors: {errors}")
    print(f"  Database: {DB_PATH}")


if __name__ == "__main__":
    main()
