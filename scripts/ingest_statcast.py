"""
ingest_statcast.py — Pull Statcast Metrics from Baseball Savant
================================================================
Downloads season-level Statcast data for all qualified batters and stores
it in the player_statcast_metrics table. Uses three data sources:

1. Baseball Savant Expected Stats API (xwOBA, xBA, xSLG, actual wOBA/BA/SLG)
2. Baseball Savant Exit Velo/Barrels API (avg EV, barrel rate, hard hit %, sweet spot %)
3. Baseball Savant Custom Leaderboard CSV (K%, BB%, BABIP, whiff%, chase%, zone contact%,
   pull%, GB/FB/LD%, and raw HR + FB counts for HR/FB ratio)

The script matches Savant player IDs to our database's players.mlb_id.
Players not in our DB are skipped (they likely had <50 PA in our game logs).

Usage:
    python scripts/ingest_statcast.py                  # All 3 seasons (2023-2025)
    python scripts/ingest_statcast.py --season 2025    # Single season
    python scripts/ingest_statcast.py --min-pa 100     # Higher PA threshold
    python scripts/ingest_statcast.py --summary        # Check what's in DB

Author: Loko
"""

import argparse
import io
import sys
import time
from pathlib import Path

# Fix Windows cp1252 encoding crashes when printing special characters
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.storage.database import engine, get_session
from src.storage.models import PlayerStatcastMetrics, Player


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 1: Expected Stats (xwOBA, xBA, xSLG + actuals)
# ═══════════════════════════════════════════════════════════════

def fetch_expected_stats(season, min_pa=50):
    """
    Pull expected stats from Baseball Savant via pybaseball.
    Returns: DataFrame with player_id, xwoba, xba, xslg, woba, ba, slg, pa.

    This is the core "luck filter" data — if xwOBA >> wOBA, the player was unlucky
    and we should project them upward.
    """
    from pybaseball import statcast_batter_expected_stats

    print(f"  Fetching expected stats for {season} (min PA={min_pa})...")
    df = statcast_batter_expected_stats(season, minPA=min_pa)

    # Rename columns to our naming convention
    # pybaseball returns: player_id, pa, ba, est_ba, slg, est_slg, woba, est_woba
    df = df.rename(columns={
        'player_id': 'mlb_player_id',
        'est_ba': 'xba',
        'est_slg': 'xslg',
        'est_woba': 'xwoba',
    })

    # Keep only what we need
    cols = ['mlb_player_id', 'pa', 'ba', 'xba', 'slg', 'xslg', 'woba', 'xwoba']
    df = df[cols].copy()
    print(f"    → {len(df)} batters with expected stats")
    return df


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 2: Exit Velo + Barrels (contact quality)
# ═══════════════════════════════════════════════════════════════

def fetch_exit_velo_barrels(season, min_bbe=50):
    """
    Pull exit velocity and barrel data from Baseball Savant via pybaseball.
    Returns: DataFrame with avg_exit_velocity, barrel_rate, hard_hit_rate,
             launch_angle_sweet_spot_pct.

    Barrels are the best single predictor of future power production.
    Sweet spot % (launch angle 8-32°) predicts consistent hard contact.
    """
    from pybaseball import statcast_batter_exitvelo_barrels

    print(f"  Fetching exit velo/barrels for {season} (min BBE={min_bbe})...")
    df = statcast_batter_exitvelo_barrels(season, minBBE=min_bbe)

    # Rename to our convention
    df = df.rename(columns={
        'player_id': 'mlb_player_id',
        'avg_hit_speed': 'avg_exit_velocity',       # Average exit velocity
        'brl_percent': 'barrel_rate',                # Barrels per BBE %
        'ev95percent': 'hard_hit_rate',              # Balls 95+ mph %
        'anglesweetspotpercent': 'launch_angle_sweet_spot_pct',  # 8-32° %
    })

    cols = ['mlb_player_id', 'avg_exit_velocity', 'barrel_rate',
            'hard_hit_rate', 'launch_angle_sweet_spot_pct']
    df = df[cols].copy()
    print(f"    → {len(df)} batters with exit velo/barrel data")
    return df


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 3: Plate Discipline + Batted Ball Profile
# ═══════════════════════════════════════════════════════════════

SAVANT_BASE = "https://baseballsavant.mlb.com/leaderboard/custom"

def fetch_discipline_and_batted_ball(season, min_pa=50):
    """
    Pull plate discipline and batted ball stats from the Savant custom leaderboard.
    Uses the CSV export endpoint — reliable and fast.

    Returns: DataFrame with K%, BB%, whiff%, chase%, z_contact%, BABIP,
             pull%, GB/FB/LD%, and raw HR + FB counts (to compute HR/FB%).

    Chase rate (O-Swing%) is the most stable plate discipline metric year-to-year.
    If it's increasing, the hitter is chasing more → expect regression.
    """
    print(f"  Fetching discipline/batted ball for {season} (min PA={min_pa})...")

    # Two requests: one for discipline/batted ball, one for HR + FB counts
    # (HR/FB ratio needs raw counts since Savant doesn't expose it directly)

    # Request 1: Discipline + batted ball rates
    selections_1 = ",".join([
        "k_percent", "bb_percent", "whiff_percent",
        "oz_swing_percent",    # = chase rate (O-Swing%)
        "iz_contact_percent",  # = zone contact rate (Z-Contact%)
        "babip",
        "pull_percent",        # Pull % of batted balls
        "groundballs_percent", "flyballs_percent", "linedrives_percent",
    ])

    url_1 = (
        f"{SAVANT_BASE}?year={season}&type=batter&filter=&min={min_pa}"
        f"&selections={selections_1}&chart=false"
        f"&x=k_percent&y=k_percent&r=no&chartType=beeswarm&csv=true"
    )

    r1 = requests.get(url_1, timeout=60)
    r1.raise_for_status()
    df1 = pd.read_csv(io.StringIO(r1.text))
    time.sleep(1)  # Be nice to Savant's servers

    # Request 2: Raw HR + FB counts for HR/FB ratio
    selections_2 = "home_run,flyballs,b_total_pa"
    url_2 = (
        f"{SAVANT_BASE}?year={season}&type=batter&filter=&min={min_pa}"
        f"&selections={selections_2}&chart=false"
        f"&x=home_run&y=home_run&r=no&chartType=beeswarm&csv=true"
    )

    r2 = requests.get(url_2, timeout=60)
    r2.raise_for_status()
    df2 = pd.read_csv(io.StringIO(r2.text))

    # Merge on player_id
    df = df1.merge(df2[['player_id', 'home_run', 'flyballs']], on='player_id', how='left')

    # Compute HR/FB rate
    df['hr_per_fb'] = df['home_run'] / df['flyballs'].replace(0, 1)

    # Rename columns to our convention
    df = df.rename(columns={
        'player_id': 'mlb_player_id',
        'k_percent': 'k_rate',
        'bb_percent': 'bb_rate',
        'whiff_percent': 'whiff_rate',
        'oz_swing_percent': 'chase_rate',
        'iz_contact_percent': 'z_contact_rate',
        'pull_percent': 'pull_air_rate',     # Pull % (best HR predictor when combined with FB)
        'groundballs_percent': 'gb_rate',
        'flyballs_percent': 'fb_rate',
        'linedrives_percent': 'ld_rate',
    })

    cols = ['mlb_player_id', 'k_rate', 'bb_rate', 'whiff_rate', 'chase_rate',
            'z_contact_rate', 'babip', 'pull_air_rate', 'gb_rate', 'fb_rate',
            'ld_rate', 'hr_per_fb']
    df = df[cols].copy()
    print(f"    → {len(df)} batters with discipline/batted ball data")
    return df


# ═══════════════════════════════════════════════════════════════
# MERGE + STORE
# ═══════════════════════════════════════════════════════════════

def build_player_id_map():
    """
    Build mapping from MLB player ID (Savant uses this) to our internal player_id.
    This is how we connect Savant data to our database records.
    """
    query = text("SELECT id, mlb_id FROM players WHERE mlb_id IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    # mlb_id → our internal id
    return {row[1]: row[0] for row in rows}


def ingest_season(season, min_pa=50):
    """
    Pull all three data sources for one season, merge them, and store in DB.

    The merge strategy:
    - Expected stats (xwOBA, xBA, xSLG) is the "base" — widest coverage
    - Exit velo/barrels joins on player_id (some players may not have enough BBE)
    - Discipline/batted ball also joins on player_id

    A player needs at least expected stats to get a row. Missing barrels or
    discipline data gets stored as NULL (better than dropping the player entirely).
    """
    print(f"\n{'='*60}")
    print(f"  INGESTING STATCAST DATA — {season}")
    print(f"{'='*60}")

    # Fetch all three sources
    expected_df = fetch_expected_stats(season, min_pa)
    barrels_df = fetch_exit_velo_barrels(season, min_bbe=30)
    discipline_df = fetch_discipline_and_batted_ball(season, min_pa)

    # Merge: expected stats as base, left join the others
    merged = expected_df.merge(barrels_df, on='mlb_player_id', how='left')
    merged = merged.merge(discipline_df, on='mlb_player_id', how='left')

    print(f"\n  Merged: {len(merged)} players with at least expected stats")

    # Map to our internal player IDs
    player_map = build_player_id_map()
    merged['player_id'] = merged['mlb_player_id'].map(player_map)

    # Only keep players that exist in our database
    matched = merged[merged['player_id'].notna()].copy()
    matched['player_id'] = matched['player_id'].astype(int)
    unmatched = len(merged) - len(matched)

    print(f"  Matched to our DB: {len(matched)} players")
    if unmatched > 0:
        print(f"  Skipped (not in DB): {unmatched} players")

    # Store in database — batch commit every 50 records (project convention)
    stored = 0
    skipped = 0

    with get_session() as session:
        for i, (_, row) in enumerate(matched.iterrows()):
            # Check if record already exists (upsert logic)
            existing = session.query(PlayerStatcastMetrics).filter_by(
                player_id=int(row['player_id']),
                season=season,
            ).first()

            if existing:
                # Update existing record with new data
                record = existing
            else:
                # Create new record
                record = PlayerStatcastMetrics(
                    player_id=int(row['player_id']),
                    season=season,
                )

            # Set all metric values (works for both insert and update)
            record.pa = int(row['pa']) if pd.notna(row.get('pa')) else None
            record.avg_exit_velocity = _safe_float(row.get('avg_exit_velocity'))
            record.barrel_rate = _safe_float(row.get('barrel_rate'))
            record.hard_hit_rate = _safe_float(row.get('hard_hit_rate'))
            record.xwoba = _safe_float(row.get('xwoba'))
            record.xslg = _safe_float(row.get('xslg'))
            record.xba = _safe_float(row.get('xba'))
            record.launch_angle_sweet_spot_pct = _safe_float(row.get('launch_angle_sweet_spot_pct'))
            record.babip = _safe_float(row.get('babip'))
            record.woba = _safe_float(row.get('woba'))
            record.slg = _safe_float(row.get('slg'))
            record.ba = _safe_float(row.get('ba'))
            record.hr_per_fb = _safe_float(row.get('hr_per_fb'))
            record.k_rate = _safe_float(row.get('k_rate'))
            record.bb_rate = _safe_float(row.get('bb_rate'))
            record.chase_rate = _safe_float(row.get('chase_rate'))
            record.whiff_rate = _safe_float(row.get('whiff_rate'))
            record.z_contact_rate = _safe_float(row.get('z_contact_rate'))
            record.pull_air_rate = _safe_float(row.get('pull_air_rate'))
            record.gb_rate = _safe_float(row.get('gb_rate'))
            record.fb_rate = _safe_float(row.get('fb_rate'))
            record.ld_rate = _safe_float(row.get('ld_rate'))

            if not existing:
                session.add(record)

            stored += 1

            # Batch commit every 50 records
            if (i + 1) % 50 == 0:
                session.commit()
                print(f"    Committed batch {(i+1)//50} ({i+1} records)...")

        # Final commit
        session.commit()

    print(f"\n  ✓ Stored {stored} player-season records for {season}")
    return stored


def _safe_float(val):
    """Convert to float, handling NaN/None gracefully."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def show_summary():
    """Show what Statcast data is currently in the database."""
    print(f"\n{'='*60}")
    print("  STATCAST DATA SUMMARY")
    print(f"{'='*60}")

    with engine.connect() as conn:
        # Count by season
        result = conn.execute(text("""
            SELECT season, COUNT(*) as players,
                   AVG(xwoba) as avg_xwoba,
                   AVG(avg_exit_velocity) as avg_ev,
                   AVG(barrel_rate) as avg_barrel,
                   SUM(CASE WHEN chase_rate IS NOT NULL THEN 1 ELSE 0 END) as has_discipline
            FROM player_statcast_metrics
            GROUP BY season
            ORDER BY season
        """)).fetchall()

        if not result:
            print("  No Statcast data in database yet.")
            print("  Run: python scripts/ingest_statcast.py")
            return

        for row in result:
            season, count, avg_xwoba, avg_ev, avg_barrel, has_disc = row
            print(f"\n  {season}:")
            print(f"    Players: {count}")
            print(f"    Avg xwOBA: {avg_xwoba:.3f}" if avg_xwoba else "    Avg xwOBA: N/A")
            print(f"    Avg EV: {avg_ev:.1f} mph" if avg_ev else "    Avg EV: N/A")
            print(f"    Avg Barrel%: {avg_barrel:.1f}%" if avg_barrel else "    Avg Barrel%: N/A")
            print(f"    With discipline data: {has_disc}/{count}")

        # Total
        total = conn.execute(text(
            "SELECT COUNT(*) FROM player_statcast_metrics"
        )).scalar()
        print(f"\n  Total records: {total}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Statcast metrics from Baseball Savant"
    )
    parser.add_argument("--season", type=int, help="Single season to ingest")
    parser.add_argument("--min-pa", type=int, default=50,
                        help="Minimum plate appearances (default: 50)")
    parser.add_argument("--summary", action="store_true",
                        help="Show current DB contents")
    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.season:
        ingest_season(args.season, min_pa=args.min_pa)
        show_summary()
    else:
        # Ingest all 3 seasons
        total = 0
        for season in [2023, 2024, 2025]:
            count = ingest_season(season, min_pa=args.min_pa)
            total += count
            time.sleep(2)  # Be nice between seasons

        print(f"\n{'='*60}")
        print(f"  DONE — Ingested {total} total player-season records")
        print(f"{'='*60}")
        show_summary()
