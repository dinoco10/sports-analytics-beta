"""
ingest_pitcher_statcast.py -- Pull Pitcher Statcast Metrics from Baseball Savant
=================================================================================
Downloads season-level and pitch-level Statcast data for all qualified pitchers.
Stores in pitcher_statcast_metrics (season-level) and pitcher_pitch_metrics (per pitch type).

Data sources:
1. Baseball Savant Expected Stats (xERA, xwOBA against, ERA vs xERA gap)
2. Baseball Savant Exit Velo/Barrels (contact suppression: EV, barrel%, hard hit%)
3. Baseball Savant Custom Leaderboard CSV (K%, BB%, whiff%, chase%, BABIP, GB/FB/LD/Pull%)
4. pybaseball pitcher arsenal stats (per-pitch run values, whiff%, usage, velocity)

The pull_air_rate_against metric is particularly important — 66% of HRs come from
pulled air balls per Statcast data. A pitcher whose pull_air_rate is rising is a
regression candidate even if their ERA looks fine.

Usage:
    python scripts/ingest_pitcher_statcast.py                  # All 3 seasons
    python scripts/ingest_pitcher_statcast.py --season 2025    # Single season
    python scripts/ingest_pitcher_statcast.py --min-pa 100     # Higher PA threshold
    python scripts/ingest_pitcher_statcast.py --summary        # Check DB contents

Author: Loko
"""

import argparse
import io
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.storage.database import engine, get_session
from src.storage.models import PitcherStatcastMetrics, PitcherPitchMetrics, Player


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 1: Expected Stats (xERA, xwOBA against)
# ═══════════════════════════════════════════════════════════════

def fetch_pitcher_expected_stats(season, min_pa=50):
    """
    Pull expected stats from Baseball Savant via pybaseball.
    This gives us the core overperformance filter:
    - If ERA << xERA, the pitcher got lucky on BABIP/sequencing
    - xwOBA against strips out luck to show true contact quality allowed
    """
    from pybaseball import statcast_pitcher_expected_stats

    print(f"  Fetching pitcher expected stats for {season} (min PA={min_pa})...")
    df = statcast_pitcher_expected_stats(season, minPA=min_pa)

    df = df.rename(columns={
        'player_id': 'mlb_player_id',
        'est_ba': 'xba_against',
        'est_slg': 'xslg_against',
        'est_woba': 'xwoba_against',
        'woba': 'woba_against',
        'ba': 'ba_against',
        'slg': 'slg_against',
    })

    cols = ['mlb_player_id', 'pa', 'ba_against', 'xba_against', 'slg_against',
            'xslg_against', 'woba_against', 'xwoba_against', 'era', 'xera']
    df = df[cols].copy()
    print(f"    --> {len(df)} pitchers with expected stats")
    return df


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 2: Exit Velo + Barrels (contact suppression)
# ═══════════════════════════════════════════════════════════════

def fetch_pitcher_exit_velo_barrels(season, min_bbe=50):
    """
    Pull exit velocity and barrel data for pitchers.
    Lower values = better contact suppression.
    Barrel rate against is the single best predictor of future HR allowed.
    """
    from pybaseball import statcast_pitcher_exitvelo_barrels

    print(f"  Fetching pitcher exit velo/barrels for {season} (min BBE={min_bbe})...")
    df = statcast_pitcher_exitvelo_barrels(season, minBBE=min_bbe)

    df = df.rename(columns={
        'player_id': 'mlb_player_id',
        'avg_hit_speed': 'avg_exit_velocity_against',
        'brl_percent': 'barrel_rate_against',
        'ev95percent': 'hard_hit_rate_against',
    })

    cols = ['mlb_player_id', 'avg_exit_velocity_against', 'barrel_rate_against',
            'hard_hit_rate_against']
    df = df[cols].copy()
    print(f"    --> {len(df)} pitchers with exit velo/barrel data")
    return df


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 3: Discipline + Batted Ball Profile
# ═══════════════════════════════════════════════════════════════

SAVANT_BASE = "https://baseballsavant.mlb.com/leaderboard/custom"

def fetch_pitcher_discipline(season, min_pa=50):
    """
    Pull plate discipline and batted ball stats from Savant custom leaderboard.

    Key metrics:
    - chase_rate_induced (O-Swing%): getting hitters to chase is elite stuff signal
    - z_contact_rate_against: lower = better (hitters can't square up strikes)
    - pull_air_rate_against: 66% of HRs come from pulled air balls — THE leading indicator
    - GB%: higher is generally better, 50%+ is elite for contact suppressors
    """
    print(f"  Fetching pitcher discipline/batted ball for {season} (min PA={min_pa})...")

    # Request 1: Discipline + batted ball rates
    selections_1 = ",".join([
        "k_percent", "bb_percent", "whiff_percent",
        "oz_swing_percent", "iz_contact_percent",
        "p_era", "babip",
        "pull_percent", "groundballs_percent", "flyballs_percent", "linedrives_percent",
    ])

    url_1 = (
        f"{SAVANT_BASE}?year={season}&type=pitcher&filter=&min={min_pa}"
        f"&selections={selections_1}&chart=false"
        f"&x=k_percent&y=k_percent&r=no&chartType=beeswarm&csv=true"
    )

    r1 = requests.get(url_1, timeout=60)
    r1.raise_for_status()
    df1 = pd.read_csv(io.StringIO(r1.text))
    time.sleep(1)

    # Request 2: HR + FB counts for HR/FB rate
    selections_2 = "home_run,flyballs,b_total_pa"
    url_2 = (
        f"{SAVANT_BASE}?year={season}&type=pitcher&filter=&min={min_pa}"
        f"&selections={selections_2}&chart=false"
        f"&x=home_run&y=home_run&r=no&chartType=beeswarm&csv=true"
    )

    r2 = requests.get(url_2, timeout=60)
    r2.raise_for_status()
    df2 = pd.read_csv(io.StringIO(r2.text))

    df = df1.merge(df2[['player_id', 'home_run', 'flyballs', 'b_total_pa']],
                   on='player_id', how='left')

    df['hr_per_fb'] = df['home_run'] / df['flyballs'].replace(0, 1)
    df['k_minus_bb'] = df['k_percent'] - df['bb_percent']

    df = df.rename(columns={
        'player_id': 'mlb_player_id',
        'k_percent': 'k_rate',
        'bb_percent': 'bb_rate',
        'whiff_percent': 'whiff_rate',
        'oz_swing_percent': 'chase_rate_induced',
        'iz_contact_percent': 'z_contact_rate_against',
        'babip': 'babip_against',
        'pull_percent': 'pull_air_rate_against',
        'groundballs_percent': 'gb_rate',
        'flyballs_percent': 'fb_rate',
        'linedrives_percent': 'ld_rate',
        'b_total_pa': 'pa_against',
    })

    cols = ['mlb_player_id', 'k_rate', 'bb_rate', 'k_minus_bb', 'whiff_rate',
            'chase_rate_induced', 'z_contact_rate_against', 'babip_against',
            'pull_air_rate_against', 'gb_rate', 'fb_rate', 'ld_rate',
            'hr_per_fb', 'pa_against']
    df = df[cols].copy()
    print(f"    --> {len(df)} pitchers with discipline/batted ball data")
    return df


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 4: Pitch Arsenal (per-pitch run values + velocity)
# ═══════════════════════════════════════════════════════════════

def fetch_pitcher_arsenal(season, min_pa=50):
    """
    Pull per-pitch-type stats: run value, whiff rate, usage, velocity.

    This identifies:
    - Negative run value fastball = major red flag
    - New plus secondary pitch (run value > 1.0/100) = breakout catalyst
    - Limited arsenal (top 2 pitches > 70% usage) = predictability risk
    """
    from pybaseball import statcast_pitcher_arsenal_stats, statcast_pitcher_pitch_arsenal

    print(f"  Fetching pitcher arsenal stats for {season} (min PA={min_pa})...")

    # Arsenal outcome stats (run values, whiff%, wOBA against per pitch)
    stats_df = statcast_pitcher_arsenal_stats(season, minPA=min_pa)
    time.sleep(1)

    # Arsenal physical properties (velocity per pitch type)
    try:
        velo_df = statcast_pitcher_pitch_arsenal(season, minP=50)
        print(f"    --> {len(velo_df)} pitchers with velocity data")
    except Exception:
        velo_df = pd.DataFrame()
        print("    --> Velocity data unavailable")

    # Rename stats columns
    stats_df = stats_df.rename(columns={
        'player_id': 'mlb_player_id',
        'pitches': 'pitches_thrown',
        'pitch_usage': 'usage_pct',
        'ba': 'ba_against',
        'slg': 'slg_against',
        'woba': 'woba_against',
        'whiff_percent': 'whiff_rate',
        'hard_hit_percent': 'hard_hit_pct',
        'est_woba': 'xwoba_against',
    })

    # Map velocity data onto arsenal stats
    if not velo_df.empty:
        # Velocity df has one row per pitcher, columns like ff_avg_speed, sl_avg_speed, etc.
        pitch_type_velo_cols = {
            'FF': 'ff_avg_speed', 'SI': 'si_avg_speed', 'FC': 'fc_avg_speed',
            'SL': 'sl_avg_speed', 'CH': 'ch_avg_speed', 'CU': 'cu_avg_speed',
            'FS': 'fs_avg_speed', 'KN': 'kn_avg_speed', 'ST': 'st_avg_speed',
            'SV': 'sv_avg_speed',
        }
        velo_df = velo_df.rename(columns={'pitcher': 'mlb_player_id'})

        # Melt velocity into pitch_type -> avg_speed
        velo_records = []
        for _, row in velo_df.iterrows():
            pid = row['mlb_player_id']
            for ptype, col in pitch_type_velo_cols.items():
                if col in velo_df.columns and pd.notna(row.get(col)):
                    velo_records.append({
                        'mlb_player_id': pid,
                        'pitch_type': ptype,
                        'avg_speed': row[col],
                    })

        velo_melted = pd.DataFrame(velo_records)
        if not velo_melted.empty:
            stats_df = stats_df.merge(
                velo_melted, on=['mlb_player_id', 'pitch_type'], how='left'
            )

    if 'avg_speed' not in stats_df.columns:
        stats_df['avg_speed'] = np.nan

    print(f"    --> {len(stats_df)} pitch-type records across "
          f"{stats_df['mlb_player_id'].nunique()} pitchers")
    return stats_df


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 5: Spin Rate (avg spin per pitch type)
# ═══════════════════════════════════════════════════════════════

def fetch_pitcher_spin_rate(season, min_pitches=50):
    """
    Pull average spin rate per pitch type via pybaseball.
    Uses statcast_pitcher_pitch_arsenal with arsenal_type='avg_spin'.
    Returns DataFrame with columns: mlb_player_id, pitch_type, avg_spin.
    """
    from pybaseball import statcast_pitcher_pitch_arsenal

    print(f"  Fetching pitcher spin rate for {season} (min pitches={min_pitches})...")

    try:
        spin_df = statcast_pitcher_pitch_arsenal(season, minP=min_pitches,
                                                  arsenal_type='avg_spin')
        time.sleep(1)
    except Exception as e:
        print(f"    --> Spin rate data unavailable: {e}")
        return pd.DataFrame()

    if spin_df.empty:
        print("    --> No spin rate data returned")
        return pd.DataFrame()

    spin_df = spin_df.rename(columns={'pitcher': 'mlb_player_id'})

    # Melt from wide format (ff_avg_spin, sl_avg_spin, ...) to long format
    pitch_type_spin_cols = {
        'FF': 'ff_avg_spin', 'SI': 'si_avg_spin', 'FC': 'fc_avg_spin',
        'SL': 'sl_avg_spin', 'CH': 'ch_avg_spin', 'CU': 'cu_avg_spin',
        'FS': 'fs_avg_spin', 'KN': 'kn_avg_spin', 'ST': 'st_avg_spin',
        'SV': 'sv_avg_spin',
    }

    records = []
    for _, row in spin_df.iterrows():
        pid = row['mlb_player_id']
        for ptype, col in pitch_type_spin_cols.items():
            if col in spin_df.columns and pd.notna(row.get(col)):
                records.append({
                    'mlb_player_id': pid,
                    'pitch_type': ptype,
                    'avg_spin': row[col],
                })

    result = pd.DataFrame(records)
    print(f"    --> {len(result)} pitch-type spin records across "
          f"{result['mlb_player_id'].nunique()} pitchers")
    return result


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE 6: Induced Vertical Break (IVB) from Savant
# ═══════════════════════════════════════════════════════════════

def fetch_pitcher_ivb(season, min_pitches=100):
    """
    Pull induced vertical break (IVB) per pitch type from Baseball Savant
    pitch-movement leaderboard. IVB measures how much a pitch defies gravity.

    Higher IVB on fastballs = more "rise" = harder to hit.
    Returns DataFrame: mlb_player_id, pitch_type, induced_vertical_break.
    """
    print(f"  Fetching IVB data for {season} (min pitches={min_pitches})...")

    # Pitch types to fetch IVB for
    pitch_types = ['FF', 'SI', 'SL', 'CH', 'CU', 'FC', 'ST', 'SV', 'FS']
    all_records = []

    for pt in pitch_types:
        url = (
            f"https://baseballsavant.mlb.com/leaderboard/pitch-movement"
            f"?year={season}&pitch_type={pt}&min={min_pitches}&csv=true"
        )
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200 or len(r.text) < 100:
                continue
            df = pd.read_csv(io.StringIO(r.text))

            # The IVB column is 'pitcher_break_z_induced' (inches)
            if 'pitcher_break_z_induced' not in df.columns:
                # Try alternate column names
                ivb_col = None
                for c in df.columns:
                    if 'break_z' in c.lower() and 'induced' in c.lower():
                        ivb_col = c
                        break
                if ivb_col is None:
                    continue
            else:
                ivb_col = 'pitcher_break_z_induced'

            # Find the pitcher ID column
            pid_col = None
            for c in ['pitcher_id', 'player_id', 'pitcher']:
                if c in df.columns:
                    pid_col = c
                    break
            if pid_col is None:
                continue

            for _, row in df.iterrows():
                pid = row.get(pid_col)
                ivb = row.get(ivb_col)
                if pd.notna(pid) and pd.notna(ivb):
                    all_records.append({
                        'mlb_player_id': int(pid),
                        'pitch_type': pt,
                        'induced_vertical_break': float(ivb),
                    })

        except Exception as e:
            print(f"    --> IVB fetch failed for {pt}: {e}")

        time.sleep(1)  # Rate limit: 1s between requests

    result = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    if not result.empty:
        print(f"    --> {len(result)} IVB records across "
              f"{result['mlb_player_id'].nunique()} pitchers")
    else:
        print("    --> No IVB data retrieved")
    return result


# ═══════════════════════════════════════════════════════════════
# ARSENAL DEPTH COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_arsenal_depth(arsenal_df, mlb_player_id):
    """
    Compute arsenal summary metrics for one pitcher:
    - arsenal_depth_score: count of pitches with usage>10% AND run_value_per_100 < 0
    - fastball_velocity: avg_speed of primary fastball (FF or SI, higher usage)
    - best_secondary_rv: lowest (best) run_value_per_100 among non-FB pitches

    Returns dict with the three fields.
    """
    pitcher_df = arsenal_df[arsenal_df['mlb_player_id'] == mlb_player_id]

    if pitcher_df.empty:
        return {'arsenal_depth_score': None, 'fastball_velocity': None,
                'best_secondary_rv': None}

    # Arsenal depth: count of meaningful + effective pitches
    meaningful = pitcher_df[pitcher_df['usage_pct'] > 10]
    effective = meaningful[meaningful['run_value_per_100'] < 0]
    depth_score = len(effective)

    # Primary fastball velocity (FF or SI, whichever has higher usage)
    fb_types = pitcher_df[pitcher_df['pitch_type'].isin(['FF', 'SI'])]
    if not fb_types.empty:
        primary_fb = fb_types.loc[fb_types['usage_pct'].idxmax()]
        fb_velo = primary_fb.get('avg_speed')
    else:
        fb_velo = None

    # Best secondary pitch run value (non-fastball)
    secondaries = pitcher_df[~pitcher_df['pitch_type'].isin(['FF', 'SI'])]
    if not secondaries.empty:
        best_rv = secondaries['run_value_per_100'].min()
    else:
        best_rv = None

    return {
        'arsenal_depth_score': float(depth_score),
        'fastball_velocity': float(fb_velo) if pd.notna(fb_velo) else None,
        'best_secondary_rv': float(best_rv) if pd.notna(best_rv) else None,
    }


# ═══════════════════════════════════════════════════════════════
# DB MIGRATION: Add new columns if they don't exist
# ═══════════════════════════════════════════════════════════════

def ensure_new_columns():
    """
    Add new columns to existing tables if they don't exist.
    Idempotent — safe to run multiple times.
    """
    alter_statements = [
        # PitcherPitchMetrics — spin rate and IVB
        "ALTER TABLE pitcher_pitch_metrics ADD COLUMN avg_spin FLOAT",
        "ALTER TABLE pitcher_pitch_metrics ADD COLUMN induced_vertical_break FLOAT",
        # PitcherStatcastMetrics — arsenal summary
        "ALTER TABLE pitcher_statcast_metrics ADD COLUMN arsenal_depth_score FLOAT",
        "ALTER TABLE pitcher_statcast_metrics ADD COLUMN fastball_velocity FLOAT",
        "ALTER TABLE pitcher_statcast_metrics ADD COLUMN best_secondary_rv FLOAT",
    ]

    with engine.connect() as conn:
        for stmt in alter_statements:
            try:
                conn.execute(text(stmt))
                col_name = stmt.split("ADD COLUMN ")[1].split(" ")[0]
                print(f"  Added column: {col_name}")
            except Exception:
                pass  # Column already exists
        conn.commit()


# ═══════════════════════════════════════════════════════════════
# MERGE + STORE
# ═══════════════════════════════════════════════════════════════

def build_player_id_map():
    """Build mapping from MLB player ID to our internal player_id."""
    query = text("SELECT id, mlb_id FROM players WHERE mlb_id IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    return {row[1]: row[0] for row in rows}


def _safe_float(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def ingest_season(season, min_pa=50):
    """
    Pull all six data sources for one season, merge, and store.
    Season-level metrics go into pitcher_statcast_metrics.
    Per-pitch metrics go into pitcher_pitch_metrics.
    """
    print(f"\n{'='*60}")
    print(f"  INGESTING PITCHER STATCAST DATA -- {season}")
    print(f"{'='*60}")

    # Fetch all sources
    expected_df = fetch_pitcher_expected_stats(season, min_pa)
    barrels_df = fetch_pitcher_exit_velo_barrels(season, min_bbe=30)
    discipline_df = fetch_pitcher_discipline(season, min_pa)
    arsenal_df = fetch_pitcher_arsenal(season, min_pa)
    spin_df = fetch_pitcher_spin_rate(season, min_pitches=50)
    ivb_df = fetch_pitcher_ivb(season, min_pitches=100)

    # Merge spin rate into arsenal_df (per pitch type)
    if not spin_df.empty:
        arsenal_df = arsenal_df.merge(
            spin_df, on=['mlb_player_id', 'pitch_type'], how='left'
        )
    if 'avg_spin' not in arsenal_df.columns:
        arsenal_df['avg_spin'] = np.nan

    # Merge IVB into arsenal_df (per pitch type)
    if not ivb_df.empty:
        arsenal_df = arsenal_df.merge(
            ivb_df, on=['mlb_player_id', 'pitch_type'], how='left'
        )
    if 'induced_vertical_break' not in arsenal_df.columns:
        arsenal_df['induced_vertical_break'] = np.nan

    # Merge season-level: expected as base, left join others
    merged = expected_df.merge(barrels_df, on='mlb_player_id', how='left')
    merged = merged.merge(discipline_df, on='mlb_player_id', how='left')

    print(f"\n  Merged: {len(merged)} pitchers with at least expected stats")

    # Map to internal player IDs
    player_map = build_player_id_map()
    merged['player_id'] = merged['mlb_player_id'].map(player_map)
    arsenal_df['player_id'] = arsenal_df['mlb_player_id'].map(player_map)

    matched = merged[merged['player_id'].notna()].copy()
    matched['player_id'] = matched['player_id'].astype(int)
    unmatched = len(merged) - len(matched)

    print(f"  Matched to DB: {len(matched)} pitchers")
    if unmatched > 0:
        print(f"  Skipped (not in DB): {unmatched} pitchers")

    # Store season-level metrics
    stored_season = 0
    with get_session() as session:
        for i, (_, row) in enumerate(matched.iterrows()):
            existing = session.query(PitcherStatcastMetrics).filter_by(
                player_id=int(row['player_id']), season=season,
            ).first()

            record = existing or PitcherStatcastMetrics(
                player_id=int(row['player_id']), season=season,
            )

            # Contact suppression
            record.avg_exit_velocity_against = _safe_float(row.get('avg_exit_velocity_against'))
            record.barrel_rate_against = _safe_float(row.get('barrel_rate_against'))
            record.hard_hit_rate_against = _safe_float(row.get('hard_hit_rate_against'))
            record.xwoba_against = _safe_float(row.get('xwoba_against'))
            record.xslg_against = _safe_float(row.get('xslg_against'))
            record.xba_against = _safe_float(row.get('xba_against'))
            record.xera = _safe_float(row.get('xera'))

            # Actual stats
            record.era = _safe_float(row.get('era'))
            record.woba_against = _safe_float(row.get('woba_against'))
            record.babip_against = _safe_float(row.get('babip_against'))
            record.hr_per_fb = _safe_float(row.get('hr_per_fb'))

            # Swing & miss
            record.k_rate = _safe_float(row.get('k_rate'))
            record.bb_rate = _safe_float(row.get('bb_rate'))
            record.k_minus_bb = _safe_float(row.get('k_minus_bb'))
            record.whiff_rate = _safe_float(row.get('whiff_rate'))
            record.chase_rate_induced = _safe_float(row.get('chase_rate_induced'))
            record.z_contact_rate_against = _safe_float(row.get('z_contact_rate_against'))

            # Batted ball
            record.pull_air_rate_against = _safe_float(row.get('pull_air_rate_against'))
            record.gb_rate = _safe_float(row.get('gb_rate'))
            record.fb_rate = _safe_float(row.get('fb_rate'))
            record.ld_rate = _safe_float(row.get('ld_rate'))

            # Metadata
            record.pa_against = int(row['pa']) if pd.notna(row.get('pa')) else None

            # Arsenal summary (computed from pitch-level data)
            arsenal_summary = compute_arsenal_depth(arsenal_df, row['mlb_player_id'])
            record.arsenal_depth_score = _safe_float(arsenal_summary.get('arsenal_depth_score'))
            record.fastball_velocity = _safe_float(arsenal_summary.get('fastball_velocity'))
            record.best_secondary_rv = _safe_float(arsenal_summary.get('best_secondary_rv'))

            if not existing:
                session.add(record)
            stored_season += 1

            if (i + 1) % 50 == 0:
                session.commit()
                print(f"    Committed season batch {(i+1)//50} ({i+1} records)...")

        session.commit()

    print(f"\n  Stored {stored_season} pitcher season-level records for {season}")

    # Store pitch-level metrics
    arsenal_matched = arsenal_df[arsenal_df['player_id'].notna()].copy()
    arsenal_matched['player_id'] = arsenal_matched['player_id'].astype(int)

    stored_pitch = 0
    with get_session() as session:
        for i, (_, row) in enumerate(arsenal_matched.iterrows()):
            pid = int(row['player_id'])
            ptype = row.get('pitch_type', '')
            if not ptype or pd.isna(ptype):
                continue

            existing = session.query(PitcherPitchMetrics).filter_by(
                player_id=pid, season=season, pitch_type=ptype,
            ).first()

            record = existing or PitcherPitchMetrics(
                player_id=pid, season=season, pitch_type=ptype,
            )

            record.pitch_name = row.get('pitch_name', '')
            record.run_value_per_100 = _safe_float(row.get('run_value_per_100'))
            record.run_value = _safe_float(row.get('run_value'))
            record.pitches_thrown = int(row['pitches_thrown']) if pd.notna(row.get('pitches_thrown')) else None
            record.usage_pct = _safe_float(row.get('usage_pct'))
            record.whiff_rate = _safe_float(row.get('whiff_rate'))
            record.ba_against = _safe_float(row.get('ba_against'))
            record.slg_against = _safe_float(row.get('slg_against'))
            record.woba_against = _safe_float(row.get('woba_against'))
            record.hard_hit_pct = _safe_float(row.get('hard_hit_pct'))
            record.avg_speed = _safe_float(row.get('avg_speed'))
            record.avg_spin = _safe_float(row.get('avg_spin'))
            record.induced_vertical_break = _safe_float(row.get('induced_vertical_break'))
            record.xwoba_against = _safe_float(row.get('xwoba_against'))

            if not existing:
                session.add(record)
            stored_pitch += 1

            if (i + 1) % 100 == 0:
                session.commit()

        session.commit()

    print(f"  Stored {stored_pitch} pitch-level records for {season}")
    return stored_season, stored_pitch


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def show_summary():
    print(f"\n{'='*60}")
    print("  PITCHER STATCAST DATA SUMMARY")
    print(f"{'='*60}")

    with engine.connect() as conn:
        # Season-level
        result = conn.execute(text("""
            SELECT season, COUNT(*) as pitchers,
                   AVG(xera) as avg_xera,
                   AVG(avg_exit_velocity_against) as avg_ev,
                   AVG(barrel_rate_against) as avg_barrel,
                   AVG(pull_air_rate_against) as avg_pull_air,
                   SUM(CASE WHEN chase_rate_induced IS NOT NULL THEN 1 ELSE 0 END) as has_disc
            FROM pitcher_statcast_metrics
            GROUP BY season ORDER BY season
        """)).fetchall()

        if not result:
            print("  No pitcher Statcast data in database yet.")
            return

        for row in result:
            season, count, avg_xera, avg_ev, avg_barrel, avg_pull, has_disc = row
            print(f"\n  {season}:")
            print(f"    Pitchers: {count}")
            print(f"    Avg xERA: {avg_xera:.2f}" if avg_xera else "    Avg xERA: N/A")
            print(f"    Avg EV against: {avg_ev:.1f} mph" if avg_ev else "    Avg EV against: N/A")
            print(f"    Avg Barrel% against: {avg_barrel:.1f}%" if avg_barrel else "    Avg Barrel%: N/A")
            print(f"    Avg Pull Air% against: {avg_pull:.1f}%" if avg_pull else "    Avg Pull Air%: N/A")
            print(f"    With discipline data: {has_disc}/{count}")

        # Pitch-level
        pitch_count = conn.execute(text(
            "SELECT COUNT(*) FROM pitcher_pitch_metrics"
        )).scalar()
        total = conn.execute(text(
            "SELECT COUNT(*) FROM pitcher_statcast_metrics"
        )).scalar()
        print(f"\n  Total season-level records: {total}")
        print(f"  Total pitch-level records: {pitch_count}")

        # New fields coverage
        spin_count = conn.execute(text(
            "SELECT COUNT(*) FROM pitcher_pitch_metrics WHERE avg_spin IS NOT NULL"
        )).scalar()
        ivb_count = conn.execute(text(
            "SELECT COUNT(*) FROM pitcher_pitch_metrics WHERE induced_vertical_break IS NOT NULL"
        )).scalar()
        arsenal_count = conn.execute(text(
            "SELECT COUNT(*) FROM pitcher_statcast_metrics WHERE arsenal_depth_score IS NOT NULL"
        )).scalar()
        print(f"\n  Pitch records with spin rate: {spin_count}/{pitch_count}")
        print(f"  Pitch records with IVB: {ivb_count}/{pitch_count}")
        print(f"  Season records with arsenal depth: {arsenal_count}/{total}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest pitcher Statcast metrics from Baseball Savant"
    )
    parser.add_argument("--season", type=int, help="Single season to ingest")
    parser.add_argument("--min-pa", type=int, default=50,
                        help="Minimum PA faced (default: 50)")
    parser.add_argument("--summary", action="store_true",
                        help="Show current DB contents")
    args = parser.parse_args()

    # Ensure new columns exist before ingesting
    if not args.summary:
        print("\nChecking for new DB columns...")
        ensure_new_columns()

    if args.summary:
        show_summary()
    elif args.season:
        ingest_season(args.season, min_pa=args.min_pa)
        show_summary()
    else:
        total_s = total_p = 0
        for season in [2023, 2024, 2025]:
            s, p = ingest_season(season, min_pa=args.min_pa)
            total_s += s
            total_p += p
            time.sleep(2)

        print(f"\n{'='*60}")
        print(f"  DONE -- {total_s} season-level + {total_p} pitch-level records")
        print(f"{'='*60}")
        show_summary()
