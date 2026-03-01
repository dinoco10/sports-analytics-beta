"""
generate_marcel_snapshots.py — Frozen Marcel Projection Snapshots
=================================================================

Generates historical Marcel+Statcast projections for each season,
respecting season boundaries to prevent lookahead bias.

Each snapshot represents "what we would have projected on Jan 1 of that year"
using only data available at that point:

  Snapshot 2023: inputs = 2021 + 2022 stats/statcast  (2 years)
  Snapshot 2024: inputs = 2021 + 2022 + 2023          (3 years, full Marcel)
  Snapshot 2025: inputs = 2022 + 2023 + 2024          (3 years, full Marcel)
  Snapshot 2026: inputs = 2023 + 2024 + 2025          (3 years, what we have now)

Output: data/features/snapshots/marcel_hitters_{year}.csv
        data/features/snapshots/marcel_pitchers_{year}.csv

These frozen snapshots are used by Floor 1 (game-level model) as stable
preseason features that don't change game-to-game. The model learns to
blend projections with rolling stats — projections dominate early season,
rolling stats take over by summer.

Usage:
    python scripts/generate_marcel_snapshots.py              # All 4 snapshots
    python scripts/generate_marcel_snapshots.py --year 2025  # Single snapshot
    python scripts/generate_marcel_snapshots.py --summary    # Check what exists
"""

import argparse
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import engine
from src.features.player_projections import (
    project_pitcher, project_hitter,
    load_statcast_multi_year, load_pitcher_statcast_multi_year,
    load_pitcher_pitch_metrics,
)

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
SNAPSHOT_DIR = FEATURES_DIR / "snapshots"


# ═══════════════════════════════════════════════════════════════
# SEASON-SPECIFIC DATA LOADERS
# ═══════════════════════════════════════════════════════════════
# These are parameterized versions of the loaders in player_projections.py.
# They accept a list of seasons instead of hardcoding 2023-2025.

def load_pitcher_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load pitcher stats for specified seasons from database."""
    season_str = ", ".join(f"'{s}'" for s in seasons)
    query = f"""
        SELECT
            p.mlb_id as mlb_player_id,
            p.name as player_name,
            p.birth_date,
            strftime('%Y', ps.date) as season,
            t.name as team_name,
            SUM(ps.ip) as ip,
            SUM(ps.earned_runs) as er,
            SUM(ps.hits) as hits,
            SUM(ps.walks) as bb,
            SUM(ps.strikeouts) as k,
            SUM(ps.home_runs) as hr,
            COUNT(DISTINCT ps.game_id) as games
        FROM pitching_game_stats ps
        JOIN players p ON ps.player_id = p.id
        JOIN teams t ON ps.team_id = t.id
        WHERE strftime('%Y', ps.date) IN ({season_str})
        GROUP BY p.mlb_id, p.name, p.birth_date, strftime('%Y', ps.date), t.name
        HAVING SUM(ps.ip) > 0
    """
    df = pd.read_sql(query, engine)
    df['season'] = df['season'].astype(int)

    df['era'] = (df['er'] / df['ip']) * 9
    df['bf'] = df['ip'] * 3 + df['hits'] + df['bb']
    df['k_pct'] = df['k'] / df['bf'] * 100
    df['bb_pct'] = df['bb'] / df['bf'] * 100
    df['k_bb_pct'] = df['k_pct'] - df['bb_pct']
    fip_const = 3.10
    df['fip'] = ((13 * df['hr'] + 3 * df['bb'] - 2 * df['k']) / df['ip']) + fip_const
    df['role'] = df['ip'].apply(lambda x: 'SP' if x >= 50 else 'RP')

    return df


def load_hitter_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load hitter stats for specified seasons from database."""
    season_str = ", ".join(f"'{s}'" for s in seasons)
    query = f"""
        SELECT
            p.mlb_id as mlb_player_id,
            p.name as player_name,
            p.birth_date,
            p.primary_position,
            strftime('%Y', hs.date) as season,
            t.name as team_name,
            SUM(hs.plate_appearances) as pa,
            SUM(hs.at_bats) as ab,
            SUM(hs.hits) as h,
            SUM(hs.doubles) as doubles,
            SUM(hs.triples) as triples,
            SUM(hs.home_runs) as hr,
            SUM(hs.rbi) as rbi,
            SUM(hs.runs) as runs,
            SUM(hs.walks) as bb,
            SUM(hs.strikeouts) as k,
            SUM(hs.stolen_bases) as sb,
            COUNT(DISTINCT hs.game_id) as games
        FROM hitting_game_stats hs
        JOIN players p ON hs.player_id = p.id
        JOIN teams t ON hs.team_id = t.id
        WHERE strftime('%Y', hs.date) IN ({season_str})
        GROUP BY p.mlb_id, p.name, p.birth_date, p.primary_position,
                 strftime('%Y', hs.date), t.name
        HAVING SUM(hs.plate_appearances) >= 10
    """
    df = pd.read_sql(query, engine)
    df['season'] = df['season'].astype(int)

    df['avg'] = df['h'] / df['ab'].replace(0, 1)
    df['obp'] = (df['h'] + df['bb']) / df['pa'].replace(0, 1)
    singles = df['h'] - df['doubles'] - df['triples'] - df['hr']
    slg_num = singles + 2*df['doubles'] + 3*df['triples'] + 4*df['hr']
    df['slg'] = slg_num / df['ab'].replace(0, 1)
    df['ops'] = df['obp'] + df['slg']
    df['woba'] = (0.69*df['bb'] + 0.88*singles + 1.24*df['doubles'] +
                  1.56*df['triples'] + 1.95*df['hr']) / df['pa'].replace(0, 1)
    df['k_pct'] = (df['k'] / df['pa'].replace(0, 1)) * 100
    df['bb_pct'] = (df['bb'] / df['pa'].replace(0, 1)) * 100

    return df


def load_statcast_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load hitter Statcast metrics for specified seasons."""
    season_str = ", ".join(str(s) for s in seasons)
    query = f"""
        SELECT
            p.mlb_id as mlb_player_id,
            sm.season,
            sm.avg_exit_velocity, sm.barrel_rate, sm.hard_hit_rate,
            sm.xwoba, sm.xslg, sm.xba,
            sm.launch_angle_sweet_spot_pct,
            sm.babip, sm.woba, sm.slg, sm.ba, sm.hr_per_fb,
            sm.k_rate, sm.bb_rate, sm.chase_rate, sm.whiff_rate,
            sm.z_contact_rate,
            sm.pull_air_rate, sm.gb_rate, sm.fb_rate, sm.ld_rate,
            sm.pa as statcast_pa
        FROM player_statcast_metrics sm
        JOIN players p ON sm.player_id = p.id
        WHERE sm.season IN ({season_str})
    """
    return pd.read_sql(query, engine)


def load_pitcher_statcast_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load pitcher Statcast metrics for specified seasons."""
    season_str = ", ".join(str(s) for s in seasons)
    query = f"""
        SELECT
            p.mlb_id as mlb_player_id,
            pm.season,
            pm.avg_exit_velocity_against, pm.barrel_rate_against,
            pm.hard_hit_rate_against,
            pm.xwoba_against, pm.xslg_against, pm.xba_against, pm.xera,
            pm.era, pm.woba_against, pm.babip_against, pm.hr_per_fb,
            pm.k_rate, pm.bb_rate, pm.k_minus_bb,
            pm.whiff_rate, pm.swstr_rate,
            pm.chase_rate_induced, pm.z_contact_rate_against,
            pm.pull_air_rate_against, pm.gb_rate, pm.fb_rate, pm.ld_rate,
            pm.pa_against, pm.ip
        FROM pitcher_statcast_metrics pm
        JOIN players p ON pm.player_id = p.id
        WHERE pm.season IN ({season_str})
    """
    return pd.read_sql(query, engine)


def load_pitch_metrics_seasons(seasons: list[int]) -> pd.DataFrame:
    """Load pitch-type arsenal metrics for specified seasons."""
    season_str = ", ".join(str(s) for s in seasons)
    query = f"""
        SELECT
            p.mlb_id as mlb_player_id,
            pp.season, pp.pitch_type, pp.pitch_name,
            pp.run_value_per_100, pp.run_value, pp.pitches_thrown,
            pp.usage_pct, pp.whiff_rate, pp.chase_rate,
            pp.ba_against, pp.slg_against, pp.woba_against,
            pp.hard_hit_pct, pp.avg_speed, pp.xwoba_against
        FROM pitcher_pitch_metrics pp
        JOIN players p ON pp.player_id = p.id
        WHERE pp.season IN ({season_str})
    """
    return pd.read_sql(query, engine)


# ═══════════════════════════════════════════════════════════════
# SNAPSHOT GENERATOR
# ═══════════════════════════════════════════════════════════════

# Which input seasons feed each projection snapshot
SNAPSHOT_INPUTS = {
    2023: [2021, 2022],             # 2 years (no 2020 — COVID)
    2024: [2021, 2022, 2023],       # 3 years, full Marcel
    2025: [2022, 2023, 2024],       # 3 years, full Marcel
    2026: [2023, 2024, 2025],       # 3 years, current production
}


def generate_snapshot(projection_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a frozen Marcel+Statcast projection snapshot for one season.

    Uses only data from seasons BEFORE projection_year (no lookahead).
    For historical snapshots (2023-2025), team assignment comes from
    the player's most recent team in the input data.
    For 2026, we'd use the live API (but that's handled by run_projections).

    Returns: (hitter_df, pitcher_df)
    """
    input_seasons = SNAPSHOT_INPUTS[projection_year]
    print(f"\n{'=' * 70}")
    print(f"  MARCEL SNAPSHOT — Projecting {projection_year}")
    print(f"  Input seasons: {input_seasons}")
    print(f"{'=' * 70}")

    # Load data for the input seasons only
    print(f"\n  Loading standard stats for {input_seasons}...")
    pitchers_raw = load_pitcher_seasons(input_seasons)
    hitters_raw = load_hitter_seasons(input_seasons)
    print(f"    Pitcher-seasons: {len(pitchers_raw)}")
    print(f"    Hitter-seasons:  {len(hitters_raw)}")

    print(f"  Loading Statcast metrics for {input_seasons}...")
    statcast_raw = load_statcast_seasons(input_seasons)
    pitcher_sc_raw = load_pitcher_statcast_seasons(input_seasons)
    pitch_metrics_raw = load_pitch_metrics_seasons(input_seasons)
    print(f"    Hitter Statcast:  {len(statcast_raw)} records")
    print(f"    Pitcher Statcast: {len(pitcher_sc_raw)} records")
    print(f"    Pitch arsenal:    {len(pitch_metrics_raw)} records")

    # For historical snapshots: assign team from most recent season in data
    # (We can't call the live API for "what team was X on in Jan 2024")
    most_recent_season = max(input_seasons)

    # Build roster map from the most recent input season
    # Pitcher teams
    pitcher_teams = (
        pitchers_raw[pitchers_raw['season'] == most_recent_season]
        .groupby('mlb_player_id')['team_name'].first()
        .to_dict()
    )
    # Hitter teams
    hitter_teams = (
        hitters_raw[hitters_raw['season'] == most_recent_season]
        .groupby('mlb_player_id')['team_name'].first()
        .to_dict()
    )
    # Fallback to any season if player wasn't active in most recent
    for pid in pitchers_raw['mlb_player_id'].unique():
        if pid not in pitcher_teams:
            pitcher_teams[pid] = pitchers_raw[
                pitchers_raw['mlb_player_id'] == pid
            ]['team_name'].iloc[-1]
    for pid in hitters_raw['mlb_player_id'].unique():
        if pid not in hitter_teams:
            hitter_teams[pid] = hitters_raw[
                hitters_raw['mlb_player_id'] == pid
            ]['team_name'].iloc[-1]

    # Project pitchers
    print(f"\n  Projecting pitchers for {projection_year}...")
    pitcher_projections = []
    for pid, group in pitchers_raw.groupby('mlb_player_id'):
        p_sc = pitcher_sc_raw[pitcher_sc_raw['mlb_player_id'] == pid]
        p_pm = pitch_metrics_raw[pitch_metrics_raw['mlb_player_id'] == pid]

        proj = project_pitcher(
            group, pitcher_statcast=p_sc, pitch_metrics=p_pm,
            projection_year=projection_year,
        )
        if proj:
            proj['current_team'] = pitcher_teams.get(pid, 'Unknown')
            proj['projection_season'] = projection_year
            pitcher_projections.append(proj)

    p_df = pd.DataFrame(pitcher_projections)
    print(f"    Projected {len(p_df)} pitchers")

    # Project hitters
    print(f"  Projecting hitters for {projection_year}...")
    hitter_projections = []
    for pid, group in hitters_raw.groupby('mlb_player_id'):
        player_sc = statcast_raw[statcast_raw['mlb_player_id'] == pid]

        proj = project_hitter(
            group, player_statcast=player_sc,
            projection_year=projection_year,
        )
        if proj:
            proj['current_team'] = hitter_teams.get(pid, 'Unknown')
            proj['projection_season'] = projection_year
            hitter_projections.append(proj)

    h_df = pd.DataFrame(hitter_projections)
    print(f"    Projected {len(h_df)} hitters")

    # Rename age column for consistency across snapshots
    if 'age_2026' in h_df.columns:
        h_df = h_df.rename(columns={'age_2026': 'proj_age'})
    if 'age_2026' in p_df.columns:
        p_df = p_df.rename(columns={'age_2026': 'proj_age'})

    # Save snapshot
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    h_path = SNAPSHOT_DIR / f"marcel_hitters_{projection_year}.csv"
    p_path = SNAPSHOT_DIR / f"marcel_pitchers_{projection_year}.csv"
    h_df.to_csv(h_path, index=False)
    p_df.to_csv(p_path, index=False)
    print(f"\n  Saved: {h_path}")
    print(f"  Saved: {p_path}")

    # Quick team WAR summary
    h_war = h_df.groupby('current_team')['proj_war'].sum()
    p_war = p_df.groupby('current_team')['proj_war'].sum()
    total = (h_war + p_war).sort_values(ascending=False)
    print(f"\n  Top 5 teams by total WAR:")
    for team, war in total.head(5).items():
        print(f"    {team:<28} {war:+.1f} WAR")
    print(f"  Bottom 3:")
    for team, war in total.tail(3).items():
        print(f"    {team:<28} {war:+.1f} WAR")

    return h_df, p_df


def show_summary():
    """Show what snapshots exist and their stats."""
    print("\n=== MARCEL SNAPSHOT SUMMARY ===\n")
    for year in SNAPSHOT_INPUTS:
        h_path = SNAPSHOT_DIR / f"marcel_hitters_{year}.csv"
        p_path = SNAPSHOT_DIR / f"marcel_pitchers_{year}.csv"
        if h_path.exists() and p_path.exists():
            h = pd.read_csv(h_path)
            p = pd.read_csv(p_path)
            print(f"  {year}: {len(h)} hitters, {len(p)} pitchers "
                  f"(inputs: {SNAPSHOT_INPUTS[year]})")
        else:
            print(f"  {year}: NOT GENERATED")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate frozen Marcel projection snapshots"
    )
    parser.add_argument("--year", type=int, help="Generate single snapshot year")
    parser.add_argument("--summary", action="store_true", help="Show existing snapshots")
    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.year:
        generate_snapshot(args.year)
    else:
        # Generate all 4 snapshots
        for year in SNAPSHOT_INPUTS:
            generate_snapshot(year)
        print("\n" + "=" * 70)
        print("  ALL SNAPSHOTS COMPLETE")
        print("=" * 70)
        show_summary()
