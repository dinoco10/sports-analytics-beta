"""
Team Feature Engineering — Layer 2
Transforms raw game/player data into team-level features for modeling.

Usage:
    python -m src.features.team_features                # Full build
    python -m src.features.team_features --season 2025  # One season
    python -m src.features.team_features --team "New York Yankees"  # One team debug
"""

import sys, os, argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from src.storage.database import get_session, engine
from config.settings import ROLLING_WINDOWS, LEAGUE_AVG_BABIP, LEAGUE_AVG_HR_FB


# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD RAW DATA FROM DATABASE
# ═══════════════════════════════════════════════════════════════

def load_games(season=None):
    """Load all games with team names."""
    query = """
        SELECT g.id as game_id, g.date, g.season,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id,
               g.innings, g.day_night,
               ht.name as home_team, ht.abbreviation as home_abbr,
               at.name as away_team, at.abbreviation as away_abbr
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.home_score IS NOT NULL
    """
    if season:
        query += f" AND g.season = {season}"
    query += " ORDER BY g.date"
    return pd.read_sql(query, engine)


def load_pitching_stats(season=None):
    """Load all pitching game stats with team/player names."""
    query = """
        SELECT ps.*, t.name as team_name, t.abbreviation as team_abbr,
               p.name as player_name
        FROM pitching_game_stats ps
        JOIN teams t ON ps.team_id = t.id
        JOIN players p ON ps.player_id = p.id
    """
    if season:
        query += f" WHERE strftime('%Y', ps.date) = '{season}'"
    return pd.read_sql(query, engine)


def load_hitting_stats(season=None):
    """Load all hitting game stats with team/player names."""
    query = """
        SELECT hs.*, t.name as team_name, t.abbreviation as team_abbr,
               p.name as player_name
        FROM hitting_game_stats hs
        JOIN teams t ON hs.team_id = t.id
        JOIN players p ON hs.player_id = p.id
    """
    if season:
        query += f" WHERE strftime('%Y', hs.date) = '{season}'"
    return pd.read_sql(query, engine)


# ═══════════════════════════════════════════════════════════════
# STEP 2: TEAM RECORD & RUN DIFFERENTIAL
# ═══════════════════════════════════════════════════════════════

def build_team_records(games_df):
    """Calculate W-L record and run differential for each team."""
    records = []

    teams = set(games_df['home_team'].unique()) | set(games_df['away_team'].unique())

    for team in teams:
        home = games_df[games_df['home_team'] == team]
        away = games_df[games_df['away_team'] == team]

        home_w = (home['home_score'] > home['away_score']).sum()
        home_l = (home['home_score'] < home['away_score']).sum()
        away_w = (away['away_score'] > away['home_score']).sum()
        away_l = (away['away_score'] < away['home_score']).sum()

        rs_home = home['home_score'].sum()
        ra_home = home['away_score'].sum()
        rs_away = away['away_score'].sum()
        ra_away = away['home_score'].sum()

        w = home_w + away_w
        l = home_l + away_l
        rs = rs_home + rs_away
        ra = ra_home + ra_away

        # Pythagorean expected wins (exponent 1.83 is standard)
        if rs + ra > 0:
            pyth_pct = rs**1.83 / (rs**1.83 + ra**1.83)
        else:
            pyth_pct = 0.500

        records.append({
            'team': team,
            'wins': w, 'losses': l,
            'win_pct': w / max(w + l, 1),
            'home_record': f"{home_w}-{home_l}",
            'away_record': f"{away_w}-{away_l}",
            'runs_scored': rs, 'runs_allowed': ra,
            'run_diff': rs - ra,
            'run_diff_per_game': (rs - ra) / max(w + l, 1),
            'pyth_win_pct': pyth_pct,
            'pyth_wins': round(pyth_pct * (w + l)),
            'games': w + l,
        })

    return pd.DataFrame(records).sort_values('win_pct', ascending=False)


# ═══════════════════════════════════════════════════════════════
# STEP 3: TEAM PITCHING FEATURES
# ═══════════════════════════════════════════════════════════════

def build_team_pitching(pitching_df):
    """Aggregate pitcher stats to team level."""
    if pitching_df.empty:
        return pd.DataFrame()

    team_groups = pitching_df.groupby('team_name')

    results = []
    for team, group in team_groups:
        total_ip = group['ip'].sum()
        if total_ip == 0:
            continue

        total_er = group['earned_runs'].sum()
        total_hits = group['hits'].sum()
        total_bb = group['walks'].sum()
        total_k = group['strikeouts'].sum()
        total_hr = group['home_runs'].sum()
        total_runs = group['runs'].sum()
        total_pitches = group['pitches'].sum()
        total_strikes = group['strikes'].sum()

        # ERA
        era = (total_er / total_ip) * 9 if total_ip > 0 else 0

        # WHIP
        whip = (total_hits + total_bb) / total_ip if total_ip > 0 else 0

        # K and BB rates (per 9 innings)
        k_per_9 = (total_k / total_ip) * 9 if total_ip > 0 else 0
        bb_per_9 = (total_bb / total_ip) * 9 if total_ip > 0 else 0

        # K% and BB% (per batter faced approximation)
        # BF ≈ IP*3 + H + BB (rough but close without actual BF data)
        approx_bf = total_ip * 3 + total_hits + total_bb
        k_pct = total_k / approx_bf * 100 if approx_bf > 0 else 0
        bb_pct = total_bb / approx_bf * 100 if approx_bf > 0 else 0
        k_bb_pct = k_pct - bb_pct

        # HR/9
        hr_per_9 = (total_hr / total_ip) * 9 if total_ip > 0 else 0

        # FIP (Fielding Independent Pitching)
        # FIP = ((13*HR + 3*BB - 2*K) / IP) + FIP_constant
        # FIP constant ≈ 3.10 (league average, adjusts yearly)
        fip_constant = 3.10
        fip = ((13 * total_hr + 3 * total_bb - 2 * total_k) / total_ip) + fip_constant if total_ip > 0 else 0

        # ERA - FIP gap (negative = ERA better than FIP, luck/defense helping)
        era_fip_gap = era - fip

        # BABIP allowed (approximation: (H - HR) / (approx_BF - K - HR - BB))
        babip_denom = approx_bf - total_k - total_hr - total_bb
        babip = (total_hits - total_hr) / babip_denom if babip_denom > 0 else LEAGUE_AVG_BABIP

        # Strike percentage
        strike_pct = total_strikes / total_pitches * 100 if total_pitches > 0 else 0

        # Pitchers used count (unique pitchers)
        unique_pitchers = group['player_id'].nunique()

        results.append({
            'team': team,
            'team_era': round(era, 2),
            'team_fip': round(fip, 2),
            'team_whip': round(whip, 2),
            'team_k_per_9': round(k_per_9, 2),
            'team_bb_per_9': round(bb_per_9, 2),
            'team_k_pct': round(k_pct, 1),
            'team_bb_pct': round(bb_pct, 1),
            'team_k_bb_pct': round(k_bb_pct, 1),
            'team_hr_per_9': round(hr_per_9, 2),
            'team_fip_era_gap': round(era_fip_gap, 2),
            'team_babip_allowed': round(babip, 3),
            'team_babip_vs_avg': round(babip - LEAGUE_AVG_BABIP, 3),
            'team_strike_pct': round(strike_pct, 1),
            'team_ip': round(total_ip, 1),
            'team_pitchers_used': unique_pitchers,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# STEP 4: TEAM HITTING FEATURES
# ═══════════════════════════════════════════════════════════════

def build_team_hitting(hitting_df):
    """Aggregate hitter stats to team level."""
    if hitting_df.empty:
        return pd.DataFrame()

    team_groups = hitting_df.groupby('team_name')

    results = []
    for team, group in team_groups:
        total_pa = group['plate_appearances'].sum()
        total_ab = group['at_bats'].sum()
        total_h = group['hits'].sum()
        total_2b = group['doubles'].sum()
        total_3b = group['triples'].sum()
        total_hr = group['home_runs'].sum()
        total_bb = group['walks'].sum()
        total_k = group['strikeouts'].sum()
        total_rbi = group['rbi'].sum()
        total_r = group['runs'].sum()
        total_sb = group['stolen_bases'].sum()

        if total_ab == 0 or total_pa == 0:
            continue

        # Basic
        avg = total_h / total_ab
        obp = (total_h + total_bb) / total_pa  # Simplified, no HBP
        slg_numer = (total_h - total_2b - total_3b - total_hr) + (2 * total_2b) + (3 * total_3b) + (4 * total_hr)
        slg = slg_numer / total_ab
        ops = obp + slg

        # Rate stats
        k_pct = total_k / total_pa * 100
        bb_pct = total_bb / total_pa * 100
        k_bb_pct = k_pct - bb_pct
        iso = slg - avg  # Isolated power
        hr_pct = total_hr / total_pa * 100

        # BABIP: (H - HR) / (AB - K - HR + SF)
        # No SF data, so approximate: (H - HR) / (AB - K - HR)
        babip_denom = total_ab - total_k - total_hr
        babip = (total_h - total_hr) / babip_denom if babip_denom > 0 else 0.300

        # wRC+ approximation (simplified, league-relative)
        # True wRC+ needs league wOBA and park factors
        # We'll compute wOBA and compare to league average
        # wOBA weights (2024 approximate): BB=0.69, 1B=0.88, 2B=1.24, 3B=1.56, HR=2.01
        singles = total_h - total_2b - total_3b - total_hr
        woba = (0.69 * total_bb + 0.88 * singles + 1.24 * total_2b +
                1.56 * total_3b + 2.01 * total_hr) / total_pa
        league_woba = 0.310  # Approximate league average
        wrc_plus_approx = (woba / league_woba) * 100

        results.append({
            'team': team,
            'team_avg': round(avg, 3),
            'team_obp': round(obp, 3),
            'team_slg': round(slg, 3),
            'team_ops': round(ops, 3),
            'team_woba': round(woba, 3),
            'team_wrc_plus': round(wrc_plus_approx, 0),
            'team_iso': round(iso, 3),
            'team_k_pct_hit': round(k_pct, 1),
            'team_bb_pct_hit': round(bb_pct, 1),
            'team_k_bb_pct_hit': round(k_bb_pct, 1),
            'team_babip': round(babip, 3),
            'team_babip_vs_avg': round(babip - LEAGUE_AVG_BABIP, 3),
            'team_hr_total': total_hr,
            'team_sb_total': total_sb,
            'team_runs': total_r,
            'team_rbi': total_rbi,
            'team_pa': total_pa,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# STEP 5: ROLLING WINDOW FEATURES
# ═══════════════════════════════════════════════════════════════

def build_rolling_features(games_df, pitching_df, hitting_df, window_days=30):
    """
    Calculate team stats over a rolling window.
    This captures RECENT form, not full-season averages.
    Essential for detecting hot/cold streaks and regression.
    """
    if games_df.empty:
        return pd.DataFrame()

    # Get the most recent date in the data
    games_df['date'] = pd.to_datetime(games_df['date'])
    latest_date = games_df['date'].max()
    cutoff = latest_date - timedelta(days=window_days)

    # Filter to window
    recent_games = games_df[games_df['date'] >= cutoff]

    pitching_df['date'] = pd.to_datetime(pitching_df['date'])
    hitting_df['date'] = pd.to_datetime(hitting_df['date'])
    recent_pitching = pitching_df[pitching_df['date'] >= cutoff]
    recent_hitting = hitting_df[hitting_df['date'] >= cutoff]

    # Build features on the window
    records = build_team_records(recent_games)
    pitching = build_team_pitching(recent_pitching)
    hitting = build_team_hitting(recent_hitting)

    # Rename columns with window prefix
    prefix = f"last_{window_days}d"
    records = records.rename(columns={c: f"{prefix}_{c}" for c in records.columns if c != 'team'})
    pitching = pitching.rename(columns={c: f"{prefix}_{c}" for c in pitching.columns if c != 'team'})
    hitting = hitting.rename(columns={c: f"{prefix}_{c}" for c in hitting.columns if c != 'team'})

    # Merge
    result = records
    if not pitching.empty:
        result = result.merge(pitching, on='team', how='left')
    if not hitting.empty:
        result = result.merge(hitting, on='team', how='left')

    return result


# ═══════════════════════════════════════════════════════════════
# STEP 6: BULLPEN FEATURES (fatigue indicator)
# ═══════════════════════════════════════════════════════════════

def build_bullpen_features(pitching_df, days=7):
    """
    Bullpen usage and fatigue over recent days.
    High bullpen IP in short window = fatigue risk.
    """
    if pitching_df.empty:
        return pd.DataFrame()

    pitching_df['date'] = pd.to_datetime(pitching_df['date'])
    latest = pitching_df['date'].max()
    cutoff = latest - timedelta(days=days)
    recent = pitching_df[pitching_df['date'] >= cutoff]

    results = []
    for team, group in recent.groupby('team_name'):
        # Relievers = anyone who pitched < 5 IP in a game (rough filter)
        relievers = group[group['ip'] < 5.0]

        bp_ip = relievers['ip'].sum()
        bp_appearances = len(relievers)
        bp_unique = relievers['player_id'].nunique()
        bp_er = relievers['earned_runs'].sum()
        bp_k = relievers['strikeouts'].sum()
        bp_bb = relievers['walks'].sum()
        bp_era = (bp_er / bp_ip) * 9 if bp_ip > 0 else 0

        # Heavy usage flag: >25 IP in 7 days is a lot
        heavy_usage = bp_ip > 25

        results.append({
            'team': team,
            f'bp_{days}d_ip': round(bp_ip, 1),
            f'bp_{days}d_appearances': bp_appearances,
            f'bp_{days}d_unique_arms': bp_unique,
            f'bp_{days}d_era': round(bp_era, 2),
            f'bp_{days}d_k': bp_k,
            f'bp_{days}d_bb': bp_bb,
            f'bp_{days}d_heavy_usage': heavy_usage,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# STEP 7: REGRESSION INDICATORS
# ═══════════════════════════════════════════════════════════════

def build_regression_flags(season_pitching, season_hitting):
    """
    Flag teams likely to regress or improve based on underlying metrics.
    This is your pitching_underlying logic, now DATA-DRIVEN.
    """
    if season_pitching.empty:
        return pd.DataFrame()

    results = []
    for _, row in season_pitching.iterrows():
        team = row['team']
        flags = []
        risk_score = 0  # Higher = more regression risk

        # BABIP allowed too low → regression UP (ERA will rise)
        if row['team_babip_allowed'] < 0.280:
            flags.append(f"BABIP {row['team_babip_allowed']:.3f} unsustainably low")
            risk_score += 3

        # BABIP allowed too high → regression DOWN (ERA might improve)
        if row['team_babip_allowed'] > 0.315:
            flags.append(f"BABIP {row['team_babip_allowed']:.3f} unluckily high")
            risk_score -= 2

        # ERA much better than FIP → luck helping
        if row['team_fip_era_gap'] > 0.30:
            flags.append(f"ERA-FIP gap {row['team_fip_era_gap']:+.2f} (FIP worse)")
            risk_score += 2

        # ERA much worse than FIP → unlucky
        if row['team_fip_era_gap'] < -0.30:
            flags.append(f"ERA-FIP gap {row['team_fip_era_gap']:+.2f} (unlucky)")
            risk_score -= 2

        # Low K-BB% → staff is weak
        if row['team_k_bb_pct'] < 8.0:
            flags.append(f"K-BB% {row['team_k_bb_pct']:.1f}% (bottom tier)")
            risk_score += 2

        # Elite K-BB% → staff is legit
        if row['team_k_bb_pct'] > 16.0:
            flags.append(f"K-BB% {row['team_k_bb_pct']:.1f}% (elite)")
            risk_score -= 1

        # Get hitting BABIP for this team
        hit_row = season_hitting[season_hitting['team'] == team]
        if not hit_row.empty:
            hit_babip = hit_row.iloc[0]['team_babip']
            if hit_babip > 0.320:
                flags.append(f"Hitting BABIP {hit_babip:.3f} (lucky)")
                risk_score += 2
            if hit_babip < 0.275:
                flags.append(f"Hitting BABIP {hit_babip:.3f} (unlucky)")
                risk_score -= 2

        regression_dir = "UP (will get worse)" if risk_score > 2 else \
                         "DOWN (will improve)" if risk_score < -2 else "STABLE"

        results.append({
            'team': team,
            'regression_risk_score': risk_score,
            'regression_direction': regression_dir,
            'regression_flags': " | ".join(flags) if flags else "None",
            'flag_count': len(flags),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# MASTER: BUILD ALL FEATURES
# ═══════════════════════════════════════════════════════════════

def build_all_features(season=None, save=True):
    """
    Build complete feature set for all teams.
    This is the main entry point.
    """
    print("=" * 70)
    print(f"BUILDING TEAM FEATURES{f' — {season}' if season else ''}")
    print("=" * 70)

    # Load raw data
    print("\nLoading data from database...")
    games = load_games(season)
    pitching = load_pitching_stats(season)
    hitting = load_hitting_stats(season)
    print(f"  Games:    {len(games)}")
    print(f"  Pitching: {len(pitching)} lines")
    print(f"  Hitting:  {len(hitting)} lines")

    if games.empty:
        print("No data found. Run backfill first.")
        return None

    # Build each feature group
    print("\nBuilding features...")

    print("  1/6 Team records & run differential...")
    records = build_team_records(games)

    print("  2/6 Team pitching stats...")
    team_pitching = build_team_pitching(pitching)

    print("  3/6 Team hitting stats...")
    team_hitting = build_team_hitting(hitting)

    print("  4/6 Rolling windows (7d, 14d, 30d)...")
    rolling_features = {}
    for window in ROLLING_WINDOWS:
        rf = build_rolling_features(games, pitching, hitting, window_days=window)
        rolling_features[window] = rf

    print("  5/6 Bullpen fatigue (7d)...")
    bullpen = build_bullpen_features(pitching, days=7)

    print("  6/6 Regression flags...")
    regression = build_regression_flags(team_pitching, team_hitting)

    # Merge everything
    print("\nMerging all features...")
    features = records
    if not team_pitching.empty:
        features = features.merge(team_pitching, on='team', how='left')
    if not team_hitting.empty:
        features = features.merge(team_hitting, on='team', how='left')
    for window, rf in rolling_features.items():
        if not rf.empty:
            features = features.merge(rf, on='team', how='left')
    if not bullpen.empty:
        features = features.merge(bullpen, on='team', how='left')
    if not regression.empty:
        features = features.merge(regression, on='team', how='left')

    print(f"\nFinal feature matrix: {features.shape[0]} teams × {features.shape[1]} features")

    # Save
    if save:
        os.makedirs('data/features', exist_ok=True)
        suffix = f"_{season}" if season else ""
        path = f"data/features/team_features{suffix}.csv"
        features.to_csv(path, index=False)
        print(f"Saved to: {path}")

    # Display summary
    print(f"\n{'=' * 70}")
    print("FEATURE SUMMARY — TOP 10 BY WIN%")
    print(f"{'=' * 70}")

    display_cols = ['team', 'wins', 'losses', 'win_pct', 'pyth_wins',
                    'run_diff', 'team_era', 'team_fip', 'team_k_bb_pct',
                    'team_obp', 'team_ops', 'team_wrc_plus',
                    'regression_direction']

    available = [c for c in display_cols if c in features.columns]
    top = features.nlargest(10, 'win_pct')

    for _, row in top.iterrows():
        print(f"\n  {row['team']}")
        print(f"    Record: {row['wins']}-{row['losses']} ({row['win_pct']:.3f}) | Pyth: {row['pyth_wins']}W")
        print(f"    Run diff: {row['run_diff']:+d} ({row.get('run_diff_per_game', 0):+.2f}/gm)")
        if 'team_era' in row:
            print(f"    Pitching: ERA {row['team_era']:.2f} | FIP {row['team_fip']:.2f} | K-BB% {row['team_k_bb_pct']:.1f}")
        if 'team_ops' in row:
            print(f"    Hitting:  OBP {row['team_obp']:.3f} | OPS {row['team_ops']:.3f} | wRC+ {row['team_wrc_plus']:.0f}")
        if 'regression_direction' in row:
            print(f"    Regression: {row['regression_direction']}")

    # Regression alerts
    if 'regression_risk_score' in features.columns:
        print(f"\n{'=' * 70}")
        print("REGRESSION ALERTS")
        print(f"{'=' * 70}")

        risers = features[features['regression_risk_score'] > 2].sort_values('regression_risk_score', ascending=False)
        fallers = features[features['regression_risk_score'] < -2].sort_values('regression_risk_score')

        if not risers.empty:
            print("\n  LIKELY TO GET WORSE:")
            for _, r in risers.iterrows():
                print(f"    {r['team']} (risk: {r['regression_risk_score']}) — {r['regression_flags']}")

        if not fallers.empty:
            print("\n  LIKELY TO IMPROVE:")
            for _, r in fallers.iterrows():
                print(f"    {r['team']} (score: {r['regression_risk_score']}) — {r['regression_flags']}")

        stable = features[(features['regression_risk_score'] >= -2) & (features['regression_risk_score'] <= 2)]
        print(f"\n  STABLE: {len(stable)} teams with no major regression signals")

    return features


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build team features from game data")
    parser.add_argument("--season", type=int, help="Season to build features for")
    parser.add_argument("--team", type=str, help="Debug: show one team's features")
    args = parser.parse_args()

    features = build_all_features(season=args.season)

    if args.team and features is not None:
        team_row = features[features['team'].str.contains(args.team, case=False)]
        if not team_row.empty:
            print(f"\n{'=' * 70}")
            print(f"FULL FEATURES: {args.team}")
            print(f"{'=' * 70}")
            for col in team_row.columns:
                val = team_row.iloc[0][col]
                print(f"  {col:40s}: {val}")