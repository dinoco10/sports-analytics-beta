"""
Power Rankings Model v2 — Data-Driven
Now feeds from your database instead of hardcoded numbers.

Usage:
    python -m src.models.power_rankings --season 2025
    python -m src.models.power_rankings --season 2024
"""

import sys, os, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.team_features import build_all_features
from config.weights import get_main_weights, get_pitching_underlying_sub_weights
from src.features.player_projections import run_projections


# ═══════════════════════════════════════════════════════════════
# FANGRAPHS PROJECTIONS (manual input until we automate scraping)
# Update these from FanGraphs preseason/in-season projections
# ═══════════════════════════════════════════════════════════════

def get_fangraphs_projections():
    """
    FanGraphs projected wins. Update manually from:
    https://www.fangraphs.com/depthcharts.aspx?position=Standings

    These are the EXPERT BASELINE your model blends with.
    """
    return {
        'Los Angeles Dodgers': 100, 'Atlanta Braves': 92,
        'New York Yankees': 91, 'Baltimore Orioles': 90,
        'Philadelphia Phillies': 89, 'Houston Astros': 88,
        'New York Mets': 87, 'Seattle Mariners': 86,
        'Milwaukee Brewers': 86, 'San Diego Padres': 85,
        'Toronto Blue Jays': 85, 'Cleveland Guardians': 84,
        'Minnesota Twins': 84, 'Boston Red Sox': 83,
        'Detroit Tigers': 83, 'Tampa Bay Rays': 82,
        'Texas Rangers': 82, 'Arizona Diamondbacks': 82,
        'Chicago Cubs': 81, 'San Francisco Giants': 80,
        'Kansas City Royals': 80, 'Cincinnati Reds': 79,
        'St. Louis Cardinals': 78, 'Pittsburgh Pirates': 76,
        'Los Angeles Angels': 74, 'Washington Nationals': 72,
        'Miami Marlins': 68, 'Oakland Athletics': 65,
        'Colorado Rockies': 64, 'Chicago White Sox': 62,
    }


# ═══════════════════════════════════════════════════════════════
# YOUR COMPONENT RATINGS (1-10, manual eye test)
# This is where YOUR baseball knowledge lives
# ═══════════════════════════════════════════════════════════════

def get_manual_ratings():
    """
    Your 1-10 ratings for each team component.
    These capture things DATA CAN'T: eye test, offseason moves,
    clubhouse vibes, coaching changes, prospect readiness.

    Update these as you watch games and follow news.
    """
    ratings = {
        #                          rot  bull  pow  cont  def  spd  dep  mgr  farm  mom
        'Los Angeles Dodgers':    [ 9,   8,    9,   9,    8,   7,   9,   8,   7,    9],
        'Atlanta Braves':         [ 8,   7,    8,   8,    7,   6,   8,   6,   8,    7],
        'New York Yankees':       [ 7,   7,    9,   7,    6,   5,   7,   5,   6,    6],
        'Baltimore Orioles':      [ 7,   6,    8,   7,    7,   7,   7,   6,   9,    7],
        'Philadelphia Phillies':  [ 8,   7,    8,   7,    7,   6,   7,   7,   6,    7],
        'Houston Astros':         [ 7,   7,    7,   8,    7,   6,   7,   8,   5,    6],
        'New York Mets':          [ 7,   6,    7,   7,    6,   6,   7,   5,   6,    7],
        'Seattle Mariners':       [ 8,   8,    5,   6,    8,   7,   6,   5,   7,    5],
        'Milwaukee Brewers':      [ 7,   7,    6,   7,    7,   7,   7,   6,   8,    7],
        'San Diego Padres':       [ 7,   6,    7,   7,    7,   7,   7,   5,   5,    6],
        'Toronto Blue Jays':      [ 7,   6,    7,   7,    6,   5,   7,   5,   7,    5],
        'Cleveland Guardians':    [ 7,   8,    5,   7,    8,   7,   7,   7,   7,    7],
        'Minnesota Twins':        [ 6,   6,    7,   7,    6,   5,   6,   5,   6,    5],
        'Boston Red Sox':         [ 6,   5,    7,   7,    5,   6,   6,   5,   7,    6],
        'Detroit Tigers':         [ 7,   7,    5,   6,    7,   6,   6,   6,   8,    7],
        'Tampa Bay Rays':         [ 6,   7,    5,   6,    7,   7,   6,   7,   8,    5],
        'Texas Rangers':          [ 6,   5,    7,   6,    6,   5,   6,   6,   5,    4],
        'Arizona Diamondbacks':   [ 6,   5,    7,   6,    5,   7,   6,   6,   6,    5],
        'Chicago Cubs':           [ 5,   5,    6,   6,    6,   5,   6,   5,   6,    5],
        'San Francisco Giants':   [ 6,   5,    5,   6,    6,   5,   6,   5,   5,    4],
        'Kansas City Royals':     [ 6,   6,    5,   6,    6,   7,   5,   5,   7,    6],
        'Cincinnati Reds':        [ 5,   5,    7,   6,    5,   6,   5,   5,   7,    5],
        'St. Louis Cardinals':    [ 5,   5,    5,   6,    6,   5,   5,   5,   6,    4],
        'Pittsburgh Pirates':     [ 5,   5,    5,   5,    5,   6,   5,   4,   8,    5],
        'Los Angeles Angels':     [ 5,   4,    6,   5,    5,   5,   5,   4,   5,    4],
        'Washington Nationals':   [ 4,   4,    5,   5,    5,   5,   4,   4,   7,    4],
        'Miami Marlins':          [ 4,   4,    4,   4,    5,   5,   3,   4,   6,    3],
        'Oakland Athletics':      [ 4,   4,    4,   4,    4,   5,   3,   4,   5,    3],
        'Colorado Rockies':       [ 4,   4,    5,   4,    4,   5,   4,   4,   4,    3],
        'Chicago White Sox':      [ 3,   3,    4,   4,    4,   4,   3,   3,   5,    2],
    }

    component_names = [
        'rotation_strength', 'bullpen_depth', 'lineup_power',
        'lineup_contact', 'defense', 'speed_baserunning',
        'depth', 'manager_coaching', 'farm_system', 'momentum'
    ]

    rows = []
    for team, vals in ratings.items():
        row = {'team': team}
        for name, val in zip(component_names, vals):
            row[name] = val
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# PERSONAL ADJUSTMENTS (-10 to +10 wins)
# ═══════════════════════════════════════════════════════════════

def get_personal_adjustments():
    """
    Your override: add/subtract wins based on gut + knowledge.
    Positive = you're more bullish than data suggests.
    Negative = you see problems data doesn't capture.
    """
    return {
        # team: (adjustment, reason, confidence 1-10)
        'Boston Red Sox':        (-1, "Bello K-BB% 5th worst in MLB, regression risk", 7),
        'Cleveland Guardians':   (-1, "Williams biggest FIP overperformer, unsustainable", 6),
        'Detroit Tigers':        (+1, "Young pitching upside, Skubal Cy Young contender", 7),
        'Baltimore Orioles':     (+1, "Farm system ready to contribute, elite prospect depth", 7),
        'Kansas City Royals':    (-1, "Low BABIP staff-wide, regression coming", 6),
    }


# ═══════════════════════════════════════════════════════════════
# PITCHING UNDERLYING SCORE (from real data now!)
# ═══════════════════════════════════════════════════════════════

def calculate_pitching_underlying_from_features(features_df):
    """
    Calculate pitching underlying score using REAL DATA from your database.
    This replaces the hardcoded values in the old model.

    Uses the same sub-weights but now with calculated metrics.
    """
    sub_weights = get_pitching_underlying_sub_weights()
    df = features_df.copy()

    # Map available features to sub-weight metrics
    # Some metrics need to be normalized to 0-1 scale

    scores = pd.DataFrame()
    scores['team'] = df['team']

    # K-BB%: Higher = better. Range 5-20%
    if 'team_k_bb_pct' in df.columns:
        scores['k_bb_norm'] = ((df['team_k_bb_pct'] - 5) / (20 - 5)).clip(0, 1)
    else:
        scores['k_bb_norm'] = 0.5

    # FIP-ERA gap: closer to 0 = more sustainable
    # Negative gap = ERA < FIP = luck helping (risky)
    if 'team_fip_era_gap' in df.columns:
        def norm_gap(gap):
            if gap > 0:  # ERA > FIP (unlucky, might improve)
                return min(1, 0.5 + gap / 1.0)
            else:  # ERA < FIP (lucky, might regress)
                return max(0, 1 - abs(gap) / 0.8)
        scores['fip_gap_norm'] = df['team_fip_era_gap'].apply(norm_gap)
    else:
        scores['fip_gap_norm'] = 0.5

    # BABIP allowed: .300 is neutral
    if 'team_babip_allowed' in df.columns:
        def norm_babip(babip):
            dev = babip - 0.300
            if dev < 0:  # Low BABIP = lucky
                return max(0, 1 - abs(dev) / 0.040)
            else:  # High BABIP = unlucky, might improve
                return min(1, 0.5 + dev / 0.040)
        scores['babip_norm'] = df['team_babip_allowed'].apply(norm_babip)
    else:
        scores['babip_norm'] = 0.5

    # GB%: Higher = better. Range 35-50%
    # We don't have GB% from basic stats yet — placeholder
    scores['gb_norm'] = 0.5

    # Barrel%: We don't have this from MLB API — placeholder
    scores['barrel_norm'] = 0.5

    # SwStr%: We don't have this from MLB API — placeholder
    scores['swstr_norm'] = 0.5

    # HR/9 as proxy for HR/FB%: Lower = better. Range 0.5-1.8
    if 'team_hr_per_9' in df.columns:
        scores['hr_norm'] = (1 - (df['team_hr_per_9'] - 0.5) / (1.8 - 0.5)).clip(0, 1)
    else:
        scores['hr_norm'] = 0.5

    # xERA gap: we don't have xERA from basic API — use FIP gap as proxy
    scores['xera_norm'] = scores['fip_gap_norm']

    # Calculate weighted score
    scores['pitching_underlying'] = (
        scores['k_bb_norm'] * sub_weights['K_BB_pct'] +
        scores['xera_norm'] * sub_weights['xERA_gap'] +
        scores['fip_gap_norm'] * sub_weights['FIP_era_gap'] +
        scores['babip_norm'] * sub_weights['BABIP_allowed'] +
        scores['hr_norm'] * sub_weights['HR_FB_pct'] +
        scores['gb_norm'] * sub_weights['GB_pct'] +
        scores['barrel_norm'] * sub_weights['barrel_pct'] +
        scores['swstr_norm'] * sub_weights['swstr_pct']
    )

    # Scale to 1-10
    scores['pitching_underlying'] = (scores['pitching_underlying'] * 9 + 1).clip(1, 10).round(1)

    return scores[['team', 'pitching_underlying']]


# ═══════════════════════════════════════════════════════════════
# CORE: CALCULATE POWER RANKINGS
# ═══════════════════════════════════════════════════════════════

def calculate_power_rankings(season=2025, use_projections=True):
    """
    The main model. Blends up to 4 sources:
    1. FanGraphs projections (expert baseline)
    2. Data-driven features (from YOUR database — pythagorean wins)
    3. Your manual ratings + adjustments (eye test)
    4. Player projections (Marcel + aging, aggregated by 2026 roster) [NEW]

    Returns DataFrame with final rankings.
    """
    print("\n" + "=" * 70)
    print(f"POWER RANKINGS MODEL v2 — {season}")
    print("=" * 70)

    weights = get_main_weights()

    # ─── Source 1: FanGraphs ──────────────────────────────
    fg = get_fangraphs_projections()
    fg_df = pd.DataFrame([
        {'team': k, 'fg_wins': v} for k, v in fg.items()
    ])
    print(f"\n  Source 1: FanGraphs projections — {len(fg_df)} teams")

    # ─── Source 2: Data-driven features ───────────────────
    features = build_all_features(season=season, save=True)
    if features is None:
        print("ERROR: No features available")
        return None
    print(f"  Source 2: Database features — {features.shape[1]} features")
    features['data_wins'] = features['pyth_wins']

    # ─── Source 3: Your ratings ───────────────────────────
    manual = get_manual_ratings()
    personal = get_personal_adjustments()
    pitching_scores = calculate_pitching_underlying_from_features(features)
    print(f"  Source 3: Your ratings — {len(manual)} teams")

    # ─── Source 4: Player Projections (NEW) ───────────────
    proj_wins_df = None
    if use_projections:
        try:
            print(f"\n  Source 4: Running player projections (Marcel + aging)...")
            p_proj, h_proj = run_projections()

            # Aggregate projected WAR by current team
            team_war = []
            all_teams = set(p_proj['current_team'].unique()) | set(h_proj['current_team'].unique())
            for team in all_teams:
                if team == 'Free Agent':
                    continue
                p_war = p_proj[p_proj['current_team'] == team]['proj_war'].sum()
                h_war = h_proj[h_proj['current_team'] == team]['proj_war'].sum()
                total = p_war + h_war
                team_war.append({
                    'team': team,
                    'proj_total_war': round(total, 1),
                    'proj_pitch_war': round(p_war, 1),
                    'proj_hit_war': round(h_war, 1),
                    'proj_wins_roster': round(48 + total, 0),  # 48 = replacement level
                })
            proj_wins_df = pd.DataFrame(team_war)
            print(f"  Source 4: Player projections — {len(proj_wins_df)} teams")
        except Exception as e:
            print(f"  Source 4: FAILED ({e}) — proceeding without")
            use_projections = False

    # ─── Merge everything ─────────────────────────────────
    df = fg_df.copy()
    df = df.merge(features[['team', 'data_wins', 'win_pct', 'wins', 'losses',
                             'run_diff', 'run_diff_per_game', 'pyth_win_pct',
                             'team_era', 'team_fip', 'team_k_bb_pct',
                             'team_obp', 'team_ops', 'team_wrc_plus',
                             'team_babip_allowed', 'team_fip_era_gap',
                             'regression_direction', 'regression_risk_score']],
                  on='team', how='left')
    df = df.merge(manual, on='team', how='left')
    df = df.merge(pitching_scores, on='team', how='left')

    if proj_wins_df is not None:
        df = df.merge(proj_wins_df, on='team', how='left')
        df['proj_wins_roster'] = df['proj_wins_roster'].fillna(df['data_wins'])
    else:
        df['proj_wins_roster'] = df['data_wins']
        df['proj_total_war'] = np.nan
        df['proj_pitch_war'] = np.nan
        df['proj_hit_war'] = np.nan

    # ─── Calculate component-based wins (your eye test) ───
    component_cols = [
        'rotation_strength', 'bullpen_depth', 'lineup_power',
        'lineup_contact', 'defense', 'speed_baserunning',
        'depth', 'manager_coaching', 'farm_system', 'momentum'
    ]

    df['component_score'] = 0
    for col in component_cols:
        w = weights.get(col, 0.05)
        df['component_score'] += (df[col] / 10) * w

    df['component_score'] += (df['pitching_underlying'] / 10) * weights.get('pitching_underlying', 0.12)

    total_component_weight = sum(weights.get(c, 0.05) for c in component_cols) + weights.get('pitching_underlying', 0.12)
    df['component_score_normalized'] = df['component_score'] / total_component_weight
    df['user_wins'] = 60 + (df['component_score_normalized'] * 40)

    # Add personal adjustments
    df['personal_adj'] = 0.0
    df['adj_reason'] = ''
    for team, (adj, reason, conf) in personal.items():
        mask = df['team'] == team
        df.loc[mask, 'personal_adj'] = adj
        df.loc[mask, 'adj_reason'] = reason
    df['user_wins'] += df['personal_adj'] * weights.get('personal_weight', 0.05) * 20

    # ─── BLEND THE SOURCES ────────────────────────────────
    if use_projections:
        # 4-source blend: FG + Data + Projections + Eye Test
        source_weights = {
            'fg': 0.35,         # Expert baseline
            'data': 0.20,       # Historical pythagorean
            'projections': 0.30, # Aging-adjusted roster WAR
            'user': 0.15,       # Your eye test
        }
        df['model_wins'] = (
            df['fg_wins'] * source_weights['fg'] +
            df['data_wins'] * source_weights['data'] +
            df['proj_wins_roster'] * source_weights['projections'] +
            df['user_wins'] * source_weights['user']
        )
    else:
        # 3-source blend (original)
        source_weights = {'fg': 0.45, 'data': 0.35, 'user': 0.20}
        df['model_wins'] = (
            df['fg_wins'] * source_weights['fg'] +
            df['data_wins'] * source_weights['data'] +
            df['user_wins'] * source_weights['user']
        )

    # Sort by model wins
    df = df.sort_values('model_wins', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['vs_fangraphs'] = df['model_wins'] - df['fg_wins']

    # ─── PRINT RESULTS ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"2026 POWER RANKINGS")
    print(f"{'=' * 70}")
    print(f"\n  Sources: ", end="")
    print(" | ".join(f"{k.upper()} {v:.0%}" for k, v in source_weights.items()))

    for _, row in df.iterrows():
        rank = row['rank']
        diff = row['vs_fangraphs']
        arrow = " >>> BULLISH" if diff > 2 else " >>> BEARISH" if diff < -2 else ""

        print(f"\n  #{rank:2d} {row['team']}")
        line = f"      MODEL: {row['model_wins']:.1f}W | FG: {row['fg_wins']}W | Data: {row['data_wins']:.0f}W"
        if use_projections and pd.notna(row.get('proj_wins_roster')):
            line += f" | Proj: {row['proj_wins_roster']:.0f}W"
        line += f" | Eye: {row['user_wins']:.1f}W"
        print(line)

        print(f"      {season}: {row['wins']:.0f}-{row['losses']:.0f} | "
              f"ERA {row['team_era']:.2f} | FIP {row['team_fip']:.2f} | "
              f"K-BB% {row['team_k_bb_pct']:.1f} | OPS {row['team_ops']:.3f}")

        if use_projections and pd.notna(row.get('proj_total_war')):
            print(f"      Proj WAR: {row['proj_total_war']:.1f} "
                  f"(P: {row['proj_pitch_war']:.1f} | H: {row['proj_hit_war']:.1f})")

        print(f"      Underlying: {row['pitching_underlying']:.1f}/10 | "
              f"Regression: {row['regression_direction']}")

        if arrow:
            print(f"      {arrow} vs FanGraphs: {diff:+.1f} wins")
        if row['adj_reason']:
            print(f"      [YOU] {row['adj_reason']} ({row['personal_adj']:+.0f}W)")

    # ─── SAVE ─────────────────────────────────────────────
    os.makedirs('data/features', exist_ok=True)
    df.to_csv(f'data/features/power_rankings_{season}.csv', index=False)
    print(f"\nSaved: data/features/power_rankings_{season}.csv")

    # ─── MODEL DIAGNOSTICS ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MODEL DIAGNOSTICS")
    print(f"{'=' * 70}")

    corr_fg_data = df['fg_wins'].corr(df['data_wins'])
    corr_fg_user = df['fg_wins'].corr(df['user_wins'])
    print(f"  Correlation FG ↔ Data:  {corr_fg_data:.3f}")
    print(f"  Correlation FG ↔ You:   {corr_fg_user:.3f}")

    if use_projections:
        corr_fg_proj = df['fg_wins'].corr(df['proj_wins_roster'])
        corr_data_proj = df['data_wins'].corr(df['proj_wins_roster'])
        print(f"  Correlation FG ↔ Proj:  {corr_fg_proj:.3f}")
        print(f"  Correlation Data ↔ Proj: {corr_data_proj:.3f}")

    print(f"\n  Win range: {df['model_wins'].min():.1f} - {df['model_wins'].max():.1f}")
    print(f"  Biggest BULL: {df.loc[df['vs_fangraphs'].idxmax(), 'team']} "
          f"({df['vs_fangraphs'].max():+.1f}W)")
    print(f"  Biggest BEAR: {df.loc[df['vs_fangraphs'].idxmin(), 'team']} "
          f"({df['vs_fangraphs'].min():+.1f}W)")

    return df
   


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Power Rankings Model v2")
    parser.add_argument("--season", type=int, default=2025, help="Season data to use")
    args = parser.parse_args()

    df = calculate_power_rankings(season=args.season)