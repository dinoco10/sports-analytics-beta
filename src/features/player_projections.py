"""
Player Projection Engine — Marcel-style aging-adjusted projections.
Takes 2023-2025 stats, applies aging curves and regression to project 2026.

The Marcel Method (named after the monkey — "so simple a monkey could do it"):
1. Weight 3 years of data: recent = 5x, middle = 4x, oldest = 3x
2. Regress toward league average based on playing time
3. Apply aging adjustment

This beats most "smart" projection systems because it avoids overfitting.

Usage:
    python -m src.features.player_projections
    python -m src.features.player_projections --team "Yankees"
    python -m src.features.player_projections --player "Gerrit Cole"
"""

import sys, os, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from src.storage.database import engine
import requests, time

MLB = "https://statsapi.mlb.com/api/v1"


def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{MLB}/{endpoint}", params=params, timeout=30)
        r.raise_for_status(); time.sleep(0.3); return r.json()
    except: return {}


# ═══════════════════════════════════════════════════════════════
# AGING CURVES
# ═══════════════════════════════════════════════════════════════

def pitcher_aging_adjustment(age):
    """
    ERA adjustment by age. Based on historical aging curves.
    Positive = expected to get WORSE (higher ERA).
    Peak age for pitchers: 26-28.

    Source: FanGraphs aging curve research, simplified.
    """
    if age <= 25:   return -0.10  # Still improving
    elif age <= 28: return  0.00  # Peak
    elif age <= 30: return  0.05  # Slight decline
    elif age <= 32: return  0.12  # Noticeable
    elif age <= 34: return  0.20  # Significant
    elif age <= 36: return  0.30  # Steep
    elif age <= 38: return  0.40  # Very steep
    else:           return  0.55  # Cliff


def hitter_aging_adjustment(age):
    """
    wOBA adjustment by age. Negative = expected decline.
    Peak age for hitters: 26-29.
    """
    if age <= 25:   return +0.008  # Still improving
    elif age <= 29: return  0.000  # Peak
    elif age <= 31: return -0.005  # Slight decline
    elif age <= 33: return -0.012  # Noticeable
    elif age <= 35: return -0.020  # Significant
    elif age <= 37: return -0.030  # Steep
    else:           return -0.045  # Cliff


def playing_time_adjustment(age, role):
    """
    Projected IP or PA multiplier based on age and role.
    Older players get fewer projected innings/PAs due to injury risk.
    """
    if role == 'SP':
        if age <= 28: return 1.00
        elif age <= 30: return 0.95
        elif age <= 32: return 0.88
        elif age <= 34: return 0.80
        elif age <= 36: return 0.72
        else: return 0.60
    elif role == 'RP':
        if age <= 30: return 1.00
        elif age <= 33: return 0.92
        elif age <= 36: return 0.82
        else: return 0.70
    else:  # Hitter
        if age <= 29: return 1.00
        elif age <= 31: return 0.96
        elif age <= 33: return 0.90
        elif age <= 35: return 0.82
        elif age <= 37: return 0.72
        else: return 0.60


# ═══════════════════════════════════════════════════════════════
# LOAD MULTI-YEAR DATA
# ═══════════════════════════════════════════════════════════════

def load_pitcher_multi_year():
    """Load pitcher stats for 2023-2025 from database."""
    query = """
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
        WHERE strftime('%Y', ps.date) IN ('2023', '2024', '2025')
        GROUP BY p.mlb_id, p.name, p.birth_date, strftime('%Y', ps.date), t.name
        HAVING SUM(ps.ip) > 0
    """
    df = pd.read_sql(query, engine)
    df['season'] = df['season'].astype(int)

    # Rate stats
    df['era'] = (df['er'] / df['ip']) * 9
    df['bf'] = df['ip'] * 3 + df['hits'] + df['bb']
    df['k_pct'] = df['k'] / df['bf'] * 100
    df['bb_pct'] = df['bb'] / df['bf'] * 100
    df['k_bb_pct'] = df['k_pct'] - df['bb_pct']
    fip_const = 3.10
    df['fip'] = ((13 * df['hr'] + 3 * df['bb'] - 2 * df['k']) / df['ip']) + fip_const
    df['role'] = df['ip'].apply(lambda x: 'SP' if x >= 50 else 'RP')

    return df


def load_hitter_multi_year():
    """Load hitter stats for 2023-2025 from database."""
    query = """
        SELECT
            p.mlb_id as mlb_player_id,
            p.name as player_name,
            p.birth_date,
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
        WHERE strftime('%Y', hs.date) IN ('2023', '2024', '2025')
        GROUP BY p.mlb_id, p.name, p.birth_date, strftime('%Y', hs.date), t.name
        HAVING SUM(hs.plate_appearances) >= 10
    """
    df = pd.read_sql(query, engine)
    df['season'] = df['season'].astype(int)

    # Rate stats
    df['avg'] = df['h'] / df['ab'].replace(0, 1)
    df['obp'] = (df['h'] + df['bb']) / df['pa'].replace(0, 1)
    singles = df['h'] - df['doubles'] - df['triples'] - df['hr']
    slg_num = singles + 2*df['doubles'] + 3*df['triples'] + 4*df['hr']
    df['slg'] = slg_num / df['ab'].replace(0, 1)
    df['ops'] = df['obp'] + df['slg']
    df['woba'] = (0.69*df['bb'] + 0.88*singles + 1.24*df['doubles'] +
                  1.56*df['triples'] + 2.01*df['hr']) / df['pa'].replace(0, 1)
    df['k_pct'] = df['k'] / df['pa'] * 100
    df['bb_pct'] = df['bb'] / df['pa'] * 100

    return df


# ═══════════════════════════════════════════════════════════════
# STATCAST DATA LOADING + REGRESSION SIGNALS
# ═══════════════════════════════════════════════════════════════

def load_statcast_multi_year():
    """
    Load Statcast metrics for 2023-2025 from our player_statcast_metrics table.
    Returns a DataFrame with one row per player per season, keyed on mlb_player_id.

    This gives us the "underlying quality" data that traditional stats miss:
    - Expected stats (xwOBA, xBA, xSLG) show what SHOULD have happened
    - Contact quality (barrel rate, hard hit %) shows real bat quality
    - Plate discipline (chase rate, whiff rate) is the most stable signal
    """
    query = """
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
        WHERE sm.season IN (2023, 2024, 2025)
    """
    df = pd.read_sql(query, engine)
    print(f"  Statcast records loaded: {len(df)}")
    return df


def compute_luck_filters(player_statcast):
    """
    Calculate "luck filters" — gaps between expected and actual performance.
    These are the core regression signals that tell us if a player was lucky/unlucky.

    A player with xwOBA >> wOBA had hard contact that didn't fall for hits.
    That's BABIP luck, not a skill change. Expect regression UPWARD.

    Returns a dict with luck metrics for the most recent season available.

    Key signals:
    - babip_vs_career: negative = unlucky, expect bounce back
    - xba_minus_ba: positive = unlucky (expected BA higher than actual)
    - xslg_minus_slg: positive = unlucky power
    - hr_per_fb_vs_norm: negative = HR due for regression up (league norm ~11-13%)
    """
    if player_statcast.empty:
        return {}

    # Career averages (weighted by PA across available seasons)
    career_babip = player_statcast['babip'].mean()
    career_hr_per_fb = player_statcast['hr_per_fb'].mean()

    # Most recent season's luck gaps
    recent = player_statcast.sort_values('season').iloc[-1]

    result = {}

    # BABIP vs career average — if current BABIP is below career, expect regression up
    if pd.notna(recent.get('babip')) and pd.notna(career_babip):
        result['babip_vs_career'] = round(recent['babip'] - career_babip, 3)

    # xBA minus actual BA — positive means unlucky (xBA > BA)
    if pd.notna(recent.get('xba')) and pd.notna(recent.get('ba')):
        result['xba_minus_ba'] = round(recent['xba'] - recent['ba'], 3)

    # xSLG minus actual SLG — positive means unlucky power
    if pd.notna(recent.get('xslg')) and pd.notna(recent.get('slg')):
        result['xslg_minus_slg'] = round(recent['xslg'] - recent['slg'], 3)

    # HR/FB vs career norm — league average is ~11-13%, if below career, expect regression up
    if pd.notna(recent.get('hr_per_fb')) and pd.notna(career_hr_per_fb):
        result['hr_per_fb_vs_career'] = round(recent['hr_per_fb'] - career_hr_per_fb, 3)

    # xwOBA minus actual wOBA — the single best luck indicator
    if pd.notna(recent.get('xwoba')) and pd.notna(recent.get('woba')):
        result['xwoba_minus_woba'] = round(recent['xwoba'] - recent['woba'], 3)

    return result


def statcast_adjustment(marcel_woba, player_statcast, luck_filters):
    """
    Apply Statcast regression signals to adjust the Marcel base projection.

    This is the key innovation: Marcel gives us a solid 3-year weighted average,
    but it treats every .250 hitter the same. Statcast tells us WHICH .250 hitters
    actually hit the ball hard (unlucky) vs which made weak contact (lucky).

    Adjustment rules (each capped to prevent extreme swings):
    1. xwOBA >> wOBA AND BABIP below career → project upward (max +.020 wOBA)
    2. Barrel rate + hard hit rate stable/improving → trust/boost projection
    3. Pull air rate trending up → add HR upside (power development signal)
    4. Chase rate increasing → flag risk, reduce projection (approach declining)

    Returns: wOBA adjustment value (positive = upside, negative = downside)
    """
    if player_statcast.empty:
        return 0.0

    adjustment = 0.0
    recent = player_statcast.sort_values('season').iloc[-1]

    # --- RULE 1: xwOBA/BABIP luck correction ---
    # If xwOBA is significantly higher than actual wOBA AND BABIP is depressed,
    # the player was genuinely unlucky — not a skill issue
    xwoba_gap = luck_filters.get('xwoba_minus_woba', 0)
    babip_gap = luck_filters.get('babip_vs_career', 0)

    if xwoba_gap > 0.010 and babip_gap < -0.010:
        # Both signals agree: unlucky. Weight by the gap size.
        # Cap at +0.020 wOBA to avoid overreacting
        luck_adj = min(xwoba_gap * 0.5, 0.020)
        adjustment += luck_adj
    elif xwoba_gap < -0.010 and babip_gap > 0.010:
        # Lucky — actual stats were inflated. Regress downward.
        luck_adj = max(xwoba_gap * 0.4, -0.015)
        adjustment += luck_adj

    # --- RULE 2: Contact quality stability ---
    # If barrel rate and hard hit rate are stable or improving across years,
    # the underlying bat quality is real — trust the projection
    if len(player_statcast) >= 2:
        sorted_sc = player_statcast.sort_values('season')
        recent_barrel = sorted_sc.iloc[-1].get('barrel_rate')
        prev_barrel = sorted_sc.iloc[-2].get('barrel_rate')

        if pd.notna(recent_barrel) and pd.notna(prev_barrel):
            barrel_trend = recent_barrel - prev_barrel
            if barrel_trend > 1.0:
                # Improving barrel rate = real power development (+0.005 to +0.010)
                adjustment += min(barrel_trend * 0.003, 0.010)
            elif barrel_trend < -2.0:
                # Declining barrel rate = real skill erosion
                adjustment += max(barrel_trend * 0.002, -0.008)

    # --- RULE 3: Pull air rate trend (HR predictor) ---
    # Pulled fly balls are the #1 predictor of future home runs.
    # If a hitter is pulling more balls in the air, power is coming.
    if len(player_statcast) >= 2:
        sorted_sc = player_statcast.sort_values('season')
        recent_pull = sorted_sc.iloc[-1].get('pull_air_rate')
        prev_pull = sorted_sc.iloc[-2].get('pull_air_rate')

        if pd.notna(recent_pull) and pd.notna(prev_pull):
            pull_trend = recent_pull - prev_pull
            if pull_trend > 2.0:
                # More pulled fly balls = HR upside
                adjustment += min(pull_trend * 0.002, 0.008)

    # --- RULE 4: Chase rate risk ---
    # Chase rate (O-Swing%) is the most stable discipline metric.
    # If it's increasing, the hitter is chasing more pitches outside the zone.
    # This is a REAL skill change, not noise. Reduce projection.
    if len(player_statcast) >= 2:
        sorted_sc = player_statcast.sort_values('season')
        recent_chase = sorted_sc.iloc[-1].get('chase_rate')
        prev_chase = sorted_sc.iloc[-2].get('chase_rate')

        if pd.notna(recent_chase) and pd.notna(prev_chase):
            chase_trend = recent_chase - prev_chase
            if chase_trend > 2.0:
                # Chasing more = approach deteriorating
                adjustment -= min(chase_trend * 0.002, 0.010)
            elif chase_trend < -2.0:
                # Chasing less = approach improving (discipline breakout)
                adjustment += min(abs(chase_trend) * 0.001, 0.005)

    # Final cap: never adjust more than +/- 0.030 wOBA
    adjustment = max(-0.030, min(0.030, adjustment))
    return round(adjustment, 3)


def compute_scores(player_statcast, luck_filters, sc_adjustment):
    """
    Compute bounce-back score and regression risk score (both 0-100).
    Also identify the single most important stat driving the projection.

    bounce_back_score: How much upside vs surface stats (high = buy low candidate)
    regression_risk_score: How much downside risk (high = sell high candidate)
    key_indicator: The one stat that matters most for this player's projection

    These scores feed directly into the content creation framework:
    - Bounce-back > 70 → "The surface numbers lied" article candidate
    - Regression risk > 70 → "Why X's 2025 Numbers Lied To You" candidate
    """
    bounce_back = 50  # Neutral starting point
    regression_risk = 50
    indicators = {}  # stat_name -> weight (highest weight = key indicator)

    if not luck_filters and player_statcast.empty:
        return 50, 50, "insufficient_data"

    recent = player_statcast.sort_values('season').iloc[-1] if not player_statcast.empty else {}

    # --- Luck signals (biggest impact on bounce-back score) ---
    xwoba_gap = luck_filters.get('xwoba_minus_woba', 0)
    babip_gap = luck_filters.get('babip_vs_career', 0)

    # xwOBA gap: positive = unlucky = bounce-back candidate
    if xwoba_gap > 0.020:
        bounce_back += min(xwoba_gap * 500, 25)  # Up to +25 points
        indicators['xwoba_gap'] = abs(xwoba_gap) * 500
    elif xwoba_gap < -0.020:
        regression_risk += min(abs(xwoba_gap) * 500, 25)
        indicators['xwoba_gap'] = abs(xwoba_gap) * 500

    # BABIP gap: negative = unlucky = bounce-back candidate
    if babip_gap < -0.020:
        bounce_back += min(abs(babip_gap) * 300, 15)
        indicators['babip_vs_career'] = abs(babip_gap) * 300
    elif babip_gap > 0.020:
        regression_risk += min(babip_gap * 300, 15)
        indicators['babip_vs_career'] = abs(babip_gap) * 300

    # --- Contact quality signals ---
    if pd.notna(recent.get('barrel_rate')):
        barrel = recent['barrel_rate']
        if barrel > 12:  # Elite barrel rate
            bounce_back += 8
            indicators['barrel_rate'] = 8
        elif barrel < 4:  # Poor contact quality
            regression_risk += 8
            indicators['barrel_rate'] = 8

    if pd.notna(recent.get('hard_hit_rate')):
        hh = recent['hard_hit_rate']
        if hh > 45:  # Elite hard hit rate
            bounce_back += 5
        elif hh < 30:  # Weak contact
            regression_risk += 5

    # --- Discipline signals ---
    if pd.notna(recent.get('chase_rate')):
        chase = recent['chase_rate']
        if chase > 35:  # High chase rate = real risk
            regression_risk += 10
            indicators['chase_rate'] = 10
        elif chase < 25:  # Elite discipline
            bounce_back += 5
            indicators['chase_rate'] = 5

    if pd.notna(recent.get('z_contact_rate')):
        z_contact = recent['z_contact_rate']
        if z_contact > 88:  # Elite zone contact
            bounce_back += 5
        elif z_contact < 78:  # Poor contact on strikes
            regression_risk += 5

    # --- Trend signals (if we have multi-year data) ---
    if len(player_statcast) >= 2:
        sorted_sc = player_statcast.sort_values('season')
        recent_pull = sorted_sc.iloc[-1].get('pull_air_rate')
        prev_pull = sorted_sc.iloc[-2].get('pull_air_rate')

        if pd.notna(recent_pull) and pd.notna(prev_pull):
            pull_trend = recent_pull - prev_pull
            if pull_trend > 3:
                bounce_back += 5
                indicators['pull_air_trend'] = 5

    # Incorporate the statcast adjustment direction
    if sc_adjustment > 0.010:
        bounce_back += 10
    elif sc_adjustment < -0.010:
        regression_risk += 10

    # Clamp to 0-100
    bounce_back = max(0, min(100, round(bounce_back)))
    regression_risk = max(0, min(100, round(regression_risk)))

    # Identify key indicator (highest weight)
    if indicators:
        key_indicator = max(indicators, key=indicators.get)
    else:
        key_indicator = "no_statcast_data"

    return bounce_back, regression_risk, key_indicator


# ═══════════════════════════════════════════════════════════════
# MARCEL PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════

def project_pitcher(player_seasons, league_avg_era=4.20, league_avg_fip=4.20):
    """
    Marcel-style projection for a single pitcher.

    Weighting: 2025 × 5, 2024 × 4, 2023 × 3
    Then regress toward league average.
    Then apply aging curve.
    """
    if player_seasons.empty:
        return None

    name = player_seasons.iloc[0]['player_name']
    mlb_id = player_seasons.iloc[0]['mlb_player_id']
    birth_date = player_seasons.iloc[0]['birth_date']
    role = player_seasons.iloc[-1]['role']  # Most recent role

    # Age in 2026
    if pd.notna(birth_date):
        age_2026 = 2026 - pd.to_datetime(birth_date).year
    else:
        age_2026 = 28  # Default assumption

    # Marcel weights
    year_weights = {2025: 5, 2024: 4, 2023: 3}

    total_weight = 0
    weighted_era = 0
    weighted_fip = 0
    weighted_k_pct = 0
    weighted_bb_pct = 0
    weighted_ip = 0
    years_available = 0

    for _, row in player_seasons.iterrows():
        season = row['season']
        w = year_weights.get(season, 0)
        if w == 0:
            continue

        # Weight by BOTH year recency AND innings (more IP = more reliable)
        ip_weight = min(row['ip'] / 150, 1.0)  # Cap at 150 IP
        effective_weight = w * ip_weight

        weighted_era += row['era'] * effective_weight
        weighted_fip += row['fip'] * effective_weight
        weighted_k_pct += row['k_pct'] * effective_weight
        weighted_bb_pct += row['bb_pct'] * effective_weight
        weighted_ip += row['ip'] * w / sum(year_weights.values())  # For PT estimate
        total_weight += effective_weight
        years_available += 1

    if total_weight == 0:
        return None

    # Weighted averages
    proj_era = weighted_era / total_weight
    proj_fip = weighted_fip / total_weight
    proj_k_pct = weighted_k_pct / total_weight
    proj_bb_pct = weighted_bb_pct / total_weight

    # Regression to mean: the less data, the more regression
    # With 3 years of 180+ IP, barely regress. With 1 year of 50 IP, regress a lot.
    total_ip = player_seasons['ip'].sum()
    regression_factor = min(total_ip / 500, 1.0)  # 500 IP = full confidence

    proj_era = proj_era * regression_factor + league_avg_era * (1 - regression_factor)
    proj_fip = proj_fip * regression_factor + league_avg_fip * (1 - regression_factor)

    # Aging adjustment
    era_aging = pitcher_aging_adjustment(age_2026)
    proj_era += era_aging
    proj_fip += era_aging * 0.7  # FIP ages slightly less than ERA

    # Projected IP
    recent_ip = player_seasons[player_seasons['season'] == 2025]['ip'].sum()
    if recent_ip == 0:
        recent_ip = player_seasons['ip'].mean()
    proj_ip = recent_ip * playing_time_adjustment(age_2026, role)

    # Simplified WAR
    league_fip = 4.20
    proj_war = ((league_fip - proj_fip) / 10) * (proj_ip / 9)

    return {
        'mlb_player_id': mlb_id,
        'player_name': name,
        'age_2026': age_2026,
        'role': role,
        'years_of_data': years_available,
        'total_ip_history': round(total_ip, 1),
        'proj_era': round(proj_era, 2),
        'proj_fip': round(proj_fip, 2),
        'proj_k_pct': round(proj_k_pct, 1),
        'proj_bb_pct': round(proj_bb_pct, 1),
        'proj_k_bb_pct': round(proj_k_pct - proj_bb_pct, 1),
        'proj_ip': round(proj_ip, 0),
        'proj_war': round(proj_war, 1),
        'era_aging_adj': round(era_aging, 2),
        'regression_factor': round(regression_factor, 2),
    }


def project_hitter(player_seasons, player_statcast=None, league_avg_woba=0.310):
    """
    Marcel-style projection for a single hitter, now with Statcast overlay.

    Step 1: Compute the base Marcel wOBA (3-year weighted average + regression + aging)
    Step 2: If Statcast data exists, compute luck filters and apply adjustment
    Step 3: Generate bounce-back and regression-risk scores

    The Statcast layer doesn't replace Marcel — it CORRECTS it. A player who hit
    .250 with elite barrel rate and depressed BABIP gets projected higher than
    a .250 hitter with weak contact quality who got lucky on BABIP.
    """
    if player_seasons.empty:
        return None

    name = player_seasons.iloc[0]['player_name']
    mlb_id = player_seasons.iloc[0]['mlb_player_id']
    birth_date = player_seasons.iloc[0]['birth_date']

    if pd.notna(birth_date):
        age_2026 = 2026 - pd.to_datetime(birth_date).year
    else:
        age_2026 = 27

    year_weights = {2025: 5, 2024: 4, 2023: 3}

    total_weight = 0
    weighted_woba = 0
    weighted_ops = 0
    weighted_k_pct = 0
    weighted_bb_pct = 0
    years_available = 0

    for _, row in player_seasons.iterrows():
        season = row['season']
        w = year_weights.get(season, 0)
        if w == 0:
            continue
        pa_weight = min(row['pa'] / 550, 1.0)
        effective_weight = w * pa_weight

        weighted_woba += row['woba'] * effective_weight
        weighted_ops += row['ops'] * effective_weight
        weighted_k_pct += row['k_pct'] * effective_weight
        weighted_bb_pct += row['bb_pct'] * effective_weight
        total_weight += effective_weight
        years_available += 1

    if total_weight == 0:
        return None

    proj_woba = weighted_woba / total_weight
    proj_ops = weighted_ops / total_weight
    proj_k_pct = weighted_k_pct / total_weight
    proj_bb_pct = weighted_bb_pct / total_weight

    # Regression to mean
    total_pa = player_seasons['pa'].sum()
    regression_factor = min(total_pa / 1500, 1.0)
    proj_woba = proj_woba * regression_factor + league_avg_woba * (1 - regression_factor)

    # Aging adjustment
    woba_aging = hitter_aging_adjustment(age_2026)
    proj_woba += woba_aging

    # This is the pure Marcel projection BEFORE Statcast
    marcel_woba = round(proj_woba, 3)

    # ── STATCAST OVERLAY ──────────────────────────────────
    # Layer Statcast on top of Marcel as regression signals.
    # If no Statcast data, these default to neutral values.
    sc_adj = 0.0
    bounce_back = 50
    regression_risk = 50
    key_indicator = "no_statcast_data"
    luck = {}

    if player_statcast is not None and not player_statcast.empty:
        luck = compute_luck_filters(player_statcast)
        sc_adj = statcast_adjustment(marcel_woba, player_statcast, luck)
        bounce_back, regression_risk, key_indicator = compute_scores(
            player_statcast, luck, sc_adj
        )

    # Apply Statcast adjustment to get final projected wOBA
    statcast_adjusted_woba = round(proj_woba + sc_adj, 3)

    # Projected PA
    recent_pa = player_seasons[player_seasons['season'] == 2025]['pa'].sum()
    if recent_pa == 0:
        recent_pa = player_seasons['pa'].mean()
    proj_pa = recent_pa * playing_time_adjustment(age_2026, 'hitter')

    # WAR (using Statcast-adjusted wOBA for the final number)
    woba_scale = 1.15
    runs_per_win = 10
    batting_runs = ((statcast_adjusted_woba - league_avg_woba) / woba_scale) * proj_pa
    proj_war = batting_runs / runs_per_win

    return {
        'mlb_player_id': mlb_id,
        'player_name': name,
        'age_2026': age_2026,
        'years_of_data': years_available,
        'total_pa_history': round(total_pa, 0),
        'marcel_woba': marcel_woba,
        'statcast_adjusted_woba': statcast_adjusted_woba,
        'statcast_adj': round(sc_adj, 3),
        'bounce_back_score': bounce_back,
        'regression_risk_score': regression_risk,
        'key_indicator': key_indicator,
        'proj_ops': round(proj_ops, 3),
        'proj_k_pct': round(proj_k_pct, 1),
        'proj_bb_pct': round(proj_bb_pct, 1),
        'proj_pa': round(proj_pa, 0),
        'proj_war': round(proj_war, 1),
        'woba_aging_adj': round(woba_aging, 3),
        'regression_factor': round(regression_factor, 2),
    }


# ═══════════════════════════════════════════════════════════════
# RUN ALL PROJECTIONS
# ═══════════════════════════════════════════════════════════════

def run_projections(team_filter=None, player_filter=None):
    """
    Project all players and aggregate by current 2026 team.
    Now loads Statcast data and passes it to hitter projections
    for luck-adjusted wOBA, bounce-back scores, and regression risk.
    """

    print("=" * 70)
    print("2026 PLAYER PROJECTIONS (Marcel + Statcast Regression Signals)")
    print("=" * 70)

    # Load multi-year data
    print("\nLoading 2023-2025 stats from database...")
    pitchers_raw = load_pitcher_multi_year()
    hitters_raw = load_hitter_multi_year()
    print(f"  Pitcher-seasons: {len(pitchers_raw)}")
    print(f"  Hitter-seasons:  {len(hitters_raw)}")

    # Load Statcast data (new in Phase 2)
    print("\nLoading Statcast metrics...")
    statcast_raw = load_statcast_multi_year()

    # Get current rosters
    print("\nFetching current 2026 rosters...")
    teams_data = api_get("teams", {"sportId": 1, "season": 2026})
    roster_map = {}  # mlb_player_id -> current_team

    for t in teams_data.get("teams", []):
        roster = api_get(f"teams/{t['id']}/roster", {"rosterType": "40Man", "season": 2026})
        for p in roster.get("roster", []):
            pid = p.get("person", {}).get("id")
            if pid:
                roster_map[pid] = t["name"]
        time.sleep(0.2)

    print(f"  Players on 2026 rosters: {len(roster_map)}")

    # Project pitchers (unchanged — Statcast hitter-only for now)
    print("\nProjecting pitchers...")
    pitcher_projections = []
    for pid, group in pitchers_raw.groupby('mlb_player_id'):
        proj = project_pitcher(group)
        if proj:
            proj['current_team'] = roster_map.get(pid, 'Free Agent')
            pitcher_projections.append(proj)

    p_df = pd.DataFrame(pitcher_projections)
    print(f"  Projected {len(p_df)} pitchers")

    # Project hitters WITH Statcast overlay
    print("Projecting hitters (with Statcast adjustment)...")
    hitter_projections = []
    statcast_hits = 0

    for pid, group in hitters_raw.groupby('mlb_player_id'):
        # Get this player's Statcast data across all seasons
        player_sc = statcast_raw[statcast_raw['mlb_player_id'] == pid]

        proj = project_hitter(group, player_statcast=player_sc)
        if proj:
            proj['current_team'] = roster_map.get(pid, 'Free Agent')
            hitter_projections.append(proj)
            if not player_sc.empty:
                statcast_hits += 1

    h_df = pd.DataFrame(hitter_projections)
    print(f"  Projected {len(h_df)} hitters ({statcast_hits} with Statcast data)")

    # Filter if requested — show individual player projection
    if player_filter:
        p_df_show = p_df[p_df['player_name'].str.contains(player_filter, case=False)]
        h_df_show = h_df[h_df['player_name'].str.contains(player_filter, case=False)]

        if not p_df_show.empty:
            print(f"\n{'=' * 70}")
            print(f"PITCHER PROJECTION: {player_filter}")
            print(f"{'=' * 70}")
            for _, r in p_df_show.iterrows():
                print(f"\n  {r['player_name']} (Age {r['age_2026']}, {r['role']})")
                print(f"  Current team: {r['current_team']}")
                print(f"  Data: {r['years_of_data']} years, {r['total_ip_history']} career IP")
                print(f"  Projected: {r['proj_era']:.2f} ERA | {r['proj_fip']:.2f} FIP | "
                      f"{r['proj_k_bb_pct']:.1f} K-BB%")
                print(f"  Projected: {r['proj_ip']:.0f} IP | {r['proj_war']:.1f} WAR")
                print(f"  Aging adj: {r['era_aging_adj']:+.2f} ERA | "
                      f"Regression: {r['regression_factor']:.0%} confidence")

        if not h_df_show.empty:
            print(f"\n{'=' * 70}")
            print(f"HITTER PROJECTION: {player_filter}")
            print(f"{'=' * 70}")
            for _, r in h_df_show.iterrows():
                print(f"\n  {r['player_name']} (Age {r['age_2026']})")
                print(f"  Current team: {r['current_team']}")
                print(f"  Data: {r['years_of_data']} years, {r['total_pa_history']:.0f} career PA")
                # Marcel vs Statcast-adjusted wOBA
                print(f"  Marcel wOBA:    {r['marcel_woba']:.3f}")
                print(f"  Statcast wOBA:  {r['statcast_adjusted_woba']:.3f} "
                      f"({r['statcast_adj']:+.003f} adjustment)")
                print(f"  Projected: {r['proj_ops']:.3f} OPS | "
                      f"{r['proj_pa']:.0f} PA | {r['proj_war']:.1f} WAR")
                # Statcast scores
                print(f"  Bounce-back:    {r['bounce_back_score']}/100")
                print(f"  Regression risk:{r['regression_risk_score']}/100")
                print(f"  Key indicator:  {r['key_indicator']}")
                print(f"  Aging adj: {r['woba_aging_adj']:+.003f} wOBA | "
                      f"Regression: {r['regression_factor']:.0%} confidence")
        return p_df, h_df

    # Team aggregation
    print(f"\n{'=' * 70}")
    print("2026 PROJECTED TEAM WAR (Marcel + Statcast)")
    print(f"{'=' * 70}")

    team_results = []
    all_teams = sorted(set(p_df['current_team'].unique()) | set(h_df['current_team'].unique()))

    if team_filter:
        all_teams = [t for t in all_teams if team_filter.lower() in t.lower()]

    for team in all_teams:
        if team == 'Free Agent':
            continue

        tp = p_df[p_df['current_team'] == team]
        th = h_df[h_df['current_team'] == team]

        p_war = tp['proj_war'].sum()
        h_war = th['proj_war'].sum()
        total_war = p_war + h_war

        # Top 5 SP by projected WAR
        sp = tp[tp['role'] == 'SP'].nlargest(5, 'proj_war')
        rp = tp[tp['role'] == 'RP'].nlargest(5, 'proj_war')

        # Lineup = top 9 hitters
        lineup = th.nlargest(9, 'proj_war')

        proj_wins = 48 + total_war  # 48 = replacement level

        # Average bounce-back score for team's hitters (new metric)
        avg_bb = th['bounce_back_score'].mean() if not th.empty else 50

        team_results.append({
            'team': team,
            'proj_wins': round(proj_wins, 0),
            'total_war': round(total_war, 1),
            'pitch_war': round(p_war, 1),
            'hit_war': round(h_war, 1),
            'avg_bounce_back': round(avg_bb, 0),
        })

        print(f"\n  {team} -- Projected {proj_wins:.0f}W (WAR: {total_war:.1f})")
        print(f"    Pitching: {p_war:.1f} WAR | Hitting: {h_war:.1f} WAR "
              f"| Avg bounce-back: {avg_bb:.0f}/100")

        if not sp.empty:
            sp_str = " | ".join(
                f"{r['player_name']} ({r['proj_era']:.2f} ERA, {r['proj_war']:.1f}W, age {r['age_2026']})"
                for _, r in sp.iterrows()
            )
            print(f"    Rotation: {sp_str}")

        if not rp.empty:
            rp_str = " | ".join(
                f"{r['player_name']} ({r['proj_war']:.1f}W)"
                for _, r in rp.head(3).iterrows()
            )
            print(f"    Bullpen:  {rp_str}")

        if not lineup.empty:
            top3 = " | ".join(
                f"{r['player_name']} ({r['statcast_adjusted_woba']:.3f} wOBA, "
                f"{r['proj_war']:.1f}W, BB:{r['bounce_back_score']})"
                for _, r in lineup.head(3).iterrows()
            )
            print(f"    Lineup:   {top3}")

    # Save
    t_df = pd.DataFrame(team_results).sort_values('proj_wins', ascending=False)
    os.makedirs('data/features', exist_ok=True)
    t_df.to_csv('data/features/team_projected_war_2026.csv', index=False)
    p_df.to_csv('data/features/pitcher_projections_2026.csv', index=False)
    h_df.to_csv('data/features/hitter_projections_2026.csv', index=False)
    print(f"\nSaved: data/features/team_projected_war_2026.csv")
    print(f"Saved: data/features/pitcher_projections_2026.csv")
    print(f"Saved: data/features/hitter_projections_2026.csv")

    return p_df, h_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", type=str, help="Filter to one team")
    parser.add_argument("--player", type=str, help="Show one player's projection")
    args = parser.parse_args()

    run_projections(team_filter=args.team, player_filter=args.player)