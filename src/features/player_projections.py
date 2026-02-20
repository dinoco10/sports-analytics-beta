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
from config.settings import (
    POSITION_ADJUSTMENT_RUNS, LEAGUE_AVG_WOBA, LEAGUE_AVG_WOBA_SCALE,
    LEAGUE_AVG_R_PA, RUNS_PER_WIN, REPLACEMENT_LEVEL_WINS,
)
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
    """Load hitter stats for 2023-2025 from database, including position."""
    query = """
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
        WHERE strftime('%Y', hs.date) IN ('2023', '2024', '2025')
        GROUP BY p.mlb_id, p.name, p.birth_date, p.primary_position,
                 strftime('%Y', hs.date), t.name
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
# PITCHER STATCAST DATA LOADING + REGRESSION SIGNALS
# ═══════════════════════════════════════════════════════════════

def load_pitcher_statcast_multi_year():
    """
    Load pitcher Statcast metrics for 2023-2025 from pitcher_statcast_metrics table.
    Returns one row per pitcher per season, keyed on mlb_player_id.

    This gives us the "underlying stuff quality" that traditional ERA hides:
    - Expected stats (xERA, xwOBA against) strip BABIP/HR luck
    - Contact suppression (barrel rate, hard hit %) shows true miss quality
    - Swing & miss (whiff%, K-BB%, chase induced) are the stickiest signals
    - Batted ball profile (pull_air_rate against) predicts future HR allowed
    """
    query = """
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
        WHERE pm.season IN (2023, 2024, 2025)
    """
    df = pd.read_sql(query, engine)
    print(f"  Pitcher Statcast records loaded: {len(df)}")
    return df


def load_pitcher_pitch_metrics():
    """
    Load per-pitch-type metrics for 2023-2025 from pitcher_pitch_metrics table.
    Returns one row per pitcher per season per pitch type.

    Key uses:
    - Identify the "best pitch" (lowest run_value_per_100)
    - Detect fastball velocity decline (regression signal)
    - Find new plus secondary pitches (breakout catalyst)
    - Measure arsenal diversity (one-pitch pitchers are riskier)
    """
    query = """
        SELECT
            p.mlb_id as mlb_player_id,
            pp.season, pp.pitch_type, pp.pitch_name,
            pp.run_value_per_100, pp.run_value, pp.pitches_thrown,
            pp.usage_pct, pp.whiff_rate, pp.chase_rate,
            pp.ba_against, pp.slg_against, pp.woba_against,
            pp.hard_hit_pct, pp.avg_speed, pp.xwoba_against
        FROM pitcher_pitch_metrics pp
        JOIN players p ON pp.player_id = p.id
        WHERE pp.season IN (2023, 2024, 2025)
    """
    df = pd.read_sql(query, engine)
    print(f"  Pitch-level metrics loaded: {len(df)} (pitch-type-seasons)")
    return df


def compute_pitcher_luck_filters(pitcher_statcast):
    """
    Calculate pitcher "luck filters" — gaps between expected and actual.
    These tell us if a pitcher's ERA was real or driven by BABIP/HR luck.

    Key signals:
    - era_minus_xera: positive = pitcher was UNLUCKY (ERA > xERA, expect improvement)
    - babip_vs_career: positive = bad luck on balls in play (expect regression down)
    - hr_per_fb_vs_norm: above ~12% = HR luck catching up, expect regression
    - xwoba_vs_woba: if xwOBA_against < actual wOBA_against, pitching better than stats show
    """
    if pitcher_statcast.empty:
        return {}

    career_babip = pitcher_statcast['babip_against'].mean()
    career_hr_fb = pitcher_statcast['hr_per_fb'].mean()
    league_hr_fb = 0.12  # League average HR/FB rate

    recent = pitcher_statcast.sort_values('season').iloc[-1]
    result = {}

    # ERA vs xERA gap — the single best pitcher luck indicator
    # Positive = unlucky (ERA higher than deserved), negative = overperforming
    if pd.notna(recent.get('era')) and pd.notna(recent.get('xera')):
        result['era_minus_xera'] = round(recent['era'] - recent['xera'], 2)

    # BABIP against vs career — above career avg = bad luck, expect improvement
    if pd.notna(recent.get('babip_against')) and pd.notna(career_babip):
        result['babip_vs_career'] = round(recent['babip_against'] - career_babip, 3)

    # HR/FB vs league norm — if above 13%, expect regression DOWN (fewer HR)
    # If below 10%, expect regression UP (more HR coming)
    if pd.notna(recent.get('hr_per_fb')):
        result['hr_per_fb_vs_norm'] = round(recent['hr_per_fb'] - league_hr_fb, 3)
        result['hr_per_fb_vs_career'] = round(recent['hr_per_fb'] - career_hr_fb, 3) if pd.notna(career_hr_fb) else 0

    # xwOBA against vs actual wOBA against
    if pd.notna(recent.get('xwoba_against')) and pd.notna(recent.get('woba_against')):
        result['xwoba_vs_woba_against'] = round(
            recent['xwoba_against'] - recent['woba_against'], 3
        )

    return result


def pitcher_statcast_adjustment(marcel_era, pitcher_statcast, luck_filters,
                                pitch_metrics=None):
    """
    Apply Statcast regression signals to adjust the Marcel ERA projection.

    This mirrors the hitter statcast_adjustment() logic, but for pitchers.
    Marcel treats every 3.50 ERA pitcher the same. Statcast tells us WHICH 3.50 ERA
    pitchers actually had elite stuff (sustainable) vs which benefited from
    lucky BABIP and low HR/FB (unsustainable).

    Adjustment rules (each capped to prevent extreme swings):
    1. ERA vs xERA + BABIP luck → ERA adjustment (max ±0.40)
    2. Contact suppression quality (barrel rate, hard hit) → trust/discount (max ±0.20)
    3. Pull air rate against trend → HR risk signal (max ±0.15)
    4. Swing & miss sustainability (K-BB%, whiff%, swstr%) → floor/ceiling signal
    5. Fastball velocity trend → aging/breakout signal (max ±0.15)

    Returns: ERA adjustment value (negative = better than surface, positive = worse)
    """
    if pitcher_statcast.empty:
        return 0.0

    adjustment = 0.0
    recent = pitcher_statcast.sort_values('season').iloc[-1]

    # --- RULE 1: ERA/xERA + BABIP luck correction ---
    # If ERA >> xERA AND BABIP above career average, the pitcher was unlucky.
    # Regress ERA downward (negative adjustment = improvement).
    era_gap = luck_filters.get('era_minus_xera', 0)
    babip_gap = luck_filters.get('babip_vs_career', 0)

    if era_gap > 0.30 and babip_gap > 0.010:
        # ERA inflated by BABIP luck — project improvement
        # Weight: 40% of the ERA gap, capped at -0.40 (don't over-correct)
        luck_adj = max(-era_gap * 0.4, -0.40)
        adjustment += luck_adj
    elif era_gap < -0.30 and babip_gap < -0.010:
        # ERA deflated by lucky BABIP — project regression upward
        luck_adj = min(abs(era_gap) * 0.35, 0.35)
        adjustment += luck_adj

    # --- RULE 2: Contact suppression quality ---
    # If barrel rate and hard hit rate are low, the pitching is real.
    # If they're high despite good ERA, the ERA is borrowed time.
    if pd.notna(recent.get('barrel_rate_against')):
        barrel = recent['barrel_rate_against']
        if barrel < 5.5:
            # Elite contact suppression — trust the ERA, slight improvement
            adjustment -= 0.10
        elif barrel > 9.0:
            # Hitters barreling this pitcher — ERA will catch up
            adjustment += 0.15

    if pd.notna(recent.get('hard_hit_rate_against')):
        hh = recent['hard_hit_rate_against']
        if hh < 32:
            adjustment -= 0.05
        elif hh > 42:
            adjustment += 0.10

    # --- RULE 3: Pull air rate against trend ---
    # Pull air rate against is the most underrated pitcher stat.
    # 66% of HRs come from pulled fly balls. If a pitcher's pull_air_rate_against
    # is trending up, home run trouble is coming even if HR/FB looks fine now.
    if len(pitcher_statcast) >= 2:
        sorted_ps = pitcher_statcast.sort_values('season')
        recent_pull = sorted_ps.iloc[-1].get('pull_air_rate_against')
        prev_pull = sorted_ps.iloc[-2].get('pull_air_rate_against')

        if pd.notna(recent_pull) and pd.notna(prev_pull):
            pull_trend = recent_pull - prev_pull
            if pull_trend > 2.0:
                # More pulled fly balls allowed = HR trouble coming
                adjustment += min(pull_trend * 0.04, 0.15)
            elif pull_trend < -2.0:
                # Fewer pulled fly balls = fewer HR coming
                adjustment -= min(abs(pull_trend) * 0.03, 0.10)

    # --- RULE 4: Swing & miss sustainability ---
    # K-BB% is the single most predictive pitcher stat year-over-year.
    # < 10% = heavy BABIP dependence (risky). > 20% = elite, trust the ERA.
    k_bb = recent.get('k_minus_bb')
    if pd.notna(k_bb):
        if k_bb > 20:
            # Elite swing & miss — ERA has a high floor
            adjustment -= 0.10
        elif k_bb < 8:
            # Very low strikeout-walk differential — BABIP dependent
            adjustment += 0.15

    # SwStr% as additional check (<8.6% = bottom 5 in the league)
    swstr = recent.get('swstr_rate')
    if pd.notna(swstr):
        if swstr < 8.6:
            adjustment += 0.08
        elif swstr > 13.0:
            adjustment -= 0.05

    # --- RULE 5: Fastball velocity trend ---
    # Declining FB velocity is the strongest aging signal for pitchers.
    # If velo drops 1+ mph year-over-year, expect decline. If it ticks up, breakout signal.
    if pitch_metrics is not None and not pitch_metrics.empty:
        fb_data = pitch_metrics[pitch_metrics['pitch_type'].isin(['FF', 'SI'])]
        if len(fb_data) >= 2:
            fb_sorted = fb_data.sort_values('season')
            recent_velo = fb_sorted.iloc[-1].get('avg_speed')
            prev_velo = fb_sorted.iloc[-2].get('avg_speed')

            if pd.notna(recent_velo) and pd.notna(prev_velo):
                velo_change = recent_velo - prev_velo
                if velo_change < -1.0:
                    # Losing velocity — age/injury regression
                    adjustment += min(abs(velo_change) * 0.10, 0.15)
                elif velo_change > 0.8:
                    # Gaining velocity — breakout/health signal
                    adjustment -= min(velo_change * 0.08, 0.10)

    # Final cap: never adjust more than ±0.60 ERA
    adjustment = max(-0.60, min(0.60, adjustment))
    return round(adjustment, 2)


def compute_pitcher_scores(pitcher_statcast, luck_filters, era_adj,
                           pitch_metrics=None):
    """
    Compute three pitcher scores (all 0-100) plus flag indicators:

    sustainability_score: How sustainable is the current ERA? (high = safe bet)
      - Driven by K-BB%, contact suppression, batted ball quality, arsenal diversity
      - 80+ means the pitcher's stuff backs up the stats

    regression_risk_score: How much ERA inflation risk? (high = sell candidate)
      - Driven by ERA < xERA, BABIP luck, low SwStr%, HR/FB below norm
      - 70+ means the surface ERA probably isn't real

    breakout_score: How much upside beyond surface stats? (high = buy candidate)
      - Driven by xERA < ERA, improving whiff/chase, declining pull_air_rate,
        new plus pitch emerging
      - 70+ means this pitcher might be significantly better than ERA shows

    Also identifies:
    - primary_risk_flag: Single biggest risk factor (e.g., "low_k_bb_pct")
    - primary_upside_flag: Single biggest upside factor (e.g., "elite_whiff_rate")
    - key_pitch: The pitcher's best pitch by run value
    """
    sustainability = 50
    regression_risk = 50
    breakout = 50
    risk_factors = {}    # name -> weight
    upside_factors = {}  # name -> weight
    key_pitch_name = "unknown"

    if pitcher_statcast.empty and not luck_filters:
        return 50, 50, 50, "insufficient_data", "insufficient_data", "unknown"

    recent = pitcher_statcast.sort_values('season').iloc[-1] if not pitcher_statcast.empty else {}

    # ── SUSTAINABILITY SIGNALS ──────────────────────────────
    # K-BB% is king — the most stable pitcher metric year-over-year
    k_bb = recent.get('k_minus_bb')
    if pd.notna(k_bb):
        if k_bb > 20:
            sustainability += 20
            upside_factors['elite_k_bb_pct'] = 20
        elif k_bb > 15:
            sustainability += 10
        elif k_bb < 10:
            sustainability -= 10
            risk_factors['low_k_bb_pct'] = 10
        elif k_bb < 5:
            sustainability -= 20
            risk_factors['very_low_k_bb_pct'] = 20

    # Barrel rate against — elite contact suppression = sustainable
    barrel = recent.get('barrel_rate_against')
    if pd.notna(barrel):
        if barrel < 5.5:
            sustainability += 12
            upside_factors['elite_barrel_suppression'] = 12
        elif barrel < 7.0:
            sustainability += 5
        elif barrel > 9.0:
            sustainability -= 10
            risk_factors['high_barrel_rate'] = 10

    # Ground ball rate — GB pitchers sustain better (fewer HR)
    gb = recent.get('gb_rate')
    if pd.notna(gb):
        if gb > 50:
            sustainability += 8
        elif gb < 38:
            sustainability -= 5

    # Chase rate induced — ability to get swings outside zone is sticky
    chase = recent.get('chase_rate_induced')
    if pd.notna(chase):
        if chase > 32:
            sustainability += 8
            upside_factors['elite_chase_induced'] = 8
        elif chase < 25:
            sustainability -= 5

    # ── REGRESSION RISK SIGNALS ─────────────────────────────
    # ERA outperforming xERA — the #1 regression signal
    era_gap = luck_filters.get('era_minus_xera', 0)
    if era_gap < -0.50:
        # ERA well below xERA — very likely to regress upward
        regression_risk += min(abs(era_gap) * 15, 25)
        risk_factors['era_below_xera'] = min(abs(era_gap) * 15, 25)
    elif era_gap < -0.25:
        regression_risk += 10
        risk_factors['era_below_xera'] = 10

    # BABIP against below career (luck-driven ERA suppression)
    babip_gap = luck_filters.get('babip_vs_career', 0)
    if babip_gap < -0.020:
        regression_risk += min(abs(babip_gap) * 400, 15)
        risk_factors['low_babip_luck'] = min(abs(babip_gap) * 400, 15)

    # HR/FB below league norm — HR luck will catch up
    hr_fb_gap = luck_filters.get('hr_per_fb_vs_norm', 0)
    if hr_fb_gap < -0.03:
        regression_risk += min(abs(hr_fb_gap) * 300, 15)
        risk_factors['low_hr_fb_luck'] = min(abs(hr_fb_gap) * 300, 15)

    # Low SwStr% — can't miss bats, dependent on defense/sequencing
    swstr = recent.get('swstr_rate')
    if pd.notna(swstr):
        if swstr < 8.6:
            regression_risk += 12
            risk_factors['low_swstr'] = 12
        elif swstr < 10.0:
            regression_risk += 5

    # High fly ball rate — HR-prone profile
    fb = recent.get('fb_rate')
    if pd.notna(fb):
        if fb > 39:
            regression_risk += 8
            risk_factors['high_fb_rate'] = 8

    # ── BREAKOUT SIGNALS ────────────────────────────────────
    # xERA better than ERA — stuff is better than results show
    if era_gap > 0.50:
        breakout += min(era_gap * 15, 25)
        upside_factors['xera_better_than_era'] = min(era_gap * 15, 25)
    elif era_gap > 0.25:
        breakout += 10
        upside_factors['xera_better_than_era'] = 10

    # BABIP against above career — unlucky, expect improvement
    if babip_gap > 0.020:
        breakout += min(babip_gap * 400, 12)
        upside_factors['high_babip_unlucky'] = min(babip_gap * 400, 12)

    # Whiff rate trending up — developing miss ability
    if len(pitcher_statcast) >= 2:
        sorted_ps = pitcher_statcast.sort_values('season')
        recent_whiff = sorted_ps.iloc[-1].get('whiff_rate')
        prev_whiff = sorted_ps.iloc[-2].get('whiff_rate')

        if pd.notna(recent_whiff) and pd.notna(prev_whiff):
            whiff_trend = recent_whiff - prev_whiff
            if whiff_trend > 2.0:
                breakout += min(whiff_trend * 3, 12)
                upside_factors['improving_whiff'] = min(whiff_trend * 3, 12)

        # Chase rate improving — getting better at inducing chases
        recent_chase = sorted_ps.iloc[-1].get('chase_rate_induced')
        prev_chase = sorted_ps.iloc[-2].get('chase_rate_induced')

        if pd.notna(recent_chase) and pd.notna(prev_chase):
            chase_trend = recent_chase - prev_chase
            if chase_trend > 2.0:
                breakout += min(chase_trend * 2, 8)

        # Pull air rate against declining — fewer HR coming
        recent_pull = sorted_ps.iloc[-1].get('pull_air_rate_against')
        prev_pull = sorted_ps.iloc[-2].get('pull_air_rate_against')

        if pd.notna(recent_pull) and pd.notna(prev_pull):
            pull_trend = recent_pull - prev_pull
            if pull_trend < -2.0:
                breakout += min(abs(pull_trend) * 2, 10)
                upside_factors['declining_pull_air'] = min(abs(pull_trend) * 2, 10)

    # ── PITCH-LEVEL SIGNALS ─────────────────────────────────
    if pitch_metrics is not None and not pitch_metrics.empty:
        recent_pitches = pitch_metrics[
            pitch_metrics['season'] == pitch_metrics['season'].max()
        ]

        if not recent_pitches.empty:
            # Find best pitch by run value per 100
            best = recent_pitches.loc[recent_pitches['run_value_per_100'].idxmin()]
            key_pitch_name = best.get('pitch_name', best.get('pitch_type', 'unknown'))

            # A truly plus pitch (run_value_per_100 < -1.0) boosts sustainability
            if best['run_value_per_100'] < -1.5:
                sustainability += 8
                upside_factors['plus_pitch'] = 8
            elif best['run_value_per_100'] < -1.0:
                sustainability += 4

            # Arsenal diversity — count pitches with >10% usage
            meaningful_pitches = recent_pitches[recent_pitches['usage_pct'] > 10]
            n_pitches = len(meaningful_pitches)
            if n_pitches >= 4:
                sustainability += 5  # Deep arsenal is harder to game-plan
            elif n_pitches <= 2:
                regression_risk += 8
                risk_factors['limited_arsenal'] = 8

            # New plus secondary — pitch not thrown much last year but elite whiff
            if len(pitch_metrics) > len(recent_pitches):
                prev_pitches = pitch_metrics[
                    pitch_metrics['season'] == pitch_metrics['season'].max() - 1
                ]
                for _, p in recent_pitches.iterrows():
                    prev_match = prev_pitches[prev_pitches['pitch_type'] == p['pitch_type']]
                    if (prev_match.empty or prev_match.iloc[0]['usage_pct'] < 5) and \
                       pd.notna(p.get('whiff_rate')) and p['whiff_rate'] > 30 and \
                       p['usage_pct'] > 10:
                        breakout += 10
                        upside_factors['new_plus_pitch'] = 10
                        break

            # Fastball velocity decline (regression signal for aging pitchers)
            fb_data = pitch_metrics[pitch_metrics['pitch_type'].isin(['FF', 'SI'])]
            if len(fb_data) >= 2:
                fb_sorted = fb_data.sort_values('season')
                recent_velo = fb_sorted.iloc[-1].get('avg_speed')
                prev_velo = fb_sorted.iloc[-2].get('avg_speed')
                if pd.notna(recent_velo) and pd.notna(prev_velo):
                    velo_change = recent_velo - prev_velo
                    if velo_change < -1.0:
                        regression_risk += 10
                        risk_factors['declining_velocity'] = 10
                    elif velo_change > 0.8:
                        breakout += 5
                        upside_factors['velocity_gain'] = 5

    # Incorporate the ERA adjustment direction
    if era_adj < -0.20:
        breakout += 8
    elif era_adj > 0.20:
        regression_risk += 8

    # Clamp all scores to 0-100
    sustainability = max(0, min(100, round(sustainability)))
    regression_risk = max(0, min(100, round(regression_risk)))
    breakout = max(0, min(100, round(breakout)))

    # Identify primary flags
    primary_risk = max(risk_factors, key=risk_factors.get) if risk_factors else "none"
    primary_upside = max(upside_factors, key=upside_factors.get) if upside_factors else "none"

    return sustainability, regression_risk, breakout, primary_risk, primary_upside, key_pitch_name


# ═══════════════════════════════════════════════════════════════
# MARCEL PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════

def project_pitcher(player_seasons, pitcher_statcast=None, pitch_metrics=None,
                    league_avg_era=4.20, league_avg_fip=4.20):
    """
    Marcel-style projection for a single pitcher, now with Statcast overlay.

    Step 1: Compute the base Marcel ERA/FIP (3-year weighted average + regression + aging)
    Step 2: If Statcast data exists, compute luck filters and apply ERA adjustment
    Step 3: Generate sustainability, regression risk, and breakout scores
    Step 4: Identify key pitch and risk/upside flags

    The Statcast layer doesn't replace Marcel — it CORRECTS it. A pitcher with 3.20 ERA
    but high BABIP luck and low K-BB% gets projected worse than a 3.20 ERA pitcher
    with elite barrel suppression and 20% K-BB%.
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

    # This is the pure Marcel ERA BEFORE Statcast
    marcel_era = round(proj_era, 2)

    # ── PITCHER STATCAST OVERLAY ──────────────────────────────
    # Layer Statcast on top of Marcel as regression signals.
    # If no Statcast data, these default to neutral values.
    sc_era_adj = 0.0
    sustainability = 50
    regression_risk = 50
    breakout = 50
    primary_risk = "no_statcast_data"
    primary_upside = "no_statcast_data"
    key_pitch = "unknown"
    luck = {}

    if pitcher_statcast is not None and not pitcher_statcast.empty:
        luck = compute_pitcher_luck_filters(pitcher_statcast)
        sc_era_adj = pitcher_statcast_adjustment(
            marcel_era, pitcher_statcast, luck, pitch_metrics
        )
        sustainability, regression_risk, breakout, primary_risk, primary_upside, key_pitch = \
            compute_pitcher_scores(pitcher_statcast, luck, sc_era_adj, pitch_metrics)

    # Apply Statcast adjustment to get final projected ERA
    statcast_adjusted_era = round(proj_era + sc_era_adj, 2)

    # Also adjust FIP proportionally (same direction, slightly less magnitude)
    proj_fip += sc_era_adj * 0.6

    # Projected IP
    recent_ip = player_seasons[player_seasons['season'] == 2025]['ip'].sum()
    if recent_ip == 0:
        recent_ip = player_seasons['ip'].mean()
    proj_ip = recent_ip * playing_time_adjustment(age_2026, role)

    # Pitcher WAR: based on FIP vs league average, scaled by innings
    league_fip = 4.20
    proj_war = ((league_fip - proj_fip) / RUNS_PER_WIN) * (proj_ip / 9)

    return {
        'mlb_player_id': mlb_id,
        'player_name': name,
        'age_2026': age_2026,
        'role': role,
        'years_of_data': years_available,
        'total_ip_history': round(total_ip, 1),
        'marcel_era': marcel_era,
        'statcast_adjusted_era': statcast_adjusted_era,
        'statcast_era_adj': round(sc_era_adj, 2),
        'proj_era': round(statcast_adjusted_era, 2),
        'proj_fip': round(proj_fip, 2),
        'proj_k_pct': round(proj_k_pct, 1),
        'proj_bb_pct': round(proj_bb_pct, 1),
        'proj_k_bb_pct': round(proj_k_pct - proj_bb_pct, 1),
        'proj_ip': round(proj_ip, 0),
        'proj_war': round(proj_war, 1),
        'sustainability_score': sustainability,
        'regression_risk_score': regression_risk,
        'breakout_score': breakout,
        'primary_risk_flag': primary_risk,
        'primary_upside_flag': primary_upside,
        'key_pitch': key_pitch,
        'era_aging_adj': round(era_aging, 2),
        'regression_factor': round(regression_factor, 2),
    }


def project_hitter(player_seasons, player_statcast=None, league_avg_woba=None):
    """
    Marcel-style projection for a single hitter, now with Statcast overlay + fWAR.

    Step 1: Compute the base Marcel wOBA (3-year weighted average + regression + aging)
    Step 2: If Statcast data exists, compute luck filters and apply adjustment
    Step 3: Generate bounce-back and regression-risk scores
    Step 4: Compute fWAR with position adjustment and wRC+

    The Statcast layer doesn't replace Marcel — it CORRECTS it. A player who hit
    .250 with elite barrel rate and depressed BABIP gets projected higher than
    a .250 hitter with weak contact quality who got lucky on BABIP.
    """
    if league_avg_woba is None:
        league_avg_woba = LEAGUE_AVG_WOBA

    if player_seasons.empty:
        return None

    name = player_seasons.iloc[0]['player_name']
    mlb_id = player_seasons.iloc[0]['mlb_player_id']
    birth_date = player_seasons.iloc[0]['birth_date']
    position = player_seasons.iloc[-1].get('primary_position', 'DH')

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

    # ── wRC+ CALCULATION ──────────────────────────────────
    # wRC+ = how many runs a hitter creates relative to league average.
    # 100 = average, 150 = 50% better than average, etc.
    # Formula: ((wOBA - lgWOBA) / wOBA_scale + lgR/PA) / lgR/PA * 100
    wrc_plus = (
        (statcast_adjusted_woba - league_avg_woba) / LEAGUE_AVG_WOBA_SCALE
        + LEAGUE_AVG_R_PA
    ) / LEAGUE_AVG_R_PA * 100

    # ── fWAR CALCULATION (with position adjustment) ──────
    # fWAR adds defensive value via position adjustment.
    # A catcher with .300 wOBA is more valuable than a DH with .300 wOBA
    # because catching is harder to fill at replacement level.
    #
    # Components:
    #   1. Batting runs = (wOBA - lgWOBA) / wOBA_scale * PA
    #   2. Position adjustment = POSITION_ADJUSTMENT_RUNS * (PA / 650)
    #   3. fWAR = (batting_runs + position_adj) / RUNS_PER_WIN

    batting_runs = ((statcast_adjusted_woba - league_avg_woba) /
                    LEAGUE_AVG_WOBA_SCALE) * proj_pa

    # Scale position adjustment by playing time (650 PA = full season)
    pos_adj_runs = POSITION_ADJUSTMENT_RUNS.get(position, 0) * (proj_pa / 650)

    proj_fwar = (batting_runs + pos_adj_runs) / RUNS_PER_WIN

    return {
        'mlb_player_id': mlb_id,
        'player_name': name,
        'position': position,
        'age_2026': age_2026,
        'years_of_data': years_available,
        'total_pa_history': round(total_pa, 0),
        'marcel_woba': marcel_woba,
        'statcast_adjusted_woba': statcast_adjusted_woba,
        'statcast_adj': round(sc_adj, 3),
        'bounce_back_score': bounce_back,
        'regression_risk_score': regression_risk,
        'key_indicator': key_indicator,
        'wrc_plus': round(wrc_plus, 0),
        'proj_ops': round(proj_ops, 3),
        'proj_k_pct': round(proj_k_pct, 1),
        'proj_bb_pct': round(proj_bb_pct, 1),
        'proj_pa': round(proj_pa, 0),
        'proj_war': round(proj_fwar, 1),
        'pos_adj_runs': round(pos_adj_runs, 1),
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

    # Load Statcast data — hitters
    print("\nLoading Statcast metrics...")
    statcast_raw = load_statcast_multi_year()

    # Load Statcast data — pitchers
    pitcher_sc_raw = load_pitcher_statcast_multi_year()
    pitch_metrics_raw = load_pitcher_pitch_metrics()

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

    # Project pitchers WITH Statcast overlay
    print("\nProjecting pitchers (with Statcast adjustment)...")
    pitcher_projections = []
    pitcher_sc_hits = 0

    for pid, group in pitchers_raw.groupby('mlb_player_id'):
        # Get this pitcher's Statcast data across all seasons
        p_sc = pitcher_sc_raw[pitcher_sc_raw['mlb_player_id'] == pid]
        p_pm = pitch_metrics_raw[pitch_metrics_raw['mlb_player_id'] == pid]

        proj = project_pitcher(group, pitcher_statcast=p_sc, pitch_metrics=p_pm)
        if proj:
            proj['current_team'] = roster_map.get(pid, 'Free Agent')
            pitcher_projections.append(proj)
            if not p_sc.empty:
                pitcher_sc_hits += 1

    p_df = pd.DataFrame(pitcher_projections)
    print(f"  Projected {len(p_df)} pitchers ({pitcher_sc_hits} with Statcast data)")

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
                # Marcel vs Statcast-adjusted ERA
                print(f"  Marcel ERA:     {r['marcel_era']:.2f}")
                print(f"  Statcast ERA:   {r['statcast_adjusted_era']:.2f} "
                      f"({r['statcast_era_adj']:+.2f} adjustment)")
                print(f"  Projected: {r['proj_fip']:.2f} FIP | "
                      f"{r['proj_k_bb_pct']:.1f} K-BB%")
                print(f"  Projected: {r['proj_ip']:.0f} IP | {r['proj_war']:.1f} WAR")
                # Statcast scores
                print(f"  Sustainability:    {r['sustainability_score']}/100")
                print(f"  Regression risk:   {r['regression_risk_score']}/100")
                print(f"  Breakout score:    {r['breakout_score']}/100")
                print(f"  Risk flag:  {r['primary_risk_flag']}")
                print(f"  Upside flag: {r['primary_upside_flag']}")
                print(f"  Key pitch:  {r['key_pitch']}")
                print(f"  Aging adj: {r['era_aging_adj']:+.2f} ERA | "
                      f"Regression: {r['regression_factor']:.0%} confidence")

        if not h_df_show.empty:
            print(f"\n{'=' * 70}")
            print(f"HITTER PROJECTION: {player_filter}")
            print(f"{'=' * 70}")
            for _, r in h_df_show.iterrows():
                print(f"\n  {r['player_name']} ({r['position']}, Age {r['age_2026']})")
                print(f"  Current team: {r['current_team']}")
                print(f"  Data: {r['years_of_data']} years, {r['total_pa_history']:.0f} career PA")
                # Marcel vs Statcast-adjusted wOBA
                print(f"  Marcel wOBA:    {r['marcel_woba']:.3f}")
                print(f"  Statcast wOBA:  {r['statcast_adjusted_woba']:.3f} "
                      f"({r['statcast_adj']:+.003f} adjustment)")
                print(f"  wRC+: {r['wrc_plus']:.0f} | "
                      f"{r['proj_pa']:.0f} PA | {r['proj_war']:.1f} fWAR "
                      f"(pos adj: {r['pos_adj_runs']:+.1f} runs)")
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

        proj_wins = REPLACEMENT_LEVEL_WINS + total_war

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
                f"{r['player_name']} ({r['statcast_adjusted_era']:.2f} ERA, "
                f"{r['proj_war']:.1f}W, sust:{r['sustainability_score']})"
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
                f"{r['player_name']} ({r['position']}, {r['wrc_plus']:.0f} wRC+, "
                f"{r['proj_war']:.1f} fWAR)"
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