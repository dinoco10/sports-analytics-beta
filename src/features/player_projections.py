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
    LEAGUE_AVG_STATCAST_HITTER, LEAGUE_AVG_STATCAST_PITCHER,
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

def hitter_aging_curves(age):
    """
    Nonlinear, tool-specific aging curves for hitters.
    Returns a dict of per-metric aging deltas.

    Different physical tools age on different schedules:
    - SPEED peaks earliest (24-26) and declines steeply.
      Think sprint speed dropping ~0.1 ft/s per year after 28.
    - POWER (barrel rate, exit velocity) peaks 27-30 and declines slowly.
      Muscle memory and strength hold up longer than fast-twitch.
    - PLATE DISCIPLINE (BB%) is the most stable skill.
      Veterans actually improve until early 30s — they learn the zone.
    - CONTACT (K%) worsens steadily from ~28.
      Bat speed declines, pitchers exploit weak zones more.

    The composite wOBA delta is derived from weighting these components:
    wOBA ≈ 40% power + 25% contact + 20% discipline + 15% speed

    Source: Lichtman delta-method aging research, FanGraphs, plus
    Statcast-era refinements showing power holds better than previously thought.
    """
    # Each row: (power_barrel_delta, speed_multiplier, discipline_bb_delta,
    #            contact_k_delta, composite_woba)
    #
    # Power = barrel rate delta (percentage points)
    # Speed = multiplier on PA projection and SB projection
    # Discipline = BB% delta (percentage points, positive = more walks)
    # Contact = K% delta (percentage points, positive = more strikeouts)
    # Composite = weighted wOBA delta derived from the components
    curves = {
        # Young — still developing, especially power. Speed at peak.
        'young':      {'power': +0.3, 'speed': 1.02, 'disc': -0.2, 'contact': +0.3, 'woba': +0.008},
        # Pre-peak — power still growing, speed starts ticking down
        'pre_peak':   {'power': +0.2, 'speed': 1.00, 'disc':  0.0, 'contact':  0.0, 'woba': +0.005},
        # Peak — everything in balance. Best overall production.
        'peak':       {'power':  0.0, 'speed': 0.97, 'disc': +0.1, 'contact': -0.2, 'woba':  0.000},
        # Early decline — power barely fading, speed dropping noticeably
        'early_dec':  {'power': -0.3, 'speed': 0.92, 'disc': +0.1, 'contact': -0.5, 'woba': -0.006},
        # Mid decline — power fading, speed falling fast, discipline holding
        'mid_dec':    {'power': -0.6, 'speed': 0.85, 'disc':  0.0, 'contact': -0.8, 'woba': -0.014},
        # Late decline — significant across all tools
        'late_dec':   {'power': -1.0, 'speed': 0.78, 'disc': -0.1, 'contact': -1.2, 'woba': -0.022},
        # Steep — only discipline/guile remain somewhat intact
        'steep':      {'power': -1.5, 'speed': 0.70, 'disc': -0.3, 'contact': -1.5, 'woba': -0.032},
        # Cliff — everything falling apart
        'cliff':      {'power': -2.0, 'speed': 0.60, 'disc': -0.5, 'contact': -2.0, 'woba': -0.045},
    }

    if age <= 25:   phase = 'young'
    elif age <= 27: phase = 'pre_peak'
    elif age <= 29: phase = 'peak'
    elif age <= 31: phase = 'early_dec'
    elif age <= 33: phase = 'mid_dec'
    elif age <= 35: phase = 'late_dec'
    elif age <= 37: phase = 'steep'
    else:           phase = 'cliff'

    c = curves[phase]
    return {
        'woba': c['woba'],                         # Composite wOBA delta
        'barrel_rate': c['power'],                  # Barrel % delta (power tool)
        'exit_velo': c['power'] * 0.15,             # Exit velo ages with power (~0.15 mph per barrel pt)
        'k_pct': c['contact'],                      # K% delta (positive = more Ks = worse)
        'bb_pct': c['disc'],                        # BB% delta (positive = more walks = better)
        'speed_mult': c['speed'],                   # Multiplier on PA/SB projections
        'whiff_rate': c['contact'] * 0.4,           # Whiff ages with contact (same direction)
        'chase_rate': c['disc'] * -0.8,             # Chase ages inversely with discipline
    }


def pitcher_aging_curves(age):
    """
    Nonlinear, tool-specific aging curves for pitchers.
    Returns a dict of per-metric aging deltas.

    Pitcher tools age differently than hitters:
    - VELOCITY drops ~0.5 mph/year after 28 (fast-twitch muscle decline).
      This is the most measurable and predictable decline.
    - COMMAND (BB%) is relatively stable, can even IMPROVE into early 30s.
      Pitchers learn sequencing, zone management, and batter tendencies.
    - STUFF (K%, whiff rate) declines with velocity, but offset somewhat
      by command gains and pitch tunneling improvements.
    - GROUND BALL TENDENCY is very stable — it's a pitch design feature,
      not a physical tool. Sinkerballers stay sinkerballers.

    The composite ERA delta is derived from:
    ERA ≈ 50% stuff/K + 30% command/BB + 20% velocity

    Source: FanGraphs pitcher aging curves, Statcast velocity tracking data.
    """
    curves = {
        # Young — still gaining velo and refining command
        'young':      {'velo': +0.3, 'command': -0.3, 'stuff': +0.5, 'era': -0.10},
        # Peak — everything balanced, best performance window
        'peak':       {'velo':  0.0, 'command':  0.0, 'stuff':  0.0, 'era':  0.00},
        # Early decline — velo starting to tick down, command compensates
        'early_dec':  {'velo': -0.3, 'command': +0.1, 'stuff': -0.3, 'era': +0.06},
        # Mid decline — velo loss noticeable, command still OK
        'mid_dec':    {'velo': -0.5, 'command': +0.1, 'stuff': -0.7, 'era': +0.14},
        # Late decline — can't overpower, relying on command + craft
        'late_dec':   {'velo': -0.8, 'command':  0.0, 'stuff': -1.0, 'era': +0.22},
        # Steep — significant across all tools
        'steep':      {'velo': -1.2, 'command': -0.2, 'stuff': -1.5, 'era': +0.32},
        # Very steep — only craftiest survive
        'very_steep': {'velo': -1.5, 'command': -0.4, 'stuff': -2.0, 'era': +0.42},
        # Cliff — everything gone
        'cliff':      {'velo': -2.0, 'command': -0.6, 'stuff': -2.5, 'era': +0.58},
    }

    if age <= 25:   phase = 'young'
    elif age <= 28: phase = 'peak'
    elif age <= 30: phase = 'early_dec'
    elif age <= 32: phase = 'mid_dec'
    elif age <= 34: phase = 'late_dec'
    elif age <= 36: phase = 'steep'
    elif age <= 38: phase = 'very_steep'
    else:           phase = 'cliff'

    c = curves[phase]
    return {
        'era': c['era'],                            # Composite ERA delta
        'k_pct': c['stuff'],                        # K% delta (negative = fewer Ks = worse)
        'bb_pct': c['command'],                     # BB% delta (negative = more walks = worse)
        'velocity': c['velo'],                      # Velocity delta in mph
        'whiff_rate': c['stuff'] * 0.5,             # Whiff rate ages with stuff
        'barrel_rate_against': c['stuff'] * -0.1,   # Barrel suppression degrades with stuff loss
    }


# Legacy wrappers — keep for backward compatibility with any external callers
def pitcher_aging_adjustment(age):
    """ERA adjustment by age. Now delegates to pitcher_aging_curves()."""
    return pitcher_aging_curves(age)['era']

def hitter_aging_adjustment(age):
    """wOBA adjustment by age. Now delegates to hitter_aging_curves()."""
    return hitter_aging_curves(age)['woba']


def playing_time_adjustment(age, role):
    """
    Projected IP or PA multiplier based on age and role.
    For hitters, incorporates the speed-based aging curve.
    For pitchers, uses role-specific tables (SP riskier with age).
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
    else:  # Hitter — use speed multiplier from aging curves
        return hitter_aging_curves(age)['speed_mult']


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


def compute_luck_filters(player_statcast, marcel_sc=None):
    """
    Calculate "luck filters" — gaps between expected and actual performance.
    Now uses Marcel-weighted Statcast as the "true talent" baseline instead
    of simple career averages. This is more accurate because Marcel accounts
    for recency weighting and sample-size regression.

    A player with xwOBA >> wOBA had hard contact that didn't fall for hits.
    That's BABIP luck, not a skill change. Expect regression UPWARD.

    Args:
        player_statcast: DataFrame of player's Statcast data (multi-year)
        marcel_sc: dict from marcel_weight_statcast() — Marcel-weighted baselines.
                   If None, falls back to simple career average (old behavior).

    Returns a dict with luck metrics for the most recent season.
    """
    if player_statcast.empty:
        return {}

    # Use Marcel-weighted values as baseline if available, else simple average
    if marcel_sc:
        career_babip = marcel_sc.get('babip', player_statcast['babip'].mean())
        career_hr_per_fb = marcel_sc.get('hr_per_fb', player_statcast['hr_per_fb'].mean())
    else:
        career_babip = player_statcast['babip'].mean()
        career_hr_per_fb = player_statcast['hr_per_fb'].mean()

    # Most recent season's luck gaps
    recent = player_statcast.sort_values('season').iloc[-1]

    result = {}

    # BABIP vs Marcel baseline — if current BABIP below true talent, expect bounce back
    if pd.notna(recent.get('babip')) and pd.notna(career_babip):
        result['babip_vs_career'] = round(recent['babip'] - career_babip, 3)

    # xBA minus actual BA — positive means unlucky (xBA > BA)
    if pd.notna(recent.get('xba')) and pd.notna(recent.get('ba')):
        result['xba_minus_ba'] = round(recent['xba'] - recent['ba'], 3)

    # xSLG minus actual SLG — positive means unlucky power
    if pd.notna(recent.get('xslg')) and pd.notna(recent.get('slg')):
        result['xslg_minus_slg'] = round(recent['xslg'] - recent['slg'], 3)

    # HR/FB vs Marcel baseline
    if pd.notna(recent.get('hr_per_fb')) and pd.notna(career_hr_per_fb):
        result['hr_per_fb_vs_career'] = round(recent['hr_per_fb'] - career_hr_per_fb, 3)

    # xwOBA minus actual wOBA — the single best luck indicator
    if pd.notna(recent.get('xwoba')) and pd.notna(recent.get('woba')):
        result['xwoba_minus_woba'] = round(recent['xwoba'] - recent['woba'], 3)

    return result


def statcast_adjustment(marcel_woba, player_statcast, luck_filters,
                        marcel_sc=None, trends=None):
    """
    Apply Statcast regression signals to adjust the Marcel base projection.

    Now uses Marcel-weighted Statcast values for quality thresholds (barrel rate,
    chase rate) and separate trend analysis for direction signals.

    Adjustment rules (each capped to prevent extreme swings):
    1. xwOBA >> wOBA AND BABIP below Marcel baseline → project upward (max +.020)
    2. Marcel barrel rate vs league avg → trust/discount quality (max ±.010)
    3. Pull air rate TREND → HR direction signal (max +.008)
    4. Marcel chase rate vs league avg + TREND → discipline signal (max ±.010)

    Returns: wOBA adjustment value (positive = upside, negative = downside)
    """
    if player_statcast.empty:
        return 0.0

    adjustment = 0.0
    if marcel_sc is None:
        marcel_sc = {}
    if trends is None:
        trends = {}

    # --- RULE 1: xwOBA/BABIP luck correction ---
    xwoba_gap = luck_filters.get('xwoba_minus_woba', 0)
    babip_gap = luck_filters.get('babip_vs_career', 0)

    if xwoba_gap > 0.010 and babip_gap < -0.010:
        luck_adj = min(xwoba_gap * 0.5, 0.020)
        adjustment += luck_adj
    elif xwoba_gap < -0.010 and babip_gap > 0.010:
        luck_adj = max(xwoba_gap * 0.4, -0.015)
        adjustment += luck_adj

    # --- RULE 2: Marcel-weighted contact quality ---
    # Use Marcel-weighted barrel rate (3-year stabilized) vs league avg
    # This is more reliable than checking just one year's barrel rate
    marcel_barrel = marcel_sc.get('barrel_rate')
    if marcel_barrel is not None:
        if marcel_barrel > 12.0:
            # Elite barrel rate across 3 years = real power, trust projection
            adjustment += 0.008
        elif marcel_barrel < 5.0:
            # Consistently poor contact = discount
            adjustment -= 0.006

    # Also check trend — is barrel rate improving or declining?
    barrel_trend = trends.get('barrel_rate_trend', 0)
    if barrel_trend > 1.5:
        adjustment += min(barrel_trend * 0.003, 0.010)
    elif barrel_trend < -2.0:
        adjustment += max(barrel_trend * 0.002, -0.008)

    # --- RULE 3: Pull air rate trend (HR predictor) ---
    pull_trend = trends.get('pull_air_rate_trend', 0)
    if pull_trend > 2.0:
        adjustment += min(pull_trend * 0.002, 0.008)

    # --- RULE 4: Marcel-weighted chase rate + trend ---
    # Marcel chase rate gives true talent discipline level.
    # Trend tells us if discipline is improving or declining.
    marcel_chase = marcel_sc.get('chase_rate')
    if marcel_chase is not None:
        if marcel_chase > 33.0:
            # High chase hitter across 3 years = real discipline problem
            adjustment -= 0.005
        elif marcel_chase < 22.0:
            # Elite discipline = trust the projection
            adjustment += 0.003

    chase_trend = trends.get('chase_rate_trend', 0)
    if chase_trend > 2.0:
        adjustment -= min(chase_trend * 0.002, 0.010)
    elif chase_trend < -2.0:
        adjustment += min(abs(chase_trend) * 0.001, 0.005)

    # Final cap: never adjust more than +/- 0.030 wOBA
    adjustment = max(-0.030, min(0.030, adjustment))
    return round(adjustment, 3)


def compute_scores(player_statcast, luck_filters, sc_adjustment,
                   marcel_sc=None, trends=None):
    """
    Compute bounce-back score and regression risk score (both 0-100).
    Now uses Marcel-weighted Statcast values for quality thresholds
    and trends for direction signals.

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
    if marcel_sc is None:
        marcel_sc = {}
    if trends is None:
        trends = {}

    if not luck_filters and player_statcast.empty:
        return 50, 50, "insufficient_data"

    # --- Luck signals (biggest impact on bounce-back score) ---
    xwoba_gap = luck_filters.get('xwoba_minus_woba', 0)
    babip_gap = luck_filters.get('babip_vs_career', 0)

    # xwOBA gap: positive = unlucky = bounce-back candidate
    if xwoba_gap > 0.020:
        bounce_back += min(xwoba_gap * 500, 25)
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

    # --- Contact quality signals (now using Marcel-weighted values) ---
    # Marcel-weighted barrel rate is more stable than a single year's value
    marcel_barrel = marcel_sc.get('barrel_rate') if marcel_sc else None
    if marcel_barrel is not None:
        if marcel_barrel > 12:
            bounce_back += 8
            indicators['barrel_rate'] = 8
        elif marcel_barrel < 4:
            regression_risk += 8
            indicators['barrel_rate'] = 8

    marcel_hh = marcel_sc.get('hard_hit_rate') if marcel_sc else None
    if marcel_hh is not None:
        if marcel_hh > 45:
            bounce_back += 5
        elif marcel_hh < 30:
            regression_risk += 5

    # --- Discipline signals (Marcel-weighted) ---
    marcel_chase = marcel_sc.get('chase_rate') if marcel_sc else None
    if marcel_chase is not None:
        if marcel_chase > 35:
            regression_risk += 10
            indicators['chase_rate'] = 10
        elif marcel_chase < 25:
            bounce_back += 5
            indicators['chase_rate'] = 5

    marcel_zc = marcel_sc.get('z_contact_rate') if marcel_sc else None
    if marcel_zc is not None:
        if marcel_zc > 88:
            bounce_back += 5
        elif marcel_zc < 78:
            regression_risk += 5

    # --- Trend signals (using compute_statcast_trends output) ---
    pull_trend = trends.get('pull_air_rate_trend', 0) if trends else 0
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


# ═══════════════════════════════════════════════════════════════
# MARCEL-WEIGHT STATCAST METRICS
# ═══════════════════════════════════════════════════════════════
#
# The Marcel Method applies to EVERYTHING, not just traditional stats.
# A player's barrel rate, chase rate, xwOBA, etc. should also be
# weighted 5/4/3 across 3 years and regressed toward league average.
# This gives more stable inputs than using a single year's Statcast data.

def marcel_weight_statcast(player_statcast_df, metrics, league_averages,
                           sample_col='statcast_pa', full_season_sample=550):
    """
    Apply Marcel 5/4/3 weighting to a player's multi-year Statcast metrics,
    then regress toward league average based on total sample size.

    How it works (same Marcel logic as traditional stats):
    1. For each Statcast metric (barrel rate, xwOBA, chase rate, etc.):
       - Weight 2025 data × 5, 2024 × 4, 2023 × 3
       - Also weight by PA within each year (more PA = more reliable)
    2. Regress toward league average based on total career sample
       - 3 full seasons (1650 PA) = no regression
       - 1 season of 300 PA = heavy regression toward league average

    Returns: dict of {metric_name: marcel_weighted_value}

    Example: A hitter with barrel rates of 15%, 12%, 14% across 3 years
    gets Marcel-weighted barrel rate of ~13.5% (recent-weighted),
    then slightly regressed toward the 8.5% league average.
    """
    if player_statcast_df.empty:
        return {m: league_averages.get(m, 0) for m in metrics}

    year_weights = {2025: 5, 2024: 4, 2023: 3}
    result = {}

    # Total sample across all years (for regression calculation)
    total_sample = player_statcast_df[sample_col].sum() if sample_col in player_statcast_df.columns else 0

    for metric in metrics:
        weighted_sum = 0.0
        total_weight = 0.0

        for _, row in player_statcast_df.iterrows():
            season = row.get('season')
            value = row.get(metric)
            sample = row.get(sample_col, 0)

            if pd.isna(value) or pd.isna(season):
                continue

            yr_w = year_weights.get(int(season), 0)
            if yr_w == 0:
                continue

            # Weight by both year recency AND sample size within that year
            pa_weight = min(sample / full_season_sample, 1.0) if pd.notna(sample) and sample > 0 else 0.5
            effective_weight = yr_w * pa_weight

            weighted_sum += value * effective_weight
            total_weight += effective_weight

        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
        else:
            # No valid data — use league average
            result[metric] = league_averages.get(metric, 0)
            continue

        # Regress toward league average based on total career sample
        # 3 full seasons = full confidence, less data = more regression
        regression_factor = min(total_sample / (full_season_sample * 3), 1.0)
        league_avg = league_averages.get(metric, weighted_avg)
        final = weighted_avg * regression_factor + league_avg * (1 - regression_factor)

        result[metric] = round(final, 3)

    return result


def compute_statcast_trends(player_statcast_df, metrics):
    """
    Compute 2-year deltas for each Statcast metric.
    Returns {metric + '_trend': recent_value - previous_value}.

    Trends are the CHANGE signal on top of the Marcel-weighted base.
    Marcel tells you "what is this player's true talent level?"
    Trends tell you "is that talent level getting better or worse?"

    Example: A pitcher with Marcel-weighted barrel_rate_against of 7.0%
    but a trend of +2.5% is GETTING WORSE despite looking elite on average.
    """
    if len(player_statcast_df) < 2:
        return {}

    sorted_df = player_statcast_df.sort_values('season')
    recent = sorted_df.iloc[-1]
    previous = sorted_df.iloc[-2]

    trends = {}
    for metric in metrics:
        recent_val = recent.get(metric)
        prev_val = previous.get(metric)
        if pd.notna(recent_val) and pd.notna(prev_val):
            trends[f'{metric}_trend'] = round(recent_val - prev_val, 3)

    return trends


def compute_pitcher_luck_filters(pitcher_statcast, marcel_sc=None):
    """
    Calculate pitcher "luck filters" — gaps between expected and actual.
    Now uses Marcel-weighted Statcast as the "true talent" baseline.

    Key signals:
    - era_minus_xera: positive = pitcher was UNLUCKY (ERA > xERA, expect improvement)
    - babip_vs_career: positive = bad luck on balls in play (expect regression down)
    - hr_per_fb_vs_norm: above ~12% = HR luck catching up, expect regression
    - xwoba_vs_woba: if xwOBA_against < actual wOBA_against, pitching better than stats show
    """
    if pitcher_statcast.empty:
        return {}

    # Use Marcel-weighted baselines if available
    if marcel_sc:
        career_babip = marcel_sc.get('babip_against', pitcher_statcast['babip_against'].mean())
        career_hr_fb = marcel_sc.get('hr_per_fb', pitcher_statcast['hr_per_fb'].mean())
    else:
        career_babip = pitcher_statcast['babip_against'].mean()
        career_hr_fb = pitcher_statcast['hr_per_fb'].mean()
    league_hr_fb = 0.12

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
                                pitch_metrics=None, marcel_sc=None, trends=None):
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
    if marcel_sc is None:
        marcel_sc = {}
    if trends is None:
        trends = {}

    # --- RULE 1: ERA/xERA + BABIP luck correction ---
    era_gap = luck_filters.get('era_minus_xera', 0)
    babip_gap = luck_filters.get('babip_vs_career', 0)

    if era_gap > 0.30 and babip_gap > 0.010:
        luck_adj = max(-era_gap * 0.4, -0.40)
        adjustment += luck_adj
    elif era_gap < -0.30 and babip_gap < -0.010:
        luck_adj = min(abs(era_gap) * 0.35, 0.35)
        adjustment += luck_adj

    # --- RULE 2: Marcel-weighted contact suppression quality ---
    marcel_barrel = marcel_sc.get('barrel_rate_against')
    if marcel_barrel is not None:
        if marcel_barrel < 5.5:
            adjustment -= 0.10
        elif marcel_barrel > 9.0:
            adjustment += 0.15

    marcel_hh = marcel_sc.get('hard_hit_rate_against')
    if marcel_hh is not None:
        if marcel_hh < 32:
            adjustment -= 0.05
        elif marcel_hh > 42:
            adjustment += 0.10

    # --- RULE 3: Pull air rate against TREND ---
    pull_trend = trends.get('pull_air_rate_against_trend', 0)
    if pull_trend > 2.0:
        adjustment += min(pull_trend * 0.04, 0.15)
    elif pull_trend < -2.0:
        adjustment -= min(abs(pull_trend) * 0.03, 0.10)

    # --- RULE 4: Marcel-weighted swing & miss sustainability ---
    marcel_k_bb = marcel_sc.get('k_minus_bb')
    if marcel_k_bb is not None:
        if marcel_k_bb > 20:
            adjustment -= 0.10
        elif marcel_k_bb < 8:
            adjustment += 0.15

    marcel_swstr = marcel_sc.get('swstr_rate')
    if marcel_swstr is not None:
        if marcel_swstr < 8.6:
            adjustment += 0.08
        elif marcel_swstr > 13.0:
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
                           pitch_metrics=None, marcel_sc=None, trends=None):
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

    if marcel_sc is None:
        marcel_sc = {}
    if trends is None:
        trends = {}

    if pitcher_statcast.empty and not luck_filters:
        return 50, 50, 50, "insufficient_data", "insufficient_data", "unknown"

    recent = pitcher_statcast.sort_values('season').iloc[-1] if not pitcher_statcast.empty else {}

    # ── SUSTAINABILITY SIGNALS ──────────────────────────────
    # Use Marcel-weighted values for level, trends for direction
    # K-BB% is king — the most stable pitcher metric year-over-year
    k_bb = marcel_sc.get('k_minus_bb', recent.get('k_minus_bb'))
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
    barrel = marcel_sc.get('barrel_rate_against', recent.get('barrel_rate_against'))
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
    gb = marcel_sc.get('gb_rate', recent.get('gb_rate'))
    if pd.notna(gb):
        if gb > 50:
            sustainability += 8
        elif gb < 38:
            sustainability -= 5

    # Chase rate induced — ability to get swings outside zone is sticky
    chase = marcel_sc.get('chase_rate_induced', recent.get('chase_rate_induced'))
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
    swstr = marcel_sc.get('swstr_rate', recent.get('swstr_rate'))
    if pd.notna(swstr):
        if swstr < 8.6:
            regression_risk += 12
            risk_factors['low_swstr'] = 12
        elif swstr < 10.0:
            regression_risk += 5

    # High fly ball rate — HR-prone profile
    fb = marcel_sc.get('fb_rate', recent.get('fb_rate'))
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
    # Use pre-computed trends if available, fallback to manual calculation
    whiff_trend = trends.get('whiff_rate_trend')
    if whiff_trend is None and len(pitcher_statcast) >= 2:
        sorted_ps = pitcher_statcast.sort_values('season')
        rw = sorted_ps.iloc[-1].get('whiff_rate')
        pw = sorted_ps.iloc[-2].get('whiff_rate')
        if pd.notna(rw) and pd.notna(pw):
            whiff_trend = rw - pw
    if whiff_trend is not None and whiff_trend > 2.0:
        breakout += min(whiff_trend * 3, 12)
        upside_factors['improving_whiff'] = min(whiff_trend * 3, 12)

    # Chase rate improving — getting better at inducing chases
    chase_trend = trends.get('chase_rate_induced_trend')
    if chase_trend is None and len(pitcher_statcast) >= 2:
        sorted_ps = pitcher_statcast.sort_values('season')
        rc = sorted_ps.iloc[-1].get('chase_rate_induced')
        pc = sorted_ps.iloc[-2].get('chase_rate_induced')
        if pd.notna(rc) and pd.notna(pc):
            chase_trend = rc - pc
    if chase_trend is not None and chase_trend > 2.0:
        breakout += min(chase_trend * 2, 8)

    # Pull air rate against declining — fewer HR coming
    pull_trend = trends.get('pull_air_rate_against_trend')
    if pull_trend is None and len(pitcher_statcast) >= 2:
        sorted_ps = pitcher_statcast.sort_values('season')
        rp = sorted_ps.iloc[-1].get('pull_air_rate_against')
        pp = sorted_ps.iloc[-2].get('pull_air_rate_against')
        if pd.notna(rp) and pd.notna(pp):
            pull_trend = rp - pp
    if pull_trend is not None and pull_trend < -2.0:
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

    # ── TOOL-SPECIFIC AGING ─────────────────────────────────
    # Different pitcher tools age at different rates:
    # - Velocity drops ~0.5 mph/year after 28
    # - Command (BB%) is stable, can even improve into early 30s
    # - Stuff (K%) declines with velocity
    aging = pitcher_aging_curves(age_2026)
    era_aging = aging['era']
    proj_era += era_aging
    proj_fip += era_aging * 0.7  # FIP ages slightly less than ERA
    proj_k_pct += aging['k_pct']
    proj_bb_pct += aging['bb_pct']

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
    marcel_sc = {}
    trends = {}

    if pitcher_statcast is not None and not pitcher_statcast.empty:
        # Marcel-weight ALL pitcher Statcast metrics across 3 years
        pitcher_sc_metrics = [
            'avg_exit_velocity_against', 'barrel_rate_against', 'hard_hit_rate_against',
            'xwoba_against', 'xera', 'whiff_rate', 'swstr_rate',
            'chase_rate_induced', 'z_contact_rate_against',
            'pull_air_rate_against', 'gb_rate', 'fb_rate', 'ld_rate',
            'babip_against', 'hr_per_fb', 'k_minus_bb',
        ]
        marcel_sc = marcel_weight_statcast(
            pitcher_statcast, pitcher_sc_metrics,
            LEAGUE_AVG_STATCAST_PITCHER,
            sample_col='statcast_pa', full_season_sample=180,
        )
        # Apply tool-specific aging to Marcel-weighted Statcast metrics
        if 'whiff_rate' in marcel_sc:
            marcel_sc['whiff_rate'] += aging['whiff_rate']
        if 'barrel_rate_against' in marcel_sc:
            marcel_sc['barrel_rate_against'] += aging['barrel_rate_against']
        trends = compute_statcast_trends(pitcher_statcast, pitcher_sc_metrics)
        luck = compute_pitcher_luck_filters(pitcher_statcast, marcel_sc=marcel_sc)
        sc_era_adj = pitcher_statcast_adjustment(
            marcel_era, pitcher_statcast, luck, pitch_metrics,
            marcel_sc=marcel_sc, trends=trends,
        )
        sustainability, regression_risk, breakout, primary_risk, primary_upside, key_pitch = \
            compute_pitcher_scores(
                pitcher_statcast, luck, sc_era_adj, pitch_metrics,
                marcel_sc=marcel_sc, trends=trends,
            )

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
        'aging_k_pct_adj': round(aging['k_pct'], 1),
        'aging_bb_pct_adj': round(aging['bb_pct'], 1),
        'aging_velo_adj': round(aging['velocity'], 1),
        'regression_factor': round(regression_factor, 2),
        # Marcel-weighted Statcast fields (for Floor 1 features if needed)
        'proj_barrel_rate_against': round(marcel_sc.get('barrel_rate_against', 8.5), 1),
        'proj_xera': round(marcel_sc.get('xera', 4.20), 2),
        'proj_whiff_rate': round(marcel_sc.get('whiff_rate', 25.0), 1),
        'proj_gb_rate': round(marcel_sc.get('gb_rate', 43.0), 1),
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

    # ── TOOL-SPECIFIC AGING ─────────────────────────────────
    # Different hitter tools age at different rates:
    # - Speed peaks earliest (24-26), declines steeply
    # - Power (barrel rate) peaks 27-30, declines slowly
    # - Plate discipline (BB%) most stable, barely moves until late 30s
    # - Contact (K%) worsens steadily from ~28
    aging = hitter_aging_curves(age_2026)
    woba_aging = aging['woba']
    proj_woba += woba_aging
    proj_k_pct += aging['k_pct']
    proj_bb_pct += aging['bb_pct']

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

    marcel_sc = {}
    trends = {}

    if player_statcast is not None and not player_statcast.empty:
        # Marcel-weight ALL Statcast metrics across 3 years
        hitter_sc_metrics = [
            'avg_exit_velocity', 'barrel_rate', 'hard_hit_rate',
            'xwoba', 'xslg', 'xba', 'launch_angle_sweet_spot_pct',
            'chase_rate', 'whiff_rate', 'z_contact_rate',
            'pull_air_rate', 'gb_rate', 'fb_rate', 'ld_rate',
            'babip', 'hr_per_fb',
        ]
        marcel_sc = marcel_weight_statcast(
            player_statcast, hitter_sc_metrics,
            LEAGUE_AVG_STATCAST_HITTER,
            sample_col='statcast_pa', full_season_sample=550,
        )
        # Apply tool-specific aging to Marcel-weighted Statcast metrics
        if 'barrel_rate' in marcel_sc:
            marcel_sc['barrel_rate'] += aging['barrel_rate']
        if 'avg_exit_velocity' in marcel_sc:
            marcel_sc['avg_exit_velocity'] += aging['exit_velo']
        if 'whiff_rate' in marcel_sc:
            marcel_sc['whiff_rate'] += aging['whiff_rate']
        if 'chase_rate' in marcel_sc:
            marcel_sc['chase_rate'] += aging['chase_rate']
        trends = compute_statcast_trends(player_statcast, hitter_sc_metrics)
        luck = compute_luck_filters(player_statcast, marcel_sc=marcel_sc)
        sc_adj = statcast_adjustment(
            marcel_woba, player_statcast, luck,
            marcel_sc=marcel_sc, trends=trends,
        )
        bounce_back, regression_risk, key_indicator = compute_scores(
            player_statcast, luck, sc_adj,
            marcel_sc=marcel_sc, trends=trends,
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
        'aging_k_pct_adj': round(aging['k_pct'], 1),
        'aging_bb_pct_adj': round(aging['bb_pct'], 1),
        'aging_barrel_adj': round(aging['barrel_rate'], 1),
        'regression_factor': round(regression_factor, 2),
        # Marcel-weighted Statcast fields (for Floor 1 features if needed)
        'proj_barrel_rate': round(marcel_sc.get('barrel_rate', 8.5), 1),
        'proj_xwoba': round(marcel_sc.get('xwoba', 0.310), 3),
        'proj_chase_rate': round(marcel_sc.get('chase_rate', 28.0), 1),
        'proj_whiff_rate': round(marcel_sc.get('whiff_rate', 25.0), 1),
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