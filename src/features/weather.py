"""
weather.py — Weather & Park Environment Features
=================================================
Computes a single composite "ball_flight_index" per game that captures
how atmospheric conditions affect ball flight distance.

Physics: Air density rho = P / (R * T) determines drag on a baseball.
Lower density = less drag = more carry = more runs/HR.

Key factors (by variance explained):
  - Altitude (80%) — static per park, captured via elevation
  - Temperature (13%) — +4 ft per 10F rise
  - Barometric pressure (4%) — correlated with altitude
  - Humidity (3%) — humid air is actually lighter (water vapor < N2/O2)
  - Wind — park-dependent, directional

Output: ball_flight_index on a 1-10 scale where:
  1 = extreme pitcher-friendly (cold, sea level, wind in)
  5 = neutral
  10 = extreme hitter-friendly (hot, high altitude, wind out)

References:
  - Alan Nathan, UIUC: "Effect of Temperature on Home Run Production"
  - homerunforecast.com methodology
  - BallparkPal post-contact flight model

Usage:
  from src.features.weather import compute_weather_features
"""

import numpy as np
import pandas as pd


# ─── Physical Constants ────────────────────────────────────
# Standard conditions (sea level, 72F) as baseline
BASELINE_TEMP_F = 72.0
BASELINE_ELEVATION_FT = 0.0
BASELINE_AIR_DENSITY = 1.225  # kg/m3 at sea level, 15C

# How much each factor affects ball flight distance (in feet per unit change)
# Source: Alan Nathan (UIUC) + SABR atmospheric research
TEMP_EFFECT_PER_10F = 4.0       # +4 ft per 10F rise
ELEVATION_EFFECT_PER_1000FT = 7.0  # +7 ft per 1000ft elevation
# At Coors (5200 ft) = +36.4 ft — consistent with Nathan's ~5% more carry

# Wind effects (approximate, in feet per mph of wind)
WIND_OUT_EFFECT_PER_MPH = 3.0    # +3 ft per mph of tailwind
WIND_IN_EFFECT_PER_MPH = -3.0    # -3 ft per mph of headwind
WIND_CROSS_EFFECT_PER_MPH = 0.5  # minimal effect for crosswind

# Scaling: map raw distance delta to 1-10 index
# A +50 ft advantage = index 10, -50 ft = index 1
DISTANCE_DELTA_RANGE = 50.0  # ft from min to max


def _wind_direction_factor(wind_dir: str) -> float:
    """
    Convert wind direction string to a -1.0 to +1.0 factor.

    +1.0 = wind blowing OUT (helps batters)
    -1.0 = wind blowing IN (helps pitchers)
     0.0 = crosswind or calm (minimal effect)

    The MLB API uses strings like:
    "Out To CF", "Out To RF", "Out To LF" → positive
    "In From CF", "In From RF", "In From LF" → negative
    "L To R", "R To L" → crosswind
    "Calm" → no effect
    """
    if not wind_dir or not isinstance(wind_dir, str):
        return 0.0

    wind_dir_lower = wind_dir.lower().strip()

    if "calm" in wind_dir_lower or wind_dir_lower == "":
        return 0.0

    # Out = helping hitters (tailwind for fly balls)
    if "out to" in wind_dir_lower:
        if "cf" in wind_dir_lower:
            return 1.0     # Dead center = maximum effect
        elif "lf" in wind_dir_lower or "rf" in wind_dir_lower:
            return 0.75    # To a corner, slightly less
        return 0.8  # Generic "out"

    # In = hurting hitters (headwind for fly balls)
    if "in from" in wind_dir_lower:
        if "cf" in wind_dir_lower:
            return -1.0
        elif "lf" in wind_dir_lower or "rf" in wind_dir_lower:
            return -0.75
        return -0.8

    # Crosswind
    if "l to r" in wind_dir_lower or "r to l" in wind_dir_lower:
        return 0.15  # Slight positive — crosswinds tend to help marginal flies

    # Varies / unknown
    if "varies" in wind_dir_lower:
        return 0.0

    return 0.0


def compute_ball_flight_index(temperature_f, wind_speed_mph, wind_direction,
                               elevation_ft, is_dome=False):
    """
    Compute a single ball_flight_index (1-10 scale) for a game.

    This pre-computes the physics into a single feature, which is better than
    5 raw weather variables at max_depth=2 (tree can't learn interactions).

    Parameters
    ----------
    temperature_f : float or None
        Game-time temperature in Fahrenheit
    wind_speed_mph : float or None
        Wind speed in mph
    wind_direction : str or None
        Wind direction string from MLB API
    elevation_ft : float or None
        Park elevation in feet
    is_dome : bool
        True if game is in a dome/closed roof

    Returns
    -------
    float : ball_flight_index on 1-10 scale
    """
    # Dome games: use neutral values (indoor climate controlled ~72F, no wind)
    if is_dome:
        temperature_f = BASELINE_TEMP_F
        wind_speed_mph = 0
        wind_direction = "Calm"

    # Default missing values to neutral
    if temperature_f is None or temperature_f <= 0 or temperature_f == -999:
        temperature_f = BASELINE_TEMP_F
    if wind_speed_mph is None:
        wind_speed_mph = 0
    if elevation_ft is None:
        elevation_ft = BASELINE_ELEVATION_FT

    # Compute distance delta from baseline
    # Temperature component
    temp_delta = (temperature_f - BASELINE_TEMP_F) / 10.0 * TEMP_EFFECT_PER_10F

    # Elevation component
    elev_delta = (elevation_ft - BASELINE_ELEVATION_FT) / 1000.0 * ELEVATION_EFFECT_PER_1000FT

    # Wind component
    wind_factor = _wind_direction_factor(wind_direction)
    if wind_factor > 0:
        wind_delta = wind_speed_mph * wind_factor * WIND_OUT_EFFECT_PER_MPH
    elif wind_factor < 0:
        wind_delta = wind_speed_mph * abs(wind_factor) * WIND_IN_EFFECT_PER_MPH
    else:
        wind_delta = wind_speed_mph * WIND_CROSS_EFFECT_PER_MPH * 0.1  # Tiny effect

    total_delta = temp_delta + elev_delta + wind_delta

    # Scale to 1-10
    # 0 delta = 5.5, +50 = 10, -50 = 1
    index = 5.5 + (total_delta / DISTANCE_DELTA_RANGE) * 4.5
    return np.clip(index, 1.0, 10.0)


def compute_weather_features(games_df, ballparks_df=None):
    """
    Compute weather features for all games.

    Parameters
    ----------
    games_df : DataFrame
        Must have columns: game_id, temperature_f, wind_speed_mph,
        wind_direction, is_dome, home_team_id
    ballparks_df : DataFrame or None
        If provided, must have: team_id, elevation_ft, park_factor,
        park_factor_hr, park_factor_lhb, park_factor_rhb

    Returns
    -------
    DataFrame with columns:
        game_id, game_temperature, ball_flight_index,
        park_factor_runs, park_factor_hr
    """
    # Build elevation lookup from ballparks
    elevation_map = {}
    park_factor_map = {}
    if ballparks_df is not None and len(ballparks_df) > 0:
        for _, bp in ballparks_df.iterrows():
            tid = bp.get("team_id")
            if tid:
                elevation_map[tid] = bp.get("elevation_ft", 0)
                park_factor_map[tid] = {
                    "park_factor": bp.get("park_factor", 1.0),
                    "park_factor_hr": bp.get("park_factor_hr", 1.0),
                    "park_factor_lhb": bp.get("park_factor_lhb", 1.0),
                    "park_factor_rhb": bp.get("park_factor_rhb", 1.0),
                }

    results = []
    for _, game in games_df.iterrows():
        game_id = game["game_id"]
        home_team = game.get("home_team_id")
        temp = game.get("temperature_f")
        wind_speed = game.get("wind_speed_mph")
        wind_dir = game.get("wind_direction")
        is_dome = bool(game.get("is_dome", False))

        elevation = elevation_map.get(home_team, 0) or 0

        bfi = compute_ball_flight_index(
            temp, wind_speed, wind_dir, elevation, is_dome
        )

        # Park factor features (static per park)
        pf = park_factor_map.get(home_team, {})

        results.append({
            "game_id": game_id,
            "game_temperature": temp if temp and temp > 0 and temp != -999 else None,
            "ball_flight_index": round(bfi, 2),
            "park_factor_runs": pf.get("park_factor", 1.0),
            "park_factor_hr": pf.get("park_factor_hr", 1.0),
        })

    result_df = pd.DataFrame(results)

    # Stats
    valid_temp = result_df["game_temperature"].notna().sum()
    avg_bfi = result_df["ball_flight_index"].mean()
    print(f"  Weather features: {len(result_df)} games, "
          f"{valid_temp} with temperature, avg BFI={avg_bfi:.2f}")

    return result_df
