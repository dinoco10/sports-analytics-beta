"""
backfill_weather.py — Populate weather data for historical games
================================================================
Fetches temperature, wind speed, wind direction from the MLB API
live feed for each game and updates the games table.

Also collects venue metadata (elevation, azimuth, coords) and
populates/updates the ballparks table.

Usage:
  python scripts/backfill_weather.py                    # All games missing weather
  python scripts/backfill_weather.py --season 2025      # Single season
  python scripts/backfill_weather.py --limit 100        # First N games only
  python scripts/backfill_weather.py --dry-run           # Show what would be updated
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.mlb_api import MLBApiClient

DB_PATH = Path(__file__).parent.parent / "data" / "mlb_analytics.db"


def get_games_needing_weather(conn, season=None, limit=None):
    """Get games where temperature_f is NULL (weather not yet populated)."""
    query = """
    SELECT id, mlb_game_id, date, season, home_team_id
    FROM games
    WHERE temperature_f IS NULL
      AND home_score IS NOT NULL
    """
    if season:
        query += f" AND season = {season}"
    query += " ORDER BY date, id"
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    return cursor.fetchall()


def update_game_weather(conn, game_id, weather_data):
    """Update a single game row with weather data."""
    conn.execute("""
        UPDATE games
        SET temperature_f = ?,
            wind_speed_mph = ?,
            wind_direction = ?,
            is_dome = ?
        WHERE id = ?
    """, (
        weather_data.get("temperature_f"),
        weather_data.get("wind_speed_mph"),
        weather_data.get("wind_direction"),
        1 if weather_data.get("condition", "").lower() in ("dome", "roof closed") else 0,
        game_id,
    ))


def upsert_ballpark(conn, weather_data, team_id):
    """Insert or update ballpark with venue data from API."""
    venue_id = weather_data.get("venue_id")
    if not venue_id:
        return

    # Check if ballpark exists
    existing = conn.execute(
        "SELECT id FROM ballparks WHERE name = ?",
        (weather_data.get("venue_name"),)
    ).fetchone()

    if existing:
        # Update elevation if we have it and it's not set
        if weather_data.get("elevation_ft"):
            conn.execute("""
                UPDATE ballparks
                SET elevation_ft = COALESCE(elevation_ft, ?)
                WHERE id = ?
            """, (weather_data["elevation_ft"], existing[0]))
    else:
        # Detect dome status from condition
        is_dome = weather_data.get("condition", "").lower() in ("dome", "roof closed")

        conn.execute("""
            INSERT INTO ballparks (name, team_id, elevation_ft, is_dome, park_factor,
                                   park_factor_hr, park_factor_lhb, park_factor_rhb)
            VALUES (?, ?, ?, ?, 1.000, 1.000, 1.000, 1.000)
        """, (
            weather_data.get("venue_name"),
            team_id,
            weather_data.get("elevation_ft"),
            is_dome,
        ))


def main():
    parser = argparse.ArgumentParser(description="Backfill weather data from MLB API")
    parser.add_argument("--season", type=int, help="Single season to backfill")
    parser.add_argument("--limit", type=int, help="Max games to process")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only")
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N games")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    games = get_games_needing_weather(conn, args.season, args.limit)

    print(f"Games needing weather: {len(games)}")
    if args.dry_run:
        # Show breakdown by season
        seasons = {}
        for g in games:
            seasons[g[3]] = seasons.get(g[3], 0) + 1
        for s, c in sorted(seasons.items()):
            print(f"  {s}: {c} games")
        conn.close()
        return

    client = MLBApiClient()
    updated = 0
    errors = 0
    seen_venues = set()

    for i, (game_id, mlb_game_id, game_date, season, home_team_id) in enumerate(games):
        try:
            weather = client.get_game_weather(mlb_game_id)

            if weather and weather.get("temperature_f") is not None:
                update_game_weather(conn, game_id, weather)
                updated += 1

                # Populate ballpark on first encounter
                venue_name = weather.get("venue_name")
                if venue_name and venue_name not in seen_venues:
                    upsert_ballpark(conn, weather, home_team_id)
                    seen_venues.add(venue_name)
            else:
                # No weather data (possibly postponed/cancelled) — mark with -999 to skip
                conn.execute(
                    "UPDATE games SET temperature_f = -999 WHERE id = ?",
                    (game_id,)
                )
                errors += 1

        except Exception as e:
            print(f"  Error game {mlb_game_id}: {e}")
            errors += 1

        # Commit periodically
        if (i + 1) % args.batch_size == 0:
            conn.commit()
            pct = (i + 1) / len(games) * 100
            print(f"  [{i+1}/{len(games)}] ({pct:.0f}%) — {updated} updated, {errors} errors")

    conn.commit()
    conn.close()

    print(f"\nDone: {updated} games updated, {errors} errors/missing, {len(seen_venues)} venues found")


if __name__ == "__main__":
    main()
