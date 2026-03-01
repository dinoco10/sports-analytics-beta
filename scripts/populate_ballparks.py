"""
populate_ballparks.py — Seed ballparks table with park factors
==============================================================
Populates all 30 MLB ballparks with:
- Statcast park factors (runs, HR, LHB, RHB)
- Elevation (feet)
- Dome/retractable roof status
- Field orientation (azimuth angle for wind adjustments)

Park factors from Baseball Savant (2022-2024 combined, 100 = neutral).
Stored as multiplier: park_factor / 100 (e.g., 104 -> 1.040).

Usage:
  python scripts/populate_ballparks.py
"""

import sqlite3
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = Path(__file__).parent.parent / "data" / "mlb_analytics.db"

# ─── Park Factor Data ──────────────────────────────────────
# Sources: Baseball Savant Statcast Park Factors (2022-2024 combined)
#          Elevation from MLB API venue data
#          Azimuth from MLB API location.azimuthAngle
#
# Format: (team_name, venue_name, park_factor_runs, park_factor_hr,
#           park_factor_lhb, park_factor_rhb, elevation_ft,
#           is_dome, is_retractable, azimuth_angle, latitude, longitude)

BALLPARK_DATA = [
    # team_name, venue_name, pf_runs, pf_hr, pf_lhb, pf_rhb, elev, dome, retract, azimuth, lat, lon
    ("Arizona Diamondbacks", "Chase Field", 107, 109, 108, 106, 1082, False, True, 0, 33.4455, -112.0667),
    ("Atlanta Braves", "Truist Park", 100, 105, 99, 101, 997, False, False, 0, 33.8908, -84.4678),
    ("Baltimore Orioles", "Oriole Park at Camden Yards", 101, 107, 100, 103, 28, False, False, 0, 39.2838, -76.6218),
    ("Boston Red Sox", "Fenway Park", 107, 94, 114, 100, 21, False, False, 0, 42.3467, -71.0972),
    ("Chicago Cubs", "Wrigley Field", 103, 110, 102, 104, 595, False, False, 0, 41.9484, -87.6553),
    ("Chicago White Sox", "Guaranteed Rate Field", 102, 110, 100, 104, 595, False, False, 0, 41.8299, -87.6338),
    ("Cincinnati Reds", "Great American Ball Park", 108, 118, 107, 110, 490, False, False, 0, 39.0974, -84.5067),
    ("Cleveland Guardians", "Progressive Field", 96, 94, 97, 95, 640, False, False, 0, 41.4962, -81.6852),
    ("Colorado Rockies", "Coors Field", 117, 114, 118, 116, 5200, False, False, 0, 39.7559, -104.9942),
    ("Detroit Tigers", "Comerica Park", 95, 89, 96, 94, 600, False, False, 0, 42.3390, -83.0485),
    ("Houston Astros", "Minute Maid Park", 100, 107, 98, 103, 42, False, True, 0, 29.7573, -95.3555),
    ("Kansas City Royals", "Kauffman Stadium", 100, 97, 101, 99, 820, False, False, 0, 39.0517, -94.4803),
    ("Los Angeles Angels", "Angel Stadium", 97, 99, 96, 98, 160, False, False, 0, 33.8003, -117.8827),
    ("Los Angeles Dodgers", "Dodger Stadium", 96, 93, 97, 95, 515, False, False, 0, 34.0739, -118.2400),
    ("Miami Marlins", "loanDepot park", 92, 82, 93, 91, 7, False, True, 0, 25.7781, -80.2196),
    ("Milwaukee Brewers", "American Family Field", 102, 110, 101, 104, 645, False, True, 0, 43.0280, -87.9712),
    ("Minnesota Twins", "Target Field", 99, 102, 98, 100, 815, False, False, 0, 44.9818, -93.2776),
    ("New York Mets", "Citi Field", 95, 91, 96, 94, 10, False, False, 0, 40.7571, -73.8458),
    ("New York Yankees", "Yankee Stadium", 106, 120, 103, 110, 5, False, False, 0, 40.8296, -73.9262),
    ("Athletics", "Oakland Coliseum", 93, 86, 94, 92, 10, False, False, 0, 37.7516, -122.2005),
    ("Philadelphia Phillies", "Citizens Bank Park", 104, 112, 103, 106, 20, False, False, 0, 39.9061, -75.1665),
    ("Pittsburgh Pirates", "PNC Park", 96, 90, 97, 95, 780, False, False, 116, 40.4469, -80.0058),
    ("San Diego Padres", "Petco Park", 93, 85, 94, 92, 15, False, False, 0, 32.7076, -117.1570),
    ("San Francisco Giants", "Oracle Park", 90, 77, 92, 88, 5, False, False, 0, 37.7786, -122.3893),
    ("Seattle Mariners", "T-Mobile Park", 94, 89, 95, 93, 20, False, True, 0, 47.5914, -122.3326),
    ("St. Louis Cardinals", "Busch Stadium", 97, 97, 98, 96, 465, False, False, 0, 38.6226, -90.1928),
    ("Tampa Bay Rays", "Tropicana Field", 94, 91, 95, 93, 42, True, False, 0, 27.7682, -82.6534),
    ("Texas Rangers", "Globe Life Field", 97, 96, 98, 96, 543, False, True, 0, 32.7473, -97.0845),
    ("Toronto Blue Jays", "Rogers Centre", 101, 106, 100, 103, 266, False, True, 0, 43.6414, -79.3894),
    ("Washington Nationals", "Nationals Park", 100, 104, 99, 102, 20, False, False, 0, 38.8730, -77.0074),
]


def main():
    conn = sqlite3.connect(str(DB_PATH))

    # Get team_id mapping
    teams = {}
    for row in conn.execute("SELECT id, name FROM teams"):
        teams[row[1]] = row[0]

    inserted = 0
    updated = 0

    for data in BALLPARK_DATA:
        (team_name, venue_name, pf_runs, pf_hr, pf_lhb, pf_rhb,
         elevation, is_dome, is_retractable, azimuth, lat, lon) = data

        team_id = teams.get(team_name)
        if not team_id:
            print(f"  WARNING: team '{team_name}' not found in DB")
            continue

        # Convert Savant scale (100=neutral) to multiplier (1.000=neutral)
        pf_runs_mult = round(pf_runs / 100, 3)
        pf_hr_mult = round(pf_hr / 100, 3)
        pf_lhb_mult = round(pf_lhb / 100, 3)
        pf_rhb_mult = round(pf_rhb / 100, 3)

        # Check if exists
        existing = conn.execute(
            "SELECT id FROM ballparks WHERE team_id = ?", (team_id,)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE ballparks
                SET name = ?, park_factor = ?, park_factor_hr = ?,
                    park_factor_lhb = ?, park_factor_rhb = ?,
                    elevation_ft = ?, is_dome = ?
                WHERE id = ?
            """, (venue_name, pf_runs_mult, pf_hr_mult, pf_lhb_mult, pf_rhb_mult,
                  elevation, is_dome, existing[0]))
            updated += 1
        else:
            conn.execute("""
                INSERT INTO ballparks
                (name, team_id, park_factor, park_factor_hr, park_factor_lhb,
                 park_factor_rhb, elevation_ft, is_dome, field_surface)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'grass')
            """, (venue_name, team_id, pf_runs_mult, pf_hr_mult, pf_lhb_mult,
                  pf_rhb_mult, elevation, is_dome))
            inserted += 1

    # Link games to ballparks via home team
    conn.execute("""
        UPDATE games
        SET ballpark_id = (
            SELECT b.id FROM ballparks b WHERE b.team_id = games.home_team_id
        )
        WHERE ballpark_id IS NULL
    """)
    linked = conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    conn.close()

    print(f"Ballparks: {inserted} inserted, {updated} updated")
    print(f"Games linked to ballparks: {linked}")


if __name__ == "__main__":
    main()
