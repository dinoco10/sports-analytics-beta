"""
Seed the database with all 30 MLB teams from the API.
Run after initialize_db.py.

Usage:
    python scripts/seed_teams.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.mlb_api import MLBApiClient
from src.storage.database import get_session
from src.storage.models import Team


def seed_teams():
    print("⚾ Seeding MLB teams...")
    
    client = MLBApiClient()
    teams_df = client.get_teams(season=2025)
    
    session = get_session()
    added = 0
    
    try:
        for _, row in teams_df.iterrows():
            existing = session.query(Team).filter_by(mlb_id=row["mlb_id"]).first()
            if not existing:
                team = Team(
                    mlb_id=row["mlb_id"],
                    name=row["name"],
                    abbreviation=row["abbreviation"],
                    league=row["league"],
                    division=row["division"],
                    active=True,
                )
                session.add(team)
                added += 1
                print(f"   ✅ {row['name']}")
            else:
                print(f"   ⏭️  {row['name']} (already exists)")
        
        session.commit()
        print(f"\n✅ Seeded {added} new teams ({len(teams_df)} total)")
    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_teams()