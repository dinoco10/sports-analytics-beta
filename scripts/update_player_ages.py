"""
Update player birth dates from MLB API.
Run once to fix age calculations in projections.
"""
import sys, os, time, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.storage.database import get_session
from src.storage.models import Player
from datetime import datetime

MLB = "https://statsapi.mlb.com/api/v1"
session = get_session()

players = session.query(Player).filter(Player.birth_date == None).all()
print(f"Updating {len(players)} players missing birth dates...\n")

updated = 0
for i, p in enumerate(players):
    try:
        r = requests.get(f"{MLB}/people/{p.mlb_id}", timeout=15)
        if r.status_code == 200:
            data = r.json().get("people", [{}])[0]
            bd = data.get("birthDate")
            if bd:
                p.birth_date = datetime.strptime(bd, "%Y-%m-%d").date()
                p.bats = data.get("batSide", {}).get("code")
                p.throws = data.get("pitchHand", {}).get("code")
                updated += 1
        time.sleep(0.25)
        if (i + 1) % 100 == 0:
            session.commit()
            print(f"  {i+1}/{len(players)} checked, {updated} updated")
    except Exception as e:
        print(f"  Error on {p.name}: {e}")

session.commit()
session.close()
print(f"\nDone: {updated} players updated with birth dates")