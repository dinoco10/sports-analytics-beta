"""
update_rosters.py -- Daily Roster Tracking
==========================================
Syncs 40-man rosters for all 30 MLB teams, creates daily roster snapshots,
and optionally fetches recent transactions.

Run daily to track:
- Roster moves (trades, DFA, call-ups, option assignments)
- IL stints (IL10, IL60)
- Player status changes

Usage:
    python scripts/update_rosters.py                  # Full sync all 30 teams
    python scripts/update_rosters.py --team NYY       # Single team
    python scripts/update_rosters.py --transactions    # Show recent transactions
    python scripts/update_rosters.py --summary         # Show current roster state

Author: Loko
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.storage.database import engine, get_session
from src.storage.models import Player, Team, RosterSnapshot
from src.ingestion.mlb_api import MLBApiClient

api = MLBApiClient()


# ═══════════════════════════════════════════════════════════════
# ROSTER SYNC
# ═══════════════════════════════════════════════════════════════

def ensure_roster_table():
    """Create roster_snapshots table if it doesn't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS roster_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL REFERENCES players(id),
                team_id INTEGER NOT NULL REFERENCES teams(id),
                date DATE NOT NULL,
                status VARCHAR(20) NOT NULL,
                roster_type VARCHAR(20),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, date)
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_roster_date ON roster_snapshots(date)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_roster_team_date ON roster_snapshots(team_id, date)"
        ))
        conn.commit()


def get_team_map():
    """Build mapping from team abbreviation/name to (internal_id, mlb_id)."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, mlb_id, abbreviation, name FROM teams WHERE active = 1"
        )).fetchall()
    result = {}
    for row in rows:
        result[row[2]] = {'id': row[0], 'mlb_id': row[1], 'name': row[3]}
        result[row[3]] = {'id': row[0], 'mlb_id': row[1], 'name': row[3]}
    return result


def get_player_map():
    """Build mapping from MLB player ID to internal player ID."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, mlb_id FROM players WHERE mlb_id IS NOT NULL"
        )).fetchall()
    return {row[1]: row[0] for row in rows}


def classify_status(api_status):
    """
    Map MLB API status descriptions to our simplified status codes.
    API returns things like "Active", "10-Day Injured List",
    "60-Day Injured List", "Paternity List", etc.
    """
    s = api_status.lower()
    if '60-day' in s or '60-day' in s:
        return 'IL60'
    if '10-day' in s or '15-day' in s or 'injured' in s:
        return 'IL10'
    if 'minor' in s or 'optional' in s or 'outright' in s:
        return 'minors'
    if 'restrict' in s or 'suspend' in s:
        return 'restricted'
    if 'paternity' in s or 'bereavement' in s:
        return 'leave'
    return 'active'


def sync_team_roster(team_info, today, player_map, session):
    """
    Sync one team's 40-man roster: create snapshot rows and update players.
    Returns (new_players, snapshots_created).
    """
    team_id = team_info['id']
    mlb_team_id = team_info['mlb_id']
    team_name = team_info['name']

    # Fetch 40-man roster from MLB API
    roster_df = api.get_roster(mlb_team_id, roster_type="40Man")
    if roster_df.empty:
        return 0, 0

    new_players = 0
    snapshots = 0

    for _, row in roster_df.iterrows():
        mlb_id = row['mlb_id']
        name = row['name']
        position = row.get('position', 'UT')
        status_raw = row.get('status', 'Active')
        status = classify_status(status_raw)

        # Find or create player
        internal_id = player_map.get(mlb_id)
        if internal_id is None:
            # New player not in our DB — add them
            player = Player(
                mlb_id=mlb_id,
                name=name,
                primary_position=position,
                current_team_id=team_id,
                active=True,
            )
            session.add(player)
            session.flush()  # Get the ID
            internal_id = player.id
            player_map[mlb_id] = internal_id
            new_players += 1
        else:
            # Update current team
            session.execute(
                text("UPDATE players SET current_team_id = :tid WHERE id = :pid"),
                {'tid': team_id, 'pid': internal_id}
            )

        # Create or update today's snapshot
        existing = session.query(RosterSnapshot).filter_by(
            player_id=internal_id, date=today,
        ).first()

        if existing:
            existing.team_id = team_id
            existing.status = status
            existing.roster_type = '40Man'
        else:
            snapshot = RosterSnapshot(
                player_id=internal_id,
                team_id=team_id,
                date=today,
                status=status,
                roster_type='40Man',
            )
            session.add(snapshot)
            snapshots += 1

    return new_players, snapshots


def sync_all_rosters(team_filter=None):
    """
    Sync 40-man rosters for all 30 teams (or a single team).
    Creates roster snapshots for today and updates player team assignments.
    """
    today = date.today()
    team_map = get_team_map()
    player_map = get_player_map()

    print(f"\n{'='*60}")
    print(f"  ROSTER SYNC -- {today}")
    print(f"{'='*60}")

    if team_filter:
        if team_filter not in team_map:
            print(f"  Team '{team_filter}' not found. Available: "
                  f"{', '.join(k for k in team_map if len(k) <= 4)}")
            return
        teams_to_sync = [team_map[team_filter]]
    else:
        # All active teams (deduplicate by id)
        seen = set()
        teams_to_sync = []
        for info in team_map.values():
            if info['id'] not in seen:
                teams_to_sync.append(info)
                seen.add(info['id'])

    total_new = 0
    total_snaps = 0

    with get_session() as session:
        for i, team_info in enumerate(teams_to_sync):
            new_p, snaps = sync_team_roster(team_info, today, player_map, session)
            total_new += new_p
            total_snaps += snaps

            if (i + 1) % 5 == 0:
                session.commit()
                print(f"  Synced {i+1}/{len(teams_to_sync)} teams...")

            time.sleep(0.3)  # Rate limit

        session.commit()

    print(f"\n  Done! {total_snaps} snapshots created, {total_new} new players added")
    print(f"  Teams synced: {len(teams_to_sync)}")


# ═══════════════════════════════════════════════════════════════
# TRANSACTIONS
# ═══════════════════════════════════════════════════════════════

def fetch_recent_transactions(days_back=7):
    """Fetch and display recent MLB transactions."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    print(f"\n{'='*60}")
    print(f"  RECENT TRANSACTIONS ({start_date} to {end_date})")
    print(f"{'='*60}")

    txns = api.get_transactions(start_date, end_date)
    if txns.empty:
        print("  No transactions found.")
        return

    # Filter to interesting transaction types
    interesting = txns[txns['type'].isin([
        'Trade', 'Signing', 'Free Agency', 'Waiver Claim',
        'Designated for Assignment', 'Released',
        'Status Change', 'Roster Move',
    ])] if 'type' in txns.columns else txns

    if interesting.empty:
        interesting = txns

    print(f"\n  {len(interesting)} transactions found:\n")
    for _, row in interesting.head(30).iterrows():
        player = row.get('player', 'Unknown')
        txn_type = row.get('type', 'Unknown')
        from_team = row.get('from_team', '')
        to_team = row.get('to_team', '')
        txn_date = row.get('date', '')

        team_info = f"{from_team} -> {to_team}" if from_team and to_team else (to_team or from_team or '')
        print(f"  {txn_date[:10] if txn_date else 'N/A':>10}  {txn_type:<30}  {player:<25}  {team_info}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def show_roster_summary():
    """Print current roster state from the most recent snapshot."""
    print(f"\n{'='*60}")
    print("  ROSTER SNAPSHOT SUMMARY")
    print(f"{'='*60}")

    with engine.connect() as conn:
        # Check if table exists
        tables = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roster_snapshots'"
        )).fetchall()
        if not tables:
            print("  No roster_snapshots table yet. Run sync first.")
            return

        # Most recent date
        latest = conn.execute(text(
            "SELECT MAX(date) FROM roster_snapshots"
        )).scalar()

        if not latest:
            print("  No roster snapshots in database yet.")
            return

        print(f"\n  Latest snapshot: {latest}")

        # Counts by status
        status_counts = conn.execute(text("""
            SELECT status, COUNT(*) as cnt
            FROM roster_snapshots
            WHERE date = :d
            GROUP BY status
            ORDER BY cnt DESC
        """), {'d': latest}).fetchall()

        print(f"\n  Status breakdown:")
        total = 0
        for status, cnt in status_counts:
            print(f"    {status:<15} {cnt:>4}")
            total += cnt
        print(f"    {'TOTAL':<15} {total:>4}")

        # Per-team counts
        team_counts = conn.execute(text("""
            SELECT t.abbreviation, t.name,
                   COUNT(*) as total,
                   SUM(CASE WHEN rs.status = 'active' THEN 1 ELSE 0 END) as active_ct,
                   SUM(CASE WHEN rs.status LIKE 'IL%' THEN 1 ELSE 0 END) as il_ct
            FROM roster_snapshots rs
            JOIN teams t ON rs.team_id = t.id
            WHERE rs.date = :d
            GROUP BY t.id
            ORDER BY t.abbreviation
        """), {'d': latest}).fetchall()

        print(f"\n  {'Team':<5} {'Name':<25} {'40-Man':>6} {'Active':>7} {'IL':>4}")
        print(f"  {'-'*4:<5} {'-'*24:<25} {'-'*5:>6} {'-'*6:>7} {'-'*3:>4}")
        for abbr, name, total, active, il in team_counts:
            print(f"  {abbr:<5} {name:<25} {total:>6} {active:>7} {il:>4}")

        # Historical coverage
        date_count = conn.execute(text(
            "SELECT COUNT(DISTINCT date) FROM roster_snapshots"
        )).scalar()
        print(f"\n  Total snapshot dates: {date_count}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily roster tracking — sync 40-man rosters and transactions"
    )
    parser.add_argument("--team", type=str, help="Sync single team (abbreviation, e.g. NYY)")
    parser.add_argument("--transactions", action="store_true",
                        help="Show recent transactions (last 7 days)")
    parser.add_argument("--days", type=int, default=7,
                        help="Days back for transactions (default: 7)")
    parser.add_argument("--summary", action="store_true",
                        help="Show current roster state")
    args = parser.parse_args()

    # Ensure table exists
    ensure_roster_table()

    if args.summary:
        show_roster_summary()
    elif args.transactions:
        fetch_recent_transactions(days_back=args.days)
    else:
        sync_all_rosters(team_filter=args.team)
        show_roster_summary()
