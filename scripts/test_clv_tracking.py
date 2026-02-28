"""
test_clv_tracking.py — Test CLV Tracking Workflow
==================================================
Demonstrates the full CLV tracking workflow with historical data.

This script simulates a complete day:
1. Morning: Log predictions + opening lines
2. Evening: Fetch closing lines + calculate CLV
3. Next day: Settle games + report results

Usage:
    python scripts/test_clv_tracking.py

Author: Loko
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "mlb_analytics.db"

def find_test_date():
    """Find a suitable date with both games and odds data."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.execute("""
        SELECT g.date, COUNT(DISTINCT g.id) as games
        FROM games g
        JOIN odds o ON g.date = o.date
        WHERE g.season = 2025 AND g.date >= '2025-03-27'
        GROUP BY g.date
        HAVING games > 5
        ORDER BY g.date
        LIMIT 1
    """)
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        print("ERROR: No suitable test date found (need games + odds)")
        return None


def run_command(cmd, description):
    """Run a CLI command and print output."""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"  ERROR: Command failed with exit code {result.returncode}")
        return False

    return True


def main():
    print("\n" + "=" * 70)
    print("  CLV TRACKING — END-TO-END TEST")
    print("=" * 70)

    # Find test date
    test_date = find_test_date()
    if not test_date:
        return

    print(f"\n  Test date: {test_date}")
    print(f"  This date has games + odds data for a realistic test")

    # Initialize CLV table
    print("\n  Step 0: Initialize CLV tracking table...")
    if not run_command(
        ["python", "scripts/track_clv.py", "--init"],
        "Initialize CLV tracking table"
    ):
        return

    # Step 1: Morning run
    if not run_command(
        ["python", "scripts/track_clv.py", "--morning", "--date", test_date],
        f"STEP 1: Morning Run — Log predictions + opening lines ({test_date})"
    ):
        return

    # Step 2: Evening run
    if not run_command(
        ["python", "scripts/track_clv.py", "--closing", "--date", test_date],
        f"STEP 2: Evening Run — Fetch closing lines + calculate CLV ({test_date})"
    ):
        return

    # Step 3: Settlement
    if not run_command(
        ["python", "scripts/track_clv.py", "--settle", "--date", test_date],
        f"STEP 3: Settlement — Mark winners + calculate P/L ({test_date})"
    ):
        return

    # Step 4: Report
    if not run_command(
        ["python", "scripts/track_clv.py", "--report"],
        "STEP 4: Report — Show CLV statistics"
    ):
        return

    print("\n" + "=" * 70)
    print("  TEST COMPLETE!")
    print("=" * 70)
    print("\n  Next steps:")
    print("  1. Review the report output above")
    print("  2. Check clv_tracking table in DB:")
    print("     sqlite3 data/mlb_analytics.db 'SELECT * FROM clv_tracking LIMIT 5'")
    print("  3. Run for more dates to build CLV history")
    print("  4. Use --report --last 30 to see rolling trends")


if __name__ == "__main__":
    main()
