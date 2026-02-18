"""
Roster Impact Analysis — Player-Level Team Projections
Pulls CURRENT 2026 rosters, matches to 2025 stats, shows impact of moves.

Usage:
    python -m src.features.roster_impact
    python -m src.features.roster_impact --team "New York Yankees"
"""

import sys, os, argparse, time
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from src.storage.database import engine, get_session
from src.storage.models import Team, Player

MLB = "https://statsapi.mlb.com/api/v1"


def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{MLB}/{endpoint}", params=params, timeout=30)
        r.raise_for_status(); time.sleep(0.3); return r.json()
    except Exception as e:
        print(f"API error: {e}"); return {}


# ═══════════════════════════════════════════════════════════════
# STEP 1: FETCH CURRENT 2026 ROSTERS
# ═══════════════════════════════════════════════════════════════

def fetch_current_rosters():
    """Pull current roster for all 30 teams from MLB API."""
    print("Fetching current 2026 rosters from MLB API...")

    teams_data = api_get("teams", {"sportId": 1, "season": 2026})
    all_players = []

    for t in teams_data.get("teams", []):
        team_name = t["name"]
        team_mlb_id = t["id"]
        abbr = t.get("abbreviation", "")

        # Get 40-man roster (includes spring training)
        roster = api_get(f"teams/{team_mlb_id}/roster", {
            "rosterType": "40Man",
            "season": 2026
        })

        for p in roster.get("roster", []):
            person = p.get("person", {})
            pos = p.get("position", {})
            status = p.get("status", {})

            all_players.append({
                "mlb_player_id": person.get("id"),
                "player_name": person.get("fullName", "Unknown"),
                "current_team": team_name,
                "current_team_abbr": abbr,
                "current_team_mlb_id": team_mlb_id,
                "position": pos.get("abbreviation", ""),
                "position_type": pos.get("type", ""),
                "status": status.get("description", "Active"),
            })

        print(f"  {abbr:4s} {team_name:30s} — {len(roster.get('roster', []))} players")
        time.sleep(0.3)

    df = pd.DataFrame(all_players)
    print(f"\nTotal players on 40-man rosters: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: GET 2025 PLAYER STATS FROM OUR DATABASE
# ═══════════════════════════════════════════════════════════════

def load_player_pitching_2025():
    """Aggregate 2025 pitching stats per player from our database."""
    query = """
        SELECT
            p.mlb_id as mlb_player_id,
            p.name as player_name,
            t.name as team_2025,
            t.abbreviation as team_2025_abbr,
            SUM(ps.ip) as ip,
            SUM(ps.earned_runs) as er,
            SUM(ps.hits) as hits,
            SUM(ps.walks) as bb,
            SUM(ps.strikeouts) as k,
            SUM(ps.home_runs) as hr,
            SUM(ps.pitches) as pitches,
            SUM(ps.runs) as runs,
            COUNT(DISTINCT ps.game_id) as games
        FROM pitching_game_stats ps
        JOIN players p ON ps.player_id = p.id
        JOIN teams t ON ps.team_id = t.id
        WHERE strftime('%Y', ps.date) = '2025'
        GROUP BY p.mlb_id, p.name, t.name, t.abbreviation
        HAVING SUM(ps.ip) > 0
    """
    df = pd.read_sql(query, engine)

    # Calculate rate stats
    df['era'] = (df['er'] / df['ip']) * 9
    df['whip'] = (df['hits'] + df['bb']) / df['ip']
    df['k_per_9'] = (df['k'] / df['ip']) * 9
    df['bb_per_9'] = (df['bb'] / df['ip']) * 9

    # Approximate batters faced
    df['bf'] = df['ip'] * 3 + df['hits'] + df['bb']
    df['k_pct'] = df['k'] / df['bf'] * 100
    df['bb_pct'] = df['bb'] / df['bf'] * 100
    df['k_bb_pct'] = df['k_pct'] - df['bb_pct']

    # FIP
    fip_const = 3.10
    df['fip'] = ((13 * df['hr'] + 3 * df['bb'] - 2 * df['k']) / df['ip']) + fip_const

    # Simplified pitcher WAR approximation
    # WAR ≈ (league_avg_FIP - pitcher_FIP) / runs_per_win * (IP/9)
    # runs_per_win ≈ 10, league FIP ≈ 4.20
    league_fip = 4.20
    df['war_approx'] = ((league_fip - df['fip']) / 10) * (df['ip'] / 9)
    df['war_approx'] = df['war_approx'].clip(-3, 10)

    # Role classification
    df['role'] = df['ip'].apply(lambda x: 'SP' if x >= 50 else 'RP')

    return df


def load_player_hitting_2025():
    """Aggregate 2025 hitting stats per player from our database."""
    query = """
        SELECT
            p.mlb_id as mlb_player_id,
            p.name as player_name,
            t.name as team_2025,
            t.abbreviation as team_2025_abbr,
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
        WHERE strftime('%Y', hs.date) = '2025'
        GROUP BY p.mlb_id, p.name, t.name, t.abbreviation
        HAVING SUM(hs.plate_appearances) >= 10
    """
    df = pd.read_sql(query, engine)

    # Rate stats
    df['avg'] = df['h'] / df['ab'].replace(0, 1)
    df['obp'] = (df['h'] + df['bb']) / df['pa'].replace(0, 1)
    singles = df['h'] - df['doubles'] - df['triples'] - df['hr']
    slg_num = singles + 2*df['doubles'] + 3*df['triples'] + 4*df['hr']
    df['slg'] = slg_num / df['ab'].replace(0, 1)
    df['ops'] = df['obp'] + df['slg']

    # wOBA (2024 weights)
    df['woba'] = (0.69*df['bb'] + 0.88*singles + 1.24*df['doubles'] +
                  1.56*df['triples'] + 2.01*df['hr']) / df['pa'].replace(0, 1)

    # K% and BB%
    df['k_pct'] = df['k'] / df['pa'] * 100
    df['bb_pct'] = df['bb'] / df['pa'] * 100

    # Simplified position WAR approximation
    # WAR ≈ (wOBA - league_wOBA) / wOBA_scale * PA / PA_per_win
    # + positional adjustment (simplified)
    league_woba = 0.310
    woba_scale = 1.15
    runs_per_win = 10
    df['batting_runs'] = ((df['woba'] - league_woba) / woba_scale) * df['pa']
    df['war_approx'] = df['batting_runs'] / runs_per_win
    df['war_approx'] = df['war_approx'].clip(-3, 10)

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: MATCH CURRENT ROSTER TO 2025 STATS
# ═══════════════════════════════════════════════════════════════

def match_rosters_to_stats(current_rosters, pitching_2025, hitting_2025):
    """Join current team assignments with 2025 performance."""

    # Pitchers: match by mlb_player_id
    pitchers = current_rosters[current_rosters['position_type'] == 'Pitcher'].copy()
    pitchers = pitchers.merge(
        pitching_2025[['mlb_player_id', 'team_2025', 'team_2025_abbr',
                        'ip', 'era', 'fip', 'k_bb_pct', 'war_approx', 'role', 'games']],
        on='mlb_player_id', how='left'
    )
    pitchers['changed_team'] = pitchers.apply(
        lambda r: r['current_team'] != r['team_2025'] if pd.notna(r['team_2025']) else False,
        axis=1
    )

    # Hitters: match by mlb_player_id
    hitters = current_rosters[current_rosters['position_type'] != 'Pitcher'].copy()
    hitters = hitters.merge(
        hitting_2025[['mlb_player_id', 'team_2025', 'team_2025_abbr',
                       'pa', 'avg', 'obp', 'ops', 'woba', 'hr', 'sb',
                       'war_approx', 'games']],
        on='mlb_player_id', how='left'
    )
    hitters['changed_team'] = hitters.apply(
        lambda r: r['current_team'] != r['team_2025'] if pd.notna(r['team_2025']) else False,
        axis=1
    )

    return pitchers, hitters


# ═══════════════════════════════════════════════════════════════
# STEP 4: CALCULATE TEAM ROSTER VALUE
# ═══════════════════════════════════════════════════════════════

def calculate_team_roster_value(pitchers, hitters):
    """
    Aggregate player WAR by CURRENT team to get roster strength.
    Compare to their 2025 team assignments to find net gains/losses.
    """
    results = []

    all_teams = sorted(set(pitchers['current_team'].unique()) |
                       set(hitters['current_team'].unique()))

    for team in all_teams:
        tp = pitchers[pitchers['current_team'] == team]
        th = hitters[hitters['current_team'] == team]

        # Current roster WAR
        pitch_war = tp['war_approx'].sum()
        hit_war = th['war_approx'].sum()
        total_war = pitch_war + hit_war

        # Players gained (changed_team == True, now on THIS team)
        gained_p = tp[tp['changed_team'] == True]
        gained_h = th[th['changed_team'] == True]
        war_gained = gained_p['war_approx'].sum() + gained_h['war_approx'].sum()

        # Players lost: were on this team in 2025, now elsewhere
        lost_p = pitchers[(pitchers['team_2025'] == team) &
                          (pitchers['current_team'] != team) &
                          (pitchers['team_2025'].notna())]
        lost_h = hitters[(hitters['team_2025'] == team) &
                         (hitters['current_team'] != team) &
                         (hitters['team_2025'].notna())]
        war_lost = lost_p['war_approx'].sum() + lost_h['war_approx'].sum()

        net_war = war_gained - war_lost

        # Top players on current roster
        top_p = tp.nlargest(3, 'war_approx')[['player_name', 'war_approx', 'role']].values.tolist()
        top_h = th.nlargest(3, 'war_approx')[['player_name', 'war_approx']].values.tolist()

        # Key acquisitions
        key_gains = []
        for _, r in gained_p.nlargest(3, 'war_approx').iterrows():
            if r['war_approx'] > 0:
                key_gains.append(f"{r['player_name']} ({r['war_approx']:.1f} WAR from {r['team_2025_abbr']})")
        for _, r in gained_h.nlargest(3, 'war_approx').iterrows():
            if r['war_approx'] > 0:
                key_gains.append(f"{r['player_name']} ({r['war_approx']:.1f} WAR from {r['team_2025_abbr']})")

        # Key losses
        key_losses = []
        for _, r in lost_p.nlargest(3, 'war_approx').iterrows():
            if r['war_approx'] > 0:
                key_losses.append(f"{r['player_name']} ({r['war_approx']:.1f} WAR to {r['current_team']})")
        for _, r in lost_h.nlargest(3, 'war_approx').iterrows():
            if r['war_approx'] > 0:
                key_losses.append(f"{r['player_name']} ({r['war_approx']:.1f} WAR to {r['current_team']})")

        # SP depth
        sp = tp[tp['role'] == 'SP'].nlargest(5, 'war_approx')
        sp_war = sp['war_approx'].sum()
        sp_names = sp['player_name'].tolist()

        # Lineup strength (top 9 hitters by WAR)
        lineup = th.nlargest(9, 'war_approx')
        lineup_war = lineup['war_approx'].sum()
        lineup_ops = lineup['ops'].mean() if not lineup.empty else 0

        # Win impact estimate: ~10 WAR = 1 win above replacement baseline
        # Baseline team ≈ 48 wins (replacement level)
        projected_wins_from_roster = 48 + total_war

        results.append({
            'team': team,
            'total_war': round(total_war, 1),
            'pitch_war': round(pitch_war, 1),
            'hit_war': round(hit_war, 1),
            'war_gained': round(war_gained, 1),
            'war_lost': round(war_lost, 1),
            'net_war_change': round(net_war, 1),
            'projected_wins_roster': round(projected_wins_from_roster, 0),
            'sp_war': round(sp_war, 1),
            'sp_names': sp_names,
            'lineup_war': round(lineup_war, 1),
            'lineup_ops': round(lineup_ops, 3),
            'roster_size_p': len(tp),
            'roster_size_h': len(th),
            'key_gains': key_gains,
            'key_losses': key_losses,
            'top_pitchers': top_p,
            'top_hitters': top_h,
        })

    return pd.DataFrame(results).sort_values('total_war', ascending=False)


# ═══════════════════════════════════════════════════════════════
# STEP 5: DISPLAY
# ═══════════════════════════════════════════════════════════════

def display_results(team_values, team_filter=None):
    """Print roster analysis."""

    if team_filter:
        team_values = team_values[team_values['team'].str.contains(team_filter, case=False)]

    print(f"\n{'=' * 70}")
    print("2026 ROSTER IMPACT ANALYSIS (based on 2025 player WAR)")
    print(f"{'=' * 70}")

    # Offseason winners and losers
    sorted_by_net = team_values.sort_values('net_war_change', ascending=False)

    print(f"\n  OFFSEASON WINNERS (most WAR gained):")
    for _, r in sorted_by_net.head(5).iterrows():
        print(f"    {r['team']:30s} Net: {r['net_war_change']:+.1f} WAR "
              f"(+{r['war_gained']:.1f} gained, -{r['war_lost']:.1f} lost)")

    print(f"\n  OFFSEASON LOSERS (most WAR lost):")
    for _, r in sorted_by_net.tail(5).iterrows():
        print(f"    {r['team']:30s} Net: {r['net_war_change']:+.1f} WAR "
              f"(+{r['war_gained']:.1f} gained, -{r['war_lost']:.1f} lost)")

    # Full team breakdown
    print(f"\n{'=' * 70}")
    print("TEAM-BY-TEAM ROSTER BREAKDOWN")
    print(f"{'=' * 70}")

    for _, row in team_values.iterrows():
        print(f"\n  {row['team']}")
        print(f"    Total WAR: {row['total_war']:.1f} "
              f"(Pitching: {row['pitch_war']:.1f} | Hitting: {row['hit_war']:.1f})")
        print(f"    Roster projection: ~{row['projected_wins_roster']:.0f} wins")
        print(f"    Lineup OPS: {row['lineup_ops']:.3f} | SP WAR: {row['sp_war']:.1f}")

        if row['net_war_change'] != 0:
            direction = "IMPROVED" if row['net_war_change'] > 0 else "WEAKENED"
            print(f"    Offseason: {direction} by {row['net_war_change']:+.1f} WAR")

        if row['key_gains']:
            print(f"    Gained: {' | '.join(row['key_gains'][:3])}")
        if row['key_losses']:
            print(f"    Lost:   {' | '.join(row['key_losses'][:3])}")

        # Top pitchers
        if row['top_pitchers']:
            names = [f"{n[0]} ({n[1]:.1f}W, {n[2]})" for n in row['top_pitchers']]
            print(f"    Top P:  {' | '.join(names)}")

        # Top hitters
        if row['top_hitters']:
            names = [f"{n[0]} ({n[1]:.1f}W)" for n in row['top_hitters']]
            print(f"    Top H:  {' | '.join(names)}")

    # Save
    os.makedirs('data/features', exist_ok=True)
    save_cols = ['team', 'total_war', 'pitch_war', 'hit_war',
                 'war_gained', 'war_lost', 'net_war_change',
                 'projected_wins_roster', 'sp_war', 'lineup_war', 'lineup_ops']
    team_values[save_cols].to_csv('data/features/roster_impact_2026.csv', index=False)
    print(f"\nSaved: data/features/roster_impact_2026.csv")

    return team_values


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_roster_analysis(team_filter=None):
    """Full roster impact pipeline."""

    # Step 1: Current rosters
    rosters = fetch_current_rosters()

    # Step 2: 2025 stats
    print("\nLoading 2025 player stats from database...")
    pitching_2025 = load_player_pitching_2025()
    hitting_2025 = load_player_hitting_2025()
    print(f"  Pitchers with 2025 data: {len(pitching_2025)}")
    print(f"  Hitters with 2025 data:  {len(hitting_2025)}")

    # Step 3: Match
    print("\nMatching current rosters to 2025 stats...")
    pitchers, hitters = match_rosters_to_stats(rosters, pitching_2025, hitting_2025)

    matched_p = pitchers['war_approx'].notna().sum()
    matched_h = hitters['war_approx'].notna().sum()
    print(f"  Pitchers matched: {matched_p}/{len(pitchers)}")
    print(f"  Hitters matched:  {matched_h}/{len(hitters)}")

    changed_p = pitchers['changed_team'].sum()
    changed_h = hitters['changed_team'].sum()
    print(f"  Changed teams: {changed_p} pitchers, {changed_h} hitters")

    # Step 4: Calculate team values
    print("\nCalculating team roster values...")
    team_values = calculate_team_roster_value(pitchers, hitters)

    # Step 5: Display
    display_results(team_values, team_filter)

    return team_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 Roster Impact Analysis")
    parser.add_argument("--team", type=str, help="Filter to one team")
    args = parser.parse_args()

    run_roster_analysis(team_filter=args.team)