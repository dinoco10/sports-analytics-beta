"""
Roster Impact Analysis — Player-Level Team Projections
Uses Marcel projections (not raw 2025 stats) to show 2026 team WAR.
Shows offseason winners/losers by comparing current team to 2025 team.

Usage:
    python -m src.features.roster_impact
    python -m src.features.roster_impact --team "New York Yankees"
"""

import sys, os, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.storage.database import engine
from src.features.player_projections import get_all_projections


# ═══════════════════════════════════════════════════════════════
# STEP 1: GET 2025 TEAM ASSIGNMENTS (for offseason comparison only)
# ═══════════════════════════════════════════════════════════════

def load_2025_team_assignments():
    """
    Load which team each player was on in 2025.

    We ONLY need this for the offseason comparison: who moved where,
    and what was the net WAR gain/loss. The actual WAR values come
    from Marcel projections (get_all_projections), not raw 2025 stats.
    """
    pitcher_query = """
        SELECT
            p.mlb_id as mlb_player_id,
            t.name as team_2025,
            t.abbreviation as team_2025_abbr
        FROM pitching_game_stats ps
        JOIN players p ON ps.player_id = p.id
        JOIN teams t ON ps.team_id = t.id
        WHERE strftime('%Y', ps.date) = '2025'
        GROUP BY p.mlb_id, t.name, t.abbreviation
        HAVING SUM(ps.ip) >= 5
    """
    hitter_query = """
        SELECT
            p.mlb_id as mlb_player_id,
            t.name as team_2025,
            t.abbreviation as team_2025_abbr
        FROM hitting_game_stats hs
        JOIN players p ON hs.player_id = p.id
        JOIN teams t ON hs.team_id = t.id
        WHERE strftime('%Y', hs.date) = '2025'
        GROUP BY p.mlb_id, t.name, t.abbreviation
        HAVING SUM(hs.plate_appearances) >= 10
    """
    pitchers_2025 = pd.read_sql(pitcher_query, engine)
    hitters_2025 = pd.read_sql(hitter_query, engine)
    return pitchers_2025, hitters_2025


# ═══════════════════════════════════════════════════════════════
# STEP 2: MERGE PROJECTIONS WITH 2025 TEAM INFO
# ═══════════════════════════════════════════════════════════════

def attach_2025_teams(p_df, h_df, pitchers_2025, hitters_2025):
    """
    Join Marcel projections with 2025 team assignments.

    p_df / h_df come from get_all_projections() — they already have
    'current_team' (2026 team). We add 'team_2025' so we can see who moved.
    """
    pitchers = p_df.merge(
        pitchers_2025[['mlb_player_id', 'team_2025', 'team_2025_abbr']],
        on='mlb_player_id', how='left'
    )
    pitchers['changed_team'] = pitchers.apply(
        lambda r: r['current_team'] != r['team_2025'] if pd.notna(r['team_2025']) else False,
        axis=1
    )

    hitters = h_df.merge(
        hitters_2025[['mlb_player_id', 'team_2025', 'team_2025_abbr']],
        on='mlb_player_id', how='left'
    )
    hitters['changed_team'] = hitters.apply(
        lambda r: r['current_team'] != r['team_2025'] if pd.notna(r['team_2025']) else False,
        axis=1
    )

    return pitchers, hitters


# ═══════════════════════════════════════════════════════════════
# STEP 3: CALCULATE TEAM ROSTER VALUE (using Marcel proj_war)
# ═══════════════════════════════════════════════════════════════

def calculate_team_roster_value(pitchers, hitters):
    """
    Aggregate Marcel projected WAR by CURRENT 2026 team.

    Uses proj_war (Marcel aging-adjusted) instead of raw 2025 war_approx.
    This gives us a forward-looking view: what is each roster worth in 2026?

    Net WAR change compares current roster WAR vs what they had in 2025.
    """
    results = []

    all_teams = sorted(set(pitchers['current_team'].unique()) |
                       set(hitters['current_team'].unique()))

    for team in all_teams:
        if team == 'Free Agent':
            continue

        tp = pitchers[pitchers['current_team'] == team]
        th = hitters[hitters['current_team'] == team]

        # Current roster projected WAR (Marcel, aging-adjusted)
        pitch_war = tp['proj_war'].sum()
        hit_war = th['proj_war'].sum()
        total_war = pitch_war + hit_war

        # Players gained this offseason (changed_team == True, now on THIS team)
        gained_p = tp[tp['changed_team'] == True]
        gained_h = th[th['changed_team'] == True]
        war_gained = gained_p['proj_war'].sum() + gained_h['proj_war'].sum()

        # Players lost: were on this team in 2025, now elsewhere
        lost_p = pitchers[(pitchers['team_2025'] == team) &
                          (pitchers['current_team'] != team) &
                          (pitchers['team_2025'].notna())]
        lost_h = hitters[(hitters['team_2025'] == team) &
                         (hitters['current_team'] != team) &
                         (hitters['team_2025'].notna())]
        war_lost = lost_p['proj_war'].sum() + lost_h['proj_war'].sum()

        net_war = war_gained - war_lost

        # Top players on current roster
        top_p = tp.nlargest(3, 'proj_war')[['player_name', 'proj_war', 'role']].values.tolist()
        top_h = th.nlargest(3, 'proj_war')[['player_name', 'proj_war']].values.tolist()

        # Key acquisitions (players who joined + have positive projected WAR)
        key_gains = []
        for _, r in gained_p.nlargest(3, 'proj_war').iterrows():
            if r['proj_war'] > 0 and pd.notna(r.get('team_2025_abbr')):
                key_gains.append(f"{r['player_name']} ({r['proj_war']:.1f} WAR from {r['team_2025_abbr']})")
        for _, r in gained_h.nlargest(3, 'proj_war').iterrows():
            if r['proj_war'] > 0 and pd.notna(r.get('team_2025_abbr')):
                key_gains.append(f"{r['player_name']} ({r['proj_war']:.1f} WAR from {r['team_2025_abbr']})")

        # Key losses
        key_losses = []
        for _, r in lost_p.nlargest(3, 'proj_war').iterrows():
            if r['proj_war'] > 0:
                key_losses.append(f"{r['player_name']} ({r['proj_war']:.1f} WAR to {r['current_team']})")
        for _, r in lost_h.nlargest(3, 'proj_war').iterrows():
            if r['proj_war'] > 0:
                key_losses.append(f"{r['player_name']} ({r['proj_war']:.1f} WAR to {r['current_team']})")

        # SP depth (top 5 by projected WAR)
        sp = tp[tp['role'] == 'SP'].nlargest(5, 'proj_war')
        sp_war = sp['proj_war'].sum()
        sp_names = sp['player_name'].tolist()

        # Lineup (top 9 hitters by projected WAR)
        lineup = th.nlargest(9, 'proj_war')
        lineup_war = lineup['proj_war'].sum()
        lineup_woba = lineup['proj_woba'].mean() if not lineup.empty else 0

        # 48 wins = replacement-level team baseline
        projected_wins = 48 + total_war

        results.append({
            'team': team,
            'total_war': round(total_war, 1),
            'pitch_war': round(pitch_war, 1),
            'hit_war': round(hit_war, 1),
            'war_gained': round(war_gained, 1),
            'war_lost': round(war_lost, 1),
            'net_war_change': round(net_war, 1),
            'projected_wins_roster': round(projected_wins, 0),
            'sp_war': round(sp_war, 1),
            'sp_names': sp_names,
            'lineup_war': round(lineup_war, 1),
            'lineup_woba': round(lineup_woba, 3),
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
        print(f"    Lineup wOBA: {row['lineup_woba']:.3f} | SP WAR: {row['sp_war']:.1f}")

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
                 'projected_wins_roster', 'sp_war', 'lineup_war', 'lineup_woba']
    team_values[save_cols].to_csv('data/features/roster_impact_2026.csv', index=False)
    print(f"\nSaved: data/features/roster_impact_2026.csv")

    return team_values


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_roster_analysis(team_filter=None):
    """
    Full roster impact pipeline using Marcel projections.

    New flow:
      1. Marcel projections → proj_war per player + current 2026 team
      2. DB query → which team each player was on in 2025
      3. Merge → detect who moved and net WAR gain/loss
      4. Aggregate by current team → projected wins
    """
    print("=" * 70)
    print("2026 ROSTER IMPACT ANALYSIS (Marcel Projections + Offseason Moves)")
    print("=" * 70)

    # Step 1: Marcel projections with 2026 team assignments (API + DB)
    print("\nRunning Marcel projections...")
    p_df, h_df = get_all_projections(verbose=True)

    # Step 2: 2025 team assignments (DB only — just for offseason comparison)
    print("\nLoading 2025 team assignments for offseason comparison...")
    pitchers_2025, hitters_2025 = load_2025_team_assignments()
    print(f"  Pitchers with 2025 data: {len(pitchers_2025)}")
    print(f"  Hitters with 2025 data:  {len(hitters_2025)}")

    # Step 3: Attach 2025 team info to Marcel projections
    print("\nMatching projections to 2025 teams...")
    pitchers, hitters = attach_2025_teams(p_df, h_df, pitchers_2025, hitters_2025)

    changed_p = pitchers['changed_team'].sum()
    changed_h = hitters['changed_team'].sum()
    print(f"  Changed teams: {changed_p} pitchers, {changed_h} hitters")

    # Step 4: Aggregate Marcel proj_war by current team
    print("\nCalculating team roster values (Marcel WAR)...")
    team_values = calculate_team_roster_value(pitchers, hitters)

    # Step 5: Display
    display_results(team_values, team_filter)

    return team_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 Roster Impact Analysis")
    parser.add_argument("--team", type=str, help="Filter to one team")
    args = parser.parse_args()

    run_roster_analysis(team_filter=args.team)