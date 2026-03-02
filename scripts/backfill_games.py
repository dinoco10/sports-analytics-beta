"""
Backfill 2024-2025 MLB games + box scores into database.
Usage:
    python scripts/backfill_games.py --season 2025
    python scripts/backfill_games.py                  (both seasons)
    python scripts/backfill_games.py --summary        (check DB)
"""
import sys, os, time, argparse, logging, requests
from datetime import date, datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import text
from src.storage.database import engine, get_session
from src.storage.models import Team, Player, Game, PitchingGameStats, HittingGameStats

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
MLB = "https://statsapi.mlb.com/api/v1"

def api_get(endpoint, params=None):
    for attempt in range(3):
        try:
            r = requests.get(f"{MLB}/{endpoint}", params=params, timeout=30)
            r.raise_for_status(); time.sleep(0.3); return r.json()
        except Exception as e:
            if attempt < 2: time.sleep(2 ** attempt)
            else: print(f"FAIL: {endpoint} - {e}"); return {}

def get_or_create_player(session, mlb_id, name, pos="Unknown"):
    p = session.query(Player).filter_by(mlb_id=mlb_id).first()
    if not p:
        p = Player(mlb_id=mlb_id, name=name, primary_position=pos, active=True)
        session.add(p); session.flush()
    return p.id

def get_team_db_id(session, mlb_id):
    t = session.query(Team).filter_by(mlb_id=mlb_id).first()
    return t.id if t else None

def parse_ip(ip_str):
    try:
        parts = str(ip_str).split('.')
        return int(parts[0]) + (int(parts[1]) / 3.0 if len(parts) > 1 else 0)
    except: return 0.0

def fetch_schedule(start, end):
    games = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=29), end)
        data = api_get("schedule", {
            "sportId": 1, "startDate": cur.strftime("%Y-%m-%d"),
            "endDate": chunk_end.strftime("%Y-%m-%d"),
            "gameType": "R,F,D,L,W", "hydrate": "linescore,probablePitcher"
        })
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("detailedState") in ("Final", "Completed Early"):
                    games.append(g)
        cur = chunk_end + timedelta(days=1)
    return games

def store_game(session, g):
    gid = g["gamePk"]
    if session.query(Game).filter_by(mlb_game_id=gid).first():
        return None
    gdate = datetime.strptime(g["gameDate"][:10], "%Y-%m-%d").date()
    home, away = g["teams"]["home"], g["teams"]["away"]
    htid = get_team_db_id(session, home["team"]["id"])
    atid = get_team_db_id(session, away["team"]["id"])
    hs, aws = home.get("score", 0), away.get("score", 0)
    wid = htid if hs > aws else (atid if aws > hs else None)
    inn = g.get("linescore", {}).get("currentInning", 9)
    hsid = asid = None
    hp = home.get("probablePitcher", {})
    ap = away.get("probablePitcher", {})
    if hp.get("id"): hsid = get_or_create_player(session, hp["id"], hp.get("fullName","?"), "P")
    if ap.get("id"): asid = get_or_create_player(session, ap["id"], ap.get("fullName","?"), "P")
    game = Game(mlb_game_id=gid, date=gdate, season=gdate.year,
        home_team_id=htid, away_team_id=atid, home_starter_id=hsid, away_starter_id=asid,
        home_score=hs, away_score=aws, winner_id=wid, innings=inn, day_night=g.get("dayNight","night"))
    session.add(game); session.flush()
    return game.id

def store_box(session, game_db_id, game_date, mlb_gid):
    box = api_get(f"game/{mlb_gid}/boxscore")
    if not box: return 0, 0
    pc, hc = 0, 0
    for side in ["home", "away"]:
        td = box.get("teams", {}).get(side, {})
        tid = get_team_db_id(session, td.get("team", {}).get("id"))
        for _, pd in td.get("players", {}).items():
            per = pd.get("person", {}); mid = per.get("id")
            if not mid: continue
            nm = per.get("fullName", "?"); pos = pd.get("position", {}).get("abbreviation", "")
            st = pd.get("stats", {})
            pit = st.get("pitching", {})
            if pit and pit.get("inningsPitched"):
                pid = get_or_create_player(session, mid, nm, "P")
                if not session.query(PitchingGameStats).filter_by(game_id=game_db_id, player_id=pid).first():
                    session.add(PitchingGameStats(game_id=game_db_id, player_id=pid, team_id=tid, date=game_date,
                        ip=parse_ip(pit.get("inningsPitched","0")), hits=int(pit.get("hits",0)),
                        runs=int(pit.get("runs",0)), earned_runs=int(pit.get("earnedRuns",0)),
                        walks=int(pit.get("baseOnBalls",0)), strikeouts=int(pit.get("strikeOuts",0)),
                        home_runs=int(pit.get("homeRuns",0)), pitches=int(pit.get("numberOfPitches",0)),
                        strikes=int(pit.get("strikes",0)))); pc += 1
            bat = st.get("batting", {})
            if bat and int(bat.get("plateAppearances", 0)) > 0:
                pid = get_or_create_player(session, mid, nm, pos)
                if not session.query(HittingGameStats).filter_by(game_id=game_db_id, player_id=pid).first():
                    session.add(HittingGameStats(game_id=game_db_id, player_id=pid, team_id=tid, date=game_date,
                        plate_appearances=int(bat.get("plateAppearances",0)), at_bats=int(bat.get("atBats",0)),
                        hits=int(bat.get("hits",0)), doubles=int(bat.get("doubles",0)),
                        triples=int(bat.get("triples",0)), home_runs=int(bat.get("homeRuns",0)),
                        rbi=int(bat.get("rbi",0)), runs=int(bat.get("runs",0)),
                        walks=int(bat.get("baseOnBalls",0)), strikeouts=int(bat.get("strikeOuts",0)),
                        stolen_bases=int(bat.get("stolenBases",0)))); hc += 1
    return pc, hc

def backfill_season(season, start_date=None):
    starts = {2024: date(2024, 3, 20), 2025: date(2025, 3, 27), 2026: date(2026, 3, 26)}
    if not start_date: start_date = starts.get(season, date(season, 3, 20))
    end_date = min(date(season, 11, 5), date.today() - timedelta(days=1))
    print(f"\n{'='*70}\nBACKFILLING {season}: {start_date} to {end_date}\n{'='*70}")
    games = fetch_schedule(start_date, end_date)
    print(f"Found {len(games)} completed games\n")
    if not games: return
    session = get_session()
    added = skipped = errors = batch = tp = th = 0
    t0 = time.time()
    for i, g in enumerate(games):
        try:
            gid = g["gamePk"]; gd = g["gameDate"][:10]
            hn = g["teams"]["home"]["team"]["name"]; an = g["teams"]["away"]["team"]["name"]
            hs = g["teams"]["home"].get("score","?"); aws = g["teams"]["away"].get("score","?")
            db_id = store_game(session, g)
            if db_id is None:
                skipped += 1
                if skipped % 200 == 0: print(f"   Skipped {skipped} existing...")
                continue
            gdate = datetime.strptime(gd, "%Y-%m-%d").date()
            pc, hc = store_box(session, db_id, gdate, gid)
            added += 1; tp += pc; th += hc; batch += 1
            elapsed = time.time() - t0
            rate = added / elapsed if elapsed > 0 else 1
            rem = (len(games) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"   [{i+1}/{len(games)}] {gd} {an} {aws} @ {hn} {hs} | +{pc}P +{hc}H | ~{rem:.0f}m left")
            if batch >= 50:
                session.commit(); batch = 0
                print(f"   --- Committed ({added} games) ---")
        except Exception as e:
            errors += 1; print(f"   ERROR: {e}")
            session.rollback(); session = get_session()
            if errors > 20: print("Too many errors, stopping."); break
    try: session.commit()
    except: session.rollback()
    finally: session.close()
    elapsed = time.time() - t0
    print(f"\n{'='*70}\n{season} DONE: {added} games, {skipped} skipped, {tp} pitching lines, {th} hitting lines, {errors} errors, {elapsed/60:.1f}min\n{'='*70}")

def show_summary():
    s = get_session()
    print(f"\nDB SUMMARY:")
    print(f"  Teams:    {s.execute(text('SELECT COUNT(*) FROM teams')).scalar()}")
    print(f"  Players:  {s.execute(text('SELECT COUNT(*) FROM players')).scalar()}")
    print(f"  Games:    {s.execute(text('SELECT COUNT(*) FROM games')).scalar()}")
    print(f"  Pitching: {s.execute(text('SELECT COUNT(*) FROM pitching_game_stats')).scalar()}")
    print(f"  Hitting:  {s.execute(text('SELECT COUNT(*) FROM hitting_game_stats')).scalar()}")
    for row in s.execute(text("SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season")).fetchall():
        print(f"    {row[0]}: {row[1]} games")
    s.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary: show_summary()
    elif args.season:
        sd = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
        backfill_season(args.season, sd); show_summary()
    else:
        backfill_season(2024); backfill_season(2025); show_summary()