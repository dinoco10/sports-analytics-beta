"""
MLB Stats API Client.

Public API, no key required. Rate limit: be respectful (~1 req/sec).
Docs: https://statsapi.mlb.com/docs/

This client is designed to be called by agents/scripts/cron jobs.
Every method returns structured data (dict or DataFrame).
"""

import requests
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from config.settings import MLB_API_BASE
import time
import logging

logger = logging.getLogger(__name__)


class MLBApiClient:
    """Clean, reusable MLB Stats API client."""
    
    def __init__(self, base_url: str = MLB_API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MLBAnalyticsPlatform/1.0"
        })
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(0.5)  # Rate limiting: max 2 req/sec
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} — {e}")
            return {}
    
    # ─── Teams ────────────────────────────────────────────
    
    def get_teams(self, season: int = None) -> pd.DataFrame:
        """Get all MLB teams."""
        params = {"sportId": 1}
        if season:
            params["season"] = season
        
        data = self._get("teams", params)
        teams = []
        for t in data.get("teams", []):
            teams.append({
                "mlb_id": t["id"],
                "name": t["name"],
                "abbreviation": t.get("abbreviation", ""),
                "league": t.get("league", {}).get("abbreviation", ""),
                "division": t.get("division", {}).get("name", ""),
            })
        
        return pd.DataFrame(teams)
    
    # ─── Schedule & Games ─────────────────────────────────
    
    def get_schedule(self, start_date: date, end_date: date = None) -> pd.DataFrame:
        """Get games for a date range."""
        if end_date is None:
            end_date = start_date
        
        params = {
            "sportId": 1,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "hydrate": "probablePitcher,linescore"
        }
        
        data = self._get("schedule", params)
        games = []
        
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                game_info = {
                    "mlb_game_id": g["gamePk"],
                    "date": date_entry["date"],
                    "status": g["status"]["detailedState"],
                    "home_team": g["teams"]["home"]["team"]["name"],
                    "home_team_id": g["teams"]["home"]["team"]["id"],
                    "away_team": g["teams"]["away"]["team"]["name"],
                    "away_team_id": g["teams"]["away"]["team"]["id"],
                    "home_score": g["teams"]["home"].get("score"),
                    "away_score": g["teams"]["away"].get("score"),
                }
                
                # Probable pitchers
                home_pitcher = g["teams"]["home"].get("probablePitcher", {})
                away_pitcher = g["teams"]["away"].get("probablePitcher", {})
                game_info["home_starter"] = home_pitcher.get("fullName", "TBD")
                game_info["home_starter_id"] = home_pitcher.get("id")
                game_info["away_starter"] = away_pitcher.get("fullName", "TBD")
                game_info["away_starter_id"] = away_pitcher.get("id")
                
                games.append(game_info)
        
        return pd.DataFrame(games)
    
    # ─── Box Scores ───────────────────────────────────────
    
    def get_box_score(self, game_id: int) -> dict:
        """Get detailed box score for a single game."""
        data = self._get(f"game/{game_id}/boxscore")
        return data
    
    def get_game_player_stats(self, game_id: int) -> tuple:
        """
        Extract individual player stats from a box score.
        Returns: (pitchers_df, hitters_df)
        """
        box = self.get_box_score(game_id)
        if not box:
            return pd.DataFrame(), pd.DataFrame()
        
        pitchers = []
        hitters = []
        
        for side in ["home", "away"]:
            team_data = box.get("teams", {}).get(side, {})
            team_name = team_data.get("team", {}).get("name", "")
            team_id = team_data.get("team", {}).get("id")
            
            for player_id_str, player_data in team_data.get("players", {}).items():
                p = player_data.get("person", {})
                stats = player_data.get("stats", {})
                
                # Pitching stats
                if "pitching" in stats and stats["pitching"]:
                    s = stats["pitching"]
                    pitchers.append({
                        "mlb_player_id": p.get("id"),
                        "name": p.get("fullName"),
                        "team": team_name,
                        "team_id": team_id,
                        "ip": s.get("inningsPitched", "0"),
                        "hits": s.get("hits", 0),
                        "runs": s.get("runs", 0),
                        "earned_runs": s.get("earnedRuns", 0),
                        "walks": s.get("baseOnBalls", 0),
                        "strikeouts": s.get("strikeOuts", 0),
                        "home_runs": s.get("homeRuns", 0),
                        "pitches": s.get("numberOfPitches", 0),
                        "strikes": s.get("strikes", 0),
                    })
                
                # Hitting stats
                if "batting" in stats and stats["batting"]:
                    s = stats["batting"]
                    if s.get("plateAppearances", 0) > 0:
                        hitters.append({
                            "mlb_player_id": p.get("id"),
                            "name": p.get("fullName"),
                            "team": team_name,
                            "team_id": team_id,
                            "pa": s.get("plateAppearances", 0),
                            "ab": s.get("atBats", 0),
                            "hits": s.get("hits", 0),
                            "doubles": s.get("doubles", 0),
                            "triples": s.get("triples", 0),
                            "home_runs": s.get("homeRuns", 0),
                            "rbi": s.get("rbi", 0),
                            "runs": s.get("runs", 0),
                            "walks": s.get("baseOnBalls", 0),
                            "strikeouts": s.get("strikeOuts", 0),
                            "stolen_bases": s.get("stolenBases", 0),
                        })
        
        return pd.DataFrame(pitchers), pd.DataFrame(hitters)
    
    # ─── Rosters ──────────────────────────────────────────
    
    def get_roster(self, team_id: int, roster_type: str = "active") -> pd.DataFrame:
        """Get team roster. Types: active, 40Man, fullSeason, allTime"""
        data = self._get(f"teams/{team_id}/roster", {"rosterType": roster_type})
        
        players = []
        for p in data.get("roster", []):
            players.append({
                "mlb_id": p["person"]["id"],
                "name": p["person"]["fullName"],
                "position": p["position"]["abbreviation"],
                "status": p.get("status", {}).get("description", "Active"),
            })
        
        return pd.DataFrame(players)
    
    # ─── Player Stats ─────────────────────────────────────
    
    def get_player_season_stats(self, player_id: int, season: int, 
                                  group: str = "hitting") -> dict:
        """Get a player's season stats. group: 'hitting' or 'pitching'"""
        params = {
            "stats": "season",
            "season": season,
            "group": group,
        }
        data = self._get(f"people/{player_id}/stats", params)
        
        stats_list = data.get("stats", [])
        if stats_list and stats_list[0].get("splits"):
            return stats_list[0]["splits"][0].get("stat", {})
        return {}
    
    # ─── Transactions ─────────────────────────────────────
    
    def get_transactions(self, start_date: date, end_date: date = None) -> pd.DataFrame:
        """Get roster transactions (trades, DFA, call-ups, etc.)"""
        if end_date is None:
            end_date = start_date
        
        params = {
            "startDate": start_date.strftime("%m/%d/%Y"),
            "endDate": end_date.strftime("%m/%d/%Y"),
        }
        data = self._get("transactions", params)
        
        transactions = []
        for t in data.get("transactions", []):
            transactions.append({
                "id": t.get("id"),
                "date": t.get("date"),
                "type": t.get("typeDesc"),
                "description": t.get("description", ""),
                "player": t.get("person", {}).get("fullName"),
                "player_id": t.get("person", {}).get("id"),
                "from_team": t.get("fromTeam", {}).get("name"),
                "to_team": t.get("toTeam", {}).get("name"),
            })
        
        return pd.DataFrame(transactions)


# ─── Standalone test ──────────────────────────────────────

if __name__ == "__main__":
    client = MLBApiClient()
    
    # Test: get teams
    teams = client.get_teams(season=2025)
    print(f"\n✅ Teams loaded: {len(teams)}")
    print(teams.head())
    
    # Test: get yesterday's games
    from datetime import timedelta
    yesterday = date.today() - timedelta(days=1)
    schedule = client.get_schedule(yesterday)
    print(f"\n✅ Games on {yesterday}: {len(schedule)}")
    if len(schedule) > 0:
        print(schedule[["date", "away_team", "home_team", "away_score", "home_score"]].head())