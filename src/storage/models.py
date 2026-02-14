"""
Database table definitions using SQLAlchemy ORM.
These map directly to the database schema in the blueprint.

IMPORTANT: After changing these models, create a migration:
    alembic revision --autogenerate -m "description"
    alembic upgrade head
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, DateTime,
    ForeignKey, UniqueConstraint, Index, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ═══════════════════════════════════════════════════════════
# DIMENSION TABLES
# ═══════════════════════════════════════════════════════════

class Team(Base):
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True)
    mlb_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    abbreviation = Column(String(5))
    league = Column(String(2))        # AL / NL
    division = Column(String(10))     # East / Central / West
    active = Column(Boolean, default=True)
    
    # Relationships
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    
    def __repr__(self):
        return f"<Team {self.abbreviation}: {self.name}>"


class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True)
    mlb_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    birth_date = Column(Date)
    bats = Column(String(1))          # L / R / S
    throws = Column(String(1))        # L / R
    primary_position = Column(String(3))
    current_team_id = Column(Integer, ForeignKey("teams.id"))
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Player {self.name} ({self.primary_position})>"


class Ballpark(Base):
    __tablename__ = "ballparks"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    team_id = Column(Integer, ForeignKey("teams.id"))
    park_factor = Column(Float, default=1.000)
    park_factor_hr = Column(Float, default=1.000)
    park_factor_lhb = Column(Float, default=1.000)
    park_factor_rhb = Column(Float, default=1.000)
    elevation_ft = Column(Integer)
    is_dome = Column(Boolean, default=False)
    field_surface = Column(String(20))  # grass / turf


class Umpire(Base):
    __tablename__ = "umpires"
    
    id = Column(Integer, primary_key=True)
    mlb_id = Column(Integer, unique=True)
    name = Column(String(100))
    career_k_pct = Column(Float)
    career_bb_pct = Column(Float)
    career_zone_pct = Column(Float)
    run_environment = Column(Float)


# ═══════════════════════════════════════════════════════════
# FACT TABLES
# ═══════════════════════════════════════════════════════════

class Game(Base):
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True)
    mlb_game_id = Column(Integer, unique=True, nullable=False)
    date = Column(Date, nullable=False)
    season = Column(Integer, nullable=False)
    
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    ballpark_id = Column(Integer, ForeignKey("ballparks.id"))
    home_starter_id = Column(Integer, ForeignKey("players.id"))
    away_starter_id = Column(Integer, ForeignKey("players.id"))
    umpire_id = Column(Integer, ForeignKey("umpires.id"))
    
    # Results
    home_score = Column(Integer)
    away_score = Column(Integer)
    winner_id = Column(Integer, ForeignKey("teams.id"))
    innings = Column(Integer, default=9)
    
    # Context (Phase 4 — columns ready, fill later)
    temperature_f = Column(Integer)
    wind_speed_mph = Column(Integer)
    wind_direction = Column(String(10))
    is_dome = Column(Boolean, default=False)
    day_night = Column(String(5))
    home_rest_days = Column(Integer)
    away_rest_days = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    
    # Indexes
    __table_args__ = (
        Index("idx_games_date", "date"),
        Index("idx_games_season", "season"),
        Index("idx_games_teams", "home_team_id", "away_team_id"),
    )
    
    def __repr__(self):
        return f"<Game {self.date}: {self.away_team_id} @ {self.home_team_id}>"


class PitchingGameStats(Base):
    __tablename__ = "pitching_game_stats"
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    date = Column(Date, nullable=False)
    
    # Basic
    ip = Column(Float)
    hits = Column(Integer)
    runs = Column(Integer)
    earned_runs = Column(Integer)
    walks = Column(Integer)
    strikeouts = Column(Integer)
    home_runs = Column(Integer)
    pitches = Column(Integer)
    strikes = Column(Integer)
    
    # Batted ball (Phase 2+, from Savant)
    avg_exit_velo = Column(Float)
    barrel_pct = Column(Float)
    hard_hit_pct = Column(Float)
    gb_pct = Column(Float)
    
    # Advanced
    whiff_pct = Column(Float)
    chase_pct = Column(Float)
    swstr_pct = Column(Float)
    
    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_pitching_game_player"),
    )


class HittingGameStats(Base):
    __tablename__ = "hitting_game_stats"
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    date = Column(Date, nullable=False)
    
    # Basic
    plate_appearances = Column(Integer)
    at_bats = Column(Integer)
    hits = Column(Integer)
    doubles = Column(Integer)
    triples = Column(Integer)
    home_runs = Column(Integer)
    rbi = Column(Integer)
    runs = Column(Integer)
    walks = Column(Integer)
    strikeouts = Column(Integer)
    stolen_bases = Column(Integer)
    
    # Batted ball (Phase 2+)
    avg_exit_velo = Column(Float)
    barrel_pct = Column(Float)
    hard_hit_pct = Column(Float)
    launch_angle = Column(Float)
    sprint_speed = Column(Float)
    
    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_hitting_game_player"),
    )


# ═══════════════════════════════════════════════════════════
# AGGREGATION & MODEL TABLES
# ═══════════════════════════════════════════════════════════

class TeamSeasonSnapshot(Base):
    """Daily snapshot of team stats. Enables time-travel queries."""
    __tablename__ = "team_season_snapshots"
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    season = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    
    wins = Column(Integer)
    losses = Column(Integer)
    run_diff = Column(Integer)
    
    # JSONB-like columns (JSON in SQLite, JSONB in PostgreSQL)
    pitching_stats = Column(JSON)
    hitting_stats = Column(JSON)
    underlying_metrics = Column(JSON)  # Your pitching underlying data
    model_ratings = Column(JSON)       # Your 1-10 component ratings
    
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("team_id", "season", "date", name="uq_team_season_date"),
    )


class ModelPrediction(Base):
    """Log every prediction the model makes. Essential for evaluation."""
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50), nullable=False)
    prediction_date = Column(Date, nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"))
    
    home_win_prob = Column(Float)
    away_win_prob = Column(Float)
    predicted_total = Column(Float)
    
    actual_winner = Column(String(4))  # 'home' / 'away'
    actual_total = Column(Integer)
    
    confidence = Column(Float)
    features_used = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Bet(Base):
    """Track all bets placed. Phase 3."""
    __tablename__ = "bets"
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    sportsbook = Column(String(50))
    bet_type = Column(String(50))
    selection = Column(String(200))
    odds = Column(Float)
    stake = Column(Float)
    
    model_prob = Column(Float)
    implied_prob = Column(Float)
    expected_value = Column(Float)
    
    result = Column(String(10))  # win / loss / push / pending
    profit_loss = Column(Float)
    
    placed_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime)