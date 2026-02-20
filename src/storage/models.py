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
# STATCAST METRICS (Season-Level from Baseball Savant)
# ═══════════════════════════════════════════════════════════

class PlayerStatcastMetrics(Base):
    """
    Season-level Statcast metrics per player, sourced from pybaseball.
    One row per player per season. Used as regression signals in Marcel projections.

    Categories:
    - Contact quality: exit velo, barrel rate, hard hit %, expected stats
    - Plate discipline: K%, BB%, chase rate, whiff rate, zone contact
    - Batted ball profile: GB/FB/LD rates, pull air rate
    - BABIP for luck filtering
    """
    __tablename__ = "player_statcast_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(Integer, nullable=False)

    # Contact quality
    avg_exit_velocity = Column(Float)      # Average exit velocity (mph)
    barrel_rate = Column(Float)            # Barrels per batted ball event
    hard_hit_rate = Column(Float)          # Balls 95+ mph / BBE
    xwoba = Column(Float)                  # Expected weighted on-base average
    xslg = Column(Float)                   # Expected slugging
    xba = Column(Float)                    # Expected batting average
    launch_angle_sweet_spot_pct = Column(Float)  # Balls 8-32 degrees / BBE

    # Actual stats (for luck gap calculations)
    babip = Column(Float)                  # Batting average on balls in play
    woba = Column(Float)                   # Actual wOBA (to compare vs xwOBA)
    slg = Column(Float)                    # Actual SLG (to compare vs xSLG)
    ba = Column(Float)                     # Actual BA (to compare vs xBA)
    hr_per_fb = Column(Float)              # HR / fly balls ratio

    # Plate discipline
    k_rate = Column(Float)                 # Strikeout rate
    bb_rate = Column(Float)                # Walk rate
    chase_rate = Column(Float)             # O-Swing% — swinging outside zone
    whiff_rate = Column(Float)             # Swinging strike rate
    z_contact_rate = Column(Float)         # Contact on strikes (elite 85%+)

    # Batted ball profile
    pull_air_rate = Column(Float)          # Pulled fly balls — best HR predictor
    gb_rate = Column(Float)                # Ground ball %
    fb_rate = Column(Float)                # Fly ball %
    ld_rate = Column(Float)                # Line drive %

    # Metadata
    pa = Column(Integer)                   # Plate appearances (for min threshold)

    __table_args__ = (
        UniqueConstraint("player_id", "season", name="uq_statcast_player_season"),
        Index("idx_statcast_season", "season"),
    )

    def __repr__(self):
        return f"<Statcast {self.player_id} {self.season}: xwOBA={self.xwoba}>"


# ═══════════════════════════════════════════════════════════
# PITCHER STATCAST METRICS (Season-Level + Pitch-Level)
# ═══════════════════════════════════════════════════════════

class PitcherStatcastMetrics(Base):
    """
    Season-level Statcast metrics per pitcher, sourced from Baseball Savant.
    One row per pitcher per season. Used as regression signals in Marcel projections.

    Categories:
    - Contact suppression: exit velo against, barrel rate, hard hit %, xwOBA against
    - Overperformance signals: ERA vs xERA, BABIP against, HR/FB%
    - Swing & miss: whiff%, K-BB%, chase rate induced, SwStr%
    - Batted ball profile: GB/FB/LD rates, pull air rate against (key HR predictor)
    """
    __tablename__ = "pitcher_statcast_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(Integer, nullable=False)

    # Contact suppression
    avg_exit_velocity_against = Column(Float)   # Avg EV allowed (lower = better)
    barrel_rate_against = Column(Float)         # Barrels per BBE allowed (lower = better)
    hard_hit_rate_against = Column(Float)       # Balls 95+ mph allowed (lower = better)
    xwoba_against = Column(Float)               # Expected wOBA allowed
    xslg_against = Column(Float)                # Expected SLG allowed
    xba_against = Column(Float)                 # Expected BA allowed
    xera = Column(Float)                        # Expected ERA (strips BABIP/HR luck)

    # Actual stats (for overperformance gaps)
    era = Column(Float)                         # Actual ERA
    woba_against = Column(Float)                # Actual wOBA allowed
    babip_against = Column(Float)               # BABIP allowed (~.300 league avg)
    hr_per_fb = Column(Float)                   # HR/FB% (regresses to 11-13%)

    # Swing & miss ability
    k_rate = Column(Float)                      # Strikeout rate
    bb_rate = Column(Float)                     # Walk rate
    k_minus_bb = Column(Float)                  # K-BB% (<10% = heavy BABIP dependence)
    whiff_rate = Column(Float)                  # Overall whiff %
    swstr_rate = Column(Float)                  # Swinging strike % (<8.6% = bottom 5)
    chase_rate_induced = Column(Float)          # O-Swing% induced (higher = better)
    z_contact_rate_against = Column(Float)      # Contact on strikes (lower = better)

    # Batted ball profile
    pull_air_rate_against = Column(Float)       # Pulled fly balls allowed (KEY: 66% of HR)
    gb_rate = Column(Float)                     # Ground ball % (higher generally better)
    fb_rate = Column(Float)                     # Fly ball % (>39% = red flag)
    ld_rate = Column(Float)                     # Line drive % (lower = better)

    # Metadata
    pa_against = Column(Integer)                # Plate appearances faced
    ip = Column(Float)                          # Innings pitched (for min threshold)

    __table_args__ = (
        UniqueConstraint("player_id", "season", name="uq_pitcher_statcast_player_season"),
        Index("idx_pitcher_statcast_season", "season"),
    )

    def __repr__(self):
        return f"<PitcherStatcast {self.player_id} {self.season}: xERA={self.xera}>"


class PitcherPitchMetrics(Base):
    """
    Per-pitch-type, per-season metrics for each pitcher.
    One row per pitcher per season per pitch type.
    Tracks run value, velocity, whiff rate, and usage for each pitch.

    This is where we identify:
    - Declining fastball velocity (regression signal)
    - New plus secondary pitches (breakout catalyst)
    - Arsenal diversity / predictability risk
    """
    __tablename__ = "pitcher_pitch_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(Integer, nullable=False)
    pitch_type = Column(String(5), nullable=False)   # FF, SI, SL, CH, CU, ST, etc.
    pitch_name = Column(String(30))                   # "4-Seam Fastball", "Slider", etc.

    # Per-pitch performance
    run_value_per_100 = Column(Float)                 # Run value per 100 pitches (+ = good)
    run_value = Column(Float)                         # Total run value
    pitches_thrown = Column(Integer)                   # Total pitches of this type
    usage_pct = Column(Float)                         # Usage percentage

    # Per-pitch outcomes
    whiff_rate = Column(Float)                        # Whiff % on this pitch
    chase_rate = Column(Float)                        # Chase % when thrown outside zone
    ba_against = Column(Float)                        # BA when put in play
    slg_against = Column(Float)                       # SLG when put in play
    woba_against = Column(Float)                      # wOBA against this pitch
    hard_hit_pct = Column(Float)                      # Hard hit % against this pitch

    # Physical properties
    avg_speed = Column(Float)                         # Average velocity (mph)
    xwoba_against = Column(Float)                     # Expected wOBA against this pitch

    __table_args__ = (
        UniqueConstraint("player_id", "season", "pitch_type",
                         name="uq_pitch_metrics_player_season_type"),
        Index("idx_pitch_metrics_season", "season"),
    )

    def __repr__(self):
        return f"<PitchMetric {self.player_id} {self.season} {self.pitch_type}: RV={self.run_value_per_100}>"


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