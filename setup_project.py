# ============================================================
# MLB ANALYTICS PLATFORM — WEEK 1 SETUP GUIDE
# ============================================================
# Seguí estos pasos en orden. Cada paso tiene verificación.
# Si algo falla, no avances al siguiente.
# ============================================================


# ════════════════════════════════════════════════════════════
# PASO 1: INSTALAR POSTGRESQL
# ════════════════════════════════════════════════════════════
#
# OPCIÓN A: PostgreSQL directo (RECOMENDADO para producción)
# ──────────────────────────────────────────────────────────
# 1. Descargá el instalador:
#    https://www.postgresql.org/download/windows/
#    → Click "Download the installer" → EDB installer
#    → Elegí la versión más reciente (16.x)
#
# 2. Durante la instalación:
#    - Dejá el puerto default: 5432
#    - Username: postgres (default)
#    - Password: elegí uno y ANOTALO (lo vas a necesitar)
#    - Locale: default
#    - ✅ Instalá pgAdmin 4 (viene incluido, es la GUI)
#    - ✅ Instalá Command Line Tools
#
# 3. Verificar instalación:
#    Abrí CMD o PowerShell:
#    > psql --version
#    Debería decir: psql (PostgreSQL) 16.x
#
# 4. Crear la base de datos del proyecto:
#    > psql -U postgres
#    (te pide el password que elegiste)
#    
#    postgres=# CREATE DATABASE mlb_analytics;
#    postgres=# CREATE USER mlb_user WITH PASSWORD 'tu_password_seguro';
#    postgres=# GRANT ALL PRIVILEGES ON DATABASE mlb_analytics TO mlb_user;
#    postgres=# \q
#
# VERIFICACIÓN:
#    > psql -U mlb_user -d mlb_analytics
#    Si entrás al prompt, todo OK.
#
#
# OPCIÓN B: SQLite (si querés empezar YA sin instalar nada)
# ──────────────────────────────────────────────────────────
# SQLite viene incluido con Python. Cero instalación.
# Funciona perfecto para Phase 1. Migrás a PostgreSQL después.
# El código que vamos a escribir soporta AMBOS (gracias a SQLAlchemy).
#
# Mi recomendación: empezá con SQLite hoy, instalá PostgreSQL 
# este fin de semana con calma, y switcheás cambiando UNA línea.
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
# PASO 2: CREAR EL REPO Y ESTRUCTURA
# ════════════════════════════════════════════════════════════
#
# En tu terminal (Git Bash, CMD, o PowerShell):
#
# cd C:\Users\TuUsuario\Projects  (o donde guardes tus proyectos)
# mkdir mlb-analytics
# cd mlb-analytics
# git init
#
# Luego creá la estructura ejecutando este script de Python:
# python setup_project.py
# (este archivo lo generamos abajo)
# ════════════════════════════════════════════════════════════


# ============================================================
# ARCHIVO 1: setup_project.py
# Corré esto UNA VEZ para crear toda la estructura de carpetas
# ============================================================

import os

def create_project_structure():
    """Crea la estructura completa del proyecto MLB Analytics."""
    
    directories = [
        # Config
        "config",
        
        # Data layers
        "data/raw",
        "data/processed", 
        "data/features",
        "data/external",
        
        # Source code
        "src",
        "src/ingestion",
        "src/storage",
        "src/storage/migrations",
        "src/features",
        "src/models",
        "src/betting",
        "src/evaluation",
        "src/output",
        
        # Scripts
        "scripts",
        
        # Tests
        "tests",
        
        # Notebooks (exploración solamente)
        "notebooks",
        
        # Experiments
        "experiments",
    ]
    
    for d in directories:
        os.makedirs(d, exist_ok=True)
        # Crear __init__.py en carpetas de código
        if d.startswith("src") or d == "config" or d == "tests":
            init_file = os.path.join(d, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write(f'"""{d.split("/")[-1]} module."""\n')
    
    print("✅ Estructura de carpetas creada")
    
    # Listar lo creado
    for d in sorted(directories):
        print(f"   📁 {d}/")


if __name__ == "__main__":
    create_project_structure()


# ============================================================
# ARCHIVO 2: requirements.txt
# Instalar con: pip install -r requirements.txt
# ============================================================
REQUIREMENTS = """
# Core
pandas>=2.1.0
numpy>=1.24.0
requests>=2.31.0
httpx>=0.25.0

# Database
sqlalchemy>=2.0.0
alembic>=1.13.0
psycopg2-binary>=2.9.0

# Data processing
pybaseball>=2.3.0

# Visualization
matplotlib>=3.8.0
plotly>=5.18.0

# ML (Phase 2, install now so it's ready)
scikit-learn>=1.3.0
xgboost>=2.0.0

# Dashboard (Phase 5, install later)
# streamlit>=1.29.0

# Experiment tracking
# mlflow>=2.9.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
"""


# ============================================================
# ARCHIVO 3: .env (NUNCA commitear a git)
# ============================================================
ENV_TEMPLATE = """
# Database
# Para SQLite (default, cero config):
DATABASE_URL=sqlite:///data/mlb_analytics.db

# Para PostgreSQL (cuando lo tengas instalado):
# DATABASE_URL=postgresql://mlb_user:tu_password@localhost:5432/mlb_analytics

# APIs
MLB_API_BASE=https://statsapi.mlb.com/api/v1

# Odds APIs (Phase 3, dejá vacío por ahora)
# ODDS_API_KEY=

# Weather (Phase 4)
# OPENWEATHER_API_KEY=
"""


# ============================================================
# ARCHIVO 4: .gitignore
# ============================================================
GITIGNORE = """
# Environment
.env
.venv/
venv/
env/

# Data (track with DVC, not git)
data/raw/*
data/processed/*
data/features/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/features/.gitkeep

# External data OK to commit (small CSVs)
# data/external/ is tracked

# Database files
*.db
*.sqlite3

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/
experiments/mlflow_runs/

# Outputs (regenerable)
*.png
*.csv
!data/external/*.csv

# Notebooks checkpoints
.ipynb_checkpoints/
"""


# ============================================================
# ARCHIVO 5: config/settings.py
# La configuración central del proyecto
# ============================================================
SETTINGS_PY = '''
"""
Central configuration for MLB Analytics Platform.
All settings in one place. No magic numbers in code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
EXTERNAL_DIR = DATA_DIR / "external"

# ─── Database ─────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{DATA_DIR / 'mlb_analytics.db'}"
)

# ─── MLB API ──────────────────────────────────────────────
MLB_API_BASE = os.getenv("MLB_API_BASE", "https://statsapi.mlb.com/api/v1")
MLB_CURRENT_SEASON = 2026
MLB_HISTORICAL_SEASONS = [2023, 2024, 2025]

# ─── Model Parameters ────────────────────────────────────
# How much each source contributes to final model
SOURCE_WEIGHTS = {
    "fangraphs": 0.50,
    "stats_based": 0.30,
    "user_criteria": 0.20,
}

# Rolling window sizes (in days)
ROLLING_WINDOWS = [7, 14, 30]

# ─── Feature Engineering ─────────────────────────────────
# Minimum sample sizes (NEVER use stats below these thresholds)
MIN_PA_HITTER = 50          # Plate appearances
MIN_IP_PITCHER = 20         # Innings pitched  
MIN_PA_SPLIT = 100          # For platoon/situational splits
MIN_IP_SPLIT = 50           # For pitcher splits

# Regression targets (league averages for mean reversion)
LEAGUE_AVG_BABIP = 0.300
LEAGUE_AVG_HR_FB = 0.120
LEAGUE_AVG_LOB_PCT = 0.720
LEAGUE_AVG_K_PCT = 0.224
LEAGUE_AVG_BB_PCT = 0.085

# ─── Betting (Phase 3) ───────────────────────────────────
MIN_EV_THRESHOLD = 0.03     # Minimum 3% EV to flag a bet
KELLY_FRACTION = 0.25       # Quarter Kelly for conservative sizing
BANKROLL = 100_000          # In ARS or your currency. Adjust.
'''


# ============================================================
# ARCHIVO 6: config/weights.py
# TUS weights migrados del script actual
# ============================================================
WEIGHTS_PY = '''
"""
═══════════════════════════════════════════════════════════════════════
Model Weights — YOUR baseball philosophy encoded as numbers.

RULES:
1. Main weights must sum to 1.00
2. Pitching underlying sub-weights must sum to 1.00
3. If you change one weight, adjust others to compensate
═══════════════════════════════════════════════════════════════════════
"""


def get_main_weights():
    """
    Main component weights for the power rankings model.
    These determine how much each team attribute matters.
    """
    return {
        # ── Offense ───────────────────────────────
        'lineup_contact': 0.13,      # OBP/contacto > power (TU PRIORIDAD)
        'lineup_power': 0.09,        # HR, SLG, run production
        'speed_baserunning': 0.09,   # Atletismo, robos, bases extra
        
        # ── Pitching ─────────────────────────────
        'bullpen_depth': 0.13,       # MUY importante (ganar 2-1)
        'pitching_underlying': 0.12, # DATA: K-BB%, xERA gap, BABIP, etc.
        'rotation_strength': 0.08,   # Eye test de rotación
        
        # ── Defense & Depth ──────────────────────
        'defense': 0.11,             # Defensa general
        'depth': 0.07,               # Sobrevivir 162 juegos
        
        # ── Intangibles ──────────────────────────
        'momentum': 0.06,            # Vibes, trayectoria
        'farm_system': 0.05,         # Call-ups potenciales
        'manager_coaching': 0.02,    # Bajo (difícil de evaluar)
        
        # ── Personal ─────────────────────────────
        'personal_weight': 0.05,     # Tu bias, eye test
    }


def get_pitching_underlying_sub_weights():
    """
    Sub-weights WITHIN the pitching_underlying component.
    These determine which underlying metric matters most.
    
    The effective weight of each metric in the total model is:
    effective = sub_weight × main_weight('pitching_underlying') × SOURCE_WEIGHTS['user_criteria']
    
    Example: K_BB_pct effective = 0.20 × 0.12 × 0.20 = 0.0048 (0.48%)
    All 8 metrics combined = 0.12 × 0.20 = 0.024 (2.4% of total model)
    """
    return {
        # ── Tier 1: Most Predictive (53%) ─────────
        'K_BB_pct': 0.20,        # Best single pitcher skill metric
        'xERA_gap': 0.18,        # Luck/regression detector
        'FIP_era_gap': 0.15,     # ERA sustainability
        
        # ── Tier 2: Very Important (33%) ──────────
        'BABIP_allowed': 0.13,   # Balls in play luck
        'HR_FB_pct': 0.12,       # Most volatile year-to-year
        'GB_pct': 0.08,          # Stable skill metric
        
        # ── Tier 3: Complementary (14%) ───────────
        'barrel_pct': 0.08,      # Contact quality allowed
        'swstr_pct': 0.06,       # Stuff quality proxy
    }


def validate_weights():
    """Run this to check your weights are valid."""
    main = get_main_weights()
    sub = get_pitching_underlying_sub_weights()
    
    main_total = sum(main.values())
    sub_total = sum(sub.values())
    
    errors = []
    
    if abs(main_total - 1.0) > 0.01:
        errors.append(f"Main weights sum to {main_total:.3f}, should be 1.00")
    if abs(sub_total - 1.0) > 0.01:
        errors.append(f"Sub weights sum to {sub_total:.3f}, should be 1.00")
    if any(v < 0 for v in main.values()):
        errors.append("Negative main weights found")
    if any(v < 0 for v in sub.values()):
        errors.append("Negative sub weights found")
    
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False
    
    print("✅ All weights valid")
    print(f"   Main: {main_total:.2f} | Sub: {sub_total:.2f}")
    return True


if __name__ == "__main__":
    validate_weights()
'''


# ============================================================
# ARCHIVO 7: src/storage/database.py
# Conexión a base de datos (soporta SQLite Y PostgreSQL)
# ============================================================
DATABASE_PY = '''
"""
Database connection and session management.
Supports both SQLite (development) and PostgreSQL (production).
Switch by changing DATABASE_URL in .env file.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from config.settings import DATABASE_URL

# Create engine (works for both SQLite and PostgreSQL)
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set True to see SQL queries (debug)
    pool_pre_ping=True,  # Check connection is alive
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session() -> Session:
    """Get a database session. Use with context manager."""
    session = SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def test_connection():
    """Verify database connection works."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✅ Database connected: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
'''


# ============================================================
# ARCHIVO 8: src/storage/models.py
# Definición de tablas (SQLAlchemy ORM)
# ============================================================
MODELS_PY = '''
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
'''


# ============================================================
# ARCHIVO 9: src/ingestion/mlb_api.py
# Cliente limpio para MLB Stats API
# ============================================================
MLB_API_PY = '''
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
    print(f"\\n✅ Teams loaded: {len(teams)}")
    print(teams.head())
    
    # Test: get yesterday's games
    from datetime import timedelta
    yesterday = date.today() - timedelta(days=1)
    schedule = client.get_schedule(yesterday)
    print(f"\\n✅ Games on {yesterday}: {len(schedule)}")
    if len(schedule) > 0:
        print(schedule[["date", "away_team", "home_team", "away_score", "home_score"]].head())
'''


# ============================================================
# ARCHIVO 10: scripts/initialize_db.py
# Correr UNA VEZ para crear todas las tablas
# ============================================================
INIT_DB_PY = '''
"""
Initialize the database — create all tables.
Run this ONCE after setting up your database.

Usage:
    python scripts/initialize_db.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.database import engine, test_connection
from src.storage.models import Base


def initialize_database():
    print("=" * 60)
    print("🗄️  INITIALIZING MLB ANALYTICS DATABASE")
    print("=" * 60)
    
    # Test connection
    if not test_connection():
        print("\\n❌ Cannot connect to database. Check your .env file.")
        return False
    
    # Create all tables
    print("\\n📦 Creating tables...")
    Base.metadata.create_all(bind=engine)
    
    # List created tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\\n✅ {len(tables)} tables created:")
    for table in sorted(tables):
        columns = inspector.get_columns(table)
        print(f"   📋 {table} ({len(columns)} columns)")
    
    print("\\n" + "=" * 60)
    print("✅ DATABASE READY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    initialize_database()
'''


# ============================================================
# ARCHIVO 11: scripts/seed_teams.py
# Cargar los 30 equipos de MLB en la base de datos
# ============================================================
SEED_TEAMS_PY = '''
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
        print(f"\\n✅ Seeded {added} new teams ({len(teams_df)} total)")
    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_teams()
'''


# ============================================================
# MAIN: Write all files to disk
# ============================================================

def write_all_files():
    """Write all project files."""
    
    files = {
        "requirements.txt": REQUIREMENTS.strip(),
        ".env": ENV_TEMPLATE.strip(),
        ".gitignore": GITIGNORE.strip(),
        "config/settings.py": SETTINGS_PY.strip(),
        "config/weights.py": WEIGHTS_PY.strip(),
        "src/storage/database.py": DATABASE_PY.strip(),
        "src/storage/models.py": MODELS_PY.strip(),
        "src/ingestion/mlb_api.py": MLB_API_PY.strip(),
        "scripts/initialize_db.py": INIT_DB_PY.strip(),
        "scripts/seed_teams.py": SEED_TEAMS_PY.strip(),
    }
    
    for filepath, content in files.items():
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Created: {filepath}")
    
    # Create .gitkeep files for empty data dirs
    for d in ["data/raw", "data/processed", "data/features"]:
        os.makedirs(d, exist_ok=True)
        gitkeep = os.path.join(d, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "w").close()
    
    print(f"\n{'='*60}")
    print("✅ ALL FILES CREATED")
    print(f"{'='*60}")
    print("\nNEXT STEPS:")
    print("1. pip install -r requirements.txt")
    print("2. python scripts/initialize_db.py")
    print("3. python scripts/seed_teams.py")
    print("4. python src/ingestion/mlb_api.py  (test API)")


if __name__ == "__main__":
    # First create folders
    create_project_structure()
    # Then write all files
    write_all_files()
