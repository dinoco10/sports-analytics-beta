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