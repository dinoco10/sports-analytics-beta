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
        print("\n❌ Cannot connect to database. Check your .env file.")
        return False
    
    # Create all tables
    print("\n📦 Creating tables...")
    Base.metadata.create_all(bind=engine)
    
    # List created tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\n✅ {len(tables)} tables created:")
    for table in sorted(tables):
        columns = inspector.get_columns(table)
        print(f"   📋 {table} ({len(columns)} columns)")
    
    print("\n" + "=" * 60)
    print("✅ DATABASE READY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    initialize_database()