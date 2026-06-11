import os
import logging

logger = logging.getLogger(__name__)

def get_db_manager():
    db_type = os.environ.get('DB_TYPE', 'sqlite').lower()
    
    if db_type == 'postgres':
        logger.info("[DB] Using PostgreSQL Manager")
        from database.postgres_manager import PostgresManager
        db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/aivigilance')
        return PostgresManager(db_url)
    else:
        logger.info("[DB] Using SQLite Manager")
        from database.sqlite_manager import SqliteManager
        db_path = os.environ.get('DATABASE_PATH', 'db.sqlite3')
        return SqliteManager(db_path)
