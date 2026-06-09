import os
import logging
import sqlite3
import pytz
from datetime import datetime

try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
except ImportError:
    psycopg2 = None

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

from data_access.connection import DEFAULT_DSN, SqliteConnectionWrapper, SqliteCursorWrapper
from data_access.crud.cameras import CamerasCRUD
from data_access.crud.persons import PersonsCRUD
from data_access.crud.detections import DetectionsCRUD
from data_access.crud.analytics import AnalyticsCRUD
from data_access.crud.recordings import RecordingsCRUD
from data_access.crud.alerts import AlertsCRUD
from data_access.crud.journeys import JourneysCRUD

class PostgresManager(CamerasCRUD, PersonsCRUD, DetectionsCRUD, AnalyticsCRUD, RecordingsCRUD, AlertsCRUD, JourneysCRUD):

    """PostgreSQL database manager for surveillance system."""

    def __init__(self, dsn: str = None):
        self.dsn = dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN)
        self._pool = None
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, dsn=self.dsn
            )
            self._init_db()
            logger.info(f"[OK] Connected to PostgreSQL")
        except Exception as e:
            logger.critical(f"[FAIL] PostgreSQL Manager Init Error: {e}")
            raise RuntimeError(f"PostgreSQL connection failed: {e}")

    def _get_connection(self):
        conn = self._pool.getconn()
        conn.autocommit = False
        return conn

    def _put_connection(self, conn):
        self._pool.putconn(conn)

    def _binary(self, value):
        if value is None:
            return None
        return psycopg2.Binary(value) if psycopg2 is not None else value

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Cameras
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS cameras (
                        camera_id TEXT PRIMARY KEY,
                        source TEXT,
                        updated_at TIMESTAMP
                    )
                ''')

                # 2. Camera Settings
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS camera_settings (
                        camera_id TEXT PRIMARY KEY,
                        recording_enabled INTEGER DEFAULT 0,
                        tracking_area TEXT
                    )
                ''')

                # 3. Persons (Registered)
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS persons (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE,
                        image_path TEXT,
                        encoding BYTEA,
                        last_seen TIMESTAMP,
                        last_camera TEXT
                    )
                ''')

                # 4. Registered Detections
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS registered_detections (
                        id SERIAL PRIMARY KEY,
                        person_name TEXT,
                        camera_id TEXT,
                        timestamp TIMESTAMP,
                        snapshot_path TEXT
                    )
                ''')

                # 5. Detection Snapshots
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS detection_snapshots (
                        id SERIAL PRIMARY KEY,
                        camera_id TEXT,
                        person_count INTEGER,
                        snapshot_path TEXT,
                        bbox_data TEXT,
                        face_encodings TEXT,
                        person_crops TEXT,
                        timestamp TIMESTAMP
                    )
                ''')

                # 6. Occupancy Logs
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS occupancy_logs (
                        id SERIAL PRIMARY KEY,
                        camera_id TEXT,
                        timestamp TIMESTAMP,
                        count INTEGER
                    )
                ''')

                # 7. Video Recordings
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS video_recordings (
                        id SERIAL PRIMARY KEY,
                        camera_id TEXT,
                        file_path TEXT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        has_registered_person INTEGER DEFAULT 0,
                        registered_person_times TEXT
                    )
                ''')

                # 8. Alerts
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        camera_id TEXT,
                        person_id TEXT,
                        snapshot_path TEXT,
                        timestamp TIMESTAMP,
                        type TEXT
                    )
                ''')

                # 9. Global Identities (Re-ID)
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS global_identities (
                        global_id TEXT PRIMARY KEY,
                        encoding BYTEA,
                        first_seen TIMESTAMP,
                        last_seen TIMESTAMP,
                        last_camera TEXT,
                        type TEXT,
                        thumbnail BYTEA
                    )
                ''')

                # 10. Journeys
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS journeys (
                        id SERIAL PRIMARY KEY,
                        global_id TEXT,
                        camera_id TEXT,
                        timestamp TIMESTAMP,
                        snapshot_path TEXT,
                        type TEXT
                    )
                ''')

                # 11. Analytics Snapshots
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS analytics_snapshots (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        metric_type TEXT,
                        camera_id TEXT,
                        value INTEGER,
                        metadata TEXT
                    )
                ''')

                # Indexes
                cur.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_cam_time ON detection_snapshots (camera_id, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_reg_det_name_time ON registered_detections (person_name, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_video_cam_time ON video_recordings (camera_id, start_time)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_alerts_cam_time ON alerts (camera_id, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_journeys_id_time ON journeys (global_id, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_analytics_type_time ON analytics_snapshots (metric_type, timestamp)')

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._put_connection(conn)


class SqliteManager(PostgresManager):

    """SQLite database manager fallback for local development."""

    def __init__(self, db_path: str = None, dsn: str = None):
        self.db_path = db_path or os.environ.get('SQLITE_DB_PATH', 'data/aiv_vigilance.db')
        self._conn = None
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._open_connection()
        self._init_db()
        logger.info(f"[OK] Connected to SQLite: {self.db_path}")

    def _open_connection(self):
        raw_conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        raw_conn.execute('PRAGMA journal_mode=WAL')
        raw_conn.execute('PRAGMA foreign_keys = ON')
        self._conn = SqliteConnectionWrapper(raw_conn)

    def _get_connection(self):
        return self._conn

    def _put_connection(self, conn):
        return None

    def _binary(self, value):
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if hasattr(value, 'tobytes'):
            return value.tobytes()
        return value

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Cameras
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS cameras (
                        camera_id TEXT PRIMARY KEY,
                        source TEXT,
                        updated_at TIMESTAMP
                    )
                ''')

                # 2. Camera Settings
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS camera_settings (
                        camera_id TEXT PRIMARY KEY,
                        recording_enabled INTEGER DEFAULT 0,
                        tracking_area TEXT
                    )
                ''')

                # 3. Persons (Registered)
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS persons (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        image_path TEXT,
                        encoding BLOB,
                        last_seen TIMESTAMP,
                        last_camera TEXT
                    )
                ''')

                # 4. Registered Detections
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS registered_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_name TEXT,
                        camera_id TEXT,
                        timestamp TIMESTAMP,
                        snapshot_path TEXT
                    )
                ''')

                # 5. Detection Snapshots
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS detection_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT,
                        person_count INTEGER,
                        snapshot_path TEXT,
                        bbox_data TEXT,
                        face_encodings TEXT,
                        person_crops TEXT,
                        timestamp TIMESTAMP
                    )
                ''')

                # 6. Occupancy Logs
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS occupancy_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT,
                        timestamp TIMESTAMP,
                        count INTEGER
                    )
                ''')

                # 7. Video Recordings
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS video_recordings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT,
                        file_path TEXT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        has_registered_person INTEGER DEFAULT 0,
                        registered_person_times TEXT
                    )
                ''')

                # 8. Alerts
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT,
                        person_id TEXT,
                        snapshot_path TEXT,
                        timestamp TIMESTAMP,
                        type TEXT
                    )
                ''')

                # 9. Global Identities (Re-ID)
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS global_identities (
                        global_id TEXT PRIMARY KEY,
                        encoding BLOB,
                        first_seen TIMESTAMP,
                        last_seen TIMESTAMP,
                        last_camera TEXT,
                        type TEXT,
                        thumbnail BLOB
                    )
                ''')

                # 10. Journeys
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS journeys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        global_id TEXT,
                        camera_id TEXT,
                        timestamp TIMESTAMP,
                        snapshot_path TEXT,
                        type TEXT
                    )
                ''')

                # 11. Analytics Snapshots
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS analytics_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP,
                        metric_type TEXT,
                        camera_id TEXT,
                        value INTEGER,
                        metadata TEXT
                    )
                ''')

                # Indexes
                cur.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_cam_time ON detection_snapshots (camera_id, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_reg_det_name_time ON registered_detections (person_name, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_video_cam_time ON video_recordings (camera_id, start_time)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_alerts_cam_time ON alerts (camera_id, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_journeys_id_time ON journeys (global_id, timestamp)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_analytics_type_time ON analytics_snapshots (metric_type, timestamp)')
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise


if os.environ.get('DATABASE_URL'):
    DatabaseManager = PostgresManager
    SqliteManager = PostgresManager
else:
    DatabaseManager = SqliteManager
    PostgresManager = SqliteManager


if os.environ.get('DATABASE_URL'):
    DatabaseManager = PostgresManager
    SqliteManager = PostgresManager
else:
    DatabaseManager = SqliteManager
    PostgresManager = SqliteManager
