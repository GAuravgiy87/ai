"""
PostgreSQL Database Manager for AI Vigilance (Microservice Edition).

Drop-in replacement for SqliteManager — identical public API, but uses
psycopg2 against a PostgreSQL server so that multiple containers can
share the same database concurrently.

Key differences from SQLite version:
  - Connection pooling (psycopg2.pool.ThreadedConnectionPool)
  - %s parameter bindings instead of ?
  - SERIAL instead of AUTOINCREMENT
  - BYTEA instead of BLOB
  - No PRAGMA statements
  - ON CONFLICT uses Postgres syntax
"""

import os
import json
import logging
import time
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import pytz

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)


class SqliteCursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def _translate(self, query, params=None):
        if params is None:
            params = ()
        if isinstance(params, (list, tuple)):
            params = tuple(self._unwrap_param(p) for p in params)
        return query.replace('%s', '?'), params

    def _unwrap_param(self, param):
        if isinstance(param, (bytes, bytearray)):
            return bytes(param)
        if isinstance(param, sqlite3.Binary):
            return param
        if hasattr(param, 'tobytes'):
            return param.tobytes()
        return param

    def execute(self, query, params=None):
        query, params = self._translate(query, params)
        return self._cursor.execute(query, params)

    def executemany(self, query, params_seq):
        query = query.replace('%s', '?')
        seq = [tuple(self._unwrap_param(p) for p in params) for params in params_seq]
        return self._cursor.executemany(query, seq)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)

    def __getattr__(self, item):
        return getattr(self._cursor, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cursor.close()


class SqliteConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn
        self._default_isolation_level = conn.isolation_level
        self._autocommit = False
        self._conn.row_factory = sqlite3.Row

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._autocommit = bool(value)
        self._conn.isolation_level = None if self._autocommit else self._default_isolation_level

    def cursor(self, cursor_factory=None):
        return SqliteCursorWrapper(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def __getattr__(self, item):
        return getattr(self._conn, item)

# Default connection string — overridden by DATABASE_URL env var
DEFAULT_DSN = "postgresql://aiv_user:aiv_password@localhost:5432/aiv_db"


class PostgresManager:
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

    # ─── Cameras ──────────────────────────────────────────────────────────────

    def add_camera_to_db(self, camera_id, source):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO cameras (camera_id, source, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (camera_id) DO UPDATE SET source = EXCLUDED.source, updated_at = EXCLUDED.updated_at
                ''', (camera_id, str(source), datetime.utcnow()))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[FAIL] Error adding camera: {e}")
        finally:
            self._put_connection(conn)

    def remove_camera_from_db(self, camera_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM cameras WHERE camera_id = %s', (camera_id,))
                cur.execute('DELETE FROM camera_settings WHERE camera_id = %s', (camera_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[FAIL] Error removing camera: {e}")
        finally:
            self._put_connection(conn)

    def get_cameras(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT camera_id, source FROM cameras')
                return [[r[0], r[1]] for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def update_camera_source(self, camera_id, new_source):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('UPDATE cameras SET source = %s, updated_at = %s WHERE camera_id = %s',
                            (str(new_source), datetime.utcnow(), camera_id))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating camera source: {e}")
            return False
        finally:
            self._put_connection(conn)

    # ─── Settings ─────────────────────────────────────────────────────────────

    def get_camera_recording_setting(self, camera_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT recording_enabled FROM camera_settings WHERE camera_id = %s', (camera_id,))
                row = cur.fetchone()
                return 1 if row and row[0] else 0
        except Exception:
            return 0
        finally:
            self._put_connection(conn)

    def set_camera_recording(self, camera_id, enabled):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO camera_settings (camera_id, recording_enabled)
                    VALUES (%s, %s)
                    ON CONFLICT (camera_id) DO UPDATE SET recording_enabled = EXCLUDED.recording_enabled
                ''', (camera_id, 1 if enabled else 0))
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)

    def set_camera_tracking_area(self, camera_id, area):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO camera_settings (camera_id, tracking_area)
                    VALUES (%s, %s)
                    ON CONFLICT (camera_id) DO UPDATE SET tracking_area = EXCLUDED.tracking_area
                ''', (camera_id, json.dumps(area)))
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)

    def get_camera_tracking_area(self, camera_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT tracking_area FROM camera_settings WHERE camera_id = %s', (camera_id,))
                row = cur.fetchone()
                return json.loads(row[0]) if row and row[0] else None
        except Exception:
            return None
        finally:
            self._put_connection(conn)

    # ─── Persons ──────────────────────────────────────────────────────────────

    def register_person(self, name, image_path, encoding):
        conn = self._get_connection()
        try:
            if hasattr(encoding, 'tobytes'):
                encoding_blob = encoding.tobytes()
            else:
                encoding_blob = encoding

            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO persons (name, image_path, encoding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET image_path = EXCLUDED.image_path, encoding = EXCLUDED.encoding
                ''', (name, image_path, self._binary(encoding_blob)))
            conn.commit()
            return name
        except Exception as e:
            conn.rollback()
            logger.error(f"Error registering person: {e}")
            return None
        finally:
            self._put_connection(conn)

    def get_registered_persons(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, name, image_path, encoding FROM persons')
                return [[str(r[0]), r[1], r[2], bytes(r[3]) if r[3] else None] for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_persons_with_last_seen(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, name, image_path, last_seen, last_camera FROM persons')
                return [{
                    "id": str(r[0]),
                    "name": r[1],
                    "image_path": r[2],
                    "last_seen": r[3],
                    "last_camera": r[4]
                } for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_detections(self, limit=20):
        return self.get_registered_detections(page_size=limit)

    def rename_person(self, person_id, new_name, new_image_path=None, new_encoding=None):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if new_image_path and new_encoding is not None:
                    encoding_blob = new_encoding.tobytes() if hasattr(new_encoding, 'tobytes') else new_encoding
                    cur.execute('UPDATE persons SET name = %s, image_path = %s, encoding = %s WHERE id = %s',
                                (new_name, new_image_path, self._binary(encoding_blob), int(person_id)))
                else:
                    cur.execute('UPDATE persons SET name = %s WHERE id = %s', (new_name, int(person_id)))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error renaming person: {e}")
            return False
        finally:
            self._put_connection(conn)

    def update_person_last_seen(self, name, camera_id, snapshot_path=None):
        conn = self._get_connection()
        try:
            now = datetime.now(IST)
            with conn.cursor() as cur:
                cur.execute('UPDATE persons SET last_seen = %s, last_camera = %s WHERE name = %s',
                            (now, camera_id, name))
                cur.execute('''INSERT INTO registered_detections (person_name, camera_id, timestamp, snapshot_path)
                               VALUES (%s, %s, %s, %s)''', (name, camera_id, now, snapshot_path))
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)

    def delete_person_from_db(self, person_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM persons WHERE id = %s', (int(person_id),))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting person: {e}")
        finally:
            self._put_connection(conn)

    # ─── Registered Detections ────────────────────────────────────────────────

    def search_detections(self, name=None, start_time=None, end_time=None):
        conn = self._get_connection()
        try:
            query = "SELECT id, person_name, camera_id, timestamp, snapshot_path FROM registered_detections WHERE 1=1"
            params = []
            if name:
                query += " AND person_name = %s"; params.append(name)
            if start_time:
                query += " AND timestamp >= %s"; params.append(start_time)
            if end_time:
                query += " AND timestamp <= %s"; params.append(end_time)
            query += " ORDER BY timestamp DESC"

            with conn.cursor() as cur:
                cur.execute(query, params)
                return [[r[0], r[1], r[2], r[3], r[4], r[1]] for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_registered_detections(self, name=None, date_from=None, date_to=None, page=1, page_size=20):
        conn = self._get_connection()
        try:
            offset = (page - 1) * page_size
            query = 'SELECT person_name, camera_id, timestamp, snapshot_path FROM registered_detections WHERE 1=1'
            params = []
            if name:
                query += ' AND person_name = %s'; params.append(name)
            if date_from:
                query += ' AND timestamp >= %s'; params.append(date_from)
            if date_to:
                dt = date_to + 'T23:59:59' if 'T' not in date_to else date_to
                query += ' AND timestamp <= %s'; params.append(dt)
            query += ' ORDER BY timestamp DESC LIMIT %s OFFSET %s'
            params += [page_size, offset]

            with conn.cursor() as cur:
                cur.execute(query, params)
                return [{
                    "person_name": r[0], "camera_id": r[1],
                    "timestamp": r[2], "snapshot_path": r[3]
                } for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def count_registered_detections(self, name=None, date_from=None, date_to=None):
        conn = self._get_connection()
        try:
            query = 'SELECT COUNT(*) FROM registered_detections WHERE 1=1'
            params = []
            if name:
                query += ' AND person_name = %s'; params.append(name)
            if date_from:
                query += ' AND timestamp >= %s'; params.append(date_from)
            if date_to:
                dt = date_to + 'T23:59:59' if 'T' not in date_to else date_to
                query += ' AND timestamp <= %s'; params.append(dt)
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()[0]
        except Exception:
            return 0
        finally:
            self._put_connection(conn)

    # ─── Detection Snapshots ──────────────────────────────────────────────────

    def log_detection_snapshot(self, camera_id, person_count, snapshot_path, bbox_data,
                               face_encodings=None, person_crops=None, timestamp=None):
        conn = self._get_connection()
        try:
            if timestamp is None:
                timestamp = datetime.now(IST)
            bbox_str = bbox_data if isinstance(bbox_data, str) else json.dumps(bbox_data)
            if face_encodings:
                if isinstance(face_encodings, str):
                    face_enc_str = face_encodings
                else:
                    face_enc_str = json.dumps([e.tolist() if hasattr(e, 'tolist') else e for e in face_encodings])
            else:
                face_enc_str = None
            person_crops_str = person_crops if isinstance(person_crops, str) else (json.dumps(person_crops) if person_crops else None)

            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO detection_snapshots (camera_id, person_count, snapshot_path, bbox_data, face_encodings, person_crops, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                ''', (camera_id, int(person_count), snapshot_path, bbox_str, face_enc_str, person_crops_str, timestamp))
                row_id = cur.fetchone()[0]
            conn.commit()
            return str(row_id)
        except Exception as e:
            conn.rollback()
            logger.error(f"[FAIL] Error logging snapshot: {e}")
            return None
        finally:
            self._put_connection(conn)

    def get_detection_snapshots(self, camera_id=None, date_from=None, date_to=None,
                                 page=1, page_size=20, start_time=None, end_time=None, limit=None):
        conn = self._get_connection()
        try:
            if start_time: date_from = start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time
            if end_time:   date_to   = end_time.isoformat()   if hasattr(end_time,   'isoformat') else end_time
            if limit:      page_size = limit; page = 1

            offset = (page - 1) * page_size
            query = "SELECT id, camera_id, timestamp, person_count, snapshot_path, bbox_data, person_crops FROM detection_snapshots WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = %s"; params.append(camera_id)
            if date_from:
                query += " AND timestamp >= %s"; params.append(date_from)
            if date_to:
                dt = str(date_to) + 'T23:59:59' if 'T' not in str(date_to) else date_to
                query += " AND timestamp <= %s"; params.append(dt)
            query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            params += [page_size, offset]

            with conn.cursor() as cur:
                cur.execute(query, params)
                return [
                    [str(r[0]), r[1], r[2], r[3], r[4],
                     json.loads(r[5]) if r[5] else [],
                     json.loads(r[6]) if r[6] else []]
                    for r in cur.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting snapshots: {e}")
            return []
        finally:
            self._put_connection(conn)

    def count_detection_snapshots(self, camera_id=None, date_from=None, date_to=None):
        conn = self._get_connection()
        try:
            query = "SELECT COUNT(*) FROM detection_snapshots WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = %s"; params.append(camera_id)
            if date_from:
                query += " AND timestamp >= %s"; params.append(date_from)
            if date_to:
                dt = str(date_to) + 'T23:59:59' if 'T' not in str(date_to) else date_to
                query += " AND timestamp <= %s"; params.append(dt)
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()[0]
        except Exception:
            return 0
        finally:
            self._put_connection(conn)

    def get_snapshot(self, snapshot_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, camera_id, timestamp, person_count, snapshot_path, bbox_data FROM detection_snapshots WHERE id = %s', (int(snapshot_id),))
                r = cur.fetchone()
                if r:
                    return [str(r[0]), r[1], r[2], r[3], r[4], json.loads(r[5]) if r[5] else []]
                return None
        except Exception:
            return None
        finally:
            self._put_connection(conn)

    # ─── Occupancy ────────────────────────────────────────────────────────────

    def log_occupancy(self, camera_id, count):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('INSERT INTO occupancy_logs (camera_id, timestamp, count) VALUES (%s, %s, %s)',
                            (camera_id, datetime.now(IST), int(count)))
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)

    def search_occupancy(self, camera_id=None, start_time=None, end_time=None):
        conn = self._get_connection()
        try:
            query = "SELECT id, camera_id, timestamp, count FROM occupancy_logs WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = %s"; params.append(camera_id)
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)
            query += " ORDER BY timestamp DESC"

            with conn.cursor() as cur:
                cur.execute(query, params)
                return [[str(r[0]), r[1], r[2].isoformat() if hasattr(r[2], 'isoformat') else r[2], r[3]] for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    # ─── Recordings ───────────────────────────────────────────────────────────

    def start_recording(self, camera_id, file_path):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO video_recordings (camera_id, file_path, start_time)
                    VALUES (%s, %s, %s) RETURNING id
                ''', (camera_id, file_path, datetime.now(IST)))
                row_id = cur.fetchone()[0]
            conn.commit()
            return str(row_id)
        except Exception:
            conn.rollback()
            return None
        finally:
            self._put_connection(conn)

    def end_recording(self, record_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('UPDATE video_recordings SET end_time = %s WHERE id = %s',
                            (datetime.now(IST), int(record_id)))
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)

    def search_recordings(self, camera_id=None, start_time=None, end_time=None):
        conn = self._get_connection()
        try:
            query = "SELECT id, camera_id, start_time, end_time, file_path, has_registered_person, registered_person_times FROM video_recordings WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = %s"; params.append(camera_id)
            if start_time:
                query += " AND start_time >= %s"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND start_time <= %s"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)
            query += " ORDER BY start_time DESC"

            with conn.cursor() as cur:
                cur.execute(query, params)
                return [[str(r[0]), r[1], r[2], r[3], r[4], bool(r[5]),
                         json.loads(r[6]) if r[6] else []] for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_recorded_videos(self):
        return self.search_recordings()

    def get_recording(self, record_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, camera_id, start_time, end_time, file_path FROM video_recordings WHERE id = %s', (int(record_id),))
                r = cur.fetchone()
                return [str(r[0]), r[1], r[2], r[3], r[4]] if r else None
        except Exception:
            return None
        finally:
            self._put_connection(conn)

    def delete_recording(self, record_id):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM video_recordings WHERE id = %s', (int(record_id),))
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)

    # ─── Analytics ────────────────────────────────────────────────────────────

    def get_hourly_analytics(self, camera_id=None):
        conn = self._get_connection()
        try:
            now = datetime.utcnow()
            start_time = now - timedelta(hours=24)
            query = "SELECT camera_id, person_count, timestamp FROM detection_snapshots WHERE timestamp >= %s"
            params = [start_time]
            if camera_id:
                query += " AND camera_id = %s"; params.append(camera_id)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            hourly_data = {}
            for r in rows:
                dt = r[2]
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt).astimezone(IST)
                else:
                    dt = dt.astimezone(IST)
                h = dt.hour
                if h not in hourly_data:
                    hourly_data[h] = {"_id": h, "max_count": 0, "camera_ids": set()}
                hourly_data[h]["max_count"] = max(hourly_data[h]["max_count"], r[1])
                hourly_data[h]["camera_ids"].add(r[0])

            result = list(hourly_data.values())
            for item in result:
                item["camera_ids"] = list(item["camera_ids"])
            return sorted(result, key=lambda x: x["_id"])
        except Exception as e:
            logger.error(f"Error in get_hourly_analytics: {e}")
            return []
        finally:
            self._put_connection(conn)

    def get_daily_analytics(self, camera_id=None, days=7):
        conn = self._get_connection()
        try:
            now = datetime.utcnow()
            start_time = now - timedelta(days=days)
            query = "SELECT camera_id, person_count, timestamp FROM detection_snapshots WHERE timestamp >= %s"
            params = [start_time]
            if camera_id:
                query += " AND camera_id = %s"; params.append(camera_id)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            daily_data = {}
            for r in rows:
                dt = r[2]
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt).astimezone(IST)
                else:
                    dt = dt.astimezone(IST)
                key = (dt.year, dt.month, dt.day)
                daily_data[key] = max(daily_data.get(key, 0), r[1])

            return sorted([
                {"_id": {"year": y, "month": m, "day": d}, "max_count": c}
                for (y, m, d), c in daily_data.items()
            ], key=lambda x: (x["_id"]["year"], x["_id"]["month"], x["_id"]["day"]))
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_camera_daily_person_stats(self):
        conn = self._get_connection()
        try:
            now = datetime.now(IST)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
            today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

            def get_unique_count(start, end):
                with conn.cursor() as cur:
                    cur.execute('''
                        SELECT camera_id, COUNT(DISTINCT global_id) as total
                        FROM journeys
                        WHERE timestamp >= %s AND timestamp <= %s
                        GROUP BY camera_id
                    ''', (start, end))
                    return {r[0]: r[1] for r in cur.fetchall()}

            am_data = get_unique_count(today_start, noon)
            pm_data = get_unique_count(noon, today_end)

            all_cameras = set(list(am_data.keys()) + list(pm_data.keys()))
            stats = {}
            for cam in all_cameras:
                am = am_data.get(cam, 0)
                pm = pm_data.get(cam, 0)
                stats[cam] = {"am": am, "pm": pm, "total": am + pm}
            return stats
        except Exception as e:
            logger.error(f"get_camera_daily_person_stats error: {e}")
            return {}
        finally:
            self._put_connection(conn)

    def get_total_unique_count_today(self, camera_id):
        conn = self._get_connection()
        try:
            today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT COUNT(DISTINCT global_id) as total
                    FROM journeys
                    WHERE camera_id = %s AND timestamp >= %s
                ''', (camera_id, today_start))
                row = cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0
        finally:
            self._put_connection(conn)

    def get_total_detections_count(self, period='day', camera_id=None):
        conn = self._get_connection()
        try:
            now = datetime.now(IST)
            if period == 'day':
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == 'week':
                start_time = now - timedelta(days=7)
            elif period == 'month':
                start_time = now - timedelta(days=30)
            else:
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

            with conn.cursor() as cur:
                if camera_id:
                    cur.execute('SELECT COUNT(DISTINCT global_id) FROM journeys WHERE camera_id = %s AND timestamp >= %s',
                                (camera_id, start_time))
                else:
                    cur.execute('SELECT COUNT(DISTINCT global_id) FROM journeys WHERE timestamp >= %s', (start_time,))
                row = cur.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error getting total detections count: {e}")
            return 0
        finally:
            self._put_connection(conn)

    # ─── Alerts ───────────────────────────────────────────────────────────────

    def log_critical_alert(self, camera_id, person_id, snapshot_path):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO alerts (camera_id, person_id, snapshot_path, timestamp, type)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (camera_id, person_id, snapshot_path, datetime.now(IST), "CRITICAL"))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"[FAIL] Error logging alert: {e}")
            return False
        finally:
            self._put_connection(conn)

    def get_recent_alerts(self, limit=10):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT id, camera_id, person_id, snapshot_path, timestamp, type FROM alerts ORDER BY timestamp DESC LIMIT %s', (limit,))
                return [{
                    "id": str(r[0]), "camera_id": r[1], "person_id": r[2],
                    "snapshot_path": r[3], "timestamp": r[4], "type": r[5]
                } for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    # ─── Global Re-ID & Journeys ──────────────────────────────────────────────

    def get_all_global_identities(self):
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute('SELECT * FROM global_identities ORDER BY last_seen DESC')
                return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def upsert_global_unknown(self, global_id, encoding, thumbnail_binary=None):
        conn = self._get_connection()
        try:
            now = datetime.now(IST)
            if hasattr(encoding, 'tobytes'):
                encoding_blob = encoding.tobytes()
            else:
                encoding_blob = encoding

            thumb = self._binary(thumbnail_binary) if thumbnail_binary else None

            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO global_identities (global_id, encoding, first_seen, last_seen, type, thumbnail)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (global_id) DO UPDATE SET
                        encoding = EXCLUDED.encoding,
                        last_seen = EXCLUDED.last_seen,
                        thumbnail = CASE WHEN EXCLUDED.thumbnail IS NOT NULL THEN EXCLUDED.thumbnail ELSE global_identities.thumbnail END
                ''', (global_id, self._binary(encoding_blob), now, now, "unknown", thumb))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[FAIL] Global ID Error: {e}")
        finally:
            self._put_connection(conn)

    def log_journey_event(self, global_id, camera_id, snapshot_path=None, person_type="unknown", timestamp=None):
        conn = self._get_connection()
        try:
            now = timestamp if timestamp is not None else datetime.now(IST)
            with conn.cursor() as cur:
                cur.execute('UPDATE global_identities SET last_seen = %s, last_camera = %s, type = %s WHERE global_id = %s',
                            (now, camera_id, person_type, global_id))
                cur.execute('''
                    INSERT INTO journeys (global_id, camera_id, timestamp, snapshot_path, type)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (global_id, camera_id, now, snapshot_path, person_type))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[FAIL] Journey log error: {e}")
        finally:
            self._put_connection(conn)

    def get_target_journey(self, global_id):
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute('SELECT * FROM journeys WHERE global_id = %s ORDER BY timestamp DESC', (global_id,))
                return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_recent_active_targets(self, hours=24):
        conn = self._get_connection()
        try:
            since = datetime.now(IST) - timedelta(hours=hours)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute('SELECT * FROM global_identities WHERE last_seen > %s ORDER BY last_seen DESC', (since,))
                return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            self._put_connection(conn)

    def get_global_identity_by_id(self, global_id):
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute('SELECT * FROM global_identities WHERE global_id = %s', (global_id,))
                r = cur.fetchone()
                return dict(r) if r else None
        except Exception:
            return None
        finally:
            self._put_connection(conn)

    # ─── Analytics Storage ────────────────────────────────────────────────────

    def store_analytics_snapshot(self, metric_type, value, camera_id=None, metadata=None):
        conn = self._get_connection()
        try:
            metadata_str = json.dumps(metadata) if metadata else None
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO analytics_snapshots (timestamp, metric_type, camera_id, value, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (datetime.now(IST), metric_type, camera_id, int(value), metadata_str))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing analytics snapshot: {e}")
            return False
        finally:
            self._put_connection(conn)

    def get_analytics_history(self, metric_type, hours=24, camera_id=None):
        conn = self._get_connection()
        try:
            since = datetime.now(IST) - timedelta(hours=hours)
            query = 'SELECT timestamp, value, camera_id, metadata FROM analytics_snapshots WHERE metric_type = %s AND timestamp >= %s'
            params = [metric_type, since]
            if camera_id:
                query += ' AND camera_id = %s'; params.append(camera_id)
            query += ' ORDER BY timestamp DESC'

            with conn.cursor() as cur:
                cur.execute(query, params)
                return [{
                    "timestamp": r[0], "value": r[1], "camera_id": r[2],
                    "metadata": json.loads(r[3]) if r[3] else None
                } for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting analytics history: {e}")
            return []
        finally:
            self._put_connection(conn)

    # ─── Similarity Search ────────────────────────────────────────────────────

    def search_detections_by_encoding(self, target_encoding, threshold=1.10, start_time=None, end_time=None):
        target_v = np.array(target_encoding, dtype=np.float32)
        norm = np.linalg.norm(target_v)
        if norm > 0:
            target_v /= norm
        return self.search_snapshots_by_similarity(target_v, start_time, end_time, threshold=threshold)

    def search_snapshots_by_similarity(self, target_encoding, start_time=None, end_time=None, threshold=1.10):
        conn = self._get_connection()
        try:
            query = "SELECT id, camera_id, timestamp, snapshot_path, bbox_data, face_encodings FROM detection_snapshots WHERE face_encodings IS NOT NULL"
            params = []
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            if not rows:
                return []

            target_v = np.array(target_encoding, dtype=np.float32)
            results = []
            for r in rows:
                try:
                    encs = json.loads(r[5])
                    if not encs:
                        continue
                    enc_matrix = np.array(encs, dtype=np.float32)
                    dists = np.linalg.norm(enc_matrix - target_v, axis=1)
                    best_dist = float(np.min(dists))

                    if best_dist < threshold:
                        ts = r[2]
                        results.append({
                            "id": str(r[0]), "camera_id": r[1],
                            "timestamp": ts.strftime("%Y-%m-%d %I:%M:%S %p") if hasattr(ts, 'strftime') else str(ts),
                            "snapshot_path": r[3],
                            "bbox_data": json.loads(r[4]) if r[4] else [],
                            "distance": round(best_dist, 3),
                            "confidence": f"{max(0, 100 - (best_dist * 50)):.1f}%"
                        })
                except Exception:
                    continue

            results.sort(key=lambda x: x["distance"])
            return results
        except Exception as e:
            logger.error(f"search_snapshots_by_similarity error: {e}")
            return []
        finally:
            self._put_connection(conn)

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup_old_data(self, snapshot_hours=24, recording_days=7):
        conn = self._get_connection()
        now = datetime.utcnow()
        deleted_files = []
        try:
            snap_cutoff = now - timedelta(hours=snapshot_hours)
            with conn.cursor() as cur:
                cur.execute('SELECT snapshot_path, person_crops FROM detection_snapshots WHERE timestamp < %s', (snap_cutoff,))
                for r in cur.fetchall():
                    if r[0]: deleted_files.append(r[0])
                    if r[1]:
                        crops = json.loads(r[1])
                        deleted_files.extend(crops)
                cur.execute('DELETE FROM detection_snapshots WHERE timestamp < %s', (snap_cutoff,))

            rec_cutoff = now - timedelta(days=recording_days)
            with conn.cursor() as cur:
                cur.execute('SELECT file_path FROM video_recordings WHERE start_time < %s', (rec_cutoff,))
                for r in cur.fetchall():
                    if r[0]: deleted_files.append(r[0])
                cur.execute('DELETE FROM video_recordings WHERE start_time < %s', (rec_cutoff,))

            occ_cutoff = now - timedelta(days=7)
            with conn.cursor() as cur:
                cur.execute('DELETE FROM occupancy_logs WHERE timestamp < %s', (occ_cutoff,))

            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self._put_connection(conn)
        return deleted_files

    def delete_all_detections(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for table in ['detection_snapshots', 'registered_detections', 'journeys',
                              'global_identities', 'occupancy_logs', 'alerts', 'analytics_snapshots']:
                    cur.execute(f'DELETE FROM {table}')
            conn.commit()
            logger.info("Database historical data wiped.")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"delete_all_detections error: {e}")
            return False
        finally:
            self._put_connection(conn)

    def vacuum_database(self):
        """PostgreSQL VACUUM (does not require exclusive lock like SQLite)."""
        conn = self._get_connection()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute('VACUUM ANALYZE')
            return True
        except Exception:
            return False
        finally:
            conn.autocommit = False
            self._put_connection(conn)


class SqliteManager(PostgresManager):
    """SQLite database manager fallback for local development."""

    def __init__(self, db_path: str = None, dsn: str = None):
        self.db_path = db_path or os.environ.get('SQLITE_DB_PATH', 'data/aiv_vigilance.db')
        self._conn = None
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
