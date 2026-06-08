import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class JourneysCRUD:

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

