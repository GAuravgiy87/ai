import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class CamerasCRUD:

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

