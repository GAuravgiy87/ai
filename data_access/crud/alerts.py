import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class AlertsCRUD:

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

