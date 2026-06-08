import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class RecordingsCRUD:

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

