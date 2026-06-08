import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class AnalyticsCRUD:

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



