import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class DetectionsCRUD:

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

