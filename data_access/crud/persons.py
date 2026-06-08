import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class PersonsCRUD:

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

