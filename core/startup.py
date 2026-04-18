import threading
import time
import os
import logging
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.state import get_ist_time
from core.pipeline import process_camera, notification_manager
from utils.detector import PersonDetector
from utils.recognizer import FaceRecognizer

logger = logging.getLogger("app.startup")

class GlobalReIDManager:
    """Manages cross-camera person re-identification using face encodings."""
    def __init__(self, db_manager):
        self.db = db_manager
        self.lock = threading.Lock()
        self.identities = [] # List of {id, encoding}
        self._load_identities()
        
    def _load_identities(self):
        with self.lock:
            try:
                data = self.db.get_recent_active_targets(hours=24)
                for item in data:
                    encoding = item["encoding"]
                    if isinstance(encoding, bytes):
                        encoding = np.frombuffer(encoding, dtype=np.float32)
                    else:
                        encoding = np.array(encoding, dtype=np.float32)
                    self.identities.append({"id": item["global_id"], "encoding": encoding})
                logger.info(f"✓ Global Re-ID: Loaded {len(self.identities)} active identities.")
            except Exception as e:
                logger.error(f"✗ Global Re-ID Load Error: {e}")

    def match(self, encoding, threshold=0.75):
        if encoding is None: return None
        with self.lock:
            best_id = None; min_dist = threshold
            for item in self.identities:
                dist = np.linalg.norm(encoding - item["encoding"])
                if dist < min_dist: min_dist = dist; best_id = item["id"]
            return best_id

    def register_new(self, encoding, thumbnail_binary=None):
        with self.lock:
            import random
            new_id = f"U-{random.randint(1000, 9999)}"
            while any(i["id"] == new_id for i in self.identities):
                new_id = f"U-{random.randint(1000, 9999)}"
            self.identities.append({"id": new_id, "encoding": encoding})
            self.db.upsert_global_unknown(new_id, encoding, thumbnail_binary)
            return new_id

def storage_optimization_task(db_manager):
    """Periodically clean old recordings and snapshots."""
    while True:
        try:
            time.sleep(3600)
            paths_to_delete = db_manager.cleanup_old_data(snapshot_hours=24, recording_days=2)
            local_deleted = 0
            for path in paths_to_delete:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        local_deleted += 1
                    except Exception: pass
            if local_deleted:
                logger.info(f"✓ Storage Cleaned: {local_deleted} local files removed.")
        except Exception as e:
            logger.error(f"✗ Storage optimization error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI, db_manager, camera_manager):
    """Reload all saved cameras from the database on startup."""
    notification_manager.set_loop(asyncio.get_event_loop())
    logger.info("[Startup] Loading persistent cameras from database...")
    cameras = db_manager.get_cameras()
    for cam_id, source in cameras:
        parsed_source = int(source) if str(source).isdigit() else source
        if camera_manager.add_camera(cam_id, parsed_source):
            threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
            logger.info(f"[Startup] Restored camera: {cam_id}")
    yield

def load_models(db_manager):
    """Initialize system models."""
    detector = PersonDetector()
    try:
        recognizer = FaceRecognizer()
        recognizer.load_known_faces(db_manager)
    except Exception as e:
        logger.critical(f"FaceRecognizer init failed: {e}")
        return None, None, None
    reid_manager = GlobalReIDManager(db_manager)
    return detector, recognizer, reid_manager
