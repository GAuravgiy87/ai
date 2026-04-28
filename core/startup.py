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
                logger.info(f"[OK] Global Re-ID: Loaded {len(self.identities)} active identities.")
            except Exception as e:
                logger.error(f"[FAIL] Global Re-ID Load Error: {e}")

    def match(self, encoding, threshold=0.55):
        # Issue 9 Fix: Tightened re-ID threshold from 0.75 to 0.55
        # Prevents merging different people into one global ID
        # More conservative matching improves unique visitor counting accuracy
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



def analytics_snapshot_task(db_manager, camera_manager):
    """Periodically store analytics snapshots for historical tracking."""
    while True:
        try:
            time.sleep(300)  # Run every 5 minutes
            
            # Store current metrics
            active_cameras = len(camera_manager.cameras)
            db_manager.store_analytics_snapshot(
                metric_type='active_cameras_periodic',
                value=active_cameras,
                metadata={'source': 'background_task'}
            )
            
            # Store daily stats (same as live stream shows)
            try:
                stats = db_manager.get_camera_daily_person_stats()
                total_count = sum(s.get("total", 0) for s in stats.values())
                
                # Store overall total
                db_manager.store_analytics_snapshot(
                    metric_type='total_count_day_periodic',
                    value=total_count,
                    metadata={'period': 'day', 'source': 'background_task', 'stats': stats}
                )
                
                # Store per-camera totals
                for cam_id, cam_stats in stats.items():
                    db_manager.store_analytics_snapshot(
                        metric_type='camera_total_count_day',
                        value=cam_stats.get("total", 0),
                        camera_id=cam_id,
                        metadata={
                            'am': cam_stats.get("am", 0),
                            'pm': cam_stats.get("pm", 0),
                            'source': 'background_task'
                        }
                    )
            except Exception as e:
                logger.error(f"Error storing daily stats: {e}")
            
            # Store total counts for different periods
            for period in ['week', 'month']:
                try:
                    count = db_manager.get_total_detections_count(period=period)
                    db_manager.store_analytics_snapshot(
                        metric_type=f'total_count_{period}_periodic',
                        value=count,
                        metadata={'period': period, 'source': 'background_task'}
                    )
                except Exception as e:
                    logger.error(f"Error storing {period} count: {e}")
            
            logger.debug("[Analytics] Periodic snapshot stored")
        except Exception as e:
            logger.error(f"[FAIL] Analytics snapshot task error: {e}")

def restore_cameras(db_manager, camera_manager):
    """Background task to restore cameras from DB."""
    try:
        # Wait a bit for server to fully bind
        time.sleep(2)
        logger.info("[Startup] Restoring persistent cameras...")
        cameras = db_manager.get_cameras()
        for cam_id, source in cameras:
            # Auto-probe bare RTSP URLs and persist the working one
            if isinstance(source, str) and source.startswith("rtsp://"):
                from cameras.camera_manager import probe_rtsp_url
                new_source = probe_rtsp_url(source)
                if new_source != source:
                    logger.info(f"[Startup] Updating {cam_id} source to probed working path: {new_source}")
                    db_manager.update_camera_source(cam_id, new_source)
                source = new_source

            parsed_source = int(source) if str(source).isdigit() else source
            if camera_manager.add_camera(cam_id, parsed_source):
                threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
                logger.info(f"[Startup] Restored camera: {cam_id}")
    except Exception as e:
        logger.error(f"[Startup] Camera restoration error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI, db_manager, camera_manager):
    """FastAPI initialization."""
    notification_manager.set_loop(asyncio.get_event_loop())
    # Start restoration in a safe background thread
    threading.Thread(target=restore_cameras, args=(db_manager, camera_manager), daemon=True).start()
    yield

def load_models(db_manager):
    """Initialize system models."""
    # Issue 3 Fix: Upgraded from yolov8n to yolov8s for better accuracy
    # yolov8s provides significantly lower false positive rate with minimal
    # performance impact on i7-8700 systems with hardware encoding offload
    detector = PersonDetector(model_path='yolov8s.pt')
    try:
        recognizer = FaceRecognizer()
        recognizer.load_known_faces(db_manager)
    except Exception as e:
        logger.critical(f"FaceRecognizer init failed: {e}")
        return None, None, None
    reid_manager = GlobalReIDManager(db_manager)
    return detector, recognizer, reid_manager
