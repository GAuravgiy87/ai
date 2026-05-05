"""
core/startup.py — Main app startup helpers.

The camera server (port 9001) is started here as a daemon thread inside
the same Python process.  No subprocess, no separate terminal needed.
"""

import threading
import time
import logging
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.state import get_ist_time
from core.pipeline import notification_manager

logger = logging.getLogger("app.startup")


# ─────────────────────────────────────────────────────────────────────────────
# Camera server — started as a daemon thread inside this process
# ─────────────────────────────────────────────────────────────────────────────

_cam_server_thread: threading.Thread = None


def start_camera_server():
    """
    Launch the camera server (port 9001) in a daemon thread.
    Returns immediately; the server starts in the background.
    """
    global _cam_server_thread

    from camera_server.client import is_alive
    if is_alive():
        logger.info("[Startup] Camera server already running on :9001")
        return

    def _run():
        from camera_server.server import start
        start()   # blocks inside uvicorn.run() — that's fine for a daemon thread

    _cam_server_thread = threading.Thread(target=_run, name="camera-server", daemon=True)
    _cam_server_thread.start()
    logger.info("[Startup] Camera server thread started — waiting for :9001 to be ready...")

    # Wait up to 15 s for the server to accept connections
    from camera_server.client import is_alive
    for _ in range(30):
        time.sleep(0.5)
        if is_alive():
            logger.info("[Startup] Camera server is ready on :9001")
            return
    logger.warning("[Startup] Camera server did not respond within 15 s — continuing anyway.")


# ─────────────────────────────────────────────────────────────────────────────
# Global Re-ID Manager
# ─────────────────────────────────────────────────────────────────────────────

class GlobalReIDManager:
    """Cross-camera person re-identification using face encodings."""

    def __init__(self, db_manager):
        self.db         = db_manager
        self.lock       = threading.Lock()
        self.identities = []
        self._load_identities()

    def _load_identities(self):
        with self.lock:
            try:
                data = self.db.get_recent_active_targets(hours=24)
                for item in data:
                    enc = item["encoding"]
                    if isinstance(enc, bytes):
                        enc = np.frombuffer(enc, dtype=np.float32)
                    else:
                        enc = np.array(enc, dtype=np.float32)
                    self.identities.append({"id": item["global_id"], "encoding": enc})
                logger.info(f"[OK] Global Re-ID: Loaded {len(self.identities)} active identities.")
            except Exception as e:
                logger.error(f"[FAIL] Global Re-ID Load Error: {e}")

    def match(self, encoding, threshold=0.55):
        if encoding is None:
            return None
        with self.lock:
            best_id, min_dist = None, threshold
            for item in self.identities:
                dist = np.linalg.norm(encoding - item["encoding"])
                if dist < min_dist:
                    min_dist, best_id = dist, item["id"]
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


# ─────────────────────────────────────────────────────────────────────────────
# Analytics background task
# ─────────────────────────────────────────────────────────────────────────────

def analytics_snapshot_task(db_manager):
    """Periodically store analytics snapshots (every 5 minutes)."""
    while True:
        try:
            time.sleep(300)

            from camera_server.client import list_cameras
            active_cameras = len(list_cameras())
            db_manager.store_analytics_snapshot(
                metric_type='active_cameras_periodic',
                value=active_cameras,
                metadata={'source': 'background_task'},
            )

            try:
                stats       = db_manager.get_camera_daily_person_stats()
                total_count = sum(s.get("total", 0) for s in stats.values())
                db_manager.store_analytics_snapshot(
                    metric_type='total_count_day_periodic',
                    value=total_count,
                    metadata={'period': 'day', 'source': 'background_task', 'stats': stats},
                )
                for cam_id, cam_stats in stats.items():
                    db_manager.store_analytics_snapshot(
                        metric_type='camera_total_count_day',
                        value=cam_stats.get("total", 0),
                        camera_id=cam_id,
                        metadata={
                            'am':     cam_stats.get("am", 0),
                            'pm':     cam_stats.get("pm", 0),
                            'source': 'background_task',
                        },
                    )
            except Exception as e:
                logger.error(f"[Analytics] Daily stats error: {e}")

            for period in ['week', 'month']:
                try:
                    count = db_manager.get_total_detections_count(period=period)
                    db_manager.store_analytics_snapshot(
                        metric_type=f'total_count_{period}_periodic',
                        value=count,
                        metadata={'period': period, 'source': 'background_task'},
                    )
                except Exception as e:
                    logger.error(f"[Analytics] {period} count error: {e}")

            logger.debug("[Analytics] Periodic snapshot stored.")
        except Exception as e:
            logger.error(f"[FAIL] Analytics snapshot task error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI, db_manager):
    """
    Called by FastAPI on startup/shutdown.
    Starts the camera server thread and wires the SSE event loop.
    """
    notification_manager.set_loop(asyncio.get_event_loop())
    start_camera_server()
    yield
    # Camera server is a daemon thread — it dies automatically with the process.


# ─────────────────────────────────────────────────────────────────────────────
# Model loader stub
# ─────────────────────────────────────────────────────────────────────────────

def load_models(db_manager):
    """
    AI models are owned by the camera server.
    Returns (None, None, None) so existing call-sites in app.py don't break.
    """
    logger.info("[Startup] AI models are managed by the camera server (:9001).")
    return None, None, None
