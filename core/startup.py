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
# Global Re-ID Manager
# ─────────────────────────────────────────────────────────────────────────────

class GlobalReIDManager:
    """Cross-camera person re-identification using face encodings."""

    def __init__(self, db_manager):
        self.db         = db_manager
        self.lock       = threading.Lock()
        self.identities = []
        self._next_uid  = 1000  # monotonic counter — BUG-11 fix
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
                # BUG-11 fix: seed counter from highest existing U-ID to avoid collisions
                existing_uids = [
                    int(i["id"].split("-")[1])
                    for i in self.identities
                    if isinstance(i["id"], str) and i["id"].startswith("U-")
                    and i["id"].split("-")[1].isdigit()
                ]
                if existing_uids:
                    self._next_uid = max(existing_uids) + 1
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
            # BUG-11 fix: use monotonic counter instead of random 4-digit int
            # (random had only 9000 unique IDs and a TOCTOU collision window)
            new_id = f"U-{self._next_uid}"
            self._next_uid += 1
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
            active_cameras = len(asyncio.run(list_cameras()))
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
    Wires the SSE event loop.
    """
    notification_manager.set_loop(asyncio.get_event_loop())
    yield


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
