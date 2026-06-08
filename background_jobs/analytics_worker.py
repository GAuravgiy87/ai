"""
workers/analytics_worker.py — Standalone Analytics & Re-ID Microservice

Extracted from core/startup.py's analytics_snapshot_task() and
GlobalReIDManager. Runs in its own container with access to PostgreSQL
and Redis.

Responsibilities:
  - Periodic analytics snapshots (every 5 minutes)
  - Global Re-ID identity management
  - Data cleanup (old snapshots, recordings)
"""

import os
import sys
import time
import signal
import logging
import threading
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_access.manager import DatabaseManager
from core.redis_manager import get_redis_state

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("analytics_worker")

_running = True
_db = None
_redis = None


class GlobalReIDManager:
    """Cross-camera person re-identification using face encodings."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.lock = threading.Lock()
        self.identities = []
        self._next_uid = 1000
        self._load_identities()

    def _load_identities(self):
        with self.lock:
            try:
                data = self.db.get_recent_active_targets(hours=24)
                for item in data:
                    enc = item["encoding"]
                    if isinstance(enc, (bytes, memoryview)):
                        enc = np.frombuffer(bytes(enc), dtype=np.float32)
                    else:
                        enc = np.array(enc, dtype=np.float32)
                    self.identities.append({"id": item["global_id"], "encoding": enc})

                existing_uids = [
                    int(i["id"].split("-")[1])
                    for i in self.identities
                    if isinstance(i["id"], str) and i["id"].startswith("U-")
                    and i["id"].split("-")[1].isdigit()
                ]
                if existing_uids:
                    self._next_uid = max(existing_uids) + 1
                logger.info(f"[ReID] Loaded {len(self.identities)} active identities.")
            except Exception as e:
                logger.error(f"[ReID] Load Error: {e}")

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
            new_id = f"U-{self._next_uid}"
            self._next_uid += 1
            self.identities.append({"id": new_id, "encoding": encoding})
            self.db.upsert_global_unknown(new_id, encoding, thumbnail_binary)
            return new_id


def _analytics_loop():
    """Periodically store analytics snapshots (every 5 minutes)."""
    while _running:
        try:
            time.sleep(300)
            if not _running:
                break

            # Count active cameras from Redis
            active_cameras = len(_redis.get_active_camera_ids())
            _db.store_analytics_snapshot(
                metric_type='active_cameras_periodic',
                value=active_cameras,
                metadata={'source': 'analytics_worker'},
            )

            # Daily stats
            try:
                stats = _db.get_camera_daily_person_stats()
                total_count = sum(s.get("total", 0) for s in stats.values())
                _db.store_analytics_snapshot(
                    metric_type='total_count_day_periodic',
                    value=total_count,
                    metadata={'period': 'day', 'source': 'analytics_worker', 'stats': stats},
                )
                for cam_id, cam_stats in stats.items():
                    _db.store_analytics_snapshot(
                        metric_type='camera_total_count_day',
                        value=cam_stats.get("total", 0),
                        camera_id=cam_id,
                        metadata={
                            'am': cam_stats.get("am", 0),
                            'pm': cam_stats.get("pm", 0),
                            'source': 'analytics_worker',
                        },
                    )
            except Exception as e:
                logger.error(f"[Analytics] Daily stats error: {e}")

            # Weekly/Monthly counts
            for period in ['week', 'month']:
                try:
                    count = _db.get_total_detections_count(period=period)
                    _db.store_analytics_snapshot(
                        metric_type=f'total_count_{period}_periodic',
                        value=count,
                        metadata={'period': period, 'source': 'analytics_worker'},
                    )
                except Exception as e:
                    logger.error(f"[Analytics] {period} count error: {e}")

            logger.debug("[Analytics] Periodic snapshot stored.")
        except Exception as e:
            logger.error(f"[Analytics] Snapshot task error: {e}")


def _cleanup_loop():
    """Run data cleanup every 6 hours."""
    while _running:
        time.sleep(6 * 3600)  # 6 hours
        if not _running:
            break
        try:
            deleted = _db.cleanup_old_data(snapshot_hours=24, recording_days=7)
            if deleted:
                logger.info(f"[Cleanup] Removed {len(deleted)} old files from DB records.")
                # Note: actual file deletion should be handled by the recording_service
                # container which has the volume mounted
        except Exception as e:
            logger.error(f"[Cleanup] Error: {e}")


def _signal_handler(sig, frame):
    global _running
    logger.info("[AnalyticsWorker] Shutdown signal received.")
    _running = False


def main():
    global _running, _db, _redis

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 60)
    logger.info("  AI Vigilance — Analytics Worker Starting")
    logger.info("=" * 60)

    # Wait for dependencies
    for i in range(30):
        try:
            _redis = get_redis_state()
            if _redis.ping():
                break
        except Exception:
            pass
        logger.info(f"[AnalyticsWorker] Waiting for Redis... ({i+1}/30)")
        time.sleep(2)

    _db = DatabaseManager()
    _redis = get_redis_state()

    # Initialize Re-ID manager (loads identities from DB)
    reid_manager = GlobalReIDManager(_db)
    logger.info("[AnalyticsWorker] Re-ID manager initialized.")

    # Start background threads
    analytics_thread = threading.Thread(target=_analytics_loop, daemon=True, name="AnalyticsLoop")
    analytics_thread.start()

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="CleanupLoop")
    cleanup_thread.start()

    logger.info("[AnalyticsWorker] Running. Press Ctrl+C to stop.")

    # Keep main thread alive
    while _running:
        time.sleep(1)

    logger.info("[AnalyticsWorker] Exited.")


if __name__ == "__main__":
    main()
