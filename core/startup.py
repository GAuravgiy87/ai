"""
core/startup.py — App startup: DB, models, GlobalReIDManager, NotificationManager, lifespan.
"""
import asyncio
import atexit
import logging
import random
import signal
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np

from core.state import (
    PRIO_LOW, PRIO_NORMAL, _set_thread_priority,
    camera_recognized_persons, recognized_lock, MAX_CACHE_SIZE, _prune_dict,
)
from core.logging_config import (
    MinimalDBLogHandler, capture_system_snapshot, format_snapshot,
)

logger = logging.getLogger(__name__)

# Re-export so app.py can import from one place
DBLogHandler = MinimalDBLogHandler


# ---------------------------------------------------------------------------
# GlobalReIDManager
# ---------------------------------------------------------------------------
class GlobalReIDManager:
    def __init__(self, db_manager):
        self.db   = db_manager
        self.lock = threading.Lock()
        self._enc_matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._ids: list = []
        self._load_identities()

    def _load_identities(self):
        with self.lock:
            try:
                data = self.db.get_recent_active_targets(hours=24)
                encs, ids = [], []
                for item in data:
                    enc = item["encoding"]
                    enc = np.frombuffer(enc, dtype=np.float32) if isinstance(enc, bytes) \
                          else np.array(enc, dtype=np.float32)
                    encs.append(enc)
                    ids.append(item["global_id"])
                self._ids        = ids
                self._enc_matrix = np.stack(encs) if encs else np.empty((0, 512), dtype=np.float32)
            except Exception as e:
                logger.error(f"Global Re-ID load error: {e}")

    def match(self, encoding, threshold=0.75):
        if encoding is None:
            return None
        with self.lock:
            if not self._ids:
                return None
            dists   = np.linalg.norm(self._enc_matrix - encoding, axis=1)
            min_idx = int(np.argmin(dists))
            return self._ids[min_idx] if dists[min_idx] < threshold else None

    def register_new(self, encoding, thumbnail_binary=None):
        with self.lock:
            new_id = f"U-{random.randint(1000, 9999)}"
            while new_id in self._ids:
                new_id = f"U-{random.randint(1000, 9999)}"
            self._ids.append(new_id)
            self._enc_matrix = (
                np.vstack([self._enc_matrix, encoding[np.newaxis]])
                if len(self._ids) > 1 else encoding[np.newaxis].copy()
            )
            self.db.upsert_global_unknown(new_id, encoding, thumbnail_binary)
            return new_id


# ---------------------------------------------------------------------------
# NotificationManager (SSE broadcast)
# ---------------------------------------------------------------------------
class NotificationManager:
    def __init__(self):
        self.clients = []
        self.lock    = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop):
        self._loop = loop

    async def subscribe(self):
        import asyncio as _asyncio
        q = _asyncio.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, data: dict):
        import json
        msg = f"data: {json.dumps(data)}\n\n"
        with self.lock:
            loop    = self._loop
            clients = list(self.clients)
        if loop is None or not loop.is_running():
            return
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Storage cleanup background task
# ---------------------------------------------------------------------------
def _storage_optimization_task(db_manager):
    _set_thread_priority(PRIO_LOW)
    while True:
        try:
            import time, os
            time.sleep(1800)
            paths   = db_manager.cleanup_old_data(snapshot_hours=24, recording_days=2)
            deleted = sum(1 for p in paths if p and os.path.exists(p) and _safe_remove(p))
            if deleted:
                # Only log if something actually happened — no noise
                db_manager.log_event("INFO", f"Storage cleanup: {deleted} files removed",
                                     source="system.cleanup")
            with recognized_lock:
                for cam in list(camera_recognized_persons.keys()):
                    _prune_dict(camera_recognized_persons[cam], MAX_CACHE_SIZE)
        except Exception as e:
            logger.error(f"Storage optimization error: {e}")


def _safe_remove(path: str) -> bool:
    try:
        import os
        os.remove(path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model loading (background — server starts immediately)
# ---------------------------------------------------------------------------
detector          = None
recognizer        = None
_detector_ready   = threading.Event()
_recognizer_ready = threading.Event()


def _load_models_bg(db_manager):
    _set_thread_priority(PRIO_NORMAL)
    global detector, recognizer
    from utils.detector   import PersonDetector
    from utils.recognizer import FaceRecognizer

    try:
        detector = PersonDetector()
        _detector_ready.set()
    except Exception as e:
        logger.error(f"YOLO detector failed to load: {e}")
        _detector_ready.set()

    try:
        recognizer = FaceRecognizer()
        recognizer.load_known_faces(db_manager)
        _recognizer_ready.set()
    except Exception as e:
        logger.error(f"FaceRecognizer failed to load: {e}")
        db_manager.log_event("ERROR", f"FaceRecognizer init failed: {e}",
                             source="system.startup")
        _recognizer_ready.set()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
def build_lifespan(db_manager, camera_manager, notification_manager):
    from core.pipeline import process_camera

    @asynccontextmanager
    async def lifespan(app):
        notification_manager.set_loop(asyncio.get_event_loop())

        def _start_one(cam_id, source):
            try:
                parsed = int(source) if str(source).isdigit() else source
                if camera_manager.add_camera(cam_id, parsed):
                    threading.Thread(
                        target=process_camera, args=(cam_id,),
                        daemon=True, name=f"cam-{cam_id}"
                    ).start()
            except Exception as e:
                logger.error(f"Camera {cam_id} restore failed: {e}")
                db_manager.log_event("ERROR", f"Camera restore failed {cam_id}: {e}",
                                     source="system.startup")

        def _start_all():
            cams = db_manager.get_cameras()
            if not cams:
                return
            threads = [
                threading.Thread(target=_start_one, args=(cid, src),
                                 daemon=True, name=f"cam-init-{cid}")
                for cid, src in cams
            ]
            for t in threads:
                t.start()

        threading.Thread(target=_start_all, daemon=True, name="startup-cameras").start()
        yield

    return lifespan


# ---------------------------------------------------------------------------
# Signal + exception hooks — with system snapshots
# ---------------------------------------------------------------------------
def install_signal_hooks(db_manager):

    def _on_shutdown():
        """Called by atexit — clean exit."""
        try:
            snap = capture_system_snapshot(reason="clean shutdown")
            db_manager.log_event(
                "INFO",
                f"System shutdown | {format_snapshot(snap)}",
                source="system.shutdown",
            )
            db_manager.purge_old_logs(keep_days=60)
        except Exception:
            pass

    atexit.register(_on_shutdown)

    def _signal_handler(signum, frame):
        names = {signal.SIGTERM: "SIGTERM (service stop / kill)",
                 signal.SIGINT:  "SIGINT (Ctrl-C)"}
        sig_name = names.get(signum, f"signal {signum}")
        try:
            snap = capture_system_snapshot(reason=sig_name)
            db_manager.log_event(
                "WARNING",
                f"Process terminated: {sig_name} | {format_snapshot(snap)}",
                source="system.signal",
            )
            # Print to terminal so operator sees it immediately
            print(f"\n\033[93m[SHUTDOWN] Received {sig_name} — shutting down cleanly.\033[0m",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    def _excepthook(exc_type, exc_value, exc_tb):
        """Unhandled exception — print full cause to terminal, save to DB."""
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        # Terminal — always visible, full detail
        print(
            f"\n\033[1m\033[91m{'='*60}\n"
            f"  CRITICAL CRASH — {exc_type.__name__}\n"
            f"  Cause: {exc_value}\n"
            f"{'='*60}\033[0m",
            file=sys.stderr, flush=True,
        )
        print(f"\033[91m{tb_str}\033[0m", file=sys.stderr, flush=True)

        # System snapshot at crash time
        try:
            snap = capture_system_snapshot(reason=f"crash: {exc_type.__name__}: {exc_value}")
            snap_str = format_snapshot(snap)
            print(f"\033[93mSystem state at crash: {snap_str}\033[0m",
                  file=sys.stderr, flush=True)
        except Exception:
            snap_str = "snapshot unavailable"

        # Save to DB
        try:
            db_manager.log_event(
                "CRITICAL",
                f"CRASH: {exc_type.__name__}: {exc_value} | {snap_str}",
                source="system.crash",
                extra=tb_str,
            )
        except Exception:
            pass

        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook


# ---------------------------------------------------------------------------
# Startup system snapshot — logged once when app starts
# ---------------------------------------------------------------------------
def log_startup_snapshot(db_manager):
    """Log system state at startup — CPU, RAM, GPU, platform."""
    try:
        snap     = capture_system_snapshot(reason="startup")
        snap_str = format_snapshot(snap)
        db_manager.log_event(
            "INFO",
            f"System started | {snap_str}",
            source="system.startup",
        )
        # Also print a clean summary to terminal
        print(
            f"\033[96m✓ AI Vigilance started\033[0m  "
            f"cpu={snap.get('cpu_pct','?')}%  "
            f"ram={snap.get('ram_pct','?')}%  "
            f"gpu={snap.get('gpu_name','none')}",
            flush=True,
        )
    except Exception:
        pass
