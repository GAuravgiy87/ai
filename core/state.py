"""
core/state.py — Single source of truth for all shared runtime state.

Every module imports from here. Nothing is duplicated.
Import order: state.py has no internal app imports — safe to import first.
"""
import os
import sys
import queue
import threading
import logging
import shutil
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Set

import numpy as np
import pytz
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")

def get_ist_time() -> datetime:
    return datetime.now(IST)

def format_12h(dt) -> str:
    if dt is None: return "N/A"
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%I:%M:%S %p")

def format_full_dt(dt) -> str:
    if dt is None: return "N/A"
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%a, %d %b %Y • %I:%M:%S %p")

def format_date_key(dt) -> str:
    if dt is None: return "Unknown"
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%A, %d %b %Y")

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
SNAPSHOTS_DIR       = "snapshots"
DATASET_DIR         = "dataset"
RECORDINGS_DIR      = "recordings"
LOCAL_RECORDINGS_DIR = "recordings"

SNAPSHOT_COOLDOWN_SECONDS = 60.0
MAX_CACHE_SIZE            = 100

os.makedirs(SNAPSHOTS_DIR,  exist_ok=True)
os.makedirs(DATASET_DIR,    exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Thread priority
# ---------------------------------------------------------------------------
PRIO_HIGH   = "high"
PRIO_NORMAL = "normal"
PRIO_LOW    = "low"

def _set_thread_priority(level: str):
    try:
        if sys.platform == "win32":
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentThread()
            prio_map = {PRIO_HIGH: 2, PRIO_NORMAL: 0, PRIO_LOW: -2}
            ctypes.windll.kernel32.SetThreadPriority(handle, prio_map.get(level, 0))
        else:
            import os as _os
            nice_map = {PRIO_HIGH: -5, PRIO_NORMAL: 0, PRIO_LOW: 10}
            _os.nice(nice_map.get(level, 0))
    except Exception:
        pass

def _make_executor(max_workers: int, priority: str, name_prefix: str) -> ThreadPoolExecutor:
    _idx = [0]
    def _init():
        _set_thread_priority(priority)
        _idx[0] += 1
        threading.current_thread().name = f"{name_prefix}-{_idx[0]}"
    return ThreadPoolExecutor(max_workers=max_workers, initializer=_init)

# ---------------------------------------------------------------------------
# Shared camera state
# ---------------------------------------------------------------------------
camera_results:             Dict[str, Any]           = {}
results_lock                                         = threading.Lock()

camera_recognized_persons:  Dict[str, Dict[int, str]] = {}
recognized_lock                                      = threading.Lock()

camera_writers:             Dict[str, Any]           = {}
writer_lock                                          = threading.Lock()

occupancy_last_count:       Dict[str, int]           = {}
occupancy_last_track_ids:   Dict[str, Set[int]]      = {}
snapshot_cooldowns:         Dict[str, float]         = {}

alert_cooldowns:            Dict[str, float]         = {}
ALERT_COOLDOWN_SECONDS                               = 30

recording_threads:          Dict[str, Any]           = {}
recording_stop_events:      Dict[str, threading.Event] = {}

global_reid_assignments:    Dict[tuple, str]         = {}
reid_lock                                            = threading.Lock()

active_search:              Dict[str, Any]           = {}
active_search_lock                                   = threading.Lock()

recognition_cooldowns:      Dict[tuple, float]       = {}
cooldown_lock                                        = threading.Lock()

# ---------------------------------------------------------------------------
# Executors & I/O queue
# ---------------------------------------------------------------------------
recognition_executor = _make_executor(max_workers=2, priority=PRIO_NORMAL, name_prefix="recog")
transfer_queue       = queue.Queue(maxsize=40)

def _prune_dict(d: dict, max_size: int):
    if len(d) > max_size:
        keys = list(d.keys())
        for k in keys[:len(keys) // 2]:
            d.pop(k, None)

def _cleanup_executor():
    try:
        recognition_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

atexit.register(_cleanup_executor)

# ---------------------------------------------------------------------------
# Transfer workers (disk I/O)
# ---------------------------------------------------------------------------
def _perform_direct_stream(data: bytes, local_path: str) -> bool:
    try:
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        logger.error(f"[LocalSave] {e}")
        return False

def _perform_actual_process(src_path: str, dest_dir: str) -> bool:
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_path, dest_dir)
        return True
    except Exception:
        return False

def _transfer_worker():
    _set_thread_priority(PRIO_LOW)
    while True:
        try:
            item = transfer_queue.get()
            if item is None:
                break
            data, destination, callback = item
            if isinstance(data, (bytes, bytearray)):
                success = _perform_direct_stream(data, destination)
            else:
                success = _perform_actual_process(data, destination)
            if callback:
                callback(success)
            transfer_queue.task_done()
        except Exception as e:
            logger.error(f"[TransferWorker] {e}")

for _i in range(2):
    threading.Thread(target=_transfer_worker, daemon=True, name=f"transfer-{_i}").start()

def stream_bytes_to_local(data: bytes, local_path: str, callback=None) -> bool:
    try:
        transfer_queue.put((data, local_path, callback), block=False)
        return True
    except queue.Full:
        return False

def save_to_local(local_path: str, destination_dir: str, callback=None) -> bool:
    try:
        transfer_queue.put((local_path, destination_dir, callback), block=False)
        return True
    except queue.Full:
        return False

# ---------------------------------------------------------------------------
# RTSP URL sanitiser (used by cameras route + camera_manager)
# ---------------------------------------------------------------------------
def sanitize_rtsp_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    url = url.strip()
    if not url.startswith("rtsp://"):
        return url
    rest    = url[7:]
    last_at = rest.rfind("@")
    if last_at == -1:
        return url
    auth_part = rest[:last_at]
    host_part = rest[last_at + 1:]
    colon = auth_part.find(":")
    if colon == -1:
        return url
    user     = auth_part[:colon]
    pwd      = auth_part[colon + 1:]
    safe_pwd = pwd.replace("@", "%40")
    return f"rtsp://{user}:{safe_pwd}@{host_part}"
