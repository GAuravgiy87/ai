import os
import threading
import pytz
from datetime import datetime
from typing import Dict, Any, Optional, Set
from fastapi.templating import Jinja2Templates

# Setup IST Timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST."""
    return datetime.now(IST)

def format_12h(dt):
    """Format datetime to 12-hour AM/PM string."""
    if dt is None: return "N/A"
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%I:%M:%S %p")

# Directories
SNAPSHOTS_DIR = "snapshots"
DATASET_DIR = "dataset"
RECORDINGS_DIR = os.path.abspath("recordings")

for d in [SNAPSHOTS_DIR, DATASET_DIR, RECORDINGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Templates setup
templates = Jinja2Templates(directory="templates")
templates.env.cache_size = 0

# --- Shared Global State ---

# Per-camera: latest tracks for video overlay
camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()

# Recording service (set by app.py after initialization)
recording_service = None

# Per-camera: recognized persons info
camera_recognized_persons: Dict[str, Dict[int, str]] = {}
recognized_lock = threading.Lock()

# Occupancy state
occupancy_last_count: Dict[str, int] = {}
occupancy_last_track_ids: Dict[str, Set[int]] = {}

# Alert & Snapshot Throttling
alert_cooldowns: Dict[str, float] = {}
ALERT_COOLDOWN_SECONDS = 30
snapshot_cooldowns = {}
SNAPSHOT_COOLDOWN_SECONDS = 60.0
MAX_CACHE_SIZE = 200

# Active Search State
active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()

# Recognition Cooldown caches
recognition_cooldowns: Dict[tuple, float] = {}
cooldown_lock = threading.Lock()

# Global Re-ID Identity Mapping
global_reid_assignments: Dict[tuple, str] = {}
reid_lock = threading.Lock()

def sanitize_rtsp_url(url: str) -> str:
    """Percent-encode special characters in the password portion of an RTSP URL."""
    if not isinstance(url, str):
        return url
    url = url.strip()
    if not url.startswith("rtsp://"):
        return url
    rest = url[7:]
    last_at = rest.rfind("@")
    if last_at == -1: return url
    auth_part = rest[:last_at]
    host_part = rest[last_at + 1:]
    colon = auth_part.find(":")
    if colon == -1: return url
    user = auth_part[:colon]
    pwd  = auth_part[colon + 1:]
    safe_pwd = pwd.replace("@", "%40")
    return f"rtsp://{user}:{safe_pwd}@{host_part}"
