import cv2
import sys
import numpy as np
import os
import shutil
import json
import torch
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from database.sqlite_manager import SqliteManager
from utils.detector import PersonDetector
from utils.tracker import ObjectTracker
from utils.recognizer import FaceRecognizer
from cameras.camera_manager import CameraManager
import threading
import time
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
import queue
import asyncio
import logging
import json
import base64
import random
import subprocess

# Setup logging — file only, terminal stays clean
LOG_FILE = "app.log"

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["PYTHONWARNINGS"] = "ignore"

print("✓ AI Vigilance System Starting...")

_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
_file_h = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a')
_file_h.setFormatter(_fmt)
_file_h.setLevel(logging.INFO)

logging.root.handlers.clear()
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_h)

logger = logging.getLogger(__name__)

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Route uvicorn logs to file only — no terminal output
for _uv in ("uvicorn", "uvicorn.access", "uvicorn.error", "uvicorn.lifespan"):
    _lg = logging.getLogger(_uv)
    _lg.handlers.clear()
    _lg.propagate = False
    _lg.addHandler(_file_h)

# Redirect print() to file so camera thread prints don't hit terminal
import builtins as _builtins
_orig_print = _builtins.print
def _silent_print(*args, **kwargs):
    # Only allow explicit terminal=True prints
    if kwargs.pop("terminal", False):
        _orig_print(*args, **kwargs)
    else:
        msg = " ".join(str(a) for a in args)
        logger.info(msg)
_builtins.print = _silent_print

# Set IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST."""
    return datetime.now(IST)

def format_12h(dt):
    """Format datetime to 12-hour AM/PM string (e.g. 05:30:15 PM)."""
    if dt is None: return "N/A"
    # Convert to IST if needed
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%I:%M:%S %p")

# Security setup
security = HTTPBasic(auto_error=False)

# Simple admin credentials (in production, use database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "deiadmin@789"

# Session storage (in production, use proper session management)
authenticated_sessions: set = set()

# Snapshot throttling & structure
snapshot_cooldowns = {}
SNAPSHOT_COOLDOWN_SECONDS = 30.0  # Max 1 snapshot per 30 seconds per camera
MAX_CACHE_SIZE = 200  # Max entries in per-camera caches before pruning

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials."""
    if credentials:
        is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if is_correct_username and is_correct_password:
            return credentials.username
    return None

def require_auth(request: Request):
    """Check if user is authenticated via session cookie."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        return True
    return False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_rtsp_url(url: str) -> str:
    """Percent-encode special characters in the password portion of an RTSP URL.
    Handles passwords containing multiple '@' signs by using rfind to locate the
    last '@' as the user:pass / host boundary.
    """
    if not isinstance(url, str):
        return url
    url = url.strip()
    if not url.startswith("rtsp://"):
        return url

    # Everything after rtsp://
    rest = url[7:]
    last_at = rest.rfind("@")
    if last_at == -1:
        return url  # No auth in URL

    auth_part = rest[:last_at]       # e.g. "test:dei@12@12"
    host_part = rest[last_at + 1:]   # e.g. "10.7.16.48:554"

    colon = auth_part.find(":")
    if colon == -1:
        return url  # No password, nothing to encode

    user = auth_part[:colon]
    pwd  = auth_part[colon + 1:]     # e.g. "dei@12@12"

    # Encode only '@' in the password — FFmpeg requires this
    safe_pwd = pwd.replace("@", "%40")

    return f"rtsp://{user}:{safe_pwd}@{host_part}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

# Local storage paths
SNAPSHOTS_DIR = "snapshots"
DATASET_DIR = "dataset"
RECORDINGS_DIR = "recordings"
LOCAL_RECORDINGS_DIR = "recordings"  # alias used throughout recording logic

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reload all saved cameras from the database on startup."""
    # Store the running event loop so worker threads can broadcast SSE events
    notification_manager.set_loop(asyncio.get_event_loop())
    logger.info("[Startup] Loading persistent cameras from database...")
    cameras = db_manager.get_cameras()
    for cam_id, source in cameras:
        # Handle webcam IDs stored as strings
        parsed_source = source
        if str(source).isdigit():
            parsed_source = int(source)
        
        if camera_manager.add_camera(cam_id, parsed_source):
            threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
            logger.info(f"[Startup] Restored camera: {cam_id}")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

# Configure Jinja2 templates with cache disabled to avoid unhashable type error
templates = Jinja2Templates(directory="templates")
templates.env.cache_size = 0

# Initialize Database Manager (SQLite)
try:
    from database.sqlite_manager import SqliteManager
    db_manager = SqliteManager()
    logger.info("✓ Connected to SQLite (Local)")
except Exception as e:
    logger.critical(f"✗ Failed to connect to SQLite: {e}")
    # Force exit if DB is unreachable
    import sys
    sys.exit(1)

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
                    # SQLite stores as BLOB (bytes), MongoDB used list
                    encoding = item["encoding"]
                    if isinstance(encoding, bytes):
                        encoding = np.frombuffer(encoding, dtype=np.float32)
                    else:
                        encoding = np.array(encoding, dtype=np.float32)
                        
                    self.identities.append({
                        "id": item["global_id"],
                        "encoding": encoding
                    })
                logger.info(f"✓ Global Re-ID: Loaded {len(self.identities)} active identities.")
            except Exception as e:
                logger.error(f"✗ Global Re-ID Load Error: {e}")

    def match(self, encoding, threshold=0.75):
        """Find matching global ID for an encoding. Threshold tuned for InceptionResnetV1."""
        if encoding is None: return None
        with self.lock:
            best_id = None
            min_dist = threshold
            for item in self.identities:
                dist = np.linalg.norm(encoding - item["encoding"])
                if dist < min_dist:
                    min_dist = dist
                    best_id = item["id"]
            return best_id

    def register_new(self, encoding, thumbnail_binary=None):
        """Register a new unknown person in the global registry."""
        with self.lock:
            # Generate a slightly random/unique ID to avoid collisions
            import random
            new_id = f"U-{random.randint(1000, 9999)}"
            while any(i["id"] == new_id for i in self.identities):
                new_id = f"U-{random.randint(1000, 9999)}"
                
            self.identities.append({"id": new_id, "encoding": encoding})
            self.db.upsert_global_unknown(new_id, encoding, thumbnail_binary)
            return new_id

detector = PersonDetector()
recognizer = FaceRecognizer()
camera_manager = CameraManager()
recognizer.load_known_faces(db_manager)
reid_manager = GlobalReIDManager(db_manager)

# Global ID mapping: (camera_id, track_id) -> global_id (Unknown ID or Registered Name)
global_reid_assignments: Dict[tuple, str] = {}
reid_lock = threading.Lock()

class NotificationManager:
    """Manages real-time event broadcasting to multiple web clients via SSE."""
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Store the running event loop so worker threads can schedule onto it."""
        self._loop = loop

    async def subscribe(self):
        """Add a new client queue for SSE."""
        q = asyncio.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        """Remove a client queue."""
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, data: dict):
        """Push a message to all connected clients (safe to call from any thread)."""
        msg = f"data: {json.dumps(data)}\n\n"
        with self.lock:
            loop = self._loop
            clients = list(self.clients)
        if loop is None or not loop.is_running():
            return
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

notification_manager = NotificationManager()

# ---------------------------------------------------------------------------
# Background task: Clean old logs (older than 8 hours)
# ---------------------------------------------------------------------------
def storage_optimization_task():
    """Periodically clean old recordings and snapshots to save disk space."""
    while True:
        try:
            # Run cleanup every hour
            time.sleep(3600)
            
            # Retention Policy (Optimized for 10GB Storage)
            SNAPSHOT_RETENTION_HOURS = 24
            RECORDING_RETENTION_DAYS = 2
            
            # 1. Clean DB and get paths to delete
            paths_to_delete = db_manager.cleanup_old_data(
                snapshot_hours=SNAPSHOT_RETENTION_HOURS, 
                recording_days=RECORDING_RETENTION_DAYS
            )
            
            # 2. Perform file deletion
            local_deleted = 0
            
            
            for path in paths_to_delete:
                if not path: continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        local_deleted += 1
                except Exception: pass
            
            if local_deleted:
                logger.info(f"✓ Storage Cleaned: {local_deleted} local files removed.")
                
        except Exception as e:
            logger.error(f"✗ Storage optimization error: {e}")

# Start combined cleanup thread
cleanup_thread = threading.Thread(target=storage_optimization_task, daemon=True)
cleanup_thread.start()

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Per-camera: latest tracks for video overlay
camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()  # Single shared lock for camera_results

# Per-camera: recognized persons info
camera_recognized_persons: Dict[str, Dict[int, str]] = {}
recognized_lock = threading.Lock()

# Recording state
camera_writers: Dict[str, Any] = {}
writer_lock = threading.Lock()
occupancy_last_count: Dict[str, int] = {}
occupancy_last_track_ids: Dict[str, Set[int]] = {}
alert_cooldowns: Dict[str, float] = {}  # {camera_id: last_alert_time}
ALERT_COOLDOWN_SECONDS = 30 # Don't log same intrusion for 30s

# Recording frame writer threads
recording_threads: Dict[str, Any] = {}
recording_stop_events: Dict[str, threading.Event] = {}

# Resource management
recognition_executor = ThreadPoolExecutor(max_workers=2)  # Reduced from 4 to save RAM
transfer_queue = queue.Queue(maxsize=50)
recognition_cooldowns: Dict[tuple, float] = {}  # (camera_id, track_id) -> last_process_time
cooldown_lock = threading.Lock()

def _prune_dict(d: dict, max_size: int):
    """Remove oldest half of dict entries when it exceeds max_size."""
    if len(d) > max_size:
        keys = list(d.keys())
        for k in keys[:len(keys)//2]:
            d.pop(k, None)

import atexit
def _cleanup_executor():
    try:
        recognition_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
atexit.register(_cleanup_executor)

def transfer_worker():
    """Background worker to process filesystem tasks sequentially."""
    print("[TransferWorker] Started")
    while True:
        try:
            # item can be (local_path, remote_dir, callback) OR (bytes_data, local_path, callback)
            item = transfer_queue.get()
            if item is None: break # Sentinel
            
            data, destination, callback = item
            
            if isinstance(data, (bytes, bytearray)):
                # Case 1: Binary data stream
                success = _perform_direct_stream(data, destination)
            else:
                # Case 2: Local file path processing
                success = _perform_actual_process(data, destination)
                
            if callback:
                callback(success)
            
            transfer_queue.task_done()
        except Exception as e:
            print(f"[TransferWorker] Error: {e}")
            time.sleep(1)

def _perform_direct_stream(data: bytes, local_path: str) -> bool:
    """Stream binary data directly to a local file."""
    try:
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"[Local Save] Error: {e}")
        return False

def _perform_actual_process(src_path: str, dest_dir: str) -> bool:
    """Copy files to local storage."""
    try:
        import shutil
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_path, dest_dir)
        return True
    except Exception:
        return False

# Start transfer worker
threading.Thread(target=transfer_worker, daemon=True).start()

def stream_bytes_to_local(data: bytes, local_path: str, callback=None) -> bool:
    """Queue binary data for direct saving to local disk."""
    try:
        transfer_queue.put((data, local_path, callback), block=False)
        return True
    except queue.Full:
        return False

# Keep for compatibility where files are still used (if any)
def save_to_local(local_path: str, destination_dir: str, callback=None) -> bool:
    """Push local file processing task to the background queue."""
    try:
        transfer_queue.put((local_path, destination_dir, callback), block=False)
        return True
    except queue.Full:
        return False


def recording_writer_thread(camera_id: str, stop_event: threading.Event):
    """Background thread to write frames to FFmpeg stdin for direct streaming."""
    logger.info(f"[Recording:{camera_id}] Writer thread started (Streaming)")
    
    FRAME_INTERVAL = 0.5  # 2 FPS
    
    while not stop_event.is_set():
        try:
            with writer_lock:
                if camera_id not in camera_writers:
                    break
                # Only need the stdin pipe
                process = camera_writers[camera_id].get("process")
            
            # Get latest frame
            with results_lock:
                data = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")
                # Clear it immediately after grabbing to free RAM
                if frame is not None and "rendered_frame" in data:
                    data["rendered_frame"] = None
            
            if frame is not None and process and process.poll() is None:
                try:
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                except (IOError, BrokenPipeError):
                    print(f"[Recording:{camera_id}] Pipe broken, stopping writer")
                    break
            
            # Sleep to maintain 2 FPS
            time.sleep(FRAME_INTERVAL)
            
        except Exception as e:
            print(f"[Recording:{camera_id}] Writer error: {e}")
            time.sleep(1)
    
    print(f"[Recording:{camera_id}] Writer thread stopped")

# Active search mission — set by /api/start_search, cleared by /api/stop_search
# {person_id, name, encoding, found_track_ids: set, running: bool}
active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Passive camera processing — ONLY detection + tracking, NO recognition
# ---------------------------------------------------------------------------

def process_camera(camera_id: str):
    """
    Pipeline architecture for stable 2 FPS live feed:
      - Main loop: frame grab + YOLO detect + track + render  (~80-150ms)
      - Face thread: MTCNN + recognition runs async via executor (non-blocking)
    No ghosting: camera_results only updated with a fully rendered frame.
    """
    logger.info(f"[Camera:{camera_id}] Processing thread started (2 FPS pipeline)")

    # Wait for camera to be ready
    warmup_frames = 0
    while warmup_frames < 5:
        frame, _ = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None:
            warmup_frames += 1
        time.sleep(0.1)
    logger.info(f"[Camera:{camera_id}] Camera ready")

    # Force Recording: Always ON
    with writer_lock:
        if camera_id not in camera_writers:
            try:
                h, w = frame.shape[:2]
                ist_now = get_ist_time()
                date_str = ist_now.strftime("%Y-%m-%d")
                timestamp = ist_now.strftime("%H%M%S")
                dir_path = f"{LOCAL_RECORDINGS_DIR}/{date_str}/{camera_id}"
                os.makedirs(dir_path, exist_ok=True)
                filename = f"{camera_id}_{date_str}_{timestamp}.mp4"
                local_path = f"{dir_path}/{filename}"
                scale_w = min(w, 1280) - (min(w, 1280) % 2)
                scale_h = int(h * scale_w / w) - (int(h * scale_w / w) % 2)
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2",
                    "-i", "-",
                    "-vf", f"scale={scale_w}:{scale_h}",
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "faster", "-crf", "32", "-tune", "fastdecode",
                    "-movflags", "+faststart", local_path
                ]
                p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                db_id = db_manager.start_recording(camera_id, local_path)
                stop_event = threading.Event()
                r_thread = threading.Thread(target=recording_writer_thread,
                                            args=(camera_id, stop_event), daemon=True)
                r_thread.start()
                camera_writers[camera_id] = {
                    "process": p_ffmpeg, "db_id": db_id, "start_time": ist_now,
                    "file_path": local_path, "camera_id": camera_id, "w": w, "h": h
                }
                recording_threads[camera_id] = r_thread
                recording_stop_events[camera_id] = stop_event
                logger.info(f"[Recording:{camera_id}] Auto-started (FFmpeg)")
            except Exception as err:
                logger.error(f"Failed to auto-start FFmpeg for {camera_id}: {err}")

    # Per-camera state
    tracker = ObjectTracker(max_age=20, n_init=1, iou_threshold=0.25)
    frame_count = 0
    FRAME_INTERVAL = 0.5          # 2 FPS
    RECOGNITION_CACHE_FRAMES = 4
    FACE_DETECT_EVERY = 3         # run MTCNN every 3rd frame (every 1.5s)

    recognition_cache:       Dict[Any, tuple]       = {}
    current_frame_track_ids: set                    = set()
    face_encoding_cache:     Dict[int, np.ndarray]  = {}
    track_merge_map:         Dict[int, int]          = {}
    track_face_crops:        Dict[int, tuple]        = {}
    identity_snap_cooldowns: Dict[tuple, float]      = {}

    # Deadline-based timer — never drifts
    next_frame_time = time.time()

    def get_person_color(pid):
        hue = (pid * 137) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])

    while True:
        # ── 1. Sleep until next deadline ──────────────────────────────────
        now = time.time()
        sleep_t = next_frame_time - now
        if sleep_t > 0:
            time.sleep(sleep_t)
        next_frame_time += FRAME_INTERVAL
        # If we fell behind (heavy load), reset deadline — don't burst
        if next_frame_time < time.time():
            next_frame_time = time.time() + FRAME_INTERVAL

        # ── 2. Grab latest frame ──────────────────────────────────────────
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None:
            continue

        frame_count += 1

        try:
            h, w = frame.shape[:2]

            # Downscale to 720p max
            if w > 1280:
                proc_w, proc_h = 1280, int(h * 1280 / w)
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                h, w = proc_h, proc_w
            else:
                proc_frame = frame

            # ── 3. YOLO detect + track (fast, ~50ms) ─────────────────────
            detections = detector.detect(proc_frame)
            tracks = tracker.update(detections, proc_frame)

            # NMS on overlapping boxes
            tracks = sorted(tracks, key=lambda x: x["id"])
            final_tracks = []
            for t1 in tracks:
                keep = True
                for t2 in final_tracks:
                    b1, b2 = t1["bbox"], t2["bbox"]
                    ix = max(0, min(b1[2],b2[2]) - max(b1[0],b2[0]))
                    iy = max(0, min(b1[3],b2[3]) - max(b1[1],b2[1]))
                    inter = ix * iy
                    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
                    if union > 0 and inter/union > 0.7:
                        keep = False; break
                if keep:
                    final_tracks.append(t1)
            tracks = final_tracks

            new_track_ids = set(t["id"] for t in tracks)
            if new_track_ids != current_frame_track_ids:
                logger.info(f"[Camera:{camera_id}] Persons: {len(tracks)}")
            current_frame_track_ids = new_track_ids

            # ── 4. Build processed list from recognition cache ────────────
            processed = []
            for t in tracks:
                tid = t["id"]
                name, conf = "Unknown", 0.0
                if tid in recognition_cache:
                    cn, cc, cf = recognition_cache[tid]
                    if (frame_count - cf) < RECOGNITION_CACHE_FRAMES:
                        name, conf = cn, cc
                processed.append({"id": tid, "bbox": t["bbox"],
                                   "name": name, "confidence": conf})

            # ── 5. Submit recognition workers (non-blocking) ─────────────
            for t in processed:
                tid = t["id"]
                if tid in recognition_cache and \
                   (frame_count - recognition_cache[tid][2]) < (RECOGNITION_CACHE_FRAMES // 2):
                    continue
                now_t = time.time()
                with cooldown_lock:
                    last_t = recognition_cooldowns.get((camera_id, tid), 0)
                    cooldown = 10.0 if t["name"] != "Unknown" else 2.0
                    if now_t - last_t < cooldown:
                        continue
                    recognition_cooldowns[(camera_id, tid)] = now_t
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                bw, bh = bx2-bx1, by2-by1
                face_box = [bx1+int(0.15*bw), by1, bx2-int(0.15*bw), by1+int(0.45*bh)]
                try:
                    recognition_executor.submit(
                        self_recognition_worker,
                        proc_frame.copy(), face_box, tid,
                        recognition_cache, frame_count,
                        face_encoding_cache, track_merge_map, camera_id
                    )
                except RuntimeError:
                    break

            # ── 6. Render overlay ─────────────────────────────────────────
            record_frame = proc_frame.copy()
            final_processed = []
            run_face_detect = (frame_count % FACE_DETECT_EVERY == 0)

            for t in processed:
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                name, conf, tid = str(t["name"]), float(t["confidence"]), int(t["id"])

                if name != "Unknown":
                    body_color = (0, 255, 0)
                    label = name
                else:
                    base_tid = tid
                    while base_tid in track_merge_map:
                        base_tid = track_merge_map[base_tid]
                    body_color = get_person_color(base_tid)
                    label = f"#{base_tid}"

                # MTCNN face detection — only on throttled frames
                face_visible = False
                face_box_coords = None
                if run_face_detect:
                    bw_t, bh_t = bx2-bx1, by2-by1
                    head_y2 = by1 + int(bh_t * 0.35)
                    head_crop = proc_frame[max(0,by1):head_y2, max(0,bx1):bx2]
                    if head_crop.size > 0:
                        try:
                            head_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                            with recognizer.ai_lock:
                                boxes_f, probs_f = recognizer.mtcnn.detect(head_rgb)
                            if boxes_f is not None and len(boxes_f) > 0:
                                best_idx = int(np.argmax(
                                    [p if p is not None else 0 for p in probs_f]))
                                best_prob = probs_f[best_idx] or 0
                                if best_prob > 0.80:
                                    fb = boxes_f[best_idx]
                                    candidate = (
                                        max(0, bx1+int(fb[0])),
                                        max(0, by1+int(fb[1])),
                                        min(w-1, bx1+int(fb[2])),
                                        min(h-1, by1+int(fb[3]))
                                    )
                                    # Deduplicate across tracks
                                    dup = False
                                    for prev in [p.get("face_box_coords")
                                                 for p in final_processed
                                                 if p.get("face_box_coords")]:
                                        px1,py1,px2,py2 = prev
                                        cx1,cy1,cx2,cy2 = candidate
                                        ix = max(0, min(px2,cx2)-max(px1,cx1))
                                        iy = max(0, min(py2,cy2)-max(py1,cy1))
                                        inter = ix*iy
                                        union = (px2-px1)*(py2-py1)+(cx2-cx1)*(cy2-cy1)-inter
                                        if union > 0 and inter/union > 0.4:
                                            dup = True; break
                                    if not dup:
                                        face_visible = True
                                        face_box_coords = candidate
                                        fx1c,fy1c,fx2c,fy2c = candidate
                                        fw, fh2 = fx2c-fx1c, fy2c-fy1c
                                        if fw >= 40 and fh2 >= 40 and best_prob > 0.92:
                                            fc_img = proc_frame[fy1c:fy2c, fx1c:fx2c]
                                            if fc_img.size > 0:
                                                fc_r = cv2.resize(fc_img, (120,120))
                                                _, fc_buf = cv2.imencode(
                                                    '.jpg', fc_r,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                                                existing = track_face_crops.get(tid)
                                                if existing is None or best_prob > existing[1]:
                                                    track_face_crops[tid] = (
                                                        fc_buf.tobytes(), float(best_prob))
                        except Exception:
                            pass

                # Draw boxes
                cv2.rectangle(record_frame, (bx1,by1), (bx2,by2), body_color, 2)
                if face_visible and face_box_coords:
                    cv2.rectangle(record_frame,
                                  (face_box_coords[0], face_box_coords[1]),
                                  (face_box_coords[2], face_box_coords[3]),
                                  (255,255,0), 1)
                face_ind = " [F]" if face_visible else " [B]"
                cv2.putText(record_frame, label+face_ind,
                            (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_color, 2)

                # Face crop for sidebar (only when face visible)
                cropped_face = None
                if face_visible and face_box_coords:
                    try:
                        fx1c,fy1c,fx2c,fy2c = face_box_coords
                        fi = proc_frame[fy1c:fy2c, fx1c:fx2c]
                        if fi.size > 0:
                            fi = cv2.resize(fi, (100,120))
                            _, buf = cv2.imencode('.jpg', fi,
                                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
                            cropped_face = base64.b64encode(buf).decode('utf-8')
                    except Exception:
                        pass

                final_processed.append({
                    "id": tid, "bbox": [bx1,by1,bx2,by2],
                    "name": name, "confidence": conf,
                    "face_crop": cropped_face,
                    "face_visible": face_visible,
                    "face_box_coords": face_box_coords
                })

            processed = final_processed
            people_count = len(processed)

            # ── 7. Prune caches ───────────────────────────────────────────
            if frame_count % 100 == 0:
                _prune_dict(face_encoding_cache, MAX_CACHE_SIZE)
                _prune_dict(track_merge_map, MAX_CACHE_SIZE)
                _prune_dict(recognition_cache, MAX_CACHE_SIZE)
                _prune_dict(track_face_crops, MAX_CACHE_SIZE)
                with cooldown_lock:
                    _prune_dict(recognition_cooldowns, MAX_CACHE_SIZE * 4)
                with reid_lock:
                    _prune_dict(global_reid_assignments, MAX_CACHE_SIZE * 4)

            # ── 8. Encode frame and publish atomically ────────────────────
            _, _enc = cv2.imencode('.jpg', record_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 82])
            enc_bytes = _enc.tobytes()

            with results_lock:
                camera_results[camera_id] = {
                    "rendered_frame": record_frame,
                    "encoded_frame":  enc_bytes,
                    "frame_id":       frame_count,
                    "tracks":         processed,
                    "count":          people_count,
                    "alert_active":   False,
                    "timestamp":      time.time()
                }

            # ── 9. Occupancy + snapshot logging ──────────────────────────
            try:
                current_ids = set(t["id"] for t in processed)
                last_ids = occupancy_last_track_ids.get(camera_id, set())
                if current_ids != last_ids:
                    occupancy_last_track_ids[camera_id] = current_ids
                    db_manager.log_occupancy(camera_id, len(current_ids))
                    occupancy_last_count[camera_id] = len(current_ids)

                    if len(current_ids) > 0:
                        snap_now = time.time()
                        if snap_now - snapshot_cooldowns.get(camera_id, 0) >= SNAPSHOT_COOLDOWN_SECONDS:
                            snapshot_cooldowns[camera_id] = snap_now
                            now_ist = get_ist_time()
                            date_str = now_ist.strftime("%Y-%m-%d")
                            ts_str   = now_ist.strftime("%H%M%S")
                            dir_path = f"{SNAPSHOTS_DIR}/{date_str}/{camera_id}/logs"
                            os.makedirs(dir_path, exist_ok=True)
                            local_snapshot_path = f"{dir_path}/{camera_id}_{date_str}_{ts_str}.jpg"

                            snap_processed = []
                            current_encodings = []
                            for t in processed:
                                snap_processed.append({
                                    "id": t["id"], "bbox": t["bbox"],
                                    "name": t["name"],
                                    "face_visible": t.get("face_visible", False),
                                    "face_box": list(t["face_box_coords"])
                                              if t.get("face_box_coords") else None
                                })
                                if t["id"] in face_encoding_cache:
                                    current_encodings.append(face_encoding_cache[t["id"]])

                            snap_w = min(record_frame.shape[1], 1280)
                            snap_h = int(record_frame.shape[0] * snap_w / record_frame.shape[1])
                            snap_frame = cv2.resize(record_frame, (snap_w, snap_h)) \
                                         if record_frame.shape[1] > 1280 else record_frame
                            _, sbuf = cv2.imencode('.jpg', snap_frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 88])
                            img_bytes = sbuf.tobytes()

                            def on_snap_done(success, _cam=camera_id,
                                             _cnt=len(current_ids),
                                             _path=local_snapshot_path,
                                             _bbox=snap_processed,
                                             _encs=current_encodings,
                                             _ts=now_ist):
                                if success:
                                    db_manager.log_detection_snapshot(
                                        _cam, _cnt, _path, _bbox,
                                        face_encodings=_encs, timestamp=_ts)

                            stream_bytes_to_local(img_bytes, local_snapshot_path,
                                                  callback=on_snap_done)
            except Exception as e:
                print(f"[Camera:{camera_id}] Snapshot error: {e}")

            # ── 10. Identity snapshot for recognised persons ──────────────
            with recognized_lock:
                recognized_dict = {}
                for t in processed:
                    if t["name"] != "Unknown" and float(t["confidence"]) > 0.40:
                        recognized_dict[t["id"]] = t["name"]
                        snap_key = (camera_id, t["name"])
                        if time.time() - identity_snap_cooldowns.get(snap_key, 0) < 30.0:
                            continue
                        identity_snap_cooldowns[snap_key] = time.time()
                        try:
                            bx1,by1,bx2,by2 = [int(v) for v in t["bbox"]]
                            face_only = None
                            tid_r = int(t["id"])
                            if tid_r in track_face_crops:
                                fc_b, _ = track_face_crops[tid_r]
                                arr = np.frombuffer(fc_b, dtype=np.uint8)
                                face_only = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            body_crop = proc_frame[max(0,by1):by2, max(0,bx1):bx2]
                            snap_path = None
                            if body_crop.size > 0:
                                ist2 = get_ist_time()
                                id_dir = (f"{SNAPSHOTS_DIR}/{ist2.strftime('%Y-%m-%d')}"
                                          f"/{camera_id}/identities")
                                os.makedirs(id_dir, exist_ok=True)
                                snap_path = (f"{id_dir}/id_{t['name']}_"
                                             f"{ist2.strftime('%H%M%S%f')[:12]}.jpg")
                                TARGET_H = 300
                                bh2 = body_crop.shape[0]
                                bsc = TARGET_H / bh2 if bh2 > 0 else 1
                                body_r = cv2.resize(body_crop,
                                                    (max(1,int(body_crop.shape[1]*bsc)),
                                                     TARGET_H))
                                if face_only is not None and face_only.size > 0:
                                    fr = cv2.resize(face_only, (TARGET_H, TARGET_H))
                                    cv2.rectangle(fr, (0,0), (TARGET_H-1,24), (0,0,0), -1)
                                    cv2.putText(fr, "FACE", (8,17),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                                (0,255,100), 1)
                                    composite = np.hstack([fr, body_r])
                                else:
                                    cv2.rectangle(body_r, (0,0),
                                                  (body_r.shape[1]-1,24), (0,0,0), -1)
                                    cv2.putText(body_r, t["name"], (8,17),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0,255,100), 1)
                                    composite = body_r
                                cv2.imwrite(snap_path, composite,
                                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                            db_manager.update_person_last_seen(
                                t["name"], camera_id, snap_path)
                        except Exception as e:
                            print(f"[Camera:{camera_id}] Identity snap error: {e}")
                            db_manager.update_person_last_seen(
                                t["name"], camera_id, None)
                camera_recognized_persons[camera_id] = recognized_dict

            # ── 11. Auto-split recording every hour ───────────────────────
            with writer_lock:
                wd = camera_writers.get(camera_id)
                if wd and "process" in wd:
                    ist_now = get_ist_time()
                    if (ist_now - wd["start_time"]).total_seconds() > 3600:
                        try:
                            wd["process"].stdin.close()
                            wd["process"].wait(timeout=10)
                            db_manager.end_recording(wd["db_id"])
                            new_ist = get_ist_time()
                            ds = new_ist.strftime("%Y-%m-%d")
                            nts = new_ist.strftime("%H%M%S")
                            dp = f"{RECORDINGS_DIR}/{ds}/{camera_id}"
                            os.makedirs(dp, exist_ok=True)
                            nlp = f"{dp}/rec_{camera_id}_{nts}.mp4"
                            sw, sh = wd['w'], wd['h']
                            spw = min(sw,1280) - (min(sw,1280)%2)
                            sph = int(sh*spw/sw) - (int(sh*spw/sw)%2)
                            fc = [
                                "ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
                                "-s",f"{sw}x{sh}","-pix_fmt","bgr24","-r","2",
                                "-i","-","-vf",f"scale={spw}:{sph}",
                                "-vcodec","libx264","-pix_fmt","yuv420p",
                                "-preset","faster","-crf","32",
                                "-tune","fastdecode","-movflags","+faststart",nlp
                            ]
                            pf = subprocess.Popen(fc, stdin=subprocess.PIPE,
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.DEVNULL)
                            nid = db_manager.start_recording(camera_id, nlp)
                            camera_writers[camera_id] = {
                                "process": pf, "db_id": nid,
                                "start_time": new_ist, "file_path": nlp,
                                "camera_id": camera_id, "w": sw, "h": sh
                            }
                        except Exception as e:
                            print(f"[Camera:{camera_id}] Auto-split error: {e}")

        except Exception as e:
            print(f"[Camera:{camera_id}] Error: {e}")
            import traceback; traceback.print_exc()
    
    # Wait for camera to be ready
    warmup_frames = 0
    while warmup_frames < 5:
        frame, _ = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None:
            warmup_frames += 1
        time.sleep(0.1)
    logger.info(f"[Camera:{camera_id}] Camera ready - Processing at 2 FPS")
    
    # Force Recording: Always ON per user requirement
    with writer_lock:
        if camera_id not in camera_writers:
            try:
                # Dimensions should be known from dummy get_camera_frame_with_id
                h, w = frame.shape[:2]
                ist_now = get_ist_time()
                date_str = ist_now.strftime("%Y-%m-%d")
                timestamp = ist_now.strftime("%H%M%S")
                
                # Organized Structure: Day -> Camera
                dir_path = f"{LOCAL_RECORDINGS_DIR}/{date_str}/{camera_id}"
                os.makedirs(dir_path, exist_ok=True)
                
                filename = f"{camera_id}_{date_str}_{timestamp}.mp4"
                local_path = f"{dir_path}/{filename}"
                
                # Scale down to 720p max to save disk + RAM
                scale_w = min(w, 1280)
                scale_h = int(h * scale_w / w) if w > 1280 else h
                # Ensure even dimensions for yuv420p
                scale_w = scale_w - (scale_w % 2)
                scale_h = scale_h - (scale_h % 2)
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                    "-r", "2",
                    "-i", "-",
                    "-vf", f"scale={scale_w}:{scale_h}",
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "faster",   # better compression than ultrafast, still fast
                    "-crf", "32",          # higher = smaller file (was 28)
                    "-tune", "fastdecode",
                    "-movflags", "+faststart",
                    local_path
                ]
                
                p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                db_id = db_manager.start_recording(camera_id, local_path)
                
                stop_event = threading.Event()
                r_thread = threading.Thread(target=recording_writer_thread, args=(camera_id, stop_event), daemon=True)
                r_thread.start()
                
                camera_writers[camera_id] = {
                    "process": p_ffmpeg,
                    "db_id": db_id,
                    "start_time": ist_now,
                    "file_path": local_path,
                    "camera_id": camera_id,
                    "w": w, "h": h
                }
                recording_threads[camera_id] = r_thread
                recording_stop_events[camera_id] = stop_event
                logger.info(f"[Recording:{camera_id}] Auto-started constant stream (FFmpeg)")
            except Exception as err:
                logger.error(f"Failed to auto-start FFmpeg for {camera_id}: {err}")



def self_recognition_worker(frame, face_box, track_id, recognition_cache, frame_count, face_encoding_cache, track_merge_map, camera_id):
    """Background task for periodic biometric verification with global Re-ID."""
    try:
        name, conf, face_encoding = recognizer.recognize_with_encoding(frame, face_box)
        
        # 1. Update local caches for track deduplication
        if face_encoding is not None:
            face_encoding_cache[track_id] = face_encoding
            
            # Check for duplicate tracks in this camera
            for other_id, other_encoding in face_encoding_cache.items():
                if other_id != track_id:
                    distance = np.linalg.norm(face_encoding - other_encoding)
                    if distance < 0.6:
                        if track_id < other_id:
                            track_merge_map[other_id] = track_id
                        else:
                            track_merge_map[track_id] = other_id
                        break
        
        # 2. Update recognition cache if registered person
        if name != "Unknown" and conf > 0.40:
            recognition_cache[track_id] = (name, conf, frame_count)
            
        # 3. GLOBAL RE-ID & JOURNEY LOGGING
        global_id = None
        
        # Case A: Person is recognized as Registered
        if name != "Unknown" and conf > 0.40:
            global_id = name
        
        # Case B: Person is Unknown - attempt Global Re-ID
        elif face_encoding is not None:
            # Check if already matched in this session
            with reid_lock:
                global_id = global_reid_assignments.get((camera_id, track_id))
            
            if not global_id:
                # Attempt to match against global registry
                matched_id = reid_manager.match(face_encoding)
                
                if matched_id:
                    global_id = matched_id
                else:
                    # Register as a new global unknown
                    # Use a thumbnail for the journey record
                    try:
                        fx1, fy1, fx2, fy2 = face_box
                        crop = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                        if crop.size > 0:
                            _, buf = cv2.imencode('.jpg', crop)
                            thumbnail = buf.tobytes()
                        else:
                            thumbnail = None
                    except: thumbnail = None
                    
                    global_id = reid_manager.register_new(face_encoding, thumbnail)
        
        # Update global mapping and log sighting
        if global_id:
            with reid_lock:
                # Only update if changed or new
                old_gid = global_reid_assignments.get((camera_id, track_id))
                if old_gid != global_id:
                    global_reid_assignments[(camera_id, track_id)] = global_id
                    
                    # 1. Log sighting to database (No physical image needed to save space)
                    now_ist = get_ist_time()
                    person_type_str = "unknown" if "U-" in str(global_id) else "registered"
                    
                    db_manager.log_journey_event(
                        global_id=global_id,
                        camera_id=camera_id,
                        snapshot_path=None,
                        person_type=person_type_str,
                        timestamp=now_ist
                    )
                    
                    # 3. Broadcast Live Notification (Only for Registered Persons)
                    try:
                        is_unknown = "U-" in str(global_id)
                        if not is_unknown:
                            thumb_url = f"https://ui-avatars.com/api/?name={str(global_id)}&background=random"
                            
                            notification_manager.broadcast({
                                "type": "detection",
                                "camera": camera_id,
                                "target": str(global_id),
                                "thumbnail": thumb_url,
                                "time": now_ist.strftime("%I:%M %p"),
                                "is_registered": True
                            })
                    except Exception: pass
                    
                    # logger.info(f"[Global Re-ID] Linked {camera_id}:{track_id} -> {global_id}")

    except Exception as e:
        logger.error(f"[Worker Error] {e}")


# Active search mission logic removed. Detection is now always active.


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Check authentication
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/journey", response_class=HTMLResponse)
async def journey_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "journey.html", {})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})

@app.post("/api/login")
async def api_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        import uuid
        session_token = str(uuid.uuid4())
        authenticated_sessions.add(session_token)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session", value=session_token, httponly=True)
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        authenticated_sessions.discard(session_token)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "dashboard.html", {})

@app.get("/api/dashboard_metrics")
async def dashboard_metrics(request: Request):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Calculate metrics
    active_cameras = len(camera_manager.cameras)
    registered_persons = len(db_manager.get_registered_persons())
    total_recordings = len(db_manager.get_recorded_videos())
    
    try:
        raw = db_manager.get_detections()  # list of dicts: person_name, camera_id, timestamp, snapshot_path
        # newest first, limit 20
        raw = sorted(raw, key=lambda x: x.get("timestamp") or "", reverse=True)[:20]

        # get profile images for registered persons
        persons_db = db_manager.get_registered_persons()
        person_images = {p[1]: p[2] for p in persons_db}
        cameras = {c[0]: c[1] for c in db_manager.get_cameras()}

        recent_detections = []
        for d in raw:
            pname = d.get("person_name", "Unknown")
            snap  = d.get("snapshot_path")
            img   = snap or person_images.get(pname)
            cam   = d.get("camera_id", "")
            ts    = d.get("timestamp")
            recent_detections.append({
                "person_name": pname,
                "person_names": [pname],
                "image_path": img,
                "camera_id": cam,
                "timestamp": format_12h(ts) if ts else "—",
            })
    except Exception as e:
        print(f"Error fetching detections: {e}")
        recent_detections = []
        
    return {
        "active_cameras": active_cameras,
        "registered_persons": registered_persons,
        "total_recordings": total_recordings,
        "recent_detections": recent_detections
    }

@app.get("/api/server_time")
async def get_server_time():
    """Return the current server time in IST for frontend clock sync."""
    now = get_ist_time()
    return {
        "iso": now.isoformat(),
        "timestamp_ms": int(now.timestamp() * 1000),
        "display": now.strftime("%d %b %Y, %I:%M:%S %p"),
        "timezone": "Asia/Kolkata (IST)"
    }

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "search.html", {})

# ═══════════════════════════════════════════════════════════════════════════
# NEW SEARCH & FORENSICS API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/search")
async def api_search(
    name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Search detection history by name and/or date range."""
    # Note: Using existing search_detections which returns [[id, name, cam, ts, ...]]
    results = db_manager.search_detections(name, start_time, end_time)
    res = []
    for r in results:
        res.append({
            "id": r[0],
            "person_name": r[1] or "Unknown",
            "camera_id": r[2],
            "timestamp": r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
            "image_path": r[4], 
            "face_path": r[4],
        })
    return res

@app.post("/api/search_by_image")
async def search_by_image_api(file: UploadFile = File(...)):
    """Upload a face image — finds all detections of the matching registered person."""
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    encoding = recognizer.get_encoding(image)
    if encoding is None:
        return []

    best_person_name = None
    min_dist = 1.0

    # Match against registered persons
    persons = db_manager.get_registered_persons()
    for p in persons:
        # p = [id, name, image_path, encoding_blob]
        if p[3] is not None:
            db_enc = np.frombuffer(p[3], dtype=np.float32)
            dist = float(np.linalg.norm(db_enc - encoding))
            if dist < min_dist:
                min_dist = dist
                best_person_name = p[1]

    if best_person_name is None:
        # If no registered person found, try similarity search in snapshots
        return db_manager.search_snapshots_by_similarity(encoding)

    # Return detections of that registered person
    results = db_manager.search_detections(name=best_person_name)
    return [
        {
            "id": r[0],
            "person_name": r[1] or "Unknown",
            "camera_id": r[2],
            "timestamp": r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
            "image_path": r[4],
            "face_path": r[4],
        }
        for r in results
    ]

def scan_video_for_person(video_path, target_encoding, sample_interval=10):
    """Scan code from the implementation guide."""
    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return results

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    current_segment = None
    last_match_frame = -1
    min_segment_gap = int(fps * 2)
    DISTANCE_THRESHOLD = 1.15

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % sample_interval == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with recognizer.ai_lock:
                    boxes, _ = recognizer.mtcnn.detect(frame_rgb)

                match_found = False
                best_confidence = 0.0

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        face_crop = frame_rgb[max(0,fy1):fy2, max(0,fx1):fx2]
                        if face_crop.size == 0: continue

                        face_resized = cv2.resize(face_crop, (160, 160))
                        face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(recognizer.device)
                        face_tensor = (face_tensor - 127.5) / 128.0

                        with recognizer.ai_lock:
                            with torch.no_grad():
                                embedding = recognizer.resnet(face_tensor).cpu().numpy()[0]

                        distance = float(np.linalg.norm(target_encoding - embedding))
                        if distance < DISTANCE_THRESHOLD:
                            match_found = True
                            conf = 1 - (distance / 2.0)
                            if conf > best_confidence: best_confidence = conf

                if match_found:
                    ts_sec = frame_count / fps
                    ts_str = f"{int(ts_sec//60)}:{int(ts_sec%60):02d}"

                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        if current_segment: results.append(current_segment)
                        current_segment = {
                            "start_seconds": ts_sec, "start_timestamp": ts_str,
                            "end_seconds": ts_sec, "end_timestamp": ts_str,
                            "confidence": best_confidence, "video_path": video_path
                        }
                    else:
                        current_segment["end_seconds"] = ts_sec
                        current_segment["end_timestamp"] = ts_str
                        if best_confidence > current_segment["confidence"]:
                            current_segment["confidence"] = best_confidence
                    last_match_frame = frame_count
            except: pass

        frame_count += 1

    if current_segment: results.append(current_segment)
    cap.release()
    return results

@app.post("/api/search_video_by_name")
async def api_search_video_by_name(request: Request):
    data = await request.json()
    name = data.get("name")
    video_ids = data.get("video_ids", [])

    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if not target:
        return {"status": "error", "message": f"Person '{name}' not found"}

    target_encoding = np.frombuffer(target[3], dtype=np.float32)
    all_results = []
    for vid_id in video_ids:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            for s in segments:
                all_results.append({**s, "video_id": vid_id, "camera_id": rec[1], "person_name": name})

    return {"status": "success", "results": all_results}

@app.post("/api/search_video_by_image")
async def api_search_video_by_image(file: UploadFile = File(...), video_ids: str = Form(...)):
    video_ids_list = json.loads(video_ids)
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"status": "error", "message": "No face detected"}

    all_results = []
    for vid_id in video_ids_list:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            for s in segments:
                all_results.append({**s, "video_id": vid_id, "camera_id": rec[1], "person_name": "Target"})

    return {"status": "success", "results": all_results}

@app.get("/api/recordings")
async def api_recordings_list():
    results = db_manager.search_recordings()
    return [{"id": r[0], "camera_id": r[1], "start_time": str(r[2]), "end_time": str(r[3]), "file_path": r[4]} for r in results]

@app.post("/clear_history")
async def api_clear_history():
    db_manager.delete_all_detections()
    # Also clean files if possible
    return {"status": "success", "message": "History cleared from database"}

@app.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "recordings.html", {})

@app.get("/detection_logs", response_class=HTMLResponse)
async def detection_logs_page(request: Request, camera_id: Optional[str] = None):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "detection_logs.html", {"camera_id": camera_id})

@app.get("/registered_detections", response_class=HTMLResponse)
async def registered_detections_page(request: Request, person_name: Optional[str] = None):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "registered_detections.html", {"person_name": person_name})

@app.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "people.html", {})

@app.get("/api/recent_alerts")
async def get_recent_alerts_api(request: Request, limit: int = 10):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    alerts = db_manager.get_recent_alerts(limit=limit)
    return [{
        "id": a["id"],
        "camera_id": a["camera_id"],
        "person_id": a["person_id"],
        "snapshot_path": a.get("snapshot_path"),
        "timestamp": format_12h(a["timestamp"]),
        "type": a["type"]
    } for a in alerts]

# --- Spatial Tracking / Re-ID APIs ---

@app.get("/api/active_targets")
async def get_active_targets(request: Request, hours: int = 24):
    """Retrieve unique people seen recently across all cameras."""
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    targets = db_manager.get_recent_active_targets(hours=hours)
    res = []
    for t in targets:
        res.append({
            "id": t["global_id"],
            "type": t.get("type", "unknown"),
            "first_seen": format_12h(t["first_seen"]),
            "last_seen": format_12h(t["last_seen"]),
            "last_camera": t.get("last_camera", "Unknown"),
            "has_thumbnail": "thumbnail" in t
        })
    return res

@app.get("/api/target_journey/{global_id}")
async def get_target_journey(request: Request, global_id: str):
    """Get the chronological path of a specific person."""
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    journey = db_manager.get_target_journey(global_id)
    res = []
    for point in journey:
        res.append({
            "camera_id": point["camera_id"],
            "timestamp": format_12h(point["timestamp"]),
            "date": point["timestamp"].strftime("%d %b"),
            "snapshot_path": point.get("snapshot_path")
        })
    return res

@app.get("/api/target_thumbnail/{global_id}")
async def get_target_thumbnail(global_id: str):
    """Return the binary thumbnail image for an unknown person."""
    # Note: thumbnail is public for easier image tag usage
    target = db_manager.get_global_identity_by_id(global_id)
    if not target or "thumbnail" not in target:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    from fastapi.responses import Response
    return Response(content=target["thumbnail"], media_type="image/jpeg")

@app.get("/api/notifications/stream")
async def notification_stream(request: Request):
    """SSE endpoint for real-time alerts."""
    q = await notification_manager.subscribe()
    
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                data = await q.get()
                yield data
        except asyncio.CancelledError:
            pass
        finally:
            notification_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "cameras.html", {})

@app.get("/add_camera", response_class=HTMLResponse)
async def add_camera_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "add_camera.html", {})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "analytics.html", {})
@app.post("/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    # Read bytes into memory
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"status": "error", "message": "Invalid image file."}

    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        # Save directly to local storage
        local_path = f"{DATASET_DIR}/{name}/{file.filename}"
        
        def on_reg_complete(success):
            if success:
                db_manager.register_person(name, local_path, encoding.tobytes())
                recognizer.load_known_faces(db_manager)
                print(f"[Register] {name} saved to local storage.")

        if stream_bytes_to_local(content, local_path, callback=on_reg_complete):
            return {"status": "success", "message": f"{name} registration queued for local saving."}
        else:
            return {"status": "error", "message": "Storage queue full."}
            
    return {"status": "error", "message": "No face detected in the image."}


@app.post("/api/add_camera")
async def add_camera(request: Request, camera_id: str = Form(None), camera_type: str = Form(None), source: str = Form(None)):
    # Support both form and JSON payload from UI
    if camera_id is None or camera_type is None or source is None:
        try:
            payload = await request.json()
            camera_id = camera_id or payload.get("camera_id")
            camera_type = camera_type or payload.get("camera_type")
            source = source or payload.get("source")
        except Exception:
            pass

    if not camera_id or not source:
        return {"status": "error", "message": "camera_id and source are required"}

    parsed = source
    if camera_type == "webcam":
        try:
            parsed = int(source)
        except ValueError:
            pass
    elif camera_type == "rtsp":
        parsed = sanitize_rtsp_url(source)
    elif camera_type == "droidcam":
        if not source.startswith("http"):
            parsed = f"http://{source}:4747/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "ipwebcam":
        if not source.startswith("http"):
            parsed = f"http://{source}:8080/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "mjpeg":
        # Direct MJPEG HTTP stream — pass as-is
        parsed = source.strip()

    if camera_manager.add_camera(camera_id, parsed):
        db_manager.add_camera_to_db(camera_id, parsed)
        threading.Thread(target=process_camera, args=(camera_id,), daemon=True).start()
        return {"status": "success"}
    return {"status": "error", "message": "Camera already exists or could not connect."}


@app.delete("/api/remove_camera/{camera_id}")
async def delete_camera(camera_id: str):
    print(f"[Delete Camera] Attempting to remove: {camera_id}")
    print(f"[Delete Camera] Active cameras: {camera_manager.get_active_cameras()}")
    
    # Stop any recording first
    with writer_lock:
        if camera_id in camera_writers:
            writer_data = camera_writers.pop(camera_id)
            # Stop the recording thread
            if camera_id in recording_stop_events:
                recording_stop_events[camera_id].set()
                if camera_id in recording_threads:
                    recording_threads[camera_id].join(timeout=5)
                    del recording_threads[camera_id]
                del recording_stop_events[camera_id]
            # Close FFmpeg process
            if "process" in writer_data:
                try:
                    writer_data["process"].stdin.close()
                    writer_data["process"].wait(timeout=10)
                except Exception:
                    writer_data["process"].kill()
            db_manager.end_recording(writer_data["db_id"])
            print(f"[Delete Camera] Stopped recording for {camera_id}")
    
    # Remove from camera manager
    cam_removed = camera_manager.remove_camera(camera_id)
    print(f"[Delete Camera] Camera manager removal result: {cam_removed}")
    
    # Remove from database (always try this even if camera not active)
    db_manager.remove_camera_from_db(camera_id)
    print(f"[Delete Camera] Removed from database")
    
    # Clean up results
    camera_results.pop(camera_id, None)
    camera_recognized_persons.pop(camera_id, None)
    occupancy_last_count.pop(camera_id, None)
    
    return {"status": "success", "message": f"Camera {camera_id} removed"}


@app.get("/api/cameras")
async def api_cameras():
    """Get all active cameras with their source info."""
    cameras = []
    for cam_id in camera_manager.get_active_cameras():
        # Get camera source from database
        cam_info = {"id": cam_id, "source": "Unknown"}
        try:
            db_cams = db_manager.get_cameras()
            for db_cam in db_cams:
                if db_cam[0] == cam_id:
                    cam_info["source"] = db_cam[1] if len(db_cam) > 1 else "Local"
                    break
        except:
            pass
        cameras.append(cam_info)
    return cameras

@app.get("/api/recognized/{camera_id}")
async def api_recognized_persons(camera_id: str):
    """Get recognized persons for a specific camera."""
    with recognized_lock:
        persons = camera_recognized_persons.get(camera_id, {})
        return [{"track_id": tid, "name": name} for tid, name in persons.items()]

@app.get("/api/occupancy")
async def api_occupancy(request_camera_id: Optional[str] = None):
    """Get occupancy data - live counts from active cameras."""
    results = {}
    for active_cam_id in camera_manager.get_active_cameras():
        # filter if a specific camera was requested
        if request_camera_id and active_cam_id != request_camera_id:
            continue
            
        with results_lock:
            data = camera_results.get(active_cam_id, {})
            # Get count from processed results, or fallback to occupancy_last_count
            live_count = data.get("count", 0)
            if live_count == 0:
                live_count = occupancy_last_count.get(active_cam_id, 0)
            
            alert_status = data.get("alert_active", False)
            
        results[active_cam_id] = {
            "id": active_cam_id,
            "camera_id": active_cam_id,
            "count": live_count,
            "head_count": live_count,
            "alert_active": alert_status,
            "total_today": db_manager.get_total_unique_count_today(active_cam_id)
        }
    return results
    
    # Historical data query
    rows = db_manager.search_occupancy(camera_id, start_time, end_time)
    return [{"id": r[0], "camera_id": r[1], "timestamp": r[2], "count": r[3]} for r in rows]

@app.get("/api/camera_daily_stats")
async def api_camera_daily_stats():
    """
    Returns today's person count stats per camera split into two 12-hour windows (IST):
      - am: 12:00 AM → 12:00 PM  (morning half)
      - pm: 12:00 PM → 12:00 AM  (evening half)
      - total: am + pm
    """
    stats = db_manager.get_camera_daily_person_stats()
    # Also include cameras currently active but with no detections yet
    for cam_id in camera_manager.get_active_cameras():
        if cam_id not in stats:
            stats[cam_id] = {"am": 0, "pm": 0, "total": 0}
    return stats

# ---------------------------------------------------------------------------
# Recording API
# ---------------------------------------------------------------------------
@app.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    with writer_lock:
        if camera_id in camera_writers:
            # Stop recording
            writer_data = camera_writers.pop(camera_id)
            
            # Stop the recording thread
            if camera_id in recording_stop_events:
                recording_stop_events[camera_id].set()
                if camera_id in recording_threads:
                    recording_threads[camera_id].join(timeout=5)
                    del recording_threads[camera_id]
                del recording_stop_events[camera_id]
            
            # Close FFmpeg process
            if "process" in writer_data:
                try:
                    writer_data["process"].stdin.close()
                    writer_data["process"].wait(timeout=10)
                except Exception:
                    writer_data["process"].kill()
            
            db_manager.end_recording(writer_data["db_id"])
            print(f"[Recording:{camera_id}] Stopped direct stream")
            return {"status": "success", "recording": False}
        else:
            # Start recording
            with results_lock:
                data = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")
            if frame is None:
                return {"status": "error", "message": "Camera offline or warming up"}
                
            h, w = frame.shape[:2]
            ist_now = get_ist_time()
            date_str = ist_now.strftime("%Y-%m-%d")
            timestamp = ist_now.strftime("%H%M%S")
            dir_path = f"{LOCAL_RECORDINGS_DIR}/{date_str}/{camera_id}"
            os.makedirs(dir_path, exist_ok=True)
            filename = f"{camera_id}_{date_str}_{timestamp}.mp4"
            local_path = f"{dir_path}/{filename}"

            import subprocess
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                "-r", "2",  # Match actual write rate (2 FPS)
                "-i", "-",
                "-vcodec", "libx264",  # H.264 = universal browser support
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "28",
                "-movflags", "+faststart",  # MP4 index at start for web playback
                local_path
            ]
            
            try:
                p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                db_id = db_manager.start_recording(camera_id, local_path)
                
                stop_event = threading.Event()
                thread = threading.Thread(target=recording_writer_thread, args=(camera_id, stop_event), daemon=True)
                thread.start()
                
                camera_writers[camera_id] = {
                    "process": p_ffmpeg,
                    "db_id": db_id,
                    "start_time": ist_now,
                    "file_path": local_path,
                    "camera_id": camera_id,
                    "w": w, "h": h
                }
                recording_threads[camera_id] = thread
                recording_stop_events[camera_id] = stop_event
                
                print(f"[Recording:{camera_id}] Started direct stream to {local_path}")
                return {"status": "success", "recording": True}
            except Exception as e:
                print(f"[Recording:{camera_id}] Start failure: {e}")
                return {"status": "error", "message": f"FFmpeg error: {e}"}

@app.get("/api/recording_status")
async def get_recording_status():
    with writer_lock:
        return {"active_recordings": list(camera_writers.keys())}

@app.get("/api/video_timeline/{record_id}")
async def video_timeline(record_id: str):
    """Get all detection timestamps relative to the start of a video recording."""
    rec = db_manager.get_recording(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
        
    camera_id = rec[1]
    start_time = rec[2]
    end_time = rec[3]
    
    # Optional logic: if video is ongoing, fetch till now
    from datetime import datetime
    import pytz
    if not end_time:
        end_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        
    # Get all snapshots for this camera in this timeframe
    # Limit to 500 to prevent huge payloads, but capture all major events
    snapshots = db_manager.get_detection_snapshots(camera_id, start_time, end_time, limit=500)
    
    events = []
    # Note: start_time might be a naive or aware datetime depending on DB. 
    # Usually it's UTC or IST natively saved. We just compute total_seconds.
    for snap in snapshots:
        snap_time = snap[2] # 2 is timestamp
        
        # Ensure timezone info doesn't break subtraction
        try:
            if start_time.tzinfo is None and snap_time.tzinfo is not None:
                snap_time = snap_time.replace(tzinfo=None)
            elif start_time.tzinfo is not None and snap_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=None)
                
            offset = (snap_time - start_time).total_seconds()
            
            # Snapshots have exact timestamp. Ensure offset is realistic
            if offset >= 0:
                events.append({
                    "offset_sec": round(offset, 2),
                    "person_count": snap[3],
                    "timestamp": snap_time.isoformat()
                })
        except Exception:
            pass
            
    # Sort chronologically by offset
    events.sort(key=lambda x: x["offset_sec"])
    return {
        "status": "success",
        "start_time": start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time),
        "events": events
    }


# ---------------------------------------------------------------------------
# Active Search API
# ---------------------------------------------------------------------------

@app.post("/api/start_search")
async def start_search(name: str = Form(...)):
    """Start an active face-search mission for the given person."""
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"'{name}' is not registered."}

    encoding = np.frombuffer(target[3], dtype=np.float32)
    with active_search_lock:
        active_search.clear()
        active_search.update({
            "running": True,
            "person_id": target[0],
            "name": target[1],
            "encoding": encoding,
            "found_track_ids": set()
        })
    print(f"[ActiveSearch] Mission started for: {target[1]}")
    return {
        "status": "success",
        "message": f"Searching for {target[1]}",
        "name": target[1],
        "image_path": target[2]  # registered photo from dataset/
    }


@app.post("/api/stop_search")
async def stop_search():
    """Stop the active search mission."""
    with active_search_lock:
        active_search.clear()
    print("[ActiveSearch] Mission stopped.")
    return {"status": "success"}


@app.get("/api/active_search")
async def get_active_search():
    """Return current active search target (if any)."""
    with active_search_lock:
        name = active_search.get("name")
    return {"active": name is not None, "name": name}


# ---------------------------------------------------------------------------
# History Search API
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def api_search(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_detections(name, start_time, end_time)
    return [{"id": r[0], "person_name": r[5] or "Unknown", "camera_id": r[2], "timestamp": format_12h(r[3]), "image_path": r[4]} for r in results]


@app.get("/api/registered_detections")
async def api_registered_detections(
    name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    """Get logs for registered detections with pagination and date filter."""
    logs = db_manager.get_registered_detections(
        name=name, date_from=date_from, date_to=date_to,
        page=page, page_size=page_size
    )
    total = db_manager.count_registered_detections(name=name, date_from=date_from, date_to=date_to)

    cameras = {c[0]: c[1] for c in db_manager.get_cameras()}
    persons_db = db_manager.get_registered_persons()
    person_images = {p[1]: p[2] for p in persons_db}

    formatted = []
    for l in logs:
        cam_id = l.get("camera_id")
        cam_source = cameras.get(cam_id, "Unknown")
        cam_ip = "Unknown"
        if cam_source and cam_source != "Unknown":
            cam_ip = cam_source
            if "@" in cam_source:
                cam_ip = cam_source.split("@")[-1].split(":")[0].split("/")[0]
        pname = l.get("person_name", "Unknown")
        pimage = person_images.get(pname)
        formatted.append({
            "id": str(l.get("_id", l.get("id"))),
            "person_name": pname,
            "snapshot_path": l.get("snapshot_path"),
            "profile_image": pimage,
            "camera_id": cam_id,
            "camera_ip": cam_ip,
            "timestamp": format_12h(l["timestamp"]),
        })
    return {"data": formatted, "total": total, "page": page, "page_size": page_size}


@app.get("/api/search_detections")
async def api_search_detections(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    """Search detection history with filters."""
    # Convert string times to datetime
    start = datetime.fromisoformat(start_time).replace(tzinfo=IST) if start_time else None
    end = datetime.fromisoformat(end_time).replace(tzinfo=IST) if end_time else None
    
    snapshots = db_manager.get_detection_snapshots(start_time=start, end_time=end)
    
    # Get all cameras to map names/IPs
    all_cams = {c[0]: c[1] for c in db_manager.get_cameras()}
    
    def extract_ip(url):
        if not url: return "Local"
        if not isinstance(url, str): return "Local"
        if "rtsp://" not in url: return url
        try:
            host_part = url.split("@")[-1].split("/")[0]
            return host_part.split(":")[0]
        except: return url

    formatted = []
    for s in snapshots:
        # filter by name if needed (bbox_data contains names)
        bbox = json.loads(s[5]) if s[5] else []
        names = [p.get("name") or f"#{p.get('id')}" for p in bbox]
        
        if name and name.lower() not in [n.lower() for n in names]:
            continue
            
        cam_id = s[1]
        cam_source = all_cams.get(cam_id, "N/A")
        
        formatted.append({
            "id": s[0],
            "camera_name": cam_id,
            "camera_id": cam_id,
            "camera_ip": extract_ip(cam_source),
            "timestamp": s[2].isoformat() if isinstance(s[2], datetime) else s[2],
            "person_count": s[3],
            "image_path": s[4],
            "person_names": names,
            "person_crops": s[6] if len(s) > 6 else []
        })
        
    return formatted

@app.post("/api/search_by_image")
async def search_by_image(
    file: UploadFile = File(...), 
    start_time: Optional[str] = Form(None), 
    end_time: Optional[str] = Form(None)
):
    """Historical similarity search across all snapshots within a time range."""
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image format"}
        
    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"error": "No face detected in uploaded image"}

    # Parse time strings to datetime objects
    start_dt = datetime.fromisoformat(start_time).replace(tzinfo=IST) if start_time else None
    end_dt = datetime.fromisoformat(end_time).replace(tzinfo=IST) if end_time else None

    # Search snapshots in DB using vector similarity
    matches = db_manager.search_snapshots_by_similarity(target_encoding, start_dt, end_dt)
    
    results = []
    for m in matches:
        person_name = "Detected Person"
        if m.get("bbox_data"):
            registered = [p.get("name") for p in m["bbox_data"] if p.get("name") and p.get("name") != "Unknown"]
            if registered:
                person_name = ", ".join(registered)

        results.append({
            "id": str(m["_id"]),
            "timestamp": m["timestamp"].isoformat() if hasattr(m["timestamp"], "isoformat") else m["timestamp"],
            "camera_id": m["camera_id"],
            "image_path": m["snapshot_path"],
            "person_name": person_name
        })
    
    return results


@app.post("/clear_history")
async def clear_history():
    try:
        db_manager.delete_all_detections()
    except Exception as e:
        print(f"DB clear error: {e}")

    snaps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    deleted: int = 0
    if os.path.isdir(snaps_dir):
        for entry in os.listdir(snaps_dir):
            entry_path = os.path.join(snaps_dir, entry)
            try:
                if os.path.isfile(entry_path):
                    os.remove(entry_path)
                    deleted += 1
                elif os.path.isdir(entry_path):
                    # Per-camera subdirectory — remove all files inside
                    for fname in os.listdir(entry_path):
                        fpath = os.path.join(entry_path, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                            deleted += 1
            except Exception:
                pass

    print(f"Cleared {deleted} snapshots.")
    return {"status": "success", "message": f"Cleared {deleted} records"}

@app.get("/api/recordings")
async def api_recordings(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_recordings(camera_id, start_time, end_time)
    return [{
        "id": r[0], 
        "camera_id": r[1], 
        "start_time": format_12h(r[2]), 
        "start_time_iso": r[2].isoformat() if hasattr(r[2], 'isoformat') else str(r[2]),
        "end_time": format_12h(r[3]) if r[3] else None, 
        "end_time_iso": (r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3])) if r[3] else None,
        "file_path": r[4], 
        "has_registered_person": r[5], 
        "registered_person_times": [format_12h(ts) for ts in (r[6] if len(r) > 6 else [])]
    } for r in results]

@app.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: str):
    rec = db_manager.get_recording(record_id)
    if rec:
        file_path = rec[4]
        # Delete local file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        db_manager.delete_recording(record_id)
    return {"status": "success"}

# ---------------------------------------------------------------------------
# Camera Recording Settings API
# ---------------------------------------------------------------------------

@app.get("/api/camera_settings/{camera_id}")
async def get_camera_settings(camera_id: str):
    """Get recording settings for a camera."""
    db_setting = db_manager.get_camera_recording_setting(camera_id)
    # Also check if actually recording
    with writer_lock:
        actually_recording = camera_id in camera_writers
    return {"camera_id": camera_id, "recording_enabled": bool(db_setting), "actually_recording": actually_recording}

@app.post("/api/camera_settings/{camera_id}")
async def set_camera_settings(camera_id: str, enabled: bool = Form(...)):
    """Set recording settings for a camera and start/stop actual recording."""
    # Save setting to database
    db_manager.set_camera_recording(camera_id, enabled)
    
    # Actually start/stop the recording
    with writer_lock:
        if enabled:
            # Start recording if not already recording
            if camera_id not in camera_writers:
                # Get frame dimensions from camera results
                with results_lock:
                    data = camera_results.get(camera_id, {})
                    frame = data.get("rendered_frame")
                    if frame is None:
                        return {"status": "error", "message": "Camera not streaming"}
                    h, w = frame.shape[:2]
                
                # Setup optimized FFmpeg recording
                ist_now = get_ist_time()
                date_str = ist_now.strftime("%Y-%m-%d")
                timestamp = ist_now.strftime("%H%M%S")
                dir_path = f"{LOCAL_RECORDINGS_DIR}/{date_str}/{camera_id}"
                os.makedirs(dir_path, exist_ok=True)
                filename = f"rec_{camera_id}_{timestamp}.mp4"
                local_path = f"{dir_path}/{filename}"
                import subprocess
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "20",
                    "-i", "-", "-vcodec", "libx265", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-x265-params", "lossless=0", "-crf", "28",
                    "-tune", "zerolatency", local_path
                ]
                
                try:
                    p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    db_id = db_manager.start_recording(camera_id, local_path)
                    
                    stop_event = threading.Event()
                    thread = threading.Thread(target=recording_writer_thread, args=(camera_id, stop_event), daemon=True)
                    thread.start()
                    
                    camera_writers[camera_id] = {
                        "process": p_ffmpeg,
                        "db_id": db_id,
                        "start_time": ist_now,
                        "file_path": local_path,
                        "camera_id": camera_id,
                        "w": w, "h": h
                    }
                    recording_threads[camera_id] = thread
                    recording_stop_events[camera_id] = stop_event
                    print(f"[Recording:{camera_id}] Constant Stream started (FFmpeg)")
                except Exception as e:
                    logger.error(f"Failed to start FFmpeg recording: {e}")
                    return {"status": "error", "message": str(e)}
        else:
            # Stop recording if currently recording
            if camera_id in camera_writers:
                writer_data = camera_writers.pop(camera_id)
                # Stop the recording thread
                if camera_id in recording_stop_events:
                    recording_stop_events[camera_id].set()
                    if camera_id in recording_threads:
                        recording_threads[camera_id].join(timeout=5)
                        del recording_threads[camera_id]
                    del recording_stop_events[camera_id]
                
                # Close FFmpeg process
                if "process" in writer_data:
                    try:
                        writer_data["process"].stdin.close()
                        writer_data["process"].wait(timeout=10)
                    except Exception:
                        writer_data["process"].kill()
                
                db_manager.end_recording(writer_data["db_id"])
                print(f"[Recording:{camera_id}] Stopped direct stream")
    
    return {"status": "success", "camera_id": camera_id, "recording_enabled": enabled}

# ---------------------------------------------------------------------------
# Detection Snapshots API
# ---------------------------------------------------------------------------

@app.get("/api/detection_snapshots")
async def get_detection_snapshots(
    camera_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    """Get detection snapshots with pagination and date filter — newest first."""
    snapshots = db_manager.get_detection_snapshots(
        camera_id=camera_id, date_from=date_from, date_to=date_to,
        page=page, page_size=page_size
    )
    total = db_manager.count_detection_snapshots(camera_id=camera_id, date_from=date_from, date_to=date_to)
    data = [
        {
            "id": s[0],
            "camera_id": s[1],
            "timestamp": format_12h(s[2]) if s[2] else "—",
            "person_count": s[3],
            "snapshot_path": s[4],
            "bbox_data": s[5]
        }
        for s in snapshots
    ]
    return {"data": data, "total": total, "page": page, "page_size": page_size}

@app.get("/api/snapshot/{snapshot_id}")
async def get_snapshot(snapshot_id: str):
    """Get a specific snapshot with bounding box data."""
    snapshot = db_manager.get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {
        "id": snapshot[0],
        "camera_id": snapshot[1],
        "timestamp": snapshot[2],
        "person_count": snapshot[3],
        "snapshot_path": snapshot[4],
        "bbox_data": snapshot[5]
    }

# ---------------------------------------------------------------------------
# Remote Image Proxy API
# ---------------------------------------------------------------------------

@app.get("/api/snapshot_image")
async def get_snapshot_image(path: str):
    """Serve images from local filesystem."""
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = f.read()
            from fastapi.responses import Response
            return Response(content=content, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recording_video")
async def get_recording_video(path: str, request: Request):
    """Serve video directly from local filesystem."""
    try:
        import os
        from fastapi.responses import FileResponse
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            
            # Handle HTTP Range requests (needed for video seeking in browsers)
            range_header = request.headers.get("range")
            if range_header:
                range_match = range_header.replace("bytes=", "").split("-")
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if range_match[1] else file_size - 1
                end = min(end, file_size - 1)
                chunk_size = end - start + 1
                
                with open(path, "rb") as f:
                    f.seek(start)
                    data = f.read(chunk_size)
                
                from fastapi.responses import Response
                return Response(
                    content=data,
                    status_code=206,
                    media_type="video/mp4",
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(chunk_size),
                    }
                )
            else:
                with open(path, "rb") as f:
                    data = f.read()
                from fastapi.responses import Response
                return Response(
                    content=data,
                    media_type="video/mp4",
                    headers={
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(file_size),
                    }
                )
        else:
            raise HTTPException(status_code=404, detail="Video not found")
    except Exception as e:
        logger.error(f"Error streaming video {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ---------------------------------------------------------------------------

import json
from fastapi import BackgroundTasks

# Store video search progress
video_search_progress: Dict[str, Any] = {}
video_search_lock = threading.Lock()

@app.get("/api/persons")
async def api_persons():
    """Get all registered persons with last seen info."""
    persons = db_manager.get_persons_with_last_seen()
    return persons


@app.get("/api/registered_persons")
async def api_registered_persons():
    """Alias for /api/persons for frontend compatibility."""
    persons = db_manager.get_persons_with_last_seen()
    return persons


@app.get("/api/analytics/hourly")
async def api_analytics_hourly(camera_id: Optional[str] = None):
    """Get max person count for each hour of last 24h."""
    analytics_data = db_manager.get_hourly_analytics(camera_id)
    
    # Map back to full 24h list
    hour_map = {int(r["_id"]): r for r in analytics_data}
    data = []
    now = get_ist_time()
    for i in range(24):
        check_time = now - timedelta(hours=(23-i))
        h = check_time.hour
        h_data = hour_map.get(h, {"max_count": 0, "camera_ids": []})
        data.append({
            "hour": h,
            "label": check_time.strftime("%I %p"), # "04 PM"
            "count": h_data["max_count"],
            "camera_id": h_data["camera_ids"][0] if h_data["camera_ids"] else (camera_id or "")
        })
    return data

@app.get("/api/analytics/daily")
async def api_analytics_daily(camera_id: Optional[str] = None, days: int = 7):
    """Get max person count for each day of last N days."""
    analytics_data = db_manager.get_daily_analytics(camera_id, days=days)
    
    # Map to days list
    day_map = {f"{r['_id']['year']}-{r['_id']['month']:02d}-{r['_id']['day']:02d}": r["max_count"] for r in analytics_data}
    data = []
    now = get_ist_time()
    for i in range(days):
        check_date = now - timedelta(days=(days-1-i))
        key = f"{check_date.year}-{check_date.month:02d}-{check_date.day:02d}"
        data.append({
            "date": key,
            "label": check_date.strftime("%d %b"),
            "count": day_map.get(key, 0)
        })
    return data


@app.post("/api/register_person")
async def api_register_person(name: str = Form(...), file: UploadFile = File(...)):
    """Register a person via API (direct stream)."""
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"status": "error", "message": "Invalid image file."}

    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        local_path = f"{DATASET_DIR}/{name}/{file.filename}"
        
        def on_api_reg_complete(success):
            if success:
                db_manager.register_person(name, local_path, encoding.tobytes())
                recognizer.load_known_faces(db_manager)
        
        if stream_bytes_to_local(content, local_path, callback=on_api_reg_complete):
            return {"status": "success", "message": f"{name} registration queued for local saving."}
        else:
            return {"status": "error", "message": "Storage queue full."}
            
    return {"status": "error", "message": "No face detected in the image."}


@app.delete("/api/delete_person/{person_id}")
async def api_delete_person(person_id: int):
    """Delete a registered person from the database."""
    try:
        # Get person info to delete files
        persons = db_manager.get_registered_persons()
        person = next((p for p in persons if str(p[0]) == str(person_id)), None)
        
        if person:
            # Delete local files
            image_path = person[2]
            if image_path:
                try:
                    import shutil
                    d = os.path.dirname(image_path)
                    if d and os.path.exists(d):
                        shutil.rmtree(d)
                except Exception as e:
                    print(f"[Delete Person] Error deleting files: {e}")
            
            # Delete from DB
            db_manager.delete_person_from_db(person_id)
            recognizer.load_known_faces(db_manager)
            return {"status": "success"}
        
        return {"status": "error", "message": "Person not found"}
    except Exception as e:
        print(f"[Delete Person] Error: {e}")
        return {"status": "error", "message": str(e)}


def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 10) -> list:
    """
    Scan a video file for ALL occurrences of a person with the target face encoding.
    Detects every face in each frame and matches against the target person.
    Groups continuous appearances into flagged segments with start/end timestamps.
    Returns list of detection segments where the person appears.
    """
    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoScan] ERROR: Could not open video {video_path}")
        return results
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Track continuous appearances
    current_segment = None
    last_match_frame = -1
    min_segment_gap = int(fps * 2)  # 2 seconds gap to create new segment
    
    # Lower threshold for better detection (same as live recognition)
    DISTANCE_THRESHOLD = 1.15
    
    print(f"[VideoScan] Starting scan of {video_path}")
    print(f"[VideoScan] Total frames: {total_frames}, FPS: {fps}, Sample interval: {sample_interval}")
    
    matches_found = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame for efficiency
        if frame_count % sample_interval == 0:
            try:
                # Detect ALL faces in frame using full frame (not just body crop)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with recognizer.ai_lock:
                    boxes, probs = recognizer.mtcnn.detect(frame_rgb)
                
                match_found = False
                best_confidence = 0.0
                best_distance = 999.0
                
                if boxes is not None and len(boxes) > 0:
                    # Check EACH face in the frame against target
                    for i, box in enumerate(boxes):
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        
                        # Ensure valid box
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(frame.shape[1], fx2), min(frame.shape[0], fy2)
                        
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fw < 30 or fh < 30:  # Skip very small faces
                            continue
                        
                        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
                        
                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (160, 160))
                            face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(recognizer.device)
                            face_tensor = (face_tensor - 127.5) / 128.0
                            
                            with recognizer.ai_lock:
                                with torch.no_grad():
                                    embedding = recognizer.resnet(face_tensor).cpu().numpy()[0]
                            
                            # Compare with target
                            distance = float(np.linalg.norm(target_encoding - embedding))
                            confidence = 1 - (distance / 2.0)
                            
                            if distance < DISTANCE_THRESHOLD:  # Match found
                                match_found = True
                                matches_found += 1
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_distance = distance
                                if frame_count % 100 == 0:  # Log every 100th match frame
                                    print(f"[VideoScan] Match at frame {frame_count}, dist: {distance:.3f}, conf: {confidence:.2f}")
                
                # Handle segment tracking
                if match_found:
                    timestamp_sec = frame_count / fps
                    
                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        # Start new segment
                        if current_segment is not None:
                            results.append(current_segment)
                        current_segment = {
                            "start_seconds": timestamp_sec,
                            "start_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "end_seconds": timestamp_sec,
                            "end_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "confidence": best_confidence,
                            "start_frame": frame_count,
                            "end_frame": frame_count
                        }
                        print(f"[VideoScan] New segment started at {current_segment['start_timestamp']}")
                    else:
                        # Extend current segment
                        current_segment["end_seconds"] = timestamp_sec
                        current_segment["end_timestamp"] = f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}"
                        current_segment["end_frame"] = frame_count
                        if best_confidence > current_segment["confidence"]:
                            current_segment["confidence"] = best_confidence
                    
                    last_match_frame = frame_count
                    
            except Exception as e:
                print(f"[VideoScan] Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
        
        frame_count += 1
        
        # Progress update every 500 frames
        if frame_count % 500 == 0 and total_frames > 0:
            progress = (frame_count / total_frames) * 100
            print(f"[VideoScan] Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Don't forget the last segment
    if current_segment is not None:
        results.append(current_segment)
    
    cap.release()
    print(f"[VideoScan] Scan complete. Found {len(results)} segments, {matches_found} total matches")
    return results


@app.post("/api/search_video_by_name")
async def search_video_by_name(request: Request):
    """Search for a person by name across selected videos."""
    data = await request.json()
    name = data.get("name")
    video_ids = data.get("video_ids", [])
    
    if not name or not video_ids:
        return {"status": "error", "message": "Name and video IDs required"}
    
    # Get person's encoding
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"Person '{name}' not found"}
    
    target_encoding = np.frombuffer(target[3], dtype=np.float32)
    
    # Search each video
    all_results = []
    total_segments = 0
    for vid_id in video_ids:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            total_segments += len(segments)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": name
                })
    
    # Sort by start time
    all_results.sort(key=lambda x: x["start_seconds"])
    
    return {
        "status": "success", 
        "results": all_results,
        "total_segments": total_segments,
        "videos_searched": len(video_ids)
    }


@app.post("/api/search_video_by_image")
async def search_video_by_image(file: UploadFile = File(...), video_ids: str = Form(...)):
    """Search for a person using an uploaded image across selected videos."""
    video_ids_list = json.loads(video_ids)
    
    if not video_ids_list:
        return {"status": "error", "message": "Video IDs required"}
    
    # Get encoding from uploaded image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"status": "error", "message": "No face detected in uploaded image"}
    
    # Search each video
    all_results = []
    total_segments = 0
    for vid_id in video_ids_list:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            total_segments += len(segments)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": "Unknown (from image)"
                })
    
    # Sort by start time
    all_results.sort(key=lambda x: x["start_seconds"])
    
    return {
        "status": "success", 
        "results": all_results,
        "total_segments": total_segments,
        "videos_searched": len(video_ids_list)
    }


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

async def gen_frames(camera_id: str):
    """Generate stable MJPEG stream at exactly 2 FPS."""
    FRAME_INTERVAL = 0.5  # 2 FPS
    last_sent_bytes_id = None  # track identity of last sent frame to avoid duplicates
    next_send_time = time.time()

    while True:
        now = time.time()
        wait = next_send_time - now
        if wait > 0:
            await asyncio.sleep(wait)

        # Snap the deadline forward regardless of how long we slept
        next_send_time += FRAME_INTERVAL
        # If we're running behind (e.g. slow client), catch up without burst
        if next_send_time < time.time():
            next_send_time = time.time() + FRAME_INTERVAL

        with results_lock:
            data = camera_results.get(camera_id, {})
            frame_bytes = data.get("encoded_frame")
            frame_id = data.get("frame_id", -1)

        if frame_bytes is None:
            continue

        # Skip if this is the exact same encoded frame we already sent
        fb_id = id(frame_bytes)
        if fb_id == last_sent_bytes_id:
            continue

        last_sent_bytes_id = fb_id
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
               b"\r\n" + frame_bytes + b"\r\n")


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")



@app.get("/api/capture_frame/{camera_id}")
async def capture_frame(camera_id: str):
    """Return the latest frame as a static JPEG for the pause feature."""
    with results_lock:
        data = camera_results.get(camera_id, {})
        frame_bytes = data.get("encoded_frame")
    if frame_bytes is None:
        raise HTTPException(status_code=404, detail="No frame available")
    from fastapi.responses import Response
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/api/live_results/{camera_id}")
async def get_live_results(camera_id: str):
    """Get the current tracking results and face crops for a camera."""
    with results_lock:
        data = camera_results.get(camera_id, {})
        persons = data.get("tracks", []) or []
    
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "face_crop": p.get("face_crop")
        } 
        for p in persons
    ]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            log_config=None   # don't let uvicorn override our logging setup
        )
    except Exception as e:
        import traceback
        print(f"\n[STARTUP ERROR] {e}")
        traceback.print_exc()
