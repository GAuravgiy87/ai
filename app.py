import os
import sys

# Silence noisy FFmpeg/OpenCV logs before any other imports
os.environ["OPENCV_LOG_LEVEL"]        = "OFF"
os.environ["FFMPEG_LOG_LEVEL"]        = "quiet"
os.environ["OPENCV_FFMPEG_LOGLEVEL"]  = "-8"
os.environ["AV_LOG_FORCE_LEVEL"]      = "0"

import threading
import traceback
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.logging_config import setup_logging
from core.startup import lifespan, load_models, analytics_snapshot_task
from database.sqlite_manager import SqliteManager
from cameras.camera_manager import CameraManager

# ── Logging ───────────────────────────────────────────────────────────────────
logger = setup_logging()

# ── Shared managers (DB + lightweight camera list for non-camera routes) ──────
db_manager     = SqliteManager()
camera_manager = CameraManager()   # no cameras loaded here — camera server owns them

# load_models() returns (None, None, None) — models live in the camera server
detector, recognizer, reid_manager = load_models(db_manager)

# Pipeline init is a no-op when all three are None
from core.pipeline import init_pipeline
init_pipeline(db_manager, camera_manager, detector, recognizer, reid_manager)

# ── Routes ────────────────────────────────────────────────────────────────────
from routes import (
    auth, dashboard, cameras, people, recordings, search, detections, journey, analytics
)

dashboard.init_routes(db_manager, camera_manager)
cameras.init_routes(db_manager, camera_manager)
people.init_routes(db_manager, recognizer)
recordings.init_routes(db_manager)
search.init_routes(db_manager, recognizer)
detections.init_routes(db_manager)
journey.init_routes(db_manager)
analytics.init_routes(db_manager)

# ── FastAPI app ───────────────────────────────────────────────────────────────
def get_app_lifespan(app: FastAPI):
    return lifespan(app, db_manager, camera_manager)

app = FastAPI(title="AI Vigilance", lifespan=get_app_lifespan)

# Static files
for d in ["snapshots", "dataset", "recordings"]:
    os.makedirs(d, exist_ok=True)
    app.mount(f"/{d}", StaticFiles(directory=d), name=d)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Routers
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(cameras.router)
app.include_router(people.router)
app.include_router(recordings.router)
app.include_router(search.router)
app.include_router(detections.router)
app.include_router(journey.router)
app.include_router(analytics.router)

# Analytics background task
threading.Thread(
    target=analytics_snapshot_task,
    args=(db_manager, camera_manager),
    daemon=True,
).start()

# ── Crash handler ─────────────────────────────────────────────────────────────
def handle_crash(exc_type, value, tb):
    from utils.hw_manager import hw
    from core.state import get_ist_time
    status    = hw.get_status()
    crash_msg = (
        f"\n{'='*40}\n!!! SYSTEM CRASH DETECTED !!!\n"
        f"Reason: {value}\n"
        f"Hardware: CPU {status.get('cpu', {}).get('usage_percent')}% | "
        f"RAM {status.get('memory', {}).get('percent')}% | "
        f"GPU {status.get('gpu', {}).get('load') if status.get('gpu') else 'N/A'}\n"
        f"{'='*40}\n"
        + "".join(traceback.format_exception(exc_type, value, tb))
    )
    with open("crash_forensics.log", "a") as f:
        f.write(f"\n[{get_ist_time()}] {crash_msg}")
    logger.critical(crash_msg)

sys.excepthook = handle_crash

# ── Entry point ───────────────────────────────────────────────────────────────
def _get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = _get_local_ip()
    print("\n" + "=" * 50)
    print("[OK] AI Vigilance System Starting...")
    print(f"[OK] Main App     : http://127.0.0.1:9000")
    print(f"[OK] Camera Server: http://127.0.0.1:9001")
    print(f"[OK] Network      : http://{local_ip}:9000")
    print("=" * 50 + "\n")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=9000,
            log_level="warning",
            access_log=False,
        )
    except Exception:
        pass
