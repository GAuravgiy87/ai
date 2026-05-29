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
from core.diagnostics import install as install_diagnostics
from database.postgres_manager import DatabaseManager

# ── Logging ───────────────────────────────────────────────────────────────────
logger = setup_logging()

# ── Install crash handler + resource monitor FIRST ───────────────────────────
# auto_restart=True  : process relaunches itself 5s after any fatal crash
# monitor_interval=60: print CPU/RAM/GPU to terminal every 60 seconds
install_diagnostics(auto_restart=True, monitor_interval=60)

# ── Shared managers (DB only — camera server owns all camera state) ─────────────
db_manager = DatabaseManager()

# load_models() returns (None, None, None) — models live in the camera server
detector, recognizer, reid_manager = load_models(db_manager)


# Pipeline init is a no-op when all three are None
from core.pipeline import init_pipeline
init_pipeline(db_manager, None, detector, recognizer, reid_manager)  # camera_manager=None - camera server owns cameras

# ── Routes ────────────────────────────────────────────────────────────────────
from routes import (
    auth, dashboard, cameras, people, recordings, search, detections, journey, analytics
)

dashboard.init_routes(db_manager)
cameras.init_routes(db_manager)
people.init_routes(db_manager, recognizer)
recordings.init_routes(db_manager)
search.init_routes(db_manager, recognizer)
detections.init_routes(db_manager)
journey.init_routes(db_manager)
analytics.init_routes(db_manager)

# ── FastAPI app ───────────────────────────────────────────────────────────────
def get_app_lifespan(app: FastAPI):
    return lifespan(app, db_manager)

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
    args=(db_manager,),
    daemon=True,
).start()

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
    
    # ── Start Camera Server Locally (Master Process Only) ──────────────────────
    # Only the master process executes this block, preventing Uvicorn workers
    # from spawning multiple conflicting camera server threads.
    def _run_camera_server():
        from camera_server.server import start
        try:
            start()
        except Exception as e:
            logger.error(f"[CameraServer] Thread crashed: {e}")
            
    _cam_server_thread = threading.Thread(target=_run_camera_server, name="camera-server-master", daemon=True)
    _cam_server_thread.start()
    
    # Optional wait to keep logs clean
    import time
    time.sleep(1)

    try:
        # For local development, reduce workers to minimize idle threads
        # Use 2 workers for the main app (enough for async FastAPI)
        # Override with --workers CLI arg or UVICORN_WORKERS env var if needed
        workers = int(os.environ.get("UVICORN_WORKERS", "2"))
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=9000,
            workers=workers,
            log_level="warning",
            access_log=False,
        )
    except Exception as e:
        logger.error(f"[App] Uvicorn exited with error: {e}", exc_info=True)
