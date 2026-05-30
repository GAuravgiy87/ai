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

# ── Install crash handler + resource monitor FIRST (Main Process Only) ───────
# Only run diagnostics in the master process to avoid starting duplicate resource
# monitors and setting clashing exception handlers in worker subprocesses.
import multiprocessing
if multiprocessing.current_process().name == "MainProcess":
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
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch AI Vigilance with optional Dynamic Auto-Scaling"
    )
    parser.add_argument("--disable-autoscale", action="store_true",
                        help="Disable auto-scaling, use fixed worker count")
    parser.add_argument("--min-workers", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-threshold", type=int, default=None)
    parser.add_argument("--camera-worker-ratio", type=float, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind")
    
    args = parser.parse_args()
    
    if not args.disable_autoscale:
        # Launch with Dynamic Auto-Scaling (runs autoscale.py logic)
        from autoscale import DynamicAutoScaler
        
        scaler_kwargs = {}
        if args.min_workers is not None:
            scaler_kwargs["min_workers"] = args.min_workers
        if args.max_workers is not None:
            scaler_kwargs["max_workers"] = args.max_workers
        if args.cpu_threshold is not None:
            scaler_kwargs["cpu_threshold"] = args.cpu_threshold
        if args.camera_worker_ratio is not None:
            scaler_kwargs["camera_worker_ratio"] = args.camera_worker_ratio
            
        scaler = DynamicAutoScaler(**scaler_kwargs)
        scaler.run(app_module="app.py", host=args.host, port=args.port)
        sys.exit(0)

    # ── If autoscale is disabled, run the app directly ───────────────────────
    from utils import get_local_ip
    local_ip = get_local_ip()
    print("\n" + "=" * 50)
    print("[OK] AI Vigilance System Starting...")
    print(f"[OK] Main App     : http://127.0.0.1:{args.port}")
    print(f"[OK] Camera Server: http://127.0.0.1:9001")
    print(f"[OK] Network      : http://{local_ip}:{args.port}")
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
    
    # ── Start Recording Worker Locally (Master Process Only) ──────────────────────
    def _run_recording_worker():
        from services.recording_worker import main as rw_main
        try:
            rw_main()
        except Exception as e:
            import logging
            logging.error(f"[RecordingWorker] Thread crashed: {e}")

    _rec_worker_thread = threading.Thread(target=_run_recording_worker, name="recording-worker-master", daemon=True)
    _rec_worker_thread.start()
    
    import time
    time.sleep(1)

    try:
        # FastAPI is fully async — 1 worker handles all concurrent requests.
        # workers > 1 on Windows uses multiprocessing spawn which is slow to
        # kill on shutdown (+3-8 s per worker). Set UVICORN_WORKERS env var
        # or use --workers flag to override for CPU-bound scaling.
        workers = int(os.environ.get("UVICORN_WORKERS", "1"))
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            workers=workers,
            log_level="warning",
            access_log=False,
        )
    except (KeyboardInterrupt, InterruptedError):
        # Normal shutdown interruption on Windows
        pass
    except Exception as e:
        logger.error(f"[App] Uvicorn exited with error: {e}", exc_info=True)
