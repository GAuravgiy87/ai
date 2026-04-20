import os
import sys

# Silence noisy FFmpeg/OpenCV logs before any other imports
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_LEVEL"] = "0"

import threading
import uvicorn
import traceback
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.logging_config import setup_logging
from core.startup import lifespan, load_models, storage_optimization_task
from core.pipeline import init_pipeline
from database.sqlite_manager import SqliteManager
from cameras.camera_manager import CameraManager

# Initialize Logging
logger = setup_logging()

# Global Managers
db_manager = SqliteManager()
camera_manager = CameraManager()

# Load Models
detector, recognizer, reid_manager = load_models(db_manager)

# Initialize Pipeline with Dependencies
init_pipeline(db_manager, camera_manager, detector, recognizer, reid_manager)

# --- Routes Initialization ---
from routes import (
    auth, dashboard, cameras, people, recordings, search, detections, journey, analytics
)

# Inject dependencies into route modules
dashboard.init_routes(db_manager, camera_manager)
cameras.init_routes(db_manager, camera_manager)
people.init_routes(db_manager, recognizer)
recordings.init_routes(db_manager)
search.init_routes(db_manager, recognizer)
detections.init_routes(db_manager)
journey.init_routes(db_manager)
analytics.init_routes(db_manager)

# --- FastAPI App Setup ---
def get_app_lifespan(app: FastAPI):
    return lifespan(app, db_manager, camera_manager)

app = FastAPI(
    title="AI Vigilance",
    lifespan=get_app_lifespan
)

# Mounting static files
for d in ["snapshots", "dataset", "recordings"]:
    os.makedirs(d, exist_ok=True)
    app.mount(f"/{d}", StaticFiles(directory=d), name=d)

# Mount Static Files (Critical for skeleton.js, script.js, etc.)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include Routers
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(cameras.router)
app.include_router(people.router)
app.include_router(recordings.router)
app.include_router(search.router)
app.include_router(detections.router)
app.include_router(journey.router)
app.include_router(analytics.router)

# Start background optimization
threading.Thread(target=storage_optimization_task, args=(db_manager,), daemon=True).start()

# Global Exception Handler for Crash Debugging
def handle_crash(type, value, tb):
    """Log hardware state and traceback on fatal crash."""
    from utils.hw_manager import hw
    status = hw.get_status()
    crash_msg = f"\n{'='*40}\n!!! SYSTEM CRASH DETECTED !!!\n"
    crash_msg += f"Reason: {value}\n"
    crash_msg += f"Hardware State: CPU {status.get('cpu', {}).get('usage_percent')}% | "
    crash_msg += f"RAM {status.get('memory', {}).get('percent')}% | "
    crash_msg += f"GPU {status.get('gpu', {}).get('load') if status.get('gpu') else 'N/A'}\n"
    crash_msg += f"{'='*40}\n"
    crash_msg += "".join(traceback.format_exception(type, value, tb))
    
    # Save to a dedicated crash log
    with open("crash_forensics.log", "a") as f:
        f.write(f"\n[{get_ist_time()}] {crash_msg}")
    
    # Also log to main logger
    logger.critical(crash_msg)

sys.excepthook = handle_crash

if __name__ == "__main__":
    try:
        print("\n[OK] Starting Uvicorn Server...")
        print(f"[OK] Dashboard Area: http://127.0.0.1:8000")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="warning", # Only show warnings/errors on console
            access_log=False      # Hide GET / requests noise
        )
    except Exception as e:
        # sys.excepthook will handle this
        pass
