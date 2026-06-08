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

from common.logging import setup_logging
from core.startup import lifespan, load_models, analytics_snapshot_task
from core.diagnostics import install as install_diagnostics
from data_access.manager import DatabaseManager

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
from api import (
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
    os.makedirs(f"database/{d}", exist_ok=True)
    app.mount(f"/{d}", StaticFiles(directory=f"database/{d}"), name=d)

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


