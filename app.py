"""
app.py — Entry point. Wires all modules together and starts uvicorn.
~60 lines. All logic lives in core/ and routes/.
"""
import os
import sys

# -- Silence noisy libs before any import -----------------------------------------
os.environ.update({
    "OPENCV_LOG_LEVEL":        "OFF",
    "FFMPEG_LOG_LEVEL":        "quiet",
    "OPENCV_FFMPEG_LOGLEVEL":  "-8",
    "PYTHONWARNINGS":          "ignore",
})

# -- Minimal logging: critical events to file, errors to terminal -----------------
from core.logging_config import setup_logging
setup_logging("app.log")

import logging
logger = logging.getLogger(__name__)

# ── Core singletons ───────────────────────────────────────────────────────
from database.sqlite_manager import SqliteManager
from cameras.camera_manager  import CameraManager
from utils.hw_manager        import hw

db_manager     = SqliteManager()
camera_manager = CameraManager()

from core.startup import (
    DBLogHandler, GlobalReIDManager, NotificationManager,
    _load_models_bg, _storage_optimization_task, install_signal_hooks,
    build_lifespan, log_startup_snapshot,
)
import threading

# Wire DB log handler (WARNING+ only — no verbose noise)
logging.root.addHandler(DBLogHandler(lambda: db_manager))

# Log startup with full system snapshot (CPU/RAM/GPU/platform)
log_startup_snapshot(db_manager)

# Singletons
reid_manager          = GlobalReIDManager(db_manager)
notification_manager  = NotificationManager()

# Background tasks
threading.Thread(target=_load_models_bg, args=(db_manager,),
                 daemon=True, name="model-init").start()
threading.Thread(target=_storage_optimization_task, args=(db_manager,),
                 daemon=True, name="cleanup").start()

install_signal_hooks(db_manager)

# ── Pipeline injection ────────────────────────────────────────────────────
from core import pipeline as _pipeline
import core.startup as _startup

_pipeline.init_pipeline(
    db_manager, camera_manager, notification_manager, reid_manager,
    _startup._detector_ready, _startup._recognizer_ready,
    lambda: _startup.detector,
    lambda: _startup.recognizer,
)

# ── FastAPI app ───────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

lifespan = build_lifespan(db_manager, camera_manager, notification_manager)
app      = FastAPI(lifespan=lifespan)
app.mount("/static",     StaticFiles(directory="static"),     name="static")
app.mount("/snapshots",  StaticFiles(directory="snapshots"),  name="snapshots")
app.mount("/dataset",    StaticFiles(directory="dataset"),    name="dataset")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

# ── Route registration ────────────────────────────────────────────────────
from routes import auth, dashboard, cameras, people, detections, recordings, search, reid

app.include_router(auth.router)

dashboard.init(db_manager, camera_manager, hw)
app.include_router(dashboard.router)

cameras.init(db_manager, camera_manager, _pipeline.process_camera)
app.include_router(cameras.router)

people.init(db_manager, lambda: _startup.recognizer)
app.include_router(people.router)

detections.init(db_manager)
app.include_router(detections.router)

recordings.init(db_manager)
app.include_router(recordings.router)

search.init(db_manager, lambda: _startup.recognizer)
app.include_router(search.router)

reid.init(db_manager, notification_manager)
app.include_router(reid.router)

# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
