import os
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

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        # Use fallback print if logger not setup or errored
        print(f"\n[STARTUP ERROR] {e}")
        traceback.print_exc()
