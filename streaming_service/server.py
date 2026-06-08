"""
camera_server/server.py — Camera Processing Server (port 9001)

Runs inside the same Python process as the main app, but on a separate
uvicorn server (port 9001) started in a background thread by run.py.
"""

import asyncio
import logging
import threading
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

import streaming_service.state as state
from core.pipeline import notification_manager
from streaming_service.routes import cameras, stream, faces, stats

logger = logging.getLogger("camera_server")
CAMERA_SERVER_PORT = 9001

@asynccontextmanager
async def _lifespan(app: FastAPI):
    if state.db_manager is None:
        state.build_singletons()
    notification_manager.set_loop(asyncio.get_event_loop())
    threading.Thread(target=state.restore_cameras, daemon=True).start()
    yield

camera_app = FastAPI(title="AI Vigilance — Camera Server", lifespan=_lifespan)

# Include Modular Routers
camera_app.include_router(cameras.router, tags=["Cameras"], prefix="/cameras")
camera_app.include_router(stream.router, tags=["Stream"])
camera_app.include_router(faces.router, tags=["Faces"])
camera_app.include_router(stats.router, tags=["Stats"])

def start(host: str = "0.0.0.0", port: int = CAMERA_SERVER_PORT):
    """
    Build singletons then run the camera server in the current thread.
    Call this inside a daemon thread from run.py so it runs alongside
    the main uvicorn server without blocking it.
    """
    state.build_singletons()
    logger.info(f"[CameraServer] Listening on {host}:{port}")
    uvicorn.run(
        camera_app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Vigilance - Camera Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=CAMERA_SERVER_PORT, help="Port to bind to")
    args = parser.parse_args()
    
    start(host=args.host, port=args.port)
