"""
camera_server/server.py — Camera Processing Server (port 9001)

Runs inside the same Python process as the main app, but on a separate
uvicorn server (port 9001) started in a background thread by app.py.

Owns everything camera-related:
  - YOLOv8s detection
  - FaceNet / MTCNN recognition
  - Re-ID manager
  - Per-camera pipeline (tracking, recording, snapshots)
  - MJPEG streaming
  - Camera add / remove / list
"""

import os
import sys
import time
import asyncio
import logging
import threading
import numpy as np
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional

from core.state import (
    camera_results, results_lock,
    occupancy_last_count,
    camera_writers, writer_lock,
    recording_threads, recording_stop_events,
    sanitize_rtsp_url,
    LOCAL_RECORDINGS_DIR,
)
from core.pipeline import init_pipeline, process_camera, notification_manager
from cameras.camera_manager import CameraManager, probe_rtsp_url
from database.sqlite_manager import SqliteManager
from utils.detector import PersonDetector
from utils.recognizer import FaceRecognizer

logger = logging.getLogger("camera_server")

# ── Port ──────────────────────────────────────────────────────────────────────
CAMERA_SERVER_PORT = 9001

# ── Singletons (created once when this module is imported) ────────────────────
_db_manager:     Optional[SqliteManager]  = None
_camera_manager: Optional[CameraManager] = None
_detector:       Optional[PersonDetector] = None
_recognizer      = None
_reid_manager    = None


def _build_singletons():
    """Initialise all heavy singletons. Called once from start()."""
    global _db_manager, _camera_manager, _detector, _recognizer, _reid_manager

    # Install diagnostics for camera server thread crashes
    # auto_restart=False here — the main app's diagnostics handles restart
    from core.diagnostics import install as _install_diag
    _install_diag(auto_restart=False, monitor_interval=0)  # monitor=0 → no duplicate monitor

    from core.startup import GlobalReIDManager

    _db_manager     = SqliteManager()
    _camera_manager = CameraManager()
    _detector       = PersonDetector(model_path='yolov8s.pt')

    try:
        _recognizer = FaceRecognizer()
        _recognizer.load_known_faces(_db_manager)
    except Exception as e:
        logger.critical(f"[CameraServer] FaceRecognizer init failed: {e}")
        _recognizer = None

    _reid_manager = GlobalReIDManager(_db_manager)

    # Wire into the shared pipeline
    init_pipeline(_db_manager, _camera_manager, _detector, _recognizer, _reid_manager)
    logger.info("[CameraServer] Models and pipeline ready.")


# ── Camera restore ────────────────────────────────────────────────────────────

def _restore_cameras():
    """Restore all persisted cameras from the database."""
    time.sleep(1)   # let the server bind first
    try:
        cameras = _db_manager.get_cameras()
        logger.info(f"[CameraServer] Restoring {len(cameras)} camera(s)...")
        for cam_id, source in cameras:
            # BUG-04 fix: skip cameras already active to prevent 409 conflict errors
            if cam_id in _camera_manager.cameras:
                logger.info(f"[CameraServer] {cam_id} already active, skipping restore")
                continue

            if isinstance(source, str) and source.startswith("rtsp://"):
                new_source = probe_rtsp_url(source)
                if new_source != source:
                    _db_manager.update_camera_source(cam_id, new_source)
                source = new_source

            parsed = int(source) if str(source).isdigit() else source
            status, final_source = _camera_manager.add_camera(cam_id, parsed)
            if status == 0:
                threading.Thread(
                    target=process_camera, args=(cam_id,), daemon=True
                ).start()
                logger.info(f"[CameraServer] Restored: {cam_id}")
            else:
                logger.warning(f"[CameraServer] Could not restore {cam_id} (status={status})")
    except Exception as e:
        logger.error(f"[CameraServer] Restore error: {e}")


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    notification_manager.set_loop(asyncio.get_event_loop())
    threading.Thread(target=_restore_cameras, daemon=True).start()
    yield


camera_app = FastAPI(title="AI Vigilance — Camera Server", lifespan=_lifespan)


# ── Request models ────────────────────────────────────────────────────────────

class AddCameraRequest(BaseModel):
    camera_id:   str
    source:      str
    camera_type: Optional[str] = "rtsp"


# ── Routes ────────────────────────────────────────────────────────────────────

@camera_app.get("/health")
def health():
    return {
        "status":  "ok",
        "cameras": _camera_manager.get_active_cameras() if _camera_manager else [],
    }


@camera_app.get("/cameras")
def list_cameras():
    db_cams = {c[0]: c[1] for c in _db_manager.get_cameras()}
    return [
        {"id": cam_id, "source": db_cams.get(cam_id, "unknown")}
        for cam_id in _camera_manager.get_active_cameras()
    ]


@camera_app.post("/cameras")
def add_camera(req: AddCameraRequest):
    cam_id      = req.camera_id.strip()
    source      = req.source.strip()
    camera_type = (req.camera_type or "rtsp").strip()

    if camera_type == "webcam" and str(source).isdigit():
        parsed = int(source)
    elif camera_type == "rtsp":
        parsed = sanitize_rtsp_url(source)
    elif camera_type == "droidcam":
        parsed = f"http://{source}:4747/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "ipwebcam":
        parsed = f"http://{source}:8080/video" if ":" not in source else f"http://{source}/video"
    else:
        parsed = source

    status, final_source = _camera_manager.add_camera(cam_id, parsed)

    if status == 0:
        _db_manager.add_camera_to_db(cam_id, final_source)
        threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
        logger.info(f"[CameraServer] Added: {cam_id}")
        return {"status": "success", "camera_id": cam_id, "source": final_source}
    elif status == 1:
        raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' already exists.")
    else:
        raise HTTPException(status_code=502, detail=f"Cannot connect to camera at '{source}'.")


@camera_app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    """Safely remove a camera without deadlocking the writer_lock."""
    stop_event = None
    thread = None
    wd = None

    # 1. Quickly extract info and signal stop while under lock
    with writer_lock:
        if camera_id in camera_writers:
            wd = camera_writers.pop(camera_id)
            stop_event = recording_stop_events.pop(camera_id, None)
            thread = recording_threads.pop(camera_id, None)
            if stop_event:
                stop_event.set()

    # 2. Perform blocking cleanup OUTSIDE the writer_lock
    if wd:
        proc = wd.get("process")
        if proc:
            try:
                proc.stdin.close()
                proc.wait(timeout=1)
            except Exception:
                proc.kill()
        _db_manager.end_recording(wd.get("db_id"))

    if thread:
        thread.join(timeout=2)

    # 3. Final removal from managers
    _camera_manager.remove_camera(camera_id)
    _db_manager.remove_camera_from_db(camera_id)
    with results_lock:
        camera_results.pop(camera_id, None)
        
    logger.info(f"[CameraServer] Removed and cleaned up: {camera_id}")
    return {"status": "success"}


@camera_app.get("/results/{camera_id}")
def get_results(camera_id: str):
    with results_lock:
        data = camera_results.get(camera_id)
    if data is None:
        raise HTTPException(status_code=404, detail="No results yet.")
    return {
        "camera_id": camera_id,
        "count":     data.get("count", 0),
        "tracks":    data.get("tracks", []),
        "timestamp": data.get("timestamp"),
    }


@camera_app.get("/occupancy")
def get_occupancy(camera_id: Optional[str] = None):
    out = {}
    for cam_id in _camera_manager.get_active_cameras():
        if camera_id and cam_id != camera_id:
            continue
        with results_lock:
            data  = camera_results.get(cam_id, {})
            count = data.get("count", 0) or occupancy_last_count.get(cam_id, 0)
        out[cam_id] = {
            "camera_id":   cam_id,
            "count":       count,
            "total_today": _db_manager.get_total_unique_count_today(cam_id),
        }
    return out


@camera_app.get("/daily_stats")
def get_daily_stats():
    stats = _db_manager.get_camera_daily_person_stats()
    for cam_id in _camera_manager.get_active_cameras():
        if cam_id not in stats:
            stats[cam_id] = {"am": 0, "pm": 0, "total": 0}
    return stats


@camera_app.get("/settings/{camera_id}")
def get_camera_settings(camera_id: str):
    enabled = bool(_db_manager.get_camera_recording_setting(camera_id))
    with writer_lock:
        recording = camera_id in camera_writers
    return {
        "camera_id":          camera_id,
        "recording_enabled":  enabled,
        "actually_recording": recording,
    }


@camera_app.post("/settings/{camera_id}")
async def set_camera_settings(camera_id: str, request: Request):
    body    = await request.json()
    enabled = bool(body.get("enabled", True))
    _db_manager.set_camera_recording(camera_id, enabled)
    return {"status": "success"}


# ── MJPEG stream ──────────────────────────────────────────────────────────────

async def _gen_frames(camera_id: str):
    INTERVAL = 1.0 / 4
    next_t   = time.time()
    last_fb  = None
    while True:
        wait = next_t - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        next_t += INTERVAL
        if next_t < time.time() - (3 * INTERVAL):
            next_t = time.time() + INTERVAL

        with results_lock:
            fb = camera_results.get(camera_id, {}).get("encoded_frame")
        fb = fb if fb is not None else last_fb
        if fb is None:
            continue
        last_fb = fb
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(fb)).encode() + b"\r\n\r\n"
            + fb + b"\r\n"
        )


@camera_app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(
        _gen_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@camera_app.get("/capture/{camera_id}")
def capture_frame(camera_id: str):
    with results_lock:
        fb = camera_results.get(camera_id, {}).get("encoded_frame")
    if fb is None:
        raise HTTPException(status_code=404, detail="No frame available.")
    return Response(content=fb, media_type="image/jpeg")


# ── Public start function (called from app.py) ────────────────────────────────

def start(host: str = "0.0.0.0", port: int = CAMERA_SERVER_PORT):
    """
    Build singletons then run the camera server in the current thread.
    Call this inside a daemon thread from app.py so it runs alongside
    the main uvicorn server without blocking it.
    """
    _build_singletons()
    logger.info(f"[CameraServer] Listening on {host}:{port}")
    uvicorn.run(
        camera_app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
