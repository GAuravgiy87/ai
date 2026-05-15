import os
import threading
import subprocess
import logging
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from core.auth import require_auth
from core.state import (
    templates, writer_lock, camera_writers, get_ist_time, LOCAL_RECORDINGS_DIR,
    recording_threads, recording_stop_events, results_lock, camera_results,
    format_12h
)
from core.pipeline import recording_writer_thread
from typing import Optional

# BUG FIX #3: Add missing logger
logger = logging.getLogger(__name__)

router = APIRouter()

_db_manager = None

def init_routes(db):
    global _db_manager
    _db_manager = db

@router.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "recordings.html", {})

@router.get("/api/recordings")
async def api_recordings(camera_id: Optional[str] = None):
    results = _db_manager.search_recordings(camera_id)
    return [{
        "id": r[0], "camera_id": r[1],
        "start_time": format_12h(r[2]),
        "end_time": format_12h(r[3]) if r[3] else None,
        "file_path": r[4],
        "has_registered_person": r[5] if len(r) > 5 else False
    } for r in results]

@router.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    # BUG FIX #4: Implement actual toggle instead of always setting True
    current = _db_manager.get_camera_recording_setting(camera_id)
    new_state = not current
    _db_manager.set_camera_recording(camera_id, new_state)
    return {"status": "success", "recording": new_state}

@router.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: str):
    rec = _db_manager.get_recording(record_id)
    if rec and os.path.exists(rec[4]):
        try: os.remove(rec[4])
        except: pass
    _db_manager.delete_recording(record_id)
    return {"status": "success"}

@router.get("/api/recording_video")
async def get_recording_video(path: str, request: Request):
    """
    Stream video with security validation and efficient range support.
    BUG-02, BUG-03 fix: Use FileResponse for automatic range-request and RAM efficiency.
    SEC-01 fix: Prevent Local File Inclusion (LFI) via path traversal.
    """
    # BUG FIX #6: Use os.path.commonpath for safer path traversal check
    abs_path = os.path.abspath(path)
    base_recordings = os.path.abspath(LOCAL_RECORDINGS_DIR)
    
    try:
        # Safer check: ensure abs_path is within base_recordings using commonpath
        if os.path.commonpath([abs_path, base_recordings]) != base_recordings:
            logger.warning(f"Blocked unauthorized file access attempt: {path}")
            raise HTTPException(status_code=403, detail="Unauthorized path")
    except ValueError:
        # Different drives on Windows or other path issues
        logger.warning(f"Blocked unauthorized file access attempt (invalid path): {path}")
        raise HTTPException(status_code=403, detail="Unauthorized path")
        
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found")

    # 2. Performance: FileResponse handles Accept-Ranges and large files via streaming
    from fastapi.responses import FileResponse
    return FileResponse(
        abs_path, 
        media_type="video/mp4", 
        filename=os.path.basename(abs_path)
    )
