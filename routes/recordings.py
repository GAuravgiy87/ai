import os
import logging
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from core.auth import require_auth
from core.state import templates, results_lock, camera_results, format_12h, RECORDINGS_DIR
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

_db_manager = None
_recording_service = None

def init_routes(db, recording_service=None):
    """
    Initialize recordings routes.
    
    Args:
        db: Database manager
        recording_service: RecordingService instance (optional, for new architecture)
    """
    global _db_manager, _recording_service
    _db_manager = db
    _recording_service = recording_service

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

@router.get("/api/recording_status/{camera_id}")
async def get_recording_status(camera_id: str):
    """
    Get recording status for a camera.
    Recording is automatic - this endpoint is for status only.
    """
    if _recording_service is not None:
        is_recording = _recording_service.is_recording(camera_id)
        return {
            "camera_id": camera_id,
            "recording": is_recording,
            "mode": "automatic",
            "chunk_duration": _recording_service.chunk_duration
        }
    else:
        return {
            "camera_id": camera_id,
            "recording": False,
            "mode": "disabled"
        }

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
    Prevents path traversal attacks and uses FileResponse for efficient streaming.
    """
    # Security: validate path is within recordings directory
    abs_path = os.path.abspath(path)
    base_recordings = os.path.abspath(RECORDINGS_DIR)
    
    try:
        # Ensure abs_path is within base_recordings using commonpath
        if os.path.commonpath([abs_path, base_recordings]) != base_recordings:
            logger.warning(f"[Security] Blocked unauthorized file access attempt: {path}")
            raise HTTPException(status_code=403, detail="Unauthorized path")
    except ValueError:
        # Different drives on Windows or other path issues
        logger.warning(f"[Security] Blocked unauthorized file access attempt (invalid path): {path}")
        raise HTTPException(status_code=403, detail="Unauthorized path")
    
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # FileResponse handles Accept-Ranges and large files via streaming
    return FileResponse(
        abs_path,
        media_type="video/mp4",
        filename=os.path.basename(abs_path)
    )
