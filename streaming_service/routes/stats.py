import os
import glob
from typing import Optional
from fastapi import APIRouter, HTTPException, Request

import streaming_service.state as state
from core.state import camera_results, results_lock, occupancy_last_count, get_ist_time

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status":  "ok",
        "cameras": state.camera_manager.get_active_cameras() if state.camera_manager else [],
    }

@router.get("/results/{camera_id}")
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

@router.get("/occupancy")
def get_occupancy(camera_id: Optional[str] = None):
    out = {}
    for cam_id in state.camera_manager.get_active_cameras():
        if camera_id and cam_id != camera_id:
            continue
        with results_lock:
            data  = camera_results.get(cam_id, {})
            count = data.get("count", 0) or occupancy_last_count.get(cam_id, 0)
        out[cam_id] = {
            "camera_id":   cam_id,
            "count":       count,
            "total_today": state.db_manager.get_total_unique_count_today(cam_id),
        }
    return out

@router.get("/daily_stats")
def get_daily_stats():
    stats = state.db_manager.get_camera_daily_person_stats()
    for cam_id in state.camera_manager.get_active_cameras():
        if cam_id not in stats:
            stats[cam_id] = {"am": 0, "pm": 0, "total": 0}
    return stats

@router.get("/settings/{camera_id}")
def get_camera_settings(camera_id: str):
    enabled = bool(state.db_manager.get_camera_recording_setting(camera_id))
    return {
        "camera_id":          camera_id,
        "recording_enabled":  enabled,
    }

@router.post("/settings/{camera_id}")
async def set_camera_settings(camera_id: str, request: Request):
    body    = await request.json()
    enabled = bool(body.get("enabled", True))
    state.db_manager.set_camera_recording(camera_id, enabled)
    return {"status": "success"}

@router.get("/recordings/{camera_id}")
def list_recordings(camera_id: str, date: str = None, page: int = 1, limit: int = 20):
    if not date:
        date = get_ist_time().strftime("%Y-%m-%d")
    
    folder_path = os.path.join("database", "recordings", date, camera_id)
    pattern = os.path.join(folder_path, "*.mkv")
    
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    
    total = len(files)
    start = (page - 1) * limit
    end = start + limit
    page_files = files[start:end]
    
    result_files = []
    for f in page_files:
        name = os.path.basename(f)
        try:
            size_mb = round(os.path.getsize(f) / (1024 * 1024), 2)
        except Exception:
            size_mb = 0
        
        hour = name.split(".")[0]
        result_files.append({
            "name": name,
            "path": f.replace("\\", "/"),
            "size_mb": size_mb,
            "hour": hour
        })
    
    return {
        "files": result_files,
        "total": total,
        "page": page,
        "limit": limit
    }
