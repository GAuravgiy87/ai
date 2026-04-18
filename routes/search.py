import cv2
import numpy as np
import os
import json
from fastapi import APIRouter, Request, Form, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from core.auth import require_auth
from core.state import templates, IST, active_search, active_search_lock
from core.pipeline import scan_video_for_person
from typing import Optional

router = APIRouter()

_db_manager = None
_recognizer = None

def init_routes(db, rec):
    global _db_manager, _recognizer
    _db_manager = db
    _recognizer = rec

@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "search.html", {})

@router.get("/api/search")
async def api_search(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = _db_manager.search_detections(name, start_time, end_time)
    return [{"id": r[0], "person_name": r[1] or "Unknown", "camera_id": r[2], "timestamp": str(r[3]), "image_path": r[4]} for r in results]

@router.post("/api/start_search")
async def start_search(name: str = Form(...)):
    persons = _db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target:
        with active_search_lock:
            active_search.clear()
            active_search.update({"running": True, "person_id": target[0], "name": target[1], "encoding": np.frombuffer(target[3], dtype=np.float32), "found_track_ids": set()})
        return {"status": "success", "name": target[1]}
    return {"status": "error"}

@router.post("/api/stop_search")
async def stop_search():
    with active_search_lock: active_search.clear()
    return {"status": "success"}

@router.post("/api/search_video_by_name")
async def api_search_video_by_name(request: Request):
    data = await request.json(); name = data.get("name"); video_ids = data.get("video_ids", [])
    persons = _db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if not target: return {"status": "error"}
    enc = np.frombuffer(target[3], dtype=np.float32); all_res = []
    for vid in video_ids:
        rec = _db_manager.get_recording(vid)
        if rec and os.path.exists(rec[4]):
            for s in scan_video_for_person(rec[4], enc):
                all_res.append({**s, "video_id": vid, "camera_id": rec[1], "person_name": name})
    return {"status": "success", "results": all_res}
