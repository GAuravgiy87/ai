import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from core.auth import require_auth
from core.state import templates, format_12h
from typing import Optional

router = APIRouter()

_db_manager = None

def init_routes(db):
    global _db_manager
    _db_manager = db

@router.get("/detection_logs", response_class=HTMLResponse)
async def detection_logs(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "detections.html", {})

@router.get("/api/detection_snapshots")
async def get_detection_snapshots(camera_id: Optional[str] = None, page: int = 1, page_size: int = 20):
    snaps = _db_manager.get_detection_snapshots(camera_id=camera_id, page=page, page_size=page_size)
    total = _db_manager.count_detection_snapshots(camera_id=camera_id)
    return {
        "data": [{"id": s[0], "camera_id": s[1], "timestamp": format_12h(s[2]), "person_count": s[3], "snapshot_path": s[4], "bbox_data": s[5]} for s in snaps],
        "total": total, "page": page, "page_size": page_size
    }

@router.get("/api/snapshot_image")
async def get_snapshot_image(path: str):
    if os.path.exists(path):
        with open(path, 'rb') as f: content = f.read()
        return Response(content=content, media_type="image/jpeg")
    raise HTTPException(status_code=404)

@router.post("/clear_history")
async def clear_history():
    _db_manager.delete_all_detections()
    return {"status": "success"}
