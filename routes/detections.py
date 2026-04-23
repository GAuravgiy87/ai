"""routes/detections.py — Detection snapshots, detection logs, registered detections."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from core.auth import require_auth
from core.state import format_full_dt, format_12h, templates

router    = APIRouter()

_db_manager = None


def init(db_manager):
    global _db_manager
    _db_manager = db_manager


@router.get("/detection_logs", response_class=HTMLResponse)
async def detection_logs_page(request: Request, camera_id: Optional[str] = None):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "detection_logs.html",
                                      {"camera_id": camera_id})


@router.get("/registered_detections", response_class=HTMLResponse)
async def registered_detections_page(request: Request,
                                     person_name: Optional[str] = None):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "registered_detections.html",
                                      {"person_name": person_name})


@router.get("/api/detection_snapshots")
async def get_detection_snapshots(
    camera_id:  Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    page:       int = 1,
    page_size:  int = 20,
):
    snaps = _db_manager.get_detection_snapshots(
        camera_id=camera_id, date_from=date_from, date_to=date_to,
        page=page, page_size=page_size)
    total = _db_manager.count_detection_snapshots(
        camera_id=camera_id, date_from=date_from, date_to=date_to)
    result = []
    for s in snaps:
        result.append({
            "id":            s[0],
            "camera_id":     s[1],
            "timestamp":     format_full_dt(s[2]),
            "person_count":  s[3],
            "snapshot_path": s[4],
            "bbox_data":     s[5],
        })
    return {"data": result, "total": total, "page": page, "page_size": page_size}


@router.get("/api/registered_detections")
async def api_registered_detections(
    name:       Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    page:       int = 1,
    page_size:  int = 20,
):
    detections = _db_manager.get_registered_detections(
        name=name, date_from=date_from, date_to=date_to,
        page=page, page_size=page_size)
    total = _db_manager.count_registered_detections(
        name=name, date_from=date_from, date_to=date_to)
    result = []
    for d in detections:
        result.append({
            "person_name":   d.get("person_name"),
            "camera_id":     d.get("camera_id"),
            "timestamp":     format_full_dt(d.get("timestamp")),
            "snapshot_path": d.get("snapshot_path"),
        })
    return {"data": result, "total": total, "page": page, "page_size": page_size}


@router.get("/api/recent_alerts")
async def get_recent_alerts_api(request: Request, limit: int = 10):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    alerts = _db_manager.get_recent_alerts(limit=limit)
    return [{
        "id":            a["id"],
        "camera_id":     a.get("camera_id"),
        "person_id":     a.get("person_id"),
        "snapshot_path": a.get("snapshot_path"),
        "timestamp":     format_full_dt(a.get("timestamp")),
        "type":          a.get("type"),
    } for a in alerts]


@router.post("/clear_history")
async def clear_history():
    import os
    from core.state import SNAPSHOTS_DIR
    try:
        _db_manager.delete_all_detections()
        deleted = 0
        for root, dirs, files in os.walk(SNAPSHOTS_DIR):
            for f in files:
                if f.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(root, f))
                        deleted += 1
                    except Exception:
                        pass
        return {"status": "success", "message": f"Cleared {deleted} records"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
