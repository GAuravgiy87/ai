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
    return templates.TemplateResponse(request, "detection_logs.html", {})

@router.get("/registered_detections", response_class=HTMLResponse)
async def registered_detections_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "registered_detections.html", {})

@router.get("/api/detection_snapshots")
async def get_detection_snapshots(
    camera_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    snaps = _db_manager.get_detection_snapshots(
        camera_id=camera_id, date_from=date_from, date_to=date_to,
        page=page, page_size=page_size
    )
    total = _db_manager.count_detection_snapshots(
        camera_id=camera_id, date_from=date_from, date_to=date_to
    )
    return {
        "data": [{"id": s[0], "camera_id": s[1], "timestamp": format_12h(s[2]), "person_count": s[3], "snapshot_path": s[4], "bbox_data": s[5]} for s in snaps],
        "total": total, "page": page, "page_size": page_size
    }

@router.get("/api/registered_detections")
async def get_registered_detections(
    name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    logs = _db_manager.get_registered_detections(
        name=name, date_from=date_from, date_to=date_to,
        page=page, page_size=page_size
    )
    total = _db_manager.count_registered_detections(
        name=name, date_from=date_from, date_to=date_to
    )

    # Enrich with camera IP from camera table
    cameras_raw = _db_manager.get_cameras()
    cam_ip_map = {}
    for cam_id, source in cameras_raw:
        ip = source or "N/A"
        if "@" in ip:
            ip = ip.split("@")[-1].split(":")[0].split("/")[0]
        elif ip.startswith("rtsp://"):
            ip = ip[7:].split(":")[0].split("/")[0]
        cam_ip_map[cam_id] = ip

    # Enrich with person profile image
    persons_raw = _db_manager.get_persons_with_last_seen()
    profile_map = {p["name"]: p["image_path"] for p in persons_raw}

    formatted = []
    for r in logs:
        ts = r["timestamp"]
        formatted.append({
            "person_name": r["person_name"],
            "camera_id": r["camera_id"],
            "camera_ip": cam_ip_map.get(r["camera_id"], "N/A"),
            "timestamp": format_12h(ts),
            "snapshot_path": r.get("snapshot_path"),
            "profile_image": profile_map.get(r["person_name"]),
        })

    return {"data": formatted, "total": total, "page": page, "page_size": page_size}

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
