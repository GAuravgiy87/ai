from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from core.auth import require_auth
from core.state import templates, get_ist_time, format_12h
from typing import Optional

router = APIRouter()

_db_manager = None
_camera_manager = None

def init_routes(db, cam):
    global _db_manager, _camera_manager
    _db_manager = db
    _camera_manager = cam

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "index.html", {})

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "dashboard.html", {})

@router.get("/api/dashboard_metrics")
async def dashboard_metrics(request: Request):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    active_cameras = len(_camera_manager.cameras)
    registered_persons = len(_db_manager.get_registered_persons())
    total_recordings = len(_db_manager.get_recorded_videos())
    
    try:
        raw = _db_manager.get_detections()
        raw = sorted(raw, key=lambda x: x.get("timestamp") or "", reverse=True)[:20]
        persons_db = _db_manager.get_registered_persons()
        person_images = {p[1]: p[2] for p in persons_db}

        recent_detections = []
        for d in raw:
            pname = d.get("person_name", "Unknown")
            ts = d.get("timestamp")
            recent_detections.append({
                "person_name": pname,
                "person_names": [pname],
                "image_path": d.get("snapshot_path") or person_images.get(pname),
                "camera_id": d.get("camera_id", ""),
                "timestamp": format_12h(ts) if ts else "—",
            })
    except Exception:
        recent_detections = []
        
    return {
        "active_cameras": active_cameras,
        "registered_persons": registered_persons,
        "total_recordings": total_recordings,
        "recent_detections": recent_detections
    }

@router.get("/api/server_time")
async def get_server_time():
    """Return the current server time in IST."""
    now = get_ist_time()
    return {
        "iso": now.isoformat(),
        "timestamp_ms": int(now.timestamp() * 1000),
        "display": now.strftime("%d %b %Y, %I:%M:%S %p"),
        "timezone": "Asia/Kolkata (IST)"
    }

@router.get("/api/hw_status")
async def hw_status():
    """Return real-time hardware utilization."""
    from utils.hw_manager import hw
    return hw.get_status()
