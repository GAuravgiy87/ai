import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from core.auth import require_auth
from core.state import templates, get_ist_time, format_12h
from camera_server import client as camera_client
from typing import Optional

router = APIRouter()

_db_manager = None

def init_routes(db, cam=None):
    global _db_manager
    _db_manager = db
    # cam is no longer used - all camera ops go through camera_client

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
    
    active_cameras = len(camera_client.list_cameras())
    registered_persons = len(_db_manager.get_registered_persons())
    total_recordings = len(_db_manager.get_recorded_videos())
    # BUG-17 fix: removed analytics DB writes from here — these were called on
    # every dashboard poll (every few seconds), generating thousands of rows/hour.
    # Metrics are now only written by the background analytics_snapshot_task.

    try:
        # database already returns newest first
        raw = _db_manager.get_detections(limit=20)
        persons_db = _db_manager.get_registered_persons()
        person_images = {p[1]: p[2] for p in persons_db}

        recent_detections = []
        for d in raw:
            # d is a dict from sqlite_manager
            pname = d.get("person_name", "Unknown")
            ts = d.get("timestamp")
            recent_detections.append({
                "person_name": pname,
                "person_names": [pname],
                "image_path": d.get("snapshot_path") or person_images.get(pname),
                "camera_id": d.get("camera_id", ""),
                "timestamp": format_12h(ts) if ts else "—",
            })
    except Exception as e:
        from core.pipeline import logger
        logger.error(f"Dashboard metrics error: {e}")
        recent_detections = []
        
    return {
        "active_cameras": active_cameras,
        "registered_persons": registered_persons,
        "total_recordings": total_recordings,
        "recent_detections": recent_detections
    }

@router.get("/api/notifications/stream")
async def stream_notifications(request: Request):
    """Event stream for real-time dashboard notifications."""
    from core.pipeline import notification_manager
    from fastapi.responses import StreamingResponse
    
    async def event_generator():
        q = await notification_manager.subscribe()
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                # Get message from queue
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=10.0)
                    yield msg
                except asyncio.TimeoutError:
                    # Keep-alive comment
                    yield ": keep-alive\n\n"
        finally:
            notification_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

@router.get("/api/recognized/{camera_id}")
async def get_recognized(camera_id: str):
    """Return session-recognized persons for a camera."""
    from core.state import camera_recognized_persons, recognized_lock
    with recognized_lock:
        persons_map = camera_recognized_persons.get(camera_id, {})
        # Convert {tid: name} to list of objects for frontend compatibility
        results = []
        unique_names = set()
        for tid, name in persons_map.items():
            if name != "Unknown" and name not in unique_names:
                results.append({"id": tid, "name": name})
                unique_names.add(name)
        return results

@router.get("/api/hw_status")
async def hw_status():
    """Return real-time hardware utilization."""
    from utils.hw_manager import hw
    return hw.get_status()

@router.get("/api/total_count")
async def get_total_count(request: Request, period: str = 'day', camera_id: Optional[str] = None):
    """Get total detection count with time-based filtering - same as live stream."""
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        if period == 'day':
            # Use the same method as live stream for daily total
            stats = _db_manager.get_camera_daily_person_stats()
            
            if camera_id:
                # Get total for specific camera
                cam_stats = stats.get(camera_id, {"total": 0})
                count = cam_stats.get("total", 0)
            else:
                # Sum all cameras
                count = sum(s.get("total", 0) for s in stats.values())
        else:
            # For week/month, use the existing method
            count = _db_manager.get_total_detections_count(period=period, camera_id=camera_id)
        
        # Store this metric for historical tracking
        _db_manager.store_analytics_snapshot(
            metric_type=f'total_count_{period}',
            value=count,
            camera_id=camera_id,
            metadata={'period': period}
        )
        
        return {
            "count": count,
            "period": period,
            "camera_id": camera_id
        }
    except Exception as e:
        from core.pipeline import logger
        logger.error(f"Total count error: {e}")
        return {"count": 0, "period": period, "camera_id": camera_id}

@router.get("/api/live_total_count")
async def get_live_total_count(request: Request):
    """Get real-time total count across all cameras - same as shown in live stream."""
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Get daily stats (same as live stream)
        stats = _db_manager.get_camera_daily_person_stats()
        
        # Calculate totals
        total_count = sum(s.get("total", 0) for s in stats.values())
        am_count = sum(s.get("am", 0) for s in stats.values())
        pm_count = sum(s.get("pm", 0) for s in stats.values())
        
        # Get per-camera breakdown
        camera_stats = {}
        for cam in camera_client.list_cameras():
            cam_id = cam['id'] if isinstance(cam, dict) else cam
            cam_stat = stats.get(cam_id, {"am": 0, "pm": 0, "total": 0})
            camera_stats[cam_id] = cam_stat
        
        return {
            "total": total_count,
            "am": am_count,
            "pm": pm_count,
            "cameras": camera_stats
        }
    except Exception as e:
        from core.pipeline import logger
        logger.error(f"Live total count error: {e}")
        return {"total": 0, "am": 0, "pm": 0, "cameras": {}}
