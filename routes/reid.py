"""routes/reid.py — Active targets, journeys, thumbnails, SSE notifications."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from core.auth import require_auth
from core.state import format_full_dt, format_12h, templates

router    = APIRouter()

_db_manager           = None
_notification_manager = None


def init(db_manager, notification_manager):
    global _db_manager, _notification_manager
    _db_manager           = db_manager
    _notification_manager = notification_manager


@router.get("/journey", response_class=HTMLResponse)
async def journey_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "journey.html", {})


@router.get("/api/active_targets")
async def get_active_targets(request: Request, hours: int = 24):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    targets = _db_manager.get_recent_active_targets(hours=hours)
    res = []
    for t in targets:
        res.append({
            "global_id":   t["global_id"],
            "type":        t.get("type", "unknown"),
            "first_seen":  format_full_dt(t.get("first_seen")),
            "last_seen":   format_full_dt(t.get("last_seen")),
            "last_camera": t.get("last_camera"),
            "has_thumbnail": bool(t.get("thumbnail")),
        })
    return res


@router.get("/api/target_journey/{global_id}")
async def get_target_journey(request: Request, global_id: str):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    journey = _db_manager.get_target_journey(global_id)
    return [{
        "camera_id":     p.get("camera_id"),
        "timestamp":     format_full_dt(p.get("timestamp")),
        "snapshot_path": p.get("snapshot_path"),
        "type":          p.get("type"),
    } for p in journey]


@router.get("/api/target_thumbnail/{global_id}")
async def get_target_thumbnail(global_id: str):
    target = _db_manager.get_global_identity_by_id(global_id)
    if not target or not target.get("thumbnail"):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return Response(content=target["thumbnail"], media_type="image/jpeg")


@router.get("/api/notifications/stream")
async def notification_stream(request: Request):
    import asyncio, json as _json
    from fastapi.responses import StreamingResponse

    async def event_generator():
        q = await _notification_manager.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"ping\"}\n\n"
        finally:
            _notification_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
