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

@router.get("/journey", response_class=HTMLResponse)
async def journey_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "journey.html", {})

@router.get("/api/journey/active_targets")
async def get_active_targets():
    return _db_manager.get_recent_active_targets(hours=24)

@router.get("/api/journey/target/{global_id}")
async def get_target_journey(global_id: str):
    # get_target_journey returns list of dicts (RealDictCursor rows)
    events = _db_manager.get_target_journey(global_id)
    return [
        {
            "camera_id":   e.get("camera_id"),
            "timestamp":   format_12h(e.get("timestamp")),
            "person_type": e.get("type"),
        }
        for e in events
    ]

@router.get("/api/journey/thumbnail/{global_id}")
async def get_target_thumbnail(global_id: str):
    # Thumbnail is stored in global_identities.thumbnail column
    identity = _db_manager.get_global_identity_by_id(global_id)
    if identity:
        thumb = identity.get("thumbnail")
        if thumb:
            return Response(content=bytes(thumb), media_type="image/jpeg")
    raise HTTPException(status_code=404)
