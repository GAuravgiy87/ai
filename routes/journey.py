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
    events = _db_manager.get_journey_events(global_id)
    return [{"camera_id": e[2], "timestamp": format_12h(e[5]), "person_type": e[4]} for e in events]

@router.get("/api/journey/thumbnail/{global_id}")
async def get_target_thumbnail(global_id: str):
    thumb = _db_manager.get_target_thumbnail(global_id)
    if thumb: return Response(content=thumb, media_type="image/jpeg")
    raise HTTPException(status_code=404)
