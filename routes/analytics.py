from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from core.auth import require_auth
from core.state import templates, get_ist_time
from datetime import timedelta
from typing import Optional

router = APIRouter()

_db_manager = None

def init_routes(db):
    global _db_manager
    _db_manager = db

@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "analytics.html", {})

@router.get("/api/analytics/hourly")
async def api_analytics_hourly(camera_id: Optional[str] = None):
    analytics_data = _db_manager.get_hourly_analytics(camera_id)
    hour_map = {int(r["_id"]): r for r in analytics_data}
    data = []; now = get_ist_time()
    for i in range(24):
        check_time = now - timedelta(hours=(23-i)); h = check_time.hour
        h_data = hour_map.get(h, {"max_count": 0, "camera_ids": []})
        data.append({"hour": h, "label": check_time.strftime("%I %p"), "count": h_data["max_count"]})
    return data

@router.get("/api/analytics/daily")
async def api_analytics_daily(camera_id: Optional[str] = None, days: int = 7):
    analytics_data = _db_manager.get_daily_analytics(camera_id, days=days)
    day_map = {f"{r['_id']['year']}-{r['_id']['month']:02d}-{r['_id']['day']:02d}": r["max_count"] for r in analytics_data}
    data = []; now = get_ist_time()
    for i in range(days):
        check_date = now - timedelta(days=(days-1-i))
        key = f"{check_date.year}-{check_date.month:02d}-{check_date.day:02d}"
        data.append({"date": key, "label": check_date.strftime("%d %b"), "count": day_map.get(key, 0)})
    return data
