"""routes/dashboard.py — Dashboard, analytics, system logs, hw status."""
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from core.auth import require_auth
from core.state import get_ist_time, format_12h, format_full_dt, format_date_key, camera_results, results_lock, templates

router    = APIRouter()

# Injected by app.py
_db_manager      = None
_camera_manager  = None
_hw              = None


def init(db_manager, camera_manager, hw):
    global _db_manager, _camera_manager, _hw
    _db_manager     = db_manager
    _camera_manager = camera_manager
    _hw             = hw


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "index.html", {})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "dashboard.html", {})


@router.get("/api/dashboard_metrics")
async def dashboard_metrics(request: Request):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    active_cameras     = len(_camera_manager.cameras)
    registered_persons = _db_manager.count_registered_persons()
    total_recordings   = _db_manager.count_recordings()

    try:
        raw          = _db_manager.get_detections(limit=20)
        persons_db   = _db_manager.get_registered_persons()
        person_images = {p[1]: p[2] for p in persons_db}
        cameras      = {c[0]: c[1] for c in _db_manager.get_cameras()}
        recent_detections = []
        for d in raw:
            ts = d.get("timestamp")
            recent_detections.append({
                "person_name":   d.get("person_name", "Unknown"),
                "camera_id":     d.get("camera_id", ""),
                "camera_source": cameras.get(d.get("camera_id", ""), ""),
                "timestamp":     format_full_dt(ts),
                "snapshot_path": d.get("snapshot_path"),
                "profile_image": person_images.get(d.get("person_name", ""), None),
            })
    except Exception:
        recent_detections = []

    with results_lock:
        live_counts = {cid: data.get("count", 0)
                       for cid, data in camera_results.items()}
    total_live = sum(live_counts.values())

    return {
        "active_cameras":     active_cameras,
        "registered_persons": registered_persons,
        "total_recordings":   total_recordings,
        "recent_detections":  recent_detections,
        "live_counts":        live_counts,
        "total_live":         total_live,
    }


@router.get("/api/server_time")
async def get_server_time():
    now = get_ist_time()
    return {
        "time":     now.strftime("%I:%M:%S %p"),
        "date":     now.strftime("%A, %d %b %Y"),
        "iso":      now.isoformat(),
        "timezone": "IST",
    }


@router.get("/api/system_logs")
async def api_system_logs(
    request: Request,
    level:      Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    source:     Optional[str] = None,
    page:       int = 1,
    page_size:  int = 50,
):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    logs  = _db_manager.get_system_logs(level=level, date_from=date_from,
                                        date_to=date_to, source=source,
                                        page=page, page_size=page_size)
    total = _db_manager.count_system_logs(level=level, date_from=date_from,
                                          date_to=date_to, source=source)
    for log in logs:
        if log.get("timestamp"):
            log["timestamp"] = format_full_dt(log["timestamp"])
    return {"data": logs, "total": total, "page": page, "page_size": page_size}


@router.get("/system_logs", response_class=HTMLResponse)
async def system_logs_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "system_logs.html", {})


@router.get("/api/hw_status")
async def hw_status():
    return _hw.get_status()


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    if not require_auth(request):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "analytics.html", {})


@router.get("/api/camera_daily_stats")
async def api_camera_daily_stats():
    stats = _db_manager.get_camera_daily_person_stats()
    for cid in _camera_manager.get_active_cameras():
        if cid not in stats:
            stats[cid] = {"am": 0, "pm": 0, "total": 0}
    return stats


@router.get("/api/hourly_counts")
async def api_hourly_counts():
    # get_hourly_analytics returns list of {hour, camera_id, count}
    raw      = _db_manager.get_hourly_analytics() or []
    hour_map = {}
    for row in raw:
        h = row.get("hour", 0)
        if h not in hour_map:
            hour_map[h] = {"max_count": 0, "camera_ids": []}
        hour_map[h]["max_count"] = max(hour_map[h]["max_count"], row.get("count", 0))
        cid = row.get("camera_id")
        if cid and cid not in hour_map[h]["camera_ids"]:
            hour_map[h]["camera_ids"].append(cid)

    now    = get_ist_time()
    result = []
    for i in range(24):
        check_time = now - timedelta(hours=(23 - i))
        h      = check_time.hour
        h_data = hour_map.get(h, {"max_count": 0, "camera_ids": []})
        result.append({
            "hour":       h,
            "label":      check_time.strftime("%I %p"),
            "max_count":  h_data.get("max_count", 0),
            "camera_ids": h_data.get("camera_ids", []),
        })
    return result


@router.get("/api/daily_counts")
async def api_daily_counts(days: int = 7):
    raw    = _db_manager.get_daily_analytics() or []
    # raw is list of {date, camera_id, count}
    day_map = {}
    for row in raw:
        d = row.get("date", "")
        if d not in day_map:
            day_map[d] = 0
        day_map[d] = max(day_map[d], row.get("count", 0))

    now    = get_ist_time()
    result = []
    for i in range(days):
        d   = now - timedelta(days=(days - 1 - i))
        key = f"{d.year}-{d.month:02d}-{d.day:02d}"
        result.append({"date": key, "label": d.strftime("%d %b"),
                        "count": day_map.get(key, 0)})
    return result
