"""
routes/cameras.py — Camera routes for the main app (port 9000).

All camera operations are proxied to the Camera Server (port 9001) via
core.camera_client.  The main app no longer owns any camera state.
"""

import asyncio
import time
import httpx
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, Response
from core.auth import require_auth
from core.state import templates, sanitize_rtsp_url
from camera_server import client as camera_client
from typing import Optional

router = APIRouter()

_db_manager = None

def init_routes(db, cam=None):
    global _db_manager
    _db_manager = db
    # cam is no longer used - all camera ops go through camera_client


# ── Pages ─────────────────────────────────────────────────────────────────────

@router.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "cameras.html", {})


@router.get("/add_camera", response_class=HTMLResponse)
async def add_camera_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "add_camera.html", {})


# ── API ───────────────────────────────────────────────────────────────────────

@router.get("/api/cameras")
async def api_cameras():
    """List all active cameras (proxied from camera server)."""
    return await camera_client.list_cameras()


@router.post("/api/add_camera")
async def add_camera(
    request: Request,
    camera_id:   str = Form(None),
    camera_type: str = Form(None),
    source:      str = Form(None),
):
    # Accept both form data and JSON body
    if camera_id is None or source is None:
        try:
            payload     = await request.json()
            camera_id   = camera_id   or payload.get("camera_id")
            camera_type = camera_type or payload.get("camera_type")
            source      = source      or payload.get("source")
        except Exception:
            pass

    if not camera_id or not source:
        return {"status": "error", "message": "camera_id and source are required."}

    result = await camera_client.add_camera(
        camera_id   = camera_id.strip(),
        source      = source.strip(),
        camera_type = (camera_type or "rtsp").strip(),
    )

    # camera_client raises HTTPException on 4xx/5xx; map to friendly messages
    if isinstance(result, dict) and result.get("detail"):
        return {"status": "error", "message": result["detail"]}
    return result


@router.delete("/api/remove_camera/{camera_id}")
async def delete_camera(camera_id: str):
    return await camera_client.remove_camera(camera_id)


@router.get("/api/occupancy")
async def api_occupancy(request_camera_id: Optional[str] = None):
    return await camera_client.get_occupancy(request_camera_id)


@router.get("/api/camera_daily_stats")
async def api_camera_daily_stats():
    return await camera_client.get_daily_stats()


@router.get("/api/live_results/{camera_id}")
async def get_live_results(camera_id: str):
    data = await camera_client.get_results(camera_id)
    if data is None:
        return []
    return [{"id": p["id"], "name": p["name"]} for p in data.get("tracks", [])]


@router.get("/api/camera_settings/{camera_id}")
async def get_camera_settings(camera_id: str):
    return await camera_client.get_camera_settings(camera_id)


@router.post("/api/camera_settings/{camera_id}")
async def set_camera_settings(camera_id: str, enabled: bool = Form(...)):
    return await camera_client.set_camera_settings(camera_id, enabled)


# ── Video streaming (proxy from camera server) ────────────────────────────────

@router.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    """
    Proxy the MJPEG stream from the camera server to the browser.
    Uses httpx async streaming so the main app doesn't buffer frames.
    """
    cam_url = camera_client.video_feed_url(camera_id)

    async def _proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", cam_url) as resp:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    yield chunk

    return StreamingResponse(
        _proxy(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/capture_frame/{camera_id}")
async def capture_frame(camera_id: str):
    """Proxy a single JPEG snapshot from the camera server."""
    cap_url = camera_client.capture_url(camera_id)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(cap_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=404, detail="No frame available.")
            return Response(content=resp.content, media_type="image/jpeg")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Camera server unreachable.")
