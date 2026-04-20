import threading
import time
import asyncio
import os
import subprocess
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, Response
from core.auth import require_auth
from core.state import (
    templates, camera_results, results_lock, camera_recognized_persons, recognized_lock,
    occupancy_last_count, sanitize_rtsp_url, writer_lock, camera_writers, get_ist_time,
    LOCAL_RECORDINGS_DIR, recording_threads, recording_stop_events
)
from core.pipeline import process_camera, recording_writer_thread
from typing import Optional

router = APIRouter()

_db_manager = None
_camera_manager = None

def init_routes(db, cam):
    global _db_manager, _camera_manager
    _db_manager = db
    _camera_manager = cam

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

@router.get("/api/cameras")
async def api_cameras():
    """Get all active cameras with their source info."""
    cameras = []
    for cam_id in _camera_manager.get_active_cameras():
        cam_info = {"id": cam_id, "source": "Unknown"}
        try:
            db_cams = _db_manager.get_cameras()
            for db_cam in db_cams:
                if db_cam[0] == cam_id:
                    cam_info["source"] = db_cam[1] if len(db_cam) > 1 else "Local"
                    break
        except: pass
        cameras.append(cam_info)
    return cameras

@router.post("/api/add_camera")
async def add_camera(request: Request, camera_id: str = Form(None), camera_type: str = Form(None), source: str = Form(None)):
    if camera_id is None or source is None:
        try:
            payload = await request.json()
            camera_id = camera_id or payload.get("camera_id")
            camera_type = camera_type or payload.get("camera_type")
            source = source or payload.get("source")
        except: pass
    if not camera_id or not source:
        return {"status": "error", "message": "camera_id and source required"}

    parsed = source
    if camera_type == "webcam" and str(source).isdigit(): parsed = int(source)
    elif camera_type == "rtsp": parsed = sanitize_rtsp_url(source)
    elif camera_type == "droidcam": parsed = f"http://{source}:4747/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "ipwebcam": parsed = f"http://{source}:8080/video" if ":" not in source else f"http://{source}/video"

    status, final_source = _camera_manager.add_camera(camera_id, parsed)
    if status == 0:
        _db_manager.add_camera_to_db(camera_id, final_source)
        threading.Thread(target=process_camera, args=(camera_id,), daemon=True).start()
        return {"status": "success"}
    elif status == 1:
        return {"status": "error", "message": f"Camera ID '{camera_id}' already exists."}
    else:
        return {"status": "error", "message": f"Failed to connect to camera at '{source}'. Please check the URL and network."}

@router.delete("/api/remove_camera/{camera_id}")
async def delete_camera(camera_id: str):
    with writer_lock:
        if camera_id in camera_writers:
            wd = camera_writers.pop(camera_id)
            if camera_id in recording_stop_events:
                recording_stop_events[camera_id].set()
                if camera_id in recording_threads: recording_threads[camera_id].join(timeout=2)
            if "process" in wd:
                try: wd["process"].stdin.close(); wd["process"].wait(timeout=2)
                except: wd["process"].kill()
            _db_manager.end_recording(wd["db_id"])
    _camera_manager.remove_camera(camera_id)
    _db_manager.remove_camera_from_db(camera_id)
    camera_results.pop(camera_id, None)
    return {"status": "success"}

@router.get("/api/occupancy")
async def api_occupancy(request_camera_id: Optional[str] = None):
    results = {}
    for cam_id in _camera_manager.get_active_cameras():
        if request_camera_id and cam_id != request_camera_id: continue
        with results_lock:
            data = camera_results.get(cam_id, {})
            l_cnt = data.get("count", 0) or occupancy_last_count.get(cam_id, 0)
            alert = data.get("alert_active", False)
        results[cam_id] = {"id": cam_id, "camera_id": cam_id, "count": l_cnt, "head_count": l_cnt, "alert_active": alert, "total_today": _db_manager.get_total_unique_count_today(cam_id)}
    return results

@router.get("/api/camera_daily_stats")
async def api_camera_daily_stats():
    stats = _db_manager.get_camera_daily_person_stats()
    for cam_id in _camera_manager.get_active_cameras():
        if cam_id not in stats: stats[cam_id] = {"am": 0, "pm": 0, "total": 0}
    return stats

async def gen_frames(camera_id: str):
    STREAM_INTERVAL = 1.0 / 4
    next_s = time.time(); last_f = None
    while True:
        wait = next_s - time.time()
        if wait > 0: await asyncio.sleep(wait)
        next_s += STREAM_INTERVAL
        if next_s < time.time() - (3 * STREAM_INTERVAL): next_s = time.time() + STREAM_INTERVAL
        with results_lock: fb = camera_results.get(camera_id, {}).get("encoded_frame")
        if fb is None: fb = last_f
        if fb is None: continue
        last_f = fb
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(fb)).encode() + b"\r\n\r\n" + fb + b"\r\n")

@router.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/api/capture_frame/{camera_id}")
async def capture_frame(camera_id: str):
    with results_lock: fb = camera_results.get(camera_id, {}).get("encoded_frame")
    if fb is None: raise HTTPException(status_code=404)
    return Response(content=fb, media_type="image/jpeg")

@router.get("/api/live_results/{camera_id}")
async def get_live_results(camera_id: str):
    with results_lock: tracks = camera_results.get(camera_id, {}).get("tracks", [])
    return [{"id": p["id"], "name": p["name"]} for p in tracks]

@router.get("/api/camera_settings/{camera_id}")
async def get_camera_settings(camera_id: str):
    enabled = bool(_db_manager.get_camera_recording_setting(camera_id))
    with writer_lock: actually = camera_id in camera_writers
    return {"camera_id": camera_id, "recording_enabled": enabled, "actually_recording": actually}

@router.post("/api/camera_settings/{camera_id}")
async def set_camera_settings(camera_id: str, enabled: bool = Form(...)):
    _db_manager.set_camera_recording(camera_id, enabled)
    # Logic to start/stop recording would go here if needed instantly
    return {"status": "success"}
