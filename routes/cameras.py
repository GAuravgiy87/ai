"""routes/cameras.py — Camera CRUD, live feed, occupancy, settings."""
import asyncio
import threading
import time
from typing import Optional

import cv2
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from core.auth import require_auth
from core.state import (
    templates,
    camera_results, results_lock,
    camera_recognized_persons, recognized_lock,
    camera_writers, writer_lock,
    occupancy_last_count, occupancy_last_track_ids,
    sanitize_rtsp_url,
)

router    = APIRouter()

_db_manager     = None
_camera_manager = None
_process_camera = None   # callable injected from pipeline


def init(db_manager, camera_manager, process_camera_fn):
    global _db_manager, _camera_manager, _process_camera
    _db_manager     = db_manager
    _camera_manager = camera_manager
    _process_camera = process_camera_fn


# ── Pages ─────────────────────────────────────────────────────────────────

@router.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "cameras.html", {})


@router.get("/add_camera", response_class=HTMLResponse)
async def add_camera_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "add_camera.html", {})


# ── API ───────────────────────────────────────────────────────────────────

@router.get("/api/cameras")
async def api_cameras():
    cameras = []
    for cam_id in _camera_manager.get_active_cameras():
        cam_info = {"id": cam_id, "source": "Unknown"}
        try:
            db_cams = _db_manager.get_cameras()
            for dc in db_cams:
                if dc[0] == cam_id:
                    cam_info["source"] = dc[1]
                    break
        except Exception:
            pass
        with results_lock:
            data = camera_results.get(cam_id, {})
        cam_info["live_count"]   = data.get("count", 0)
        cam_info["alert_active"] = data.get("alert_active", False)
        cameras.append(cam_info)
    return cameras


@router.post("/api/add_camera")
async def add_camera(request: Request,
                     camera_id:   str = Form(None),
                     camera_type: str = Form(None),
                     source:      str = Form(None)):
    import threading
    # Support JSON payload too
    if camera_id is None:
        try:
            body      = await request.json()
            camera_id = body.get("camera_id")
            source    = body.get("source")
        except Exception:
            pass
    if not camera_id or not source:
        return {"status": "error", "message": "camera_id and source required"}

    parsed = sanitize_rtsp_url(source.strip())
    try:
        parsed = int(parsed) if str(parsed).isdigit() else parsed
    except Exception:
        pass

    if _camera_manager.add_camera(camera_id, parsed):
        _db_manager.add_camera_to_db(camera_id, parsed)
        threading.Thread(target=_process_camera, args=(camera_id,), daemon=True).start()
        # Log camera add as a critical operational event
        _db_manager.log_event("INFO", f"Camera added: {camera_id} source={parsed}",
                              source="camera.add")
        return {"status": "success"}
    return {"status": "error", "message": "Camera already exists or could not connect."}


@router.delete("/api/remove_camera/{camera_id}")
async def delete_camera(camera_id: str):
    with writer_lock:
        wd = camera_writers.get(camera_id)
        if wd and "process" in wd:
            try:
                wd["process"].stdin.close()
                wd["process"].wait(timeout=5)
            except Exception:
                try:
                    wd["process"].kill()
                except Exception:
                    pass
            _db_manager.end_recording(wd["db_id"])
            camera_writers.pop(camera_id, None)

    _camera_manager.remove_camera(camera_id)
    _db_manager.remove_camera_from_db(camera_id)
    _db_manager.log_event("INFO", f"Camera removed: {camera_id}", source="camera.remove")
    with results_lock:
        camera_results.pop(camera_id, None)
    return {"status": "success"}


@router.get("/api/recognized/{camera_id}")
async def api_recognized_persons(camera_id: str):
    with recognized_lock:
        persons = camera_recognized_persons.get(camera_id, {})
        return [{"track_id": tid, "name": name} for tid, name in persons.items()]


@router.get("/api/occupancy")
async def api_occupancy(request_camera_id: Optional[str] = None,
                        start_time: Optional[str] = None,
                        end_time:   Optional[str] = None):
    if start_time or end_time:
        rows = _db_manager.search_occupancy(request_camera_id, start_time, end_time)
        return [{"id": r[0], "camera_id": r[1], "timestamp": r[2], "count": r[3]}
                for r in rows]

    results = []
    active_cams = ([request_camera_id] if request_camera_id
                   else _camera_manager.get_active_cameras())
    for cam_id in active_cams:
        with results_lock:
            data       = camera_results.get(cam_id, {})
            live_count = data.get("count", 0)
            alert      = data.get("alert_active", False)
        results.append({
            "camera_id":    cam_id,
            "head_count":   live_count,
            "alert_active": alert,
            "total_today":  _db_manager.get_total_unique_count_today(cam_id),
        })
    return results


# ── Live video feed ───────────────────────────────────────────────────────

async def _gen_frames(camera_id: str):
    STREAM_INTERVAL = 1.0 / 4
    next_send       = time.time()
    last_bytes      = None
    while True:
        wait = next_send - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        next_send += STREAM_INTERVAL
        if next_send < time.time() - 3 * STREAM_INTERVAL:
            next_send = time.time() + STREAM_INTERVAL
        with results_lock:
            frame_bytes = camera_results.get(camera_id, {}).get("encoded_frame")
        if frame_bytes is None:
            frame_bytes = last_bytes
        if frame_bytes is None:
            continue
        last_bytes = frame_bytes
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(frame_bytes)).encode()
               + b"\r\n\r\n" + frame_bytes + b"\r\n")


@router.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(_gen_frames(camera_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/api/capture_frame/{camera_id}")
async def capture_frame(camera_id: str):
    from fastapi.responses import Response
    with results_lock:
        fb = camera_results.get(camera_id, {}).get("encoded_frame")
    if fb is None:
        raise HTTPException(status_code=404, detail="No frame available")
    return Response(content=fb, media_type="image/jpeg")


# ── Camera settings ───────────────────────────────────────────────────────

@router.get("/api/camera_settings/{camera_id}")
async def get_camera_settings(camera_id: str):
    db_setting        = _db_manager.get_camera_recording_setting(camera_id)
    actually_recording = camera_id in camera_writers
    return {"camera_id": camera_id,
            "recording_enabled": bool(db_setting),
            "actually_recording": actually_recording}


@router.post("/api/camera_settings/{camera_id}")
async def set_camera_settings(camera_id: str, enabled: bool = Form(...)):
    import subprocess, threading
    from core.state import (recording_threads, recording_stop_events,
                             LOCAL_RECORDINGS_DIR, get_ist_time)
    _db_manager.set_camera_recording(camera_id, enabled)
    if enabled:
        with writer_lock:
            if camera_id not in camera_writers:
                with results_lock:
                    data  = camera_results.get(camera_id, {})
                    frame = data.get("rendered_frame")
                if frame is not None:
                    h, w    = frame.shape[:2]
                    ist_now = get_ist_time()
                    ds      = ist_now.strftime("%Y-%m-%d")
                    ts      = ist_now.strftime("%H%M%S")
                    dp      = f"{LOCAL_RECORDINGS_DIR}/{ds}/{camera_id}"
                    import os; os.makedirs(dp, exist_ok=True)
                    lp  = f"{dp}/rec_{camera_id}_{ts}.mp4"
                    sw  = min(w,1280)-(min(w,1280)%2)
                    sh  = int(h*sw/w)-(int(h*sw/w)%2)
                    cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
                           "-s",f"{w}x{h}","-pix_fmt","bgr24","-r","2","-i","-",
                           "-vf",f"scale={sw}:{sh}","-vcodec","libx264",
                           "-pix_fmt","yuv420p","-preset","faster","-crf","32",
                           "-tune","fastdecode","-movflags","+faststart",lp]
                    proc  = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    db_id = _db_manager.start_recording(camera_id, lp)
                    se    = threading.Event()
                    from core.pipeline import recording_writer_thread
                    rt = threading.Thread(target=recording_writer_thread,
                                         args=(camera_id, se), daemon=True)
                    rt.start()
                    camera_writers[camera_id] = {
                        "process": proc, "db_id": db_id, "start_time": ist_now,
                        "file_path": lp, "camera_id": camera_id, "w": w, "h": h,
                    }
                    recording_threads[camera_id]     = rt
                    recording_stop_events[camera_id] = se
    else:
        with writer_lock:
            wd = camera_writers.pop(camera_id, None)
        if wd:
            se = recording_stop_events.pop(camera_id, None)
            if se:
                se.set()
            try:
                wd["process"].stdin.close()
                wd["process"].wait(timeout=10)
            except Exception:
                pass
            _db_manager.end_recording(wd["db_id"])
    return {"status": "success", "recording": enabled}
