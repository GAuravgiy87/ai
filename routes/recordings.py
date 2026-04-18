import os
import subprocess
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from core.auth import require_auth
from core.state import (
    templates, writer_lock, camera_writers, get_ist_time, LOCAL_RECORDINGS_DIR,
    recording_threads, recording_stop_events, results_lock, camera_results,
    format_12h
)
from core.pipeline import recording_writer_thread
from typing import Optional

router = APIRouter()

_db_manager = None

def init_routes(db):
    global _db_manager
    _db_manager = db

@router.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "recordings.html", {})

@router.get("/api/recordings")
async def api_recordings(camera_id: Optional[str] = None):
    results = _db_manager.search_recordings(camera_id)
    return [{
        "id": r[0], "camera_id": r[1],
        "start_time": format_12h(r[2]),
        "end_time": format_12h(r[3]) if r[3] else None,
        "file_path": r[4],
        "has_registered_person": r[5] if len(r) > 5 else False
    } for r in results]

@router.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    with writer_lock:
        if camera_id in camera_writers:
            wd = camera_writers.pop(camera_id)
            if camera_id in recording_stop_events:
                recording_stop_events[camera_id].set()
            if "process" in wd:
                try: wd["process"].stdin.close(); wd["process"].wait(timeout=2)
                except: wd["process"].kill()
            _db_manager.end_recording(wd["db_id"])
            return {"status": "success", "recording": False}
        else:
            with results_lock: frame = camera_results.get(camera_id, {}).get("rendered_frame")
            if frame is None: return {"status": "error", "message": "Offline"}
            h, w = frame.shape[:2]; ist = get_ist_time()
            l_path = f"{LOCAL_RECORDINGS_DIR}/{ist.strftime('%Y-%m-%d')}/{camera_id}/{camera_id}_{ist.strftime('%H%M%S')}.mp4"
            os.makedirs(os.path.dirname(l_path), exist_ok=True)
            cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2", "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "28", "-movflags", "+faststart", l_path]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            db_id = _db_manager.start_recording(camera_id, l_path)
            se = threading.Event(); rt = threading.Thread(target=recording_writer_thread, args=(camera_id, se), daemon=True)
            rt.start()
            camera_writers[camera_id] = {"process": p, "db_id": db_id, "start_time": ist, "file_path": l_path, "camera_id": camera_id, "w": w, "h": h}
            recording_threads[camera_id] = rt; recording_stop_events[camera_id] = se
            return {"status": "success", "recording": True}

@router.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: str):
    rec = _db_manager.get_recording(record_id)
    if rec and os.path.exists(rec[4]):
        try: os.remove(rec[4])
        except: pass
    _db_manager.delete_recording(record_id)
    return {"status": "success"}

@router.get("/api/recording_video")
async def get_recording_video(path: str, request: Request):
    if not os.path.exists(path): raise HTTPException(status_code=404)
    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")
    if range_header:
        start = int(range_header.replace("bytes=", "").split("-")[0]); end = file_size - 1
        with open(path, "rb") as f:
            f.seek(start); data = f.read(end - start + 1)
        return Response(content=data, status_code=206, media_type="video/mp4", headers={"Content-Range": f"bytes {start}-{end}/{file_size}", "Accept-Ranges": "bytes"})
    with open(path, "rb") as f: data = f.read()
    return Response(content=data, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
