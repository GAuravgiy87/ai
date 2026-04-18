"""routes/recordings.py — Recording toggle, list, delete, video timeline."""
import os
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.auth import require_auth
from core.state import (
    format_full_dt, camera_writers, writer_lock,
    recording_threads, recording_stop_events,
)

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

_db_manager = None


def init(db_manager):
    global _db_manager
    _db_manager = db_manager


@router.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "recordings.html", {})


@router.get("/api/recordings")
async def api_recordings(
    camera_id:  Optional[str] = None,
    start_time: Optional[str] = None,
    end_time:   Optional[str] = None,
    page:       int = 1,
    page_size:  int = 50,
):
    recs  = _db_manager.search_recordings(camera_id=camera_id,
                                          start_time=start_time, end_time=end_time,
                                          page=page, page_size=page_size)
    total = _db_manager.count_recordings(camera_id=camera_id,
                                         start_time=start_time, end_time=end_time)
    result = []
    for r in recs:
        result.append({
            "id":         r[0],
            "camera_id":  r[1],
            "start_time": format_full_dt(r[2]),
            "end_time":   format_full_dt(r[3]),
            "file_path":  r[4],
            "has_person": r[5],
        })
    return {"data": result, "total": total, "page": page, "page_size": page_size}


@router.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: str):
    rec = _db_manager.get_recording(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    try:
        if rec[4] and os.path.exists(rec[4]):
            os.remove(rec[4])
    except Exception:
        pass
    _db_manager.delete_recording(record_id)
    return {"status": "success"}


@router.get("/api/recording_status")
async def get_recording_status():
    with writer_lock:
        return {"active_recordings": list(camera_writers.keys())}


@router.get("/api/video_timeline/{record_id}")
async def video_timeline(record_id: str):
    rec = _db_manager.get_recording(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    start_time = rec[2]
    end_time   = rec[3]
    camera_id  = rec[1]
    snaps = _db_manager.get_detection_snapshots(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        limit=500,
    )
    events = []
    for s in snaps:
        ts = s[2]
        if start_time and ts:
            try:
                from datetime import datetime
                st = start_time if isinstance(start_time, datetime) else \
                     datetime.fromisoformat(str(start_time))
                offset = (ts - st).total_seconds() if ts > st else 0
                events.append({
                    "offset_seconds": round(offset, 1),
                    "person_count":   s[3],
                    "snapshot_path":  s[4],
                    "timestamp":      format_full_dt(ts),
                })
            except Exception:
                pass
    return {"recording_id": record_id, "events": events,
            "start_time": format_full_dt(start_time),
            "end_time":   format_full_dt(end_time)}


@router.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    """Toggle recording on/off for a camera."""
    import subprocess, threading, os as _os
    from core.state import camera_results, results_lock, LOCAL_RECORDINGS_DIR, get_ist_time
    from core.pipeline import recording_writer_thread

    with writer_lock:
        currently = camera_id in camera_writers

    if not currently:
        with results_lock:
            data  = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
        if frame is None:
            return {"status": "error", "message": "No frame available — camera not active"}
        h, w    = frame.shape[:2]
        ist_now = get_ist_time()
        ds      = ist_now.strftime("%Y-%m-%d")
        ts      = ist_now.strftime("%H%M%S")
        dp      = f"{LOCAL_RECORDINGS_DIR}/{ds}/{camera_id}"
        _os.makedirs(dp, exist_ok=True)
        lp  = f"{dp}/rec_{camera_id}_{ts}.mp4"
        sw  = min(w, 1280) - (min(w, 1280) % 2)
        sh  = int(h * sw / w) - (int(h * sw / w) % 2)
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2", "-i", "-",
               "-vf", f"scale={sw}:{sh}", "-vcodec", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "faster", "-crf", "32",
               "-tune", "fastdecode", "-movflags", "+faststart", lp]
        proc  = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        db_id = _db_manager.start_recording(camera_id, lp)
        se    = threading.Event()
        rt    = threading.Thread(target=recording_writer_thread,
                                 args=(camera_id, se), daemon=True)
        rt.start()
        with writer_lock:
            camera_writers[camera_id] = {
                "process": proc, "db_id": db_id, "start_time": ist_now,
                "file_path": lp, "camera_id": camera_id, "w": w, "h": h,
            }
        recording_threads[camera_id]     = rt
        recording_stop_events[camera_id] = se
        _db_manager.set_camera_recording(camera_id, True)
        return {"status": "success", "recording": True}
    else:
        with writer_lock:
            wd = camera_writers.pop(camera_id, None)
        se = recording_stop_events.pop(camera_id, None)
        if se:
            se.set()
        if wd:
            try:
                wd["process"].stdin.close()
                wd["process"].wait(timeout=10)
            except Exception:
                pass
            _db_manager.end_recording(wd["db_id"])
        _db_manager.set_camera_recording(camera_id, False)
        return {"status": "success", "recording": False}
