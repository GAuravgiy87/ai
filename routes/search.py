"""routes/search.py — Face search (live + video)."""
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from core.auth import require_auth
from core.state import (
    templates,
    active_search, active_search_lock,
    camera_results, results_lock,
    format_full_dt,
)

router    = APIRouter()

_db_manager = None
_recognizer = None   # lambda: recognizer


def init(db_manager, get_recognizer):
    global _db_manager, _recognizer
    _db_manager = db_manager
    _recognizer = get_recognizer


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "search.html", {})


@router.get("/api/search")
async def api_search(name: Optional[str] = None,
                     start_time: Optional[str] = None,
                     end_time:   Optional[str] = None,
                     limit: int = 100):
    results = _db_manager.search_detections(name=name,
                                            start_time=start_time,
                                            end_time=end_time,
                                            limit=limit)
    return [{"id": r[0], "person_name": r[1], "camera_id": r[2],
             "timestamp": format_full_dt(r[3])} for r in results]


@router.get("/api/search_detections")
async def api_search_detections(name: Optional[str] = None,
                                start_time: Optional[str] = None,
                                end_time:   Optional[str] = None):
    results = _db_manager.search_detections(name=name,
                                            start_time=start_time,
                                            end_time=end_time,
                                            limit=200)
    formatted = []
    for r in results:
        formatted.append({
            "id":          r[0],
            "person_name": r[1],
            "camera_id":   r[2],
            "timestamp":   format_full_dt(r[3]),
            "snapshot":    r[4],
        })
    return formatted


@router.post("/api/start_search")
async def start_search(name: str = Form(...)):
    persons = _db_manager.get_registered_persons()
    target  = next((p for p in persons if p[1].lower() == name.lower()), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Person '{name}' not found")
    encoding = np.frombuffer(target[3], dtype=np.float32)
    with active_search_lock:
        active_search.clear()
        active_search.update({
            "person_id":      str(target[0]),
            "name":           name,
            "encoding":       encoding,
            "found_track_ids": set(),
            "running":        True,
        })
    return {"status": "success", "name": name}


@router.post("/api/stop_search")
async def stop_search():
    with active_search_lock:
        active_search.clear()
    return {"status": "stopped"}


@router.get("/api/active_search")
async def get_active_search():
    with active_search_lock:
        if not active_search.get("running"):
            return {"active": False}
        return {"active": True, "name": active_search.get("name"),
                "person_id": active_search.get("person_id")}


@router.post("/api/search_by_image")
async def search_by_image(file: UploadFile = File(...),
                          camera_ids: str = Form("")):
    recognizer = _recognizer() if _recognizer else None
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not ready")
    img_bytes = await file.read()
    nparr     = np.frombuffer(img_bytes, np.uint8)
    image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    encoding = recognizer.get_encoding(image)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected")

    results = []
    with results_lock:
        cams = list(camera_results.keys())
    for cam_id in cams:
        with results_lock:
            data   = camera_results.get(cam_id, {})
            tracks = data.get("tracks", [])
        for t in tracks:
            results.append({"camera_id": cam_id, "track_id": t["id"],
                             "name": t.get("name", "Unknown")})
    return {"status": "success", "results": results}


# ── Video search ──────────────────────────────────────────────────────────

def _scan_video_for_person(video_path: str, target_encoding: np.ndarray) -> list:
    recognizer = _recognizer() if _recognizer else None
    if recognizer is None:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps            = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps * 2))
    DISTANCE_THRESHOLD = 1.15
    results, current_segment, last_match_frame = [], None, -1
    min_segment_gap = int(fps * 2)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with recognizer.ai_lock:
                    boxes, _ = recognizer.mtcnn.detect(frame_rgb)
                match_found, best_conf = False, 0.0
                if boxes is not None:
                    for box in boxes:
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        crop = frame_rgb[max(0,fy1):fy2, max(0,fx1):fx2]
                        if crop.size == 0 or (fx2-fx1) < 30 or (fy2-fy1) < 30:
                            continue
                        face_160 = cv2.resize(crop, (160, 160))
                        emb = recognizer._get_embedding(face_160)
                        if emb is None:
                            continue
                        dist = float(np.linalg.norm(target_encoding - emb))
                        if dist < DISTANCE_THRESHOLD:
                            match_found = True
                            conf = 1 - (dist / 2.0)
                            if conf > best_conf:
                                best_conf = conf
                if match_found:
                    ts_sec = frame_count / fps
                    ts_str = f"{int(ts_sec//60)}:{int(ts_sec%60):02d}"
                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        if current_segment:
                            results.append(current_segment)
                        current_segment = {
                            "start_seconds": ts_sec, "start_timestamp": ts_str,
                            "end_seconds": ts_sec, "end_timestamp": ts_str,
                            "confidence": best_conf, "video_path": video_path,
                        }
                    else:
                        current_segment["end_seconds"]   = ts_sec
                        current_segment["end_timestamp"] = ts_str
                        if best_conf > current_segment["confidence"]:
                            current_segment["confidence"] = best_conf
                    last_match_frame = frame_count
            except Exception:
                pass
        frame_count += 1
    if current_segment:
        results.append(current_segment)
    cap.release()
    return results


@router.post("/api/search_video_by_name")
async def search_video_by_name(request: Request):
    data      = await request.json()
    name      = data.get("name")
    video_ids = data.get("video_ids", [])
    if not name or not video_ids:
        return {"status": "error", "message": "Name and video IDs required"}
    persons = _db_manager.get_registered_persons()
    target  = next((p for p in persons if p[1].lower() == name.lower()), None)
    if not target:
        return {"status": "error", "message": f"Person '{name}' not found"}
    enc = np.frombuffer(target[3], dtype=np.float32)
    all_results = []
    for vid_id in video_ids:
        rec = _db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            for seg in _scan_video_for_person(rec[4], enc):
                all_results.append({**seg, "video_id": vid_id,
                                    "video_name": os.path.basename(rec[4]),
                                    "camera_id": rec[1], "person_name": name})
    all_results.sort(key=lambda x: x["start_seconds"])
    return {"status": "success", "results": all_results,
            "total_segments": len(all_results), "videos_searched": len(video_ids)}


@router.post("/api/search_video_by_image")
async def search_video_by_image(file: UploadFile = File(...),
                                video_ids: str = Form(...)):
    import json as _json
    recognizer = _recognizer() if _recognizer else None
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not ready")
    video_ids_list = _json.loads(video_ids)
    img_bytes = await file.read()
    nparr     = np.frombuffer(img_bytes, np.uint8)
    image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    enc       = recognizer.get_encoding(image)
    if enc is None:
        return {"status": "error", "message": "No face detected"}
    all_results = []
    for vid_id in video_ids_list:
        rec = _db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            for seg in _scan_video_for_person(rec[4], enc):
                all_results.append({**seg, "video_id": vid_id,
                                    "video_name": os.path.basename(rec[4]),
                                    "camera_id": rec[1],
                                    "person_name": "Unknown (from image)"})
    all_results.sort(key=lambda x: x["start_seconds"])
    return {"status": "success", "results": all_results,
            "total_segments": len(all_results), "videos_searched": len(video_ids_list)}
