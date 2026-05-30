import cv2
import logging
import time
import numpy as np
import threading
import queue
import base64
import json
import subprocess
import asyncio
import torch
import os
from typing import Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from core.state import (
    get_ist_time, camera_results, results_lock,
    occupancy_last_count, occupancy_last_track_ids, recognition_cooldowns,
    cooldown_lock, MAX_CACHE_SIZE, SNAPSHOTS_DIR, SNAPSHOT_COOLDOWN_SECONDS,
    snapshot_cooldowns, RECORDINGS_DIR,
    camera_recognized_persons, recognized_lock,
    reid_lock, global_reid_assignments,
    active_search, active_search_lock
)
from utils.tracker import ObjectTracker

# Singletons for models (Injected via init)
_detector = None
_recognizer = None
_reid_manager = None
_db_manager = None
logger = logging.getLogger(__name__)
_camera_manager = None

def init_pipeline(db, cam, det, rec, reid, num_detection_workers: int = 1):
    global _db_manager, _camera_manager, _detector, _recognizer, _reid_manager, _detection_pool
    _db_manager = db
    _camera_manager = cam
    _detector = det
    _recognizer = rec
    _reid_manager = reid
    # BUG FIX #2a: Always initialize detection pool regardless of det being None
    # This ensures recording can work even when detection models aren't loaded
    _detection_pool = DetectionWorkerPool(_detector, num_workers=1)
    # Start resource guard
    from core.resource_guard import start as _rg_start
    _rg_start()
    init_search_models(det, rec)

from core.notifications import notification_manager

# Resource management
recognition_executor = ThreadPoolExecutor(max_workers=1)
transfer_queue = queue.Queue(maxsize=50)

# Shared Detection Worker Pool (replaces per-camera detection threads)
from core.detection_pool import DetectionWorkerPool, DetectionTask, DetectionResult

_detection_pool: Optional[DetectionWorkerPool] = None



def transfer_worker():
    """Background worker for sequential file tasks."""
    while True:
        try:
            item = transfer_queue.get()
            if item is None: break
            data, destination, callback = item
            success = _perform_direct_stream(data, destination)
            if callback:
                callback(success)
            transfer_queue.task_done()
        except Exception:
            time.sleep(1)

def _perform_direct_stream(data: bytes, local_path: str) -> bool:
    try:
        parent = os.path.dirname(local_path)
        if parent: os.makedirs(parent, exist_ok=True)
        with open(local_path, 'wb') as f: f.write(data)
        return True
    except Exception: return False



threading.Thread(target=transfer_worker, daemon=True).start()

def stream_bytes_to_local(data: bytes, local_path: str, callback=None) -> bool:
    try:
        transfer_queue.put((data, local_path, callback), block=False)
        return True
    except queue.Full: return False

def process_camera(camera_id: str):
    """Main camera processing pipeline."""
    warmup_frames = 0
    max_warmup_attempts = 30 # ~3 seconds
    attempts = 0
    frame = None
    while warmup_frames < 5 and attempts < max_warmup_attempts:
        frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None: warmup_frames += 1
        else: attempts += 1
        time.sleep(0.1)

    if frame is None:
        # BUG-21 fix: log clearly so the camera shows as offline, then exit
        logger.warning(
            f"[Pipeline] Camera {camera_id} warmup failed after {max_warmup_attempts} attempts. "
            f"Stream is offline — pipeline thread exiting."
        )
        return

    # Detection FPS — controlled dynamically by resource guard
    _DET_FPS = 6.0

    tracker = ObjectTracker(max_age=6, n_init=2, iou_threshold=0.15)
    frame_count = 0
    RECOGNITION_CACHE_FRAMES = 18
    face_encoding_cache: Dict[int, np.ndarray] = {}
    track_merge_map:     Dict[int, int]        = {}
    track_face_crops:    Dict[int, tuple]      = {}
    identity_snap_cooldowns: Dict[tuple, float] = {}
    recognition_cache:   Dict[Any, tuple]      = {}
    next_render_time  = time.time()
    _next_submit_time = time.time()
    last_frame_time   = time.time()

    _color_cache = {}
    def get_color(pid):
        if pid not in _color_cache:
            _color_cache[pid] = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[(pid * 137) % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        return _color_cache[pid]

    while _camera_manager and camera_id in _camera_manager.cameras:
        # ── Dynamic FPS from resource guard ──────────────────────────────
        from core.resource_guard import get_det_fps, is_paused, should_skip_clahe, get_jpeg_quality
        _current_fps = get_det_fps()
        if _current_fps <= 0 or is_paused():
            time.sleep(0.1)
            continue
        RENDER_INTERVAL = 1.0 / _current_fps
        _SUBMIT_INTERVAL = 1.0 / _current_fps

        wait = next_render_time - time.time()
        if wait > 0: time.sleep(wait)
        next_render_time += RENDER_INTERVAL
        if next_render_time < time.time() - (3 * RENDER_INTERVAL):
            next_render_time = time.time() + RENDER_INTERVAL

        # Submit frame to shared detection pool at controlled rate
        now = time.time()
        if now >= _next_submit_time:
            raw_frame_submit, _ = _camera_manager.get_camera_frame_with_id(camera_id)
            
            # Submit frame to detection pool
            if raw_frame_submit is not None:
                last_frame_time = now
                if _detection_pool is not None:
                    _detection_pool.submit_frame(camera_id, raw_frame_submit)
            
            _next_submit_time = now + _SUBMIT_INTERVAL

        # Get result from shared detection pool — consume-once (pop, not get)
        if _detection_pool is None:
            time.sleep(0.05)
            continue
        result = _detection_pool.get_result(camera_id)
        if result is None:
            # No new detection yet — wait a bit longer to prevent high CPU spin
            time.sleep(0.05)
            continue
        proc_frame = result.processed_frame
        dets = list(result.detections)
        submit_t = result.submit_time

        # Grab the freshest raw frame for rendering (newer than proc_frame)
        raw_frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
        if raw_frame is None: raw_frame = proc_frame
        rh, rw = raw_frame.shape[:2]

        # Apply lighting normalization to the display frame.
        # Skip CLAHE when CPU is high (resource guard says so) — saves ~5ms/frame.
        try:
            from utils.detector import _analyze_frame, _normalize_frame
            _b, _c, _dark, _over = _analyze_frame(raw_frame)
            if not should_skip_clahe():
                display_frame = _normalize_frame(raw_frame, _b, _c, _dark, _over)
            else:
                # Lightweight: gamma only (no CLAHE), much cheaper
                if _b > 5:
                    _gamma = float(np.clip(np.log(120.0/_b) / np.log(255.0/_b), 0.4, 2.5))
                else:
                    _gamma = 0.4
                if abs(_gamma - 1.0) > 0.1:
                    _lut = np.array([min(255, int((i/255.0)**_gamma*255))
                                     for i in range(256)], dtype=np.uint8)
                    display_frame = cv2.LUT(raw_frame, _lut)
                else:
                    display_frame = raw_frame
        except Exception:
            display_frame = raw_frame

        # ── Correct scale factors: detection-space → raw-frame resolution ──
        # The detection frame is letterboxed: the image is scaled so the
        # WIDER dimension fits in 640.  Both sw and sh must use the SAME
        # scale factor (r) so boxes are not stretched vertically.
        #   r  = scale applied when resizing raw → proc (640-wide)
        #   pad_top = vertical padding added by letterbox
        # We undo exactly what the detector did.
        det_h, det_w = proc_frame.shape[:2]   # actual proc frame size (640 × scaled_h)
        r_scale = rw / 640.0                  # horizontal scale: det→raw
        # vertical: proc_frame height may be < 640 (letterbox), raw height is rh
        # correct vertical scale = rh / det_h  (not rh / (rh*640/rw))
        r_scale_y = rh / det_h

        frame_count += 1
        try:
            h, w = proc_frame.shape[:2]
            # tracker.update() now returns vx/vy per track (needed below)
            tracks = tracker.update(dets, proc_frame)
            tracks = sorted(tracks, key=lambda x: x["id"])

            # Build processed list — carry vx/vy through for lag fix
            processed = []
            for t in tracks:
                name, conf = "Unknown", 0.0
                if t['id'] in recognition_cache:
                    cn, cc, cf = recognition_cache[t['id']]
                    if (frame_count - cf) < RECOGNITION_CACHE_FRAMES:
                        name, conf = cn, cc
                processed.append({
                    "id":         t['id'],
                    "bbox":       t['bbox'],
                    "vx":         t.get('vx', 0.0),   # ← propagated from tracker
                    "vy":         t.get('vy', 0.0),
                    "name":       name,
                    "confidence": conf,
                })

            # Submit recognition jobs (use detection-space bbox, no lag comp needed here)
            for t in processed:
                if t['id'] in recognition_cache and (frame_count - recognition_cache[t['id']][2]) < (RECOGNITION_CACHE_FRAMES // 2):
                    continue
                with cooldown_lock:
                    last_t = recognition_cooldowns.get((camera_id, t['id']), 0)
                    if time.time() - last_t < (15.0 if t["name"] != "Unknown" else 3.0):
                        continue
                    recognition_cooldowns[(camera_id, t['id'])] = time.time()
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                bw_box, bh_box = bx2 - bx1, by2 - by1
                face_box = [bx1 + int(0.15 * bw_box), by1,
                            bx2 - int(0.15 * bw_box), by1 + int(0.45 * bh_box)]
                try:
                    recognition_executor.submit(
                        self_recognition_worker, proc_frame.copy(), face_box,
                        t['id'], recognition_cache, frame_count,
                        face_encoding_cache, track_merge_map, camera_id
                    )
                except RuntimeError:
                    break

            # ── Render on the normalized display frame ────────────────────
            record_frame = display_frame.copy()
            final_processed = []

            # ── Step 1: compute all scaled bboxes ────────────────────────
            # No lag compensation applied to bbox position.
            # The tracker stores the LAST DETECTED position (not predicted).
            # The detection just completed (submit_t is fresh), so the bbox
            # is already as current as it can be.
            # Applying velocity-based forward prediction causes the box to
            # overshoot the person, especially when they move toward/away
            # from the camera (size changes, not just position).
            render_items = []
            for t in processed:
                bx1, by1, bx2, by2 = t["bbox"]

                rbx1 = max(0, min(rw-1, int(bx1 * r_scale)))
                rby1 = max(0, min(rh-1, int(by1 * r_scale_y)))
                rbx2 = max(rbx1+2, min(rw-1, int(bx2 * r_scale)))
                rby2 = max(rby1+2, min(rh-1, int(by2 * r_scale_y)))

                tid  = t['id']
                name = t['name']
                if name != "Unknown":
                    color = (0, 255, 0);  label = name
                else:
                    btid = tid
                    while btid in track_merge_map:
                        btid = track_merge_map[btid]
                    color = get_color(btid);  label = f"#{btid}"

                render_items.append({
                    "id": tid, "rbx1": rbx1, "rby1": rby1,
                    "rbx2": rbx2, "rby2": rby2,
                    "color": color, "label": label, "name": name,
                })

            # ── Step 2: depth-sort — far (small rby2) drawn first ────────
            render_items.sort(key=lambda x: x["rby2"])
            for idx, item in enumerate(render_items):
                rbx1  = item["rbx1"];  rby1 = item["rby1"]
                rbx2  = item["rbx2"];  rby2 = item["rby2"]
                color = item["color"]; label = item["label"]
                box_h = rby2 - rby1
                thick = max(1, min(3, box_h // 80))
                fscale = max(0.4, min(0.8, box_h / 200.0))

                # Collect front-person rects (those drawn after this one)
                front_rects = [
                    (o["rbx1"], o["rby1"], o["rbx2"], o["rby2"])
                    for o in render_items[idx+1:]
                    if o["rbx1"] < rbx2 and o["rbx2"] > rbx1  # horizontal overlap
                ]

                # Fast and efficient drawing: only draw full rectangles if no major overlap
                # This significantly reduces CPU overhead on older processors like i7-4790
                if len(front_rects) == 0:
                    cv2.rectangle(record_frame, (rbx1, rby1), (rbx2, rby2), color, thick)
                else:
                    # Simple dashed-like approach for occluded boxes (much cheaper than pixel-masking)
                    cv2.rectangle(record_frame, (rbx1, rby1), (rbx2, rby2), color, 1)

                # Label
                label_y = rby1 - 8 if rby1 > 20 else rby1 + int(fscale*20) + 4
                cv2.putText(record_frame, label, (rbx1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick)

                final_processed.append({
                    "id":   item["id"],
                    "bbox": [rbx1, rby1, rbx2, rby2],
                    "name": item["name"],
                    "face_crop": None, "face_visible": False, "face_box_coords": None,
                })

            with results_lock:
                _jq = get_jpeg_quality()
                camera_results[camera_id] = {
                    "rendered_frame": record_frame,
                    "encoded_frame":  cv2.imencode('.jpg', record_frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, _jq])[1].tobytes(),
                    "tracks":         final_processed,
                    "count":          len(final_processed),
                    "timestamp":      time.time(),
                }

            c_ids = set(t['id'] for t in final_processed)
            l_ids = occupancy_last_track_ids.get(camera_id, set())
            if c_ids != l_ids:
                occupancy_last_track_ids[camera_id] = c_ids
                occupancy_last_count[camera_id]     = len(c_ids)
                _db_manager.log_occupancy(camera_id, len(c_ids))
                if len(c_ids) > 0 and (time.time() - snapshot_cooldowns.get(camera_id, 0)) >= 60:
                    snapshot_cooldowns[camera_id] = time.time()
                    ist   = get_ist_time()
                    spath = (f"{SNAPSHOTS_DIR}/{ist.strftime('%Y-%m-%d')}/{camera_id}/logs/"
                             f"{camera_id}_{ist.strftime('%Y-%m-%d_%H%M%S')}.jpg")
                    def _on_s(ok):
                        if ok:
                            _db_manager.log_detection_snapshot(
                                camera_id, len(c_ids), spath, final_processed, timestamp=ist)
                    stream_bytes_to_local(
                        cv2.imencode('.jpg', record_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(),
                        spath, callback=_on_s)

        except Exception as e:
            logger.error(f"[Pipeline:{camera_id}] Error: {e}", exc_info=True)
            time.sleep(1)
    
    logger.info(f"[Pipeline:{camera_id}] Pipeline loop exited.")

def self_recognition_worker(frame, face_box, track_id, recognition_cache, frame_count, face_encoding_cache, track_merge_map, camera_id):
    # BUG-07 fix: guard against recognizer or reid_manager being None
    if _recognizer is None:
        return
    try:
        name, conf, enc = _recognizer.recognize_with_encoding(frame, face_box)
        if enc is not None:
            face_encoding_cache[track_id] = enc
            for o_id, o_enc in face_encoding_cache.items():
                if o_id != track_id and np.linalg.norm(enc - o_enc) < 0.45:
                    if track_id < o_id: track_merge_map[o_id] = track_id
                    else: track_merge_map[track_id] = o_id
                    break
        if name != "Unknown" and conf >= 0.90: recognition_cache[track_id] = (name, conf, frame_count)
        gid = None
        if name != "Unknown" and conf >= 0.90: gid = name
        elif enc is not None:
            if _reid_manager is None:
                return
            with reid_lock: gid = global_reid_assignments.get((camera_id, track_id))
            if not gid:
                gid = _reid_manager.match(enc) or _reid_manager.register_new(enc)
        if gid:
            with reid_lock:
                if global_reid_assignments.get((camera_id, track_id)) != gid:
                    global_reid_assignments[(camera_id, track_id)] = gid
                    ist = get_ist_time()
                    _db_manager.log_journey_event(gid, camera_id, None, "unknown" if "U-" in str(gid) else "registered", ist)
                    if "U-" not in str(gid):
                        notification_manager.broadcast({"type": "detection", "camera": camera_id, "target": str(gid), "time": ist.strftime("%I:%M %p"), "is_registered": True})
    except Exception: pass

from core.search_pipeline import scan_video_for_person, init_search_models


def get_recognizer():
    """Return the active recognizer instance (injected or pipeline-level)."""
    return _recognizer
