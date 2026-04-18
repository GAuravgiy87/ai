"""
core/pipeline.py — Camera processing pipeline (detection, tracking, recognition, recording).
All shared state is imported from core.state. DB and managers injected at runtime.
"""
import base64
import logging
import os
import subprocess
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np

from core.state import (
    PRIO_HIGH, PRIO_LOW, PRIO_NORMAL, MAX_CACHE_SIZE,
    SNAPSHOT_COOLDOWN_SECONDS, SNAPSHOTS_DIR, LOCAL_RECORDINGS_DIR,
    _set_thread_priority, _prune_dict, stream_bytes_to_local,
    camera_results, results_lock,
    camera_recognized_persons, recognized_lock,
    camera_writers, writer_lock,
    occupancy_last_count, occupancy_last_track_ids, snapshot_cooldowns,
    recording_threads, recording_stop_events,
    global_reid_assignments, reid_lock,
    recognition_executor, recognition_cooldowns, cooldown_lock,
)
from core.state import get_ist_time

logger = logging.getLogger(__name__)

# These are set by core.startup after models load
_detector_ready   = None   # threading.Event — set by startup
_recognizer_ready = None   # threading.Event — set by startup

# Injected by app.py after init
_db_manager          = None
_camera_manager      = None
_notification_manager = None
_reid_manager        = None
_detector_ref        = None   # lambda: detector
_recognizer_ref      = None   # lambda: recognizer


def init_pipeline(db_manager, camera_manager, notification_manager,
                  reid_manager, detector_ready, recognizer_ready,
                  get_detector, get_recognizer):
    """Called once from app.py after all singletons are created."""
    global _db_manager, _camera_manager, _notification_manager, _reid_manager
    global _detector_ready, _recognizer_ready, _detector_ref, _recognizer_ref
    _db_manager           = db_manager
    _camera_manager       = camera_manager
    _notification_manager = notification_manager
    _reid_manager         = reid_manager
    _detector_ready       = detector_ready
    _recognizer_ready     = recognizer_ready
    _detector_ref         = get_detector
    _recognizer_ref       = get_recognizer


# ---------------------------------------------------------------------------
# Recording writer thread
# ---------------------------------------------------------------------------
def recording_writer_thread(camera_id: str, stop_event: threading.Event):
    _set_thread_priority(PRIO_LOW)
    logger.info(f"[Recording:{camera_id}] Writer thread started")
    FRAME_INTERVAL = 0.5  # 2 FPS to FFmpeg
    while not stop_event.is_set():
        try:
            with writer_lock:
                if camera_id not in camera_writers:
                    break
                process = camera_writers[camera_id].get("process")
            with results_lock:
                data  = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")
                if frame is not None and "rendered_frame" in data:
                    data["rendered_frame"] = None
            if frame is not None and process and process.poll() is None:
                try:
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                except (IOError, BrokenPipeError):
                    logger.warning(f"[Recording:{camera_id}] Pipe broken")
                    break
            time.sleep(FRAME_INTERVAL)
        except Exception as e:
            logger.error(f"[Recording:{camera_id}] Writer error: {e}")
            time.sleep(1)
    logger.info(f"[Recording:{camera_id}] Writer stopped")


# ---------------------------------------------------------------------------
# Recognition worker (runs in recognition_executor)
# ---------------------------------------------------------------------------
def self_recognition_worker(frame, face_box, track_id, recognition_cache,
                             frame_count, face_encoding_cache, track_merge_map, camera_id):
    recognizer = _recognizer_ref() if _recognizer_ref else None
    if recognizer is None:
        return
    try:
        name, conf, face_encoding = recognizer.recognize_with_encoding(frame, face_box)

        if face_encoding is not None:
            face_encoding_cache[track_id] = face_encoding
            for other_id, other_enc in list(face_encoding_cache.items()):
                if other_id != track_id:
                    if np.linalg.norm(face_encoding - other_enc) < 0.6:
                        if track_id < other_id:
                            track_merge_map[other_id] = track_id
                        else:
                            track_merge_map[track_id] = other_id
                        break

        if name != "Unknown" and conf >= 0.90:
            recognition_cache[track_id] = (name, conf, frame_count)

        global_id = None
        if name != "Unknown" and conf >= 0.90:
            global_id = name
        elif face_encoding is not None:
            with reid_lock:
                global_id = global_reid_assignments.get((camera_id, track_id))
            if not global_id:
                global_id = _reid_manager.match(face_encoding)
                if not global_id:
                    try:
                        fx1, fy1, fx2, fy2 = face_box
                        crop = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                        _, buf = cv2.imencode(".jpg", crop) if crop.size > 0 else (None, None)
                        thumbnail = buf.tobytes() if buf is not None else None
                    except Exception:
                        thumbnail = None
                    global_id = _reid_manager.register_new(face_encoding, thumbnail)

        if global_id:
            with reid_lock:
                old = global_reid_assignments.get((camera_id, track_id))
                if old != global_id:
                    global_reid_assignments[(camera_id, track_id)] = global_id
                    now_ist = get_ist_time()
                    ptype   = "unknown" if "U-" in str(global_id) else "registered"
                    _db_manager.log_journey_event(
                        global_id=global_id, camera_id=camera_id,
                        snapshot_path=None, person_type=ptype, timestamp=now_ist)
                    if "U-" not in str(global_id):
                        _notification_manager.broadcast({
                            "type": "detection", "camera": camera_id,
                            "target": str(global_id),
                            "thumbnail": f"https://ui-avatars.com/api/?name={global_id}&background=random",
                            "time": now_ist.strftime("%I:%M %p"),
                            "is_registered": True,
                        })
    except Exception as e:
        logger.error(f"[RecogWorker] {e}")


# ---------------------------------------------------------------------------
# Main camera pipeline
# ---------------------------------------------------------------------------
def process_camera(camera_id: str):
    """
    Per-camera 2-thread pipeline at fixed 10 FPS:
      Thread A (det-{id})  : capture → YOLO on GPU → pipe result
      Thread B (this)      : read pipe → track → render → publish
      recognition_executor : face recognition (non-blocking, GPU)
    """
    from utils.tracker import ObjectTracker

    logger.info(f"[Camera:{camera_id}] Starting pipeline")
    deadline = time.time() + 30
    frame = None
    while time.time() < deadline:
        frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None:
            break
        time.sleep(0.2)

    if frame is None:
        logger.error(f"[Camera:{camera_id}] No frames after 30 s — aborting")
        _db_manager.log_event("ERROR", f"Camera {camera_id} no frames", source="camera.startup")
        return

    logger.info(f"[Camera:{camera_id}] Camera ready")

    # ── Start recording (non-blocking) ────────────────────────────────
    def _start_recording():
        with writer_lock:
            if camera_id in camera_writers:
                return
        try:
            h, w    = frame.shape[:2]
            ist_now = get_ist_time()
            ds      = ist_now.strftime("%Y-%m-%d")
            ts      = ist_now.strftime("%H%M%S")
            dp      = f"{LOCAL_RECORDINGS_DIR}/{ds}/{camera_id}"
            os.makedirs(dp, exist_ok=True)
            lp      = f"{dp}/{camera_id}_{ds}_{ts}.mp4"
            sw      = min(w, 1280) - (min(w, 1280) % 2)
            sh      = int(h * sw / w) - (int(h * sw / w) % 2)
            cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                   "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2", "-i", "-",
                   "-vf", f"scale={sw}:{sh}", "-vcodec", "libx264",
                   "-pix_fmt", "yuv420p", "-preset", "faster", "-crf", "32",
                   "-tune", "fastdecode", "-movflags", "+faststart", lp]
            proc   = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            db_id  = _db_manager.start_recording(camera_id, lp)
            se     = threading.Event()
            rt     = threading.Thread(target=recording_writer_thread,
                                      args=(camera_id, se), daemon=True, name=f"rec-{camera_id}")
            rt.start()
            with writer_lock:
                camera_writers[camera_id] = {
                    "process": proc, "db_id": db_id, "start_time": ist_now,
                    "file_path": lp, "camera_id": camera_id, "w": w, "h": h,
                }
            recording_threads[camera_id]     = rt
            recording_stop_events[camera_id] = se
            logger.info(f"[Recording:{camera_id}] Started")
        except Exception as e:
            logger.error(f"[Recording:{camera_id}] Failed: {e}")

    threading.Thread(target=_start_recording, daemon=True,
                     name=f"rec-init-{camera_id}").start()

    # ── Pipe between Thread A and Thread B ────────────────────────────
    _pipe_lock     = threading.Lock()
    _pipe_frame    = [None]
    _pipe_dets     = [[]]
    _pipe_submit_t = [0.0]

    # ── Thread A: Detection ───────────────────────────────────────────
    def _detection_thread():
        _set_thread_priority(PRIO_HIGH)
        _detector_ready.wait(timeout=60)
        detector = _detector_ref() if _detector_ref else None
        if detector is None:
            logger.error(f"[Camera:{camera_id}] Detector never loaded")
            return
        logger.info(f"[Camera:{camera_id}] Detector ready — 10 FPS")
        _interval      = 1.0 / 10
        _next          = time.time()
        _last_frame_id = -1
        while True:
            try:
                wait = _next - time.time()
                if wait > 0:
                    time.sleep(wait)
                _next = max(_next + _interval, time.time())
                raw, fid = _camera_manager.get_camera_frame_with_id(camera_id)
                if raw is None:
                    time.sleep(0.05); continue
                if fid == _last_frame_id:
                    continue
                _last_frame_id = fid
                fh, fw = raw.shape[:2]
                if fw > 1280:
                    proc = cv2.resize(raw, (1280, int(fh * 1280 / fw)),
                                      interpolation=cv2.INTER_LINEAR)
                else:
                    proc = raw
                t0   = time.time()
                dets = detector.detect(proc)
                with _pipe_lock:
                    _pipe_frame[0]    = proc
                    _pipe_dets[0]     = dets
                    _pipe_submit_t[0] = t0
            except Exception as e:
                logger.error(f"[Camera:{camera_id}] Detection error: {e}")
                time.sleep(0.1)

    threading.Thread(target=_detection_thread, daemon=True,
                     name=f"det-{camera_id}").start()

    # ── Thread B: Render (this thread) ───────────────────────────────
    _set_thread_priority(PRIO_NORMAL)
    tracker = ObjectTracker(max_age=3, n_init=2, iou_threshold=0.25)
    frame_count            = 0
    RENDER_INTERVAL        = 1.0 / 10
    RECOG_CACHE_FRAMES     = 30
    FACE_DETECT_EVERY      = 10
    recognition_cache:     Dict[Any, tuple]      = {}
    face_encoding_cache:   Dict[int, np.ndarray] = {}
    track_merge_map:       Dict[int, int]        = {}
    track_face_crops:      Dict[int, tuple]      = {}
    id_snap_cooldowns:     Dict[tuple, float]    = {}
    cur_track_ids:         set                   = set()
    next_render            = time.time()

    def _person_color(pid):
        hue = (pid * 137) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])

    while True:
        wait = next_render - time.time()
        if wait > 0:
            time.sleep(wait)
        next_render = max(next_render + RENDER_INTERVAL, time.time())

        with _pipe_lock:
            proc_frame = _pipe_frame[0]
            dets       = list(_pipe_dets[0])
            submit_t   = _pipe_submit_t[0]
        if proc_frame is None:
            continue

        frame_count += 1
        try:
            h, w   = proc_frame.shape[:2]
            tracks = tracker.update(dets, proc_frame)

            # Latency compensation
            lag = min(time.time() - submit_t if submit_t > 0 else 0.0, RENDER_INTERVAL)
            if lag > 0.02:
                comp = []
                for t in tracks:
                    tr = next((x for x in tracker.tracks if x["id"] == t["id"]), None)
                    if tr:
                        vx, vy = tr.get("vx", 0.0), tr.get("vy", 0.0)
                        fe = min(lag / RENDER_INTERVAL, 1.0)
                        bw = t["bbox"][2] - t["bbox"][0]
                        bh = t["bbox"][3] - t["bbox"][1]
                        sx = max(-bw*.2, min(bw*.2, vx*fe))
                        sy = max(-bh*.2, min(bh*.2, vy*fe))
                        b  = t["bbox"]
                        comp.append({"id": t["id"],
                                     "bbox": [b[0]+sx, b[1]+sy, b[2]+sx, b[3]+sy]})
                    else:
                        comp.append(t)
                tracks = comp

            # NMS
            tracks = sorted(tracks, key=lambda x: x["id"])
            final_tracks = []
            for t1 in tracks:
                keep = True
                for t2 in final_tracks:
                    b1, b2 = t1["bbox"], t2["bbox"]
                    ix = max(0, min(b1[2],b2[2]) - max(b1[0],b2[0]))
                    iy = max(0, min(b1[3],b2[3]) - max(b1[1],b2[1]))
                    inter = ix * iy
                    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
                    if union > 0 and inter/union > 0.7:
                        keep = False; break
                if keep:
                    final_tracks.append(t1)
            tracks = final_tracks

            new_ids = set(t["id"] for t in tracks)
            if new_ids != cur_track_ids:
                logger.info(f"[Camera:{camera_id}] Persons: {len(tracks)}")
            cur_track_ids = new_ids

            # Build processed list from cache
            processed = []
            for t in tracks:
                tid = t["id"]
                name, conf = "Unknown", 0.0
                if tid in recognition_cache:
                    cn, cc, cf = recognition_cache[tid]
                    if (frame_count - cf) < RECOG_CACHE_FRAMES:
                        name, conf = cn, cc
                processed.append({"id": tid, "bbox": t["bbox"],
                                   "name": name, "confidence": conf})

            # Submit recognition (non-blocking)
            recognizer = _recognizer_ref() if _recognizer_ref else None
            if _recognizer_ready.is_set() and recognizer is not None:
                for t in processed:
                    tid = t["id"]
                    if tid in recognition_cache and \
                       (frame_count - recognition_cache[tid][2]) < (RECOG_CACHE_FRAMES // 2):
                        continue
                    now_t = time.time()
                    with cooldown_lock:
                        last_t   = recognition_cooldowns.get((camera_id, tid), 0)
                        cooldown = 20.0 if t["name"] != "Unknown" else 5.0
                        if now_t - last_t < cooldown:
                            continue
                        recognition_cooldowns[(camera_id, tid)] = now_t
                    bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                    bw, bh = bx2-bx1, by2-by1
                    fb = [bx1+int(.15*bw), by1, bx2-int(.15*bw), by1+int(.45*bh)]
                    try:
                        recognition_executor.submit(
                            self_recognition_worker, proc_frame, fb, tid,
                            recognition_cache, frame_count,
                            face_encoding_cache, track_merge_map, camera_id)
                    except RuntimeError:
                        break

            # Render overlay
            record_frame    = proc_frame.copy()
            final_processed = []
            run_face_detect = (frame_count % FACE_DETECT_EVERY == 0)

            for t in processed:
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                name, conf, tid = str(t["name"]), float(t["confidence"]), int(t["id"])
                if name != "Unknown":
                    body_color = (0, 255, 0); label = name
                else:
                    base_tid = tid
                    while base_tid in track_merge_map:
                        base_tid = track_merge_map[base_tid]
                    body_color = _person_color(base_tid); label = f"#{base_tid}"

                face_visible, face_box_coords = False, None
                if run_face_detect and recognizer is not None:
                    bw_t, bh_t = bx2-bx1, by2-by1
                    head_crop  = proc_frame[max(0,by1):by1+int(bh_t*.35), max(0,bx1):bx2]
                    if head_crop.size > 0:
                        try:
                            hr = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                            md = min(hr.shape[:2])
                            if md < 80:
                                sc = 80.0/md
                                hr = cv2.resize(hr, (max(80,int(hr.shape[1]*sc)),
                                                     max(80,int(hr.shape[0]*sc))),
                                                interpolation=cv2.INTER_LINEAR)
                            with recognizer.ai_lock:
                                boxes_f, probs_f = recognizer.mtcnn.detect(hr)
                            if boxes_f is not None and len(boxes_f) > 0:
                                bi = int(np.argmax([p or 0 for p in probs_f]))
                                bp = probs_f[bi] or 0
                                if bp > 0.80:
                                    fb2 = boxes_f[bi]
                                    cand = (max(0,bx1+int(fb2[0])), max(0,by1+int(fb2[1])),
                                            min(w-1,bx1+int(fb2[2])), min(h-1,by1+int(fb2[3])))
                                    dup = any(
                                        (lambda p: max(0,min(p[2],cand[2])-max(p[0],cand[0])) *
                                                   max(0,min(p[3],cand[3])-max(p[1],cand[1])) /
                                                   max(1,(p[2]-p[0])*(p[3]-p[1])+(cand[2]-cand[0])*(cand[3]-cand[1])-
                                                       max(0,min(p[2],cand[2])-max(p[0],cand[0]))*
                                                       max(0,min(p[3],cand[3])-max(p[1],cand[1]))) > 0.4)(prev)
                                        for prev in [fp.get("face_box_coords")
                                                     for fp in final_processed
                                                     if fp.get("face_box_coords")]
                                    )
                                    if not dup:
                                        face_visible = True; face_box_coords = cand
                                        fx1c,fy1c,fx2c,fy2c = cand
                                        if (fx2c-fx1c) >= 40 and (fy2c-fy1c) >= 40 and bp > 0.92:
                                            fc_img = proc_frame[fy1c:fy2c, fx1c:fx2c]
                                            if fc_img.size > 0:
                                                _, fc_buf = cv2.imencode(".jpg",
                                                    cv2.resize(fc_img,(120,120)),
                                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                                                ex = track_face_crops.get(tid)
                                                if ex is None or bp > ex[1]:
                                                    track_face_crops[tid] = (fc_buf.tobytes(), float(bp))
                        except Exception:
                            pass

                cv2.rectangle(record_frame, (bx1,by1), (bx2,by2), body_color, 2)
                if face_visible and face_box_coords:
                    cv2.rectangle(record_frame,
                                  (face_box_coords[0],face_box_coords[1]),
                                  (face_box_coords[2],face_box_coords[3]), (255,255,0), 1)
                cv2.putText(record_frame, label + (" [F]" if face_visible else " [B]"),
                            (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_color, 2)

                cropped_face = None
                if face_visible and face_box_coords:
                    try:
                        fx1c,fy1c,fx2c,fy2c = face_box_coords
                        fi = proc_frame[fy1c:fy2c, fx1c:fx2c]
                        if fi.size > 0:
                            _, buf = cv2.imencode(".jpg", cv2.resize(fi,(100,120)),
                                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
                            cropped_face = base64.b64encode(buf).decode("utf-8")
                    except Exception:
                        pass

                final_processed.append({
                    "id": tid, "bbox": [bx1,by1,bx2,by2],
                    "name": name, "confidence": conf,
                    "face_crop": cropped_face,
                    "face_visible": face_visible,
                    "face_box_coords": face_box_coords,
                })

            processed    = final_processed
            people_count = len(processed)

            # Prune caches every 30 frames
            if frame_count % 30 == 0:
                for d in (face_encoding_cache, track_merge_map, recognition_cache, track_face_crops):
                    _prune_dict(d, MAX_CACHE_SIZE)
                _prune_dict(id_snap_cooldowns, MAX_CACHE_SIZE * 2)
                with cooldown_lock:
                    _prune_dict(recognition_cooldowns, MAX_CACHE_SIZE * 4)
                with reid_lock:
                    _prune_dict(global_reid_assignments, MAX_CACHE_SIZE * 4)
                occupancy_last_track_ids[camera_id] = set(t["id"] for t in processed)

            # Encode + publish
            _, _enc   = cv2.imencode(".jpg", record_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            enc_bytes = _enc.tobytes()
            slim_tracks = [{"id": t["id"], "bbox": t["bbox"],
                            "name": t["name"], "confidence": t["confidence"]}
                           for t in processed]
            with results_lock:
                camera_results[camera_id] = {
                    "rendered_frame": record_frame,
                    "encoded_frame":  enc_bytes,
                    "frame_id":       frame_count,
                    "tracks":         slim_tracks,
                    "count":          people_count,
                    "alert_active":   False,
                    "timestamp":      time.time(),
                }

            # SSE broadcast on count change
            prev = getattr(process_camera, f"_prev_{camera_id}", -1)
            if people_count != prev:
                setattr(process_camera, f"_prev_{camera_id}", people_count)
                with results_lock:
                    total = sum(d.get("count", 0) for d in camera_results.values())
                _notification_manager.broadcast({
                    "type": "count_update", "camera_id": camera_id,
                    "count": people_count, "total": total,
                })

            # Occupancy + snapshot
            try:
                cur_ids  = set(t["id"] for t in processed)
                last_ids = occupancy_last_track_ids.get(camera_id, set())
                if cur_ids != last_ids:
                    occupancy_last_track_ids[camera_id] = cur_ids
                    occupancy_last_count[camera_id]     = len(cur_ids)
                    recognition_executor.submit(_db_manager.log_occupancy, camera_id, len(cur_ids))
                    if cur_ids:
                        snap_now = time.time()
                        if snap_now - snapshot_cooldowns.get(camera_id, 0) >= SNAPSHOT_COOLDOWN_SECONDS:
                            snapshot_cooldowns[camera_id] = snap_now
                            now_ist  = get_ist_time()
                            ds       = now_ist.strftime("%Y-%m-%d")
                            ts_str   = now_ist.strftime("%H%M%S")
                            dir_path = f"{SNAPSHOTS_DIR}/{ds}/{camera_id}/logs"
                            os.makedirs(dir_path, exist_ok=True)
                            snap_path = f"{dir_path}/{camera_id}_{ds}_{ts_str}.jpg"
                            snap_proc = [{"id": t["id"], "bbox": t["bbox"], "name": t["name"],
                                          "face_visible": t.get("face_visible", False),
                                          "face_box": list(t["face_box_coords"])
                                                      if t.get("face_box_coords") else None}
                                         for t in processed]
                            cur_encs  = [face_encoding_cache[t["id"]]
                                         for t in processed if t["id"] in face_encoding_cache]
                            sw = min(record_frame.shape[1], 960)
                            sh = int(record_frame.shape[0] * sw / record_frame.shape[1])
                            _, sbuf = cv2.imencode(".jpg", cv2.resize(record_frame,(sw,sh)),
                                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
                            img_bytes = sbuf.tobytes()

                            def _on_snap(ok, _c=camera_id, _n=len(cur_ids), _p=snap_path,
                                         _b=snap_proc, _e=cur_encs, _t=now_ist):
                                if ok:
                                    _db_manager.log_detection_snapshot(
                                        _c, _n, _p, _b, face_encodings=_e, timestamp=_t)

                            stream_bytes_to_local(img_bytes, snap_path, callback=_on_snap)
            except Exception as e:
                logger.error(f"[Camera:{camera_id}] Snapshot error: {e}")

            record_frame = None  # free RAM

            # Identity snapshots for recognised persons
            with recognized_lock:
                rec_dict = {}
                for t in processed:
                    if t["name"] != "Unknown" and float(t["confidence"]) > 0.40:
                        rec_dict[t["id"]] = t["name"]
                        sk = (camera_id, t["name"])
                        if time.time() - id_snap_cooldowns.get(sk, 0) < 60.0:
                            continue
                        id_snap_cooldowns[sk] = time.time()
                        try:
                            bx1,by1,bx2,by2 = [int(v) for v in t["bbox"]]
                            tid_r = int(t["id"])
                            face_only = None
                            if tid_r in track_face_crops:
                                fc_b, _ = track_face_crops[tid_r]
                                arr = np.frombuffer(fc_b, dtype=np.uint8)
                                face_only = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            body_crop = proc_frame[max(0,by1):by2, max(0,bx1):bx2]
                            if body_crop.size > 0:
                                ist2   = get_ist_time()
                                id_dir = (f"{SNAPSHOTS_DIR}/{ist2.strftime('%Y-%m-%d')}"
                                          f"/{camera_id}/identities")
                                os.makedirs(id_dir, exist_ok=True)
                                sp = (f"{id_dir}/id_{t['name']}_"
                                      f"{ist2.strftime('%H%M%S%f')[:12]}.jpg")
                                TH = 300
                                bsc = TH / body_crop.shape[0] if body_crop.shape[0] > 0 else 1
                                br  = cv2.resize(body_crop,
                                                 (max(1,int(body_crop.shape[1]*bsc)), TH))
                                if face_only is not None and face_only.size > 0:
                                    fr2 = cv2.resize(face_only, (TH, TH))
                                    cv2.rectangle(fr2,(0,0),(TH-1,24),(0,0,0),-1)
                                    cv2.putText(fr2,"FACE",(8,17),
                                                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,100),1)
                                    comp = np.hstack([fr2, br])
                                else:
                                    cv2.rectangle(br,(0,0),(br.shape[1]-1,24),(0,0,0),-1)
                                    cv2.putText(br,t["name"],(8,17),
                                                cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,100),1)
                                    comp = br
                                _cb = cv2.imencode(".jpg",comp,[cv2.IMWRITE_JPEG_QUALITY,82])[1].tobytes()
                                _sn, _sc, _sp = t["name"], camera_id, sp

                                def _write(_b=_cb, _p=_sp, _n=_sn, _c=_sc):
                                    try:
                                        with open(_p, "wb") as f: f.write(_b)
                                        _db_manager.update_person_last_seen(_n, _c, _p)
                                    except Exception:
                                        pass

                                recognition_executor.submit(_write)
                        except Exception as e:
                            logger.error(f"[Camera:{camera_id}] Identity snap error: {e}")
                camera_recognized_persons[camera_id] = rec_dict

            # Auto-split recording every hour
            with writer_lock:
                wd = camera_writers.get(camera_id)
                if wd and "process" in wd:
                    if (get_ist_time() - wd["start_time"]).total_seconds() > 3600:
                        try:
                            wd["process"].stdin.close()
                            wd["process"].wait(timeout=10)
                            _db_manager.end_recording(wd["db_id"])
                            ni = get_ist_time()
                            ds2 = ni.strftime("%Y-%m-%d"); ts2 = ni.strftime("%H%M%S")
                            dp2 = f"{RECORDINGS_DIR}/{ds2}/{camera_id}"
                            os.makedirs(dp2, exist_ok=True)
                            nlp = f"{dp2}/rec_{camera_id}_{ts2}.mp4"
                            sw2 = min(wd["w"],1280)-(min(wd["w"],1280)%2)
                            sh2 = int(wd["h"]*sw2/wd["w"])-(int(wd["h"]*sw2/wd["w"])%2)
                            cmd2 = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo",
                                    "-s",f"{wd['w']}x{wd['h']}","-pix_fmt","bgr24","-r","2",
                                    "-i","-","-vf",f"scale={sw2}:{sh2}","-vcodec","libx264",
                                    "-pix_fmt","yuv420p","-preset","faster","-crf","32",
                                    "-tune","fastdecode","-movflags","+faststart",nlp]
                            pf  = subprocess.Popen(cmd2, stdin=subprocess.PIPE,
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            nid = _db_manager.start_recording(camera_id, nlp)
                            camera_writers[camera_id] = {
                                "process": pf, "db_id": nid, "start_time": ni,
                                "file_path": nlp, "camera_id": camera_id,
                                "w": wd["w"], "h": wd["h"],
                            }
                        except Exception as e:
                            logger.error(f"[Camera:{camera_id}] Auto-split error: {e}")

        except Exception as e:
            logger.error(f"[Camera:{camera_id}] Render error: {e}")
