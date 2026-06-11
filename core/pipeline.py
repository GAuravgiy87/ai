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
    snapshot_cooldowns, LOCAL_RECORDINGS_DIR, RECORDINGS_DIR,
    camera_recognized_persons, recognized_lock,
    reid_lock, global_reid_assignments,
    active_search, active_search_lock
)
from utils.tracker import ObjectTracker
from background_jobs.recording_worker import start_recorder, stop_recorder

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
    if det is not None:
        _detection_pool = DetectionWorkerPool(num_workers=1)
    # Start resource guard
    from core.resource_guard import start as _rg_start
    _rg_start()

class NotificationManager:
    """Manages real-time event broadcasting via SSE."""
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def subscribe(self):
        q = asyncio.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, data: dict):
        msg = f"data: {json.dumps(data)}\n\n"
        with self.lock:
            loop = self._loop
            clients = list(self.clients)
        if loop is None or not loop.is_running():
            return
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

notification_manager = NotificationManager()

# Resource management
recognition_executor = ThreadPoolExecutor(max_workers=1)
transfer_queue = queue.Queue(maxsize=50)

# Shared Detection Worker Pool (replaces per-camera detection threads)
@dataclass
class DetectionTask:
    """Frame submitted to detection worker pool."""
    camera_id: str
    frame: np.ndarray
    submit_time: float

@dataclass
class DetectionResult:
    """Detection result from worker pool."""
    camera_id: str
    processed_frame: np.ndarray
    detections: list
    submit_time: float

class DetectionWorkerPool:
    """
    One detection worker per pool — the detector has a global lock anyway
    so multiple workers just block each other and waste threads.
    Results are consumed exactly once: the render loop clears the result
    after reading it so stale detections are never re-processed.
    """

    def __init__(self, num_workers: int = 1, queue_size: int = 4):
        # queue_size=4: only keep the 4 most recent frames.
        # Old frames are dropped (try_nowait) so we never process stale data.
        self.frame_queue  = queue.Queue(maxsize=queue_size)
        self.results:      Dict[str, DetectionResult] = {}
        self.results_lock  = threading.Lock()
        self.running       = True

        for i in range(max(1, num_workers)):
            w = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            w.start()
        logger.info(f"[DetectionPool] Started {max(1,num_workers)} detection worker(s)")

    def _worker_loop(self, worker_id: int):
        # Enable OpenCL (AMD GPU via OpenCL) for OpenCV operations.
        # This offloads resize, color conversion, and CLAHE to the GPU,
        # reducing CPU load significantly.
        try:
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.haveOpenCL():
                cv2.ocl.useOpenCL()
                logger.info(f"[DetectionWorker:{worker_id}] OpenCL enabled — "
                            f"preprocessing on GPU")
        except Exception:
            pass

        while self.running:
            try:
                task = self.frame_queue.get(timeout=0.5)
                if task is None:
                    continue

                fh, fw = task.frame.shape[:2]

                # ── GPU-accelerated resize via OpenCL UMat ────────────────
                # Upload frame to GPU memory once, resize on GPU, download
                # only the small 640-wide result back to CPU for ONNX.
                try:
                    if cv2.ocl.haveOpenCL() and fw > 640:
                        u_frame = cv2.UMat(task.frame)
                        u_proc  = cv2.resize(u_frame,
                                             (640, int(fh * 640 / fw)),
                                             interpolation=cv2.INTER_LINEAR)
                        proc = u_proc.get()   # download result
                    else:
                        proc = cv2.resize(task.frame, (640, int(fh * 640 / fw))) \
                               if fw > 640 else task.frame.copy()
                except Exception:
                    proc = cv2.resize(task.frame, (640, int(fh * 640 / fw))) \
                           if fw > 640 else task.frame.copy()

                dets   = _detector.detect(proc) if _detector else []
                result = DetectionResult(
                    camera_id=task.camera_id,
                    processed_frame=proc,
                    detections=dets,
                    submit_time=time.time(),
                )
                with self.results_lock:
                    self.results[task.camera_id] = result
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[DetectionWorker:{worker_id}] {e}")

    def submit_frame(self, camera_id: str, frame: np.ndarray) -> bool:
        """Drop oldest frame if queue full — always keep freshest."""
        task = DetectionTask(camera_id=camera_id, frame=frame, submit_time=time.time())
        try:
            self.frame_queue.put_nowait(task)
            return True
        except queue.Full:
            # Drain one stale frame and try again
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(task)
                return True
            except queue.Full:
                return False

    def get_result(self, camera_id: str) -> Optional[DetectionResult]:
        """Return AND CLEAR the latest result — never reuse stale detections."""
        with self.results_lock:
            return self.results.pop(camera_id, None)

# Global detection pool (initialized in init_pipeline)
_detection_pool: Optional[DetectionWorkerPool] = None

def _prune_dict(d: dict, max_size: int):
    if len(d) > max_size:
        keys = list(d.keys())
        for k in keys[:len(keys)//2]:
            d.pop(k, None)

def transfer_worker():
    """Background worker for sequential file tasks."""
    while True:
        try:
            item = transfer_queue.get()
            if item is None: break
            data, destination, callback = item
            if isinstance(data, (bytes, bytearray)):
                success = _perform_direct_stream(data, destination)
            else:
                success = _perform_actual_process(data, destination)
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

def _perform_actual_process(src_path: str, dest_dir: str) -> bool:
    try:
        import shutil
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_path, dest_dir)
        return True
    except Exception: return False

threading.Thread(target=transfer_worker, daemon=True).start()

def stream_bytes_to_local(data: bytes, local_path: str, callback=None) -> bool:
    try:
        transfer_queue.put((data, local_path, callback), block=False)
        return True
    except queue.Full: return False

def process_camera(camera_id: str):
    frame = None
    warmup_frames = 0
    max_warmup_attempts = 50
    attempts = 0
    while warmup_frames < 5 and attempts < max_warmup_attempts:
        frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None: warmup_frames += 1
        else: attempts += 1
        time.sleep(0.1)

    if frame is None:
        logger.warning(f"[Pipeline] Camera {camera_id} failed to warmup. Stream may be offline.")
        return

    # Start the new crash-safe MKV recorder
    start_recorder(camera_id, frame, _db_manager)

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

    _color_cache = {}
    def get_color(pid):
        if pid not in _color_cache:
            _color_cache[pid] = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[(pid * 137) % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        return _color_cache[pid]

    while True:
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
            if raw_frame_submit is not None and _detection_pool is not None:
                _detection_pool.submit_frame(camera_id, raw_frame_submit)
            _next_submit_time = now + _SUBMIT_INTERVAL

        # Get result from shared detection pool — consume-once (pop, not get)
        if _detection_pool is None:
            time.sleep(0.05)
            continue
        result = _detection_pool.get_result(camera_id)
        if result is None:
            # No new detection yet — skip this render cycle entirely.
            # Do NOT re-run the tracker with stale data (causes hesitation).
            time.sleep(0.02)
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

            # ── Step 3: occlusion-aware drawing with numpy mask ───────────
            #
            # For each track (back→front order), build a boolean mask of all
            # FRONT person bboxes, then draw the box outline only on pixels
            # NOT covered by any front person.  This is O(N²) in bbox area
            # but uses numpy vectorised ops — orders of magnitude faster than
            # the previous pixel-by-pixel Python loop.

            # Pre-build a "front mask" for each track index using numpy
            # We accumulate front-person regions into a single mask per track.

            n = len(render_items)
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

                if not front_rects:
                    # Fast path — no occlusion
                    cv2.rectangle(record_frame, (rbx1, rby1), (rbx2, rby2), color, thick)
                else:
                    # Build a small boolean mask covering just this box's bounding rect
                    # mask[y, x] = True means this pixel is blocked by a front person
                    bw = rbx2 - rbx1 + 1
                    bh_box = rby2 - rby1 + 1
                    mask = np.zeros((bh_box, bw), dtype=bool)

                    for (fx1, fy1, fx2, fy2) in front_rects:
                        # Clip front rect to this box's coordinate space
                        lx1 = max(0,    fx1 - rbx1)
                        lx2 = min(bw-1, fx2 - rbx1)
                        ly1 = max(0,    fy1 - rby1)
                        ly2 = min(bh_box-1, fy2 - rby1)
                        if lx2 >= lx1 and ly2 >= ly1:
                            mask[ly1:ly2+1, lx1:lx2+1] = True

                    # Draw each of the 4 sides using the mask
                    # Top edge: y=0 in local coords
                    def _draw_masked_hline(y_local, x_start, x_end):
                        y_abs = rby1 + y_local
                        if y_abs < 0 or y_abs >= rh:
                            return
                        row_mask = mask[y_local, x_start:x_end+1]
                        xs = np.where(~row_mask)[0] + rbx1 + x_start
                        if len(xs) == 0:
                            return
                        # Draw contiguous segments
                        gaps = np.where(np.diff(xs) > 1)[0]
                        segs = np.split(xs, gaps+1)
                        for seg in segs:
                            if len(seg) > 0:
                                cv2.line(record_frame,
                                         (int(seg[0]), y_abs),
                                         (int(seg[-1]), y_abs),
                                         color, thick)

                    def _draw_masked_vline(x_local, y_start, y_end):
                        x_abs = rbx1 + x_local
                        if x_abs < 0 or x_abs >= rw:
                            return
                        col_mask = mask[y_start:y_end+1, x_local]
                        ys = np.where(~col_mask)[0] + rby1 + y_start
                        if len(ys) == 0:
                            return
                        gaps = np.where(np.diff(ys) > 1)[0]
                        segs = np.split(ys, gaps+1)
                        for seg in segs:
                            if len(seg) > 0:
                                cv2.line(record_frame,
                                         (x_abs, int(seg[0])),
                                         (x_abs, int(seg[-1])),
                                         color, thick)

                    _draw_masked_hline(0,       0, bw-1)          # top
                    _draw_masked_hline(bh_box-1, 0, bw-1)         # bottom
                    _draw_masked_vline(0,       0, bh_box-1)       # left
                    _draw_masked_vline(bw-1,    0, bh_box-1)       # right

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

def self_recognition_worker(frame, face_box, track_id, recognition_cache, frame_count, face_encoding_cache, track_merge_map, camera_id):
    try:
        name, conf, enc = _recognizer.recognize_with_encoding(frame, face_box)
        if enc is not None:
            face_encoding_cache[track_id] = enc
            for o_id, o_enc in face_encoding_cache.items():
                if o_id != track_id and np.linalg.norm(enc - o_enc) < 0.6:
                    if track_id < o_id: track_merge_map[o_id] = track_id
                    else: track_merge_map[track_id] = o_id
                    break
        if name != "Unknown" and conf >= 0.90: recognition_cache[track_id] = (name, conf, frame_count)
        gid = None
        if name != "Unknown" and conf >= 0.90: gid = name
        elif enc is not None:
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

def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 10) -> list:
    if not _recognizer:
        logger.warning("[Pipeline] Video scan requested but Recognizer is not initialized.")
        return []
    res = []; cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return res
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    f_cnt = 0
    c_seg = None
    l_m_f = -1
    g_gap = int(fps * 2)
    while True:
        ret, frame = cap.read()
        if not ret: break
        if f_cnt % sample_interval == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with _recognizer.ai_lock: bxs, prbs = _recognizer.mtcnn.detect(rgb)
                m_f = False; b_c = 0.0
                if bxs is not None:
                    for box in bxs:
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        if (fx2-fx1)<30 or (fy2-fy1)<30: continue
                        f_r = cv2.resize(rgb[max(0,fy1):fy2, max(0,fx1):fx2], (160, 160))
                        f_t = (torch.tensor(np.transpose(f_r, (2, 0, 1))).float().unsqueeze(0).to(_recognizer._face_device)-127.5)/128.0
                        with _recognizer.ai_lock, torch.no_grad():
                            e = _recognizer.resnet(f_t).cpu().numpy()[0]
                        d = float(np.linalg.norm(target_encoding - e))
                        if d < 1.15: m_f = True; b_c = max(b_c, 1 - (d/2.0))
                if m_f:
                    sec = f_cnt/fps; tstr = f"{int(sec//60)}:{int(sec%60):02d}"
                    if c_seg is None or (f_cnt - l_m_f) > g_gap:
                        if c_seg: res.append(c_seg)
                        c_seg = {"start_seconds": sec, "start_timestamp": tstr, "end_seconds": sec, "end_timestamp": tstr, "confidence": b_c, "start_frame": f_cnt, "end_frame": f_cnt}
                    else:
                        c_seg["end_seconds"] = sec; c_seg["end_timestamp"] = tstr; c_seg["end_frame"] = f_cnt; c_seg["confidence"] = max(c_seg["confidence"], b_c)
                    l_m_f = f_cnt
            except: pass
        f_cnt += 1
    if c_seg: res.append(c_seg)
    cap.release(); return res
