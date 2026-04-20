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
from typing import Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from core.state import (
    get_ist_time, camera_results, results_lock, camera_writers, writer_lock,
    occupancy_last_count, occupancy_last_track_ids, recognition_cooldowns,
    cooldown_lock, MAX_CACHE_SIZE, SNAPSHOTS_DIR, SNAPSHOT_COOLDOWN_SECONDS,
    snapshot_cooldowns, LOCAL_RECORDINGS_DIR, RECORDINGS_DIR,
    camera_recognized_persons, recognized_lock, recording_threads,
    recording_stop_events, reid_lock, global_reid_assignments,
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

def init_pipeline(db, cam, det, rec, reid):
    global _db_manager, _camera_manager, _detector, _recognizer, _reid_manager
    _db_manager = db
    _camera_manager = cam
    _detector = det
    _recognizer = rec
    _reid_manager = reid

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

def recording_writer_thread(camera_id: str, stop_event: threading.Event):
    """Writes frames to FFmpeg stdin."""
    FRAME_INTERVAL = 0.5 # 2 FPS
    while not stop_event.is_set():
        try:
            with writer_lock:
                if camera_id not in camera_writers: break
                process = camera_writers[camera_id].get("process")
            with results_lock:
                data = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")
                if frame is not None and "rendered_frame" in data:
                    data["rendered_frame"] = None
            if frame is not None and process and process.poll() is None:
                try:
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                except (IOError, BrokenPipeError): break
            time.sleep(FRAME_INTERVAL)
        except Exception: time.sleep(1)

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
        logger.warning(f"[Pipeline] Camera {camera_id} failed to warmup. Stream may be offline.")
        # Optional: continue to main loop or return? 
        # For now, return to avoid crash on frame.shape
        return

    with writer_lock:
        if camera_id not in camera_writers:
            try:
                h, w = frame.shape[:2]
                ist_now = get_ist_time()
                date_str = ist_now.strftime("%Y-%m-%d")
                timestamp = ist_now.strftime("%H%M%S")
                dir_path = f"{LOCAL_RECORDINGS_DIR}/{date_str}/{camera_id}"
                os.makedirs(dir_path, exist_ok=True)
                local_path = f"{dir_path}/{camera_id}_{date_str}_{timestamp}.mp4"
                scale_w = min(w, 1280) - (min(w, 1280) % 2)
                scale_h = int(h * scale_w / w) - (int(h * scale_w / w) % 2)
                from utils.hw_manager import hw
                encoder = hw.encoder_codec
                
                # Dynamic flags based on encoder type
                v_params = ["-vcodec", encoder]
                if encoder == "h264_qsv": # Intel QuickSync
                    v_params += ["-global_quality", "28", "-look_ahead", "0", "-preset", "faster"] # Less intense QSV settings
                elif encoder == "h264_amf": # AMD AMF
                    v_params += ["-quality", "balanced", "-rc", "cbr"]
                else: # libx264 or others
                    v_params += ["-preset", "faster", "-crf", "32", "-tune", "fastdecode"]

                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2",
                    "-i", "-", "-vf", f"scale={scale_w}:{scale_h}",
                    *v_params, "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", local_path
                ]
                p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                db_id = _db_manager.start_recording(camera_id, local_path)
                stop_event = threading.Event()
                r_thread = threading.Thread(target=recording_writer_thread, args=(camera_id, stop_event), daemon=True)
                r_thread.start()
                camera_writers[camera_id] = {"process": p_ffmpeg, "db_id": db_id, "start_time": ist_now, "file_path": local_path, "camera_id": camera_id, "w": w, "h": h}
                recording_threads[camera_id] = r_thread
                recording_stop_events[camera_id] = stop_event
            except Exception: pass

    _pipe_lock = threading.Lock()
    _pipe_frame = [None]
    _pipe_dets = [[]]
    _pipe_submit_t = [0.0]

    def _detection_thread():
        _det_interval = 1.0 / 6 # Target 6 FPS for detection
        _next_det = time.time()
        while True:
            try:
                now_d = time.time(); wait_d = _next_det - now_d
                if wait_d > 0: time.sleep(wait_d)
                _next_det += _det_interval
                if _next_det < time.time() - (3 * _det_interval): _next_det = time.time() + _det_interval
                raw_frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
                if raw_frame is None: time.sleep(0.05); continue
                fh, fw = raw_frame.shape[:2]
                proc = cv2.resize(raw_frame, (640, int(fh * 640 / fw))) if fw > 640 else raw_frame.copy()
                submit_t = time.time(); dets = _detector.detect(proc)
                with _pipe_lock: _pipe_frame[0] = proc; _pipe_dets[0] = dets; _pipe_submit_t[0] = submit_t
            except Exception: time.sleep(0.1)

    threading.Thread(target=_detection_thread, daemon=True).start()

    tracker = ObjectTracker(max_age=20, n_init=2, iou_threshold=0.2)
    frame_count = 0; RENDER_INTERVAL = 1.0 / 6; RECOGNITION_CACHE_FRAMES = 18 # Synchronized 6 FPS
    face_encoding_cache: Dict[int, np.ndarray] = {}; track_merge_map: Dict[int, int] = {}
    track_face_crops: Dict[int, tuple] = {}; identity_snap_cooldowns: Dict[tuple, float] = {}
    recognition_cache: Dict[Any, tuple] = {}
    next_render_time = time.time()

    def get_color(pid): return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(pid * 137) % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])

    while True:
        wait = next_render_time - time.time()
        if wait > 0: time.sleep(wait)
        next_render_time += RENDER_INTERVAL
        if next_render_time < time.time() - (3 * RENDER_INTERVAL): next_render_time = time.time() + RENDER_INTERVAL
        with _pipe_lock: proc_frame = _pipe_frame[0]; dets = list(_pipe_dets[0]); submit_t = _pipe_submit_t[0]
        if proc_frame is None: continue
        
        # High-HQ Rendering: Map detections from 640px back to Original resolution
        raw_frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
        if raw_frame is None: raw_frame = proc_frame 
        rh, rw = raw_frame.shape[:2]
        sw, sh = rw/640, rh/(int(rh*640/rw))
        
        frame_count += 1
        try:
            h, w = proc_frame.shape[:2]; tracks = tracker.update(dets, proc_frame)
            
            # Remove the redundant overlap filter loop — trust the detector and tracker
            tracks = sorted(tracks, key=lambda x: x["id"])
            
            processed = []
            for t in tracks:
                name, conf = "Unknown", 0.0
                if t['id'] in recognition_cache:
                    cn, cc, cf = recognition_cache[t['id']]
                    if (frame_count - cf) < RECOGNITION_CACHE_FRAMES: name, conf = cn, cc
                processed.append({"id": t['id'], "bbox": t['bbox'], "name": name, "confidence": conf})

            for t in processed:
                if t['id'] in recognition_cache and (frame_count - recognition_cache[t['id']][2]) < (RECOGNITION_CACHE_FRAMES // 2): continue
                with cooldown_lock:
                    last_t = recognition_cooldowns.get((camera_id, t['id']), 0)
                    if time.time() - last_t < (15.0 if t["name"] != "Unknown" else 3.0): continue
                    recognition_cooldowns[(camera_id, t['id'])] = time.time()
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]; bw, bh = bx2-bx1, by2-by1
                face_box = [bx1+int(0.15*bw), by1, bx2-int(0.15*bw), by1+int(0.45*bh)]
                try: recognition_executor.submit(self_recognition_worker, proc_frame.copy(), face_box, t['id'], recognition_cache, frame_count, face_encoding_cache, track_merge_map, camera_id)
                except RuntimeError: break

            # EXCLUSIVE: Render on the Full HD raw_frame
            record_frame = raw_frame.copy()
            final_processed = []; run_face_detect = (frame_count % 6 == 0)
            
            for t in processed:
                # Scale 640px detection coordinates back to raw_frame resolution
                sw, sh = rw/640.0, rh / (rh * 640.0 / rw) if rw != 0 else 1.0
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                rbx1, rby1 = int(bx1 * sw), int(by1 * sh)
                rbx2, rby2 = int(bx2 * sw), int(by2 * sh)
                tid = t['id']; name = t['name']
                
                if name != "Unknown": color = (0, 255, 0); label = name
                else:
                    btid = tid
                    while btid in track_merge_map: btid = track_merge_map[btid]
                    color = get_color(btid); label = f"#{btid}"
                
                fv = False; fbc = None
                # Drawing on High-Res frame with slightly thicker lines
                cv2.rectangle(record_frame, (rbx1, rby1), (rbx2, rby2), color, 3)
                cv2.putText(record_frame, label, (rbx1, rby1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                
                final_processed.append({"id": tid, "bbox": [rbx1, rby1, rbx2, rby2], "name": name, "face_crop": None, "face_visible": fv, "face_box_coords": fbc})

            with results_lock: camera_results[camera_id] = {"rendered_frame": record_frame, "encoded_frame": cv2.imencode('.jpg', record_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), "tracks": final_processed, "count": len(final_processed), "timestamp": time.time()}
            
            c_ids = set(t['id'] for t in final_processed); l_ids = occupancy_last_track_ids.get(camera_id, set())
            if c_ids != l_ids:
                occupancy_last_track_ids[camera_id] = c_ids; occupancy_last_count[camera_id] = len(c_ids)
                _db_manager.log_occupancy(camera_id, len(c_ids))
                if len(c_ids) > 0 and (time.time() - snapshot_cooldowns.get(camera_id, 0)) >= 60:
                    snapshot_cooldowns[camera_id] = time.time(); ist = get_ist_time()
                    spath = f"{SNAPSHOTS_DIR}/{ist.strftime('%Y-%m-%d')}/{camera_id}/logs/{camera_id}_{ist.strftime('%Y-%m-%d_%H%M%S')}.jpg"
                    def _on_s(ok): 
                        if ok: _db_manager.log_detection_snapshot(camera_id, len(c_ids), spath, final_processed, timestamp=ist)
                    stream_bytes_to_local(cv2.imencode('.jpg', record_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), spath, callback=_on_s)

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
                    global_reid_assignments[(camera_id, track_id)] = gid; ist = get_ist_time()
                    _db_manager.log_journey_event(gid, camera_id, None, "unknown" if "U-" in str(gid) else "registered", ist)
                    if "U-" not in str(gid): notification_manager.broadcast({"type": "detection", "camera": camera_id, "target": str(gid), "time": ist.strftime("%I:%M %p"), "is_registered": True})
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
        if not ret:
            break
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
