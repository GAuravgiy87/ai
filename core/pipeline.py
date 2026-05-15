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

def init_pipeline(db, cam, det, rec, reid, num_detection_workers: int = 1):
    global _db_manager, _camera_manager, _detector, _recognizer, _reid_manager, _detection_pool
    _db_manager = db
    _camera_manager = cam
    _detector = det
    _recognizer = rec
    _reid_manager = reid
    # BUG FIX #2a: Always initialize detection pool regardless of det being None
    # This ensures recording can work even when detection models aren't loaded
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
        """Register a new SSE client queue. Thread-safe via GIL (list.append is atomic)."""
        q = asyncio.Queue()
        self.clients.append(q)  # BUG-15 fix: no threading.Lock in async context
        return q

    def unsubscribe(self, q):
        """Remove a client queue. Safe without lock — list.remove is GIL-protected."""
        try:
            self.clients.remove(q)
        except ValueError:
            pass

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

def recording_writer_thread(camera_id: str, stop_event: threading.Event):
    """Writes frames to FFmpeg stdin at a fixed 15fps. BUG FIX #2b: Pull frames directly from camera."""
    logger.info(f"[Recording] Writer thread started for {camera_id}")
    frame_count = 0
    last_frame = None
    last_frame_time = time.time()
    
    while not stop_event.is_set():
        try:
            with writer_lock:
                if camera_id not in camera_writers: 
                    logger.info(f"[Recording] Camera {camera_id} not in writers, stopping thread")
                    break
                writer_data = camera_writers[camera_id]
                process = writer_data.get("process")
            
            # BUG FIX #2b: Get frame directly from camera manager instead of camera_results
            # This decouples recording from detection - recording works even if detection is disabled
            frame, _ = _camera_manager.get_camera_frame_with_id(camera_id)
            
            # Use last frame if current is None (prevents gaps in recording)
            if frame is None and last_frame is not None:
                # BUG FIX #2b: Close recording if no frames for >10 seconds
                if time.time() - last_frame_time > 10:
                    logger.warning(f"[Recording] No frames for 10s on {camera_id}, closing recording")
                    break
                frame = last_frame
            elif frame is not None:
                last_frame_time = time.time()
            
            if frame is not None and process and process.poll() is None:
                try:
                    # Ensure frame dimensions match what FFmpeg expects
                    expected_h, expected_w = writer_data.get("h"), writer_data.get("w")
                    actual_h, actual_w = frame.shape[:2]
                    
                    if actual_h != expected_h or actual_w != expected_w:
                        frame = cv2.resize(frame, (expected_w, expected_h))
                    
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                    last_frame = frame
                    frame_count += 1
                    
                    if frame_count % 150 == 0:  # Log every 10 seconds
                        logger.info(f"[Recording] {camera_id}: {frame_count} frames written")
                        
                except (IOError, BrokenPipeError) as e:
                    logger.error(f"[Recording] Pipe error for {camera_id}: {e}")
                    break
                except Exception as e:
                    logger.error(f"[Recording] Write error for {camera_id}: {e}")
                    break
            elif process and process.poll() is not None:
                logger.warning(f"[Recording] FFmpeg process died for {camera_id}")
                break
                
            time.sleep(0.066)  # 15fps (1/15 ≈ 0.066)
        except Exception as e:
            logger.error(f"[Recording] Thread error for {camera_id}: {e}")
            time.sleep(1)
    
    logger.info(f"[Recording] Writer thread stopped for {camera_id}, wrote {frame_count} frames")

def _close_recording(camera_id):
    """Closes FFmpeg process and updates database. BUG FIX #5: Verify file size."""
    logger.info(f"[Recording] Closing recording for {camera_id}")
    
    with writer_lock:
        wd = camera_writers.pop(camera_id, None)
        stop_event = recording_stop_events.pop(camera_id, None)
        thread = recording_threads.pop(camera_id, None)

    if stop_event:
        stop_event.set()
        logger.debug(f"[Recording] Stop event set for {camera_id}")
    
    if wd:
        process = wd.get("process")
        db_id = wd.get("db_id")
        file_path = wd.get("file_path")
        
        if process:
            try:
                # Close stdin to signal FFmpeg to finalize the file
                if process.stdin:
                    process.stdin.close()
                    logger.debug(f"[Recording] Closed stdin for {camera_id}")
                
                # Wait for FFmpeg to finish writing
                process.wait(timeout=5)
                logger.info(f"[Recording] FFmpeg process terminated gracefully for {camera_id}")
            except subprocess.TimeoutExpired:
                logger.warning(f"[Recording] FFmpeg timeout for {camera_id}, killing process")
                if process: 
                    process.kill()
                    process.wait()
            except Exception as e:
                logger.error(f"[Recording] Error closing FFmpeg for {camera_id}: {e}")
                if process: 
                    process.kill()
        
        # BUG FIX #5: Verify file size and clean up if too small
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size < 100 * 1024:  # Less than 100KB
                logger.warning(f"[Recording] File too small ({file_size} bytes), likely corrupt: {file_path}")
                try:
                    os.remove(file_path)
                    logger.info(f"[Recording] Deleted corrupt file: {file_path}")
                except Exception as e:
                    logger.error(f"[Recording] Failed to delete corrupt file: {e}")
                # Don't update database for corrupt files
                if db_id and _db_manager:
                    _db_manager.delete_recording(db_id)
            else:
                logger.info(f"[Recording] File saved: {file_path} ({file_size / (1024*1024):.2f} MB)")
                # Update database only for valid files
                if db_id and _db_manager:
                    _db_manager.end_recording(db_id)
                    logger.info(f"[Recording] Database updated for {camera_id}, ID={db_id}")
        else:
            logger.warning(f"[Recording] File not found: {file_path}")
            # Clean up database entry for missing file
            if db_id and _db_manager:
                _db_manager.delete_recording(db_id)
    
    if thread:
        thread.join(timeout=2)
        logger.debug(f"[Recording] Writer thread joined for {camera_id}")

def cleanup_all_recordings():
    """Closes all active recordings. Called on system shutdown."""
    with writer_lock:
        cids = list(camera_writers.keys())
    
    if not cids:
        return

    logger.info(f"[Cleanup] Closing {len(cids)} active recording(s)...")
    for cid in cids:
        try:
            _close_recording(cid)
        except Exception as e:
            logger.error(f"[Cleanup] Error closing recording for {cid}: {e}")

def _start_hourly_recording(camera_id, frame_shape):
    """Starts a new hourly recording chunk. BUG FIX #5: Ensure parent dir exists before FFmpeg."""
    h, w = frame_shape[:2]
    ist_now = get_ist_time()
    date_str = ist_now.strftime("%Y-%m-%d")
    hour_str = ist_now.strftime("%H")
    
    # BUG FIX #5: Ensure recordings directory exists BEFORE starting FFmpeg
    dir_path = f"{RECORDINGS_DIR}/{date_str}/{camera_id}"
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        logger.error(f"[Recording] Failed to create directory {dir_path}: {e}")
        return
    local_path = f"{dir_path}/{hour_str}.mp4"
    
    logger.info(f"[Recording] Starting recording for {camera_id}: {local_path}")
    logger.info(f"[Recording] Input frame size: {w}x{h}")
    
    # Scale down to max 1280 width while maintaining aspect ratio
    scale_w = min(w, 1280) - (min(w, 1280) % 2)
    scale_h = int(h * scale_w / w) - (int(h * scale_w / w) % 2)
    
    logger.info(f"[Recording] Output video size: {scale_w}x{scale_h}")
    
    from utils.hw_manager import hw
    encoder = hw.encoder_codec
    v_params = ["-profile:v", "high", "-level", "4.1"]
    
    if encoder == "h264_qsv":
        v_params += ["-vcodec", "h264_qsv", "-global_quality", "25", "-preset", "veryfast", "-look_ahead", "0"]
        logger.info(f"[Recording] Using Intel QSV hardware encoder")
    elif encoder == "h264_amf":
        v_params += ["-vcodec", "h264_amf", "-quality", "speed", "-rc", "cbr", "-usage", "transcoding"]
        logger.info(f"[Recording] Using AMD AMF hardware encoder")
    else:
        v_params += ["-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23", "-tune", "zerolatency"]
        logger.info(f"[Recording] Using software encoder (libx264)")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "15",
        "-thread_queue_size", "1024",
        "-i", "-", "-vf", f"scale={scale_w}:{scale_h}",
        *v_params, "-pix_fmt", "yuv420p",
        "-movflags", "+faststart+frag_keyframe+empty_moov+default_base_moof",
        local_path
    ]
    
    try:
        # Start FFmpeg process
        p_ffmpeg = subprocess.Popen(
            ffmpeg_cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer for stdin
        )
        
        # BUG FIX #5: Check if FFmpeg started successfully, retry once if failed
        time.sleep(0.1)  # Give FFmpeg a moment to start
        if p_ffmpeg.poll() is not None:
            logger.error(f"[Recording] FFmpeg failed to start for {camera_id}, retrying once...")
            p_ffmpeg = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            time.sleep(0.1)
            if p_ffmpeg.poll() is not None:
                logger.error(f"[Recording] FFmpeg failed to start after retry for {camera_id}")
                return
        
        # Register in database
        db_id = _db_manager.start_recording(camera_id, local_path)
        logger.info(f"[Recording] Database entry created: ID={db_id}")
        
        # CRITICAL FIX: Create stop event and store writer info BEFORE starting thread
        # This prevents race condition where thread checks camera_writers before it's populated
        stop_event = threading.Event()
        
        # Store writer info FIRST (before starting thread)
        with writer_lock:
            camera_writers[camera_id] = {
                "process": p_ffmpeg, 
                "db_id": db_id, 
                "start_time": ist_now, 
                "file_path": local_path, 
                "camera_id": camera_id, 
                "w": w, "h": h
            }
            recording_stop_events[camera_id] = stop_event
        
        # NOW start writer thread (after camera_writers is populated)
        r_thread = threading.Thread(
            target=recording_writer_thread, 
            args=(camera_id, stop_event), 
            daemon=True,
            name=f"RecWriter-{camera_id}"
        )
        r_thread.start()
        
        # Store thread reference
        with writer_lock:
            recording_threads[camera_id] = r_thread
        
        # Consume stderr in background to prevent FFmpeg from hanging
        def _log_ffmpeg_err(pipe, cid):
            try:
                for line in iter(pipe.readline, b''):
                    msg = line.decode().strip()
                    if msg:  # Log all FFmpeg output for debugging
                        if "Error" in msg or "error" in msg:
                            logger.error(f"[FFmpeg:{cid}] {msg}")
                        else:
                            logger.debug(f"[FFmpeg:{cid}] {msg}")
            except Exception as e:
                logger.error(f"[FFmpeg:{cid}] Error reading stderr: {e}")
            finally: 
                pipe.close()
        
        threading.Thread(
            target=_log_ffmpeg_err, 
            args=(p_ffmpeg.stderr, camera_id), 
            daemon=True,
            name=f"FFmpegLog-{camera_id}"
        ).start()
        
        logger.info(f"[Recording] Successfully started recording for {camera_id}")
        
    except Exception as e:
        logger.error(f"[Pipeline] Failed to start hourly recording for {camera_id}: {e}", exc_info=True)

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

    # ALWAYS enable recording for all cameras (automatic recording)
    _db_manager.set_camera_recording(camera_id, True)
    logger.info(f"[Pipeline:{camera_id}] Automatic recording enabled")

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
            
            # ── Recording Management ──────────────────────────────────────────
            enabled = bool(_db_manager.get_camera_recording_setting(camera_id))
            with writer_lock:
                wd = camera_writers.get(camera_id)
                has_active_writer = wd is not None
            
            # Debug logging every 30 seconds
            if frame_count % 180 == 0:
                logger.info(f"[Pipeline:{camera_id}] Recording status: enabled={enabled}, has_writer={has_active_writer}")

            if raw_frame_submit is not None:
                last_frame_time = now
                if enabled:
                    with writer_lock:
                        writer_missing = wd is None
                        age = (get_ist_time() - wd["start_time"]).total_seconds() if wd else 0
                        process_died = wd["process"].poll() is not None if wd else False

                    if writer_missing:
                        logger.info(f"[Pipeline:{camera_id}] Recording enabled, starting new recording")
                        _start_hourly_recording(camera_id, raw_frame_submit.shape)
                    elif age >= 3600:
                        logger.info(f"[Pipeline:{camera_id}] Hourly rotation (age={age:.0f}s), starting new recording")
                        _close_recording(camera_id)
                        _start_hourly_recording(camera_id, raw_frame_submit.shape)
                    elif process_died:
                        logger.warning(f"[Pipeline:{camera_id}] FFmpeg process died, restarting recording")
                        _close_recording(camera_id)
                        _start_hourly_recording(camera_id, raw_frame_submit.shape)
                elif has_active_writer:
                    # Recording was just disabled, close it
                    logger.info(f"[Pipeline:{camera_id}] Recording disabled via settings. Closing.")
                    _close_recording(camera_id)

                if _detection_pool is not None:
                    _detection_pool.submit_frame(camera_id, raw_frame_submit)
            else:
                # Camera is offline/None
                if has_active_writer:
                    # Close after 10s timeout OR immediately if disabled
                    if not enabled or (now - last_frame_time) > 10:
                        logger.warning(f"[Pipeline:{camera_id}] Camera offline or disabled. Closing recording.")
                        _close_recording(camera_id)
            
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
    
    # Ensure recording is closed when the loop exits (camera removed or stream lost)
    logger.info(f"[Pipeline:{camera_id}] Pipeline loop exited. Cleaning up recording.")
    _close_recording(camera_id)

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

def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 15) -> list:
    """
    Optimized high-speed video search using GPU:
    1. YOLOv8 (GPU) detects persons first (fast skip for empty frames).
    2. Crops persons and uses Batch Face Recognition (GPU).
    3. Results are aggregated into segments.
    """
    if not _recognizer or not _detector:
        logger.warning("[Pipeline] Video scan requested but models are not initialized.")
        return []

    res = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return res

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_cnt = 0
    c_seg = None
    l_m_f = -1
    g_gap = int(fps * 3)  # 3-second gap for segmenting
    
    # Batch settings — increased to 32 for massive GPU speedup in forensic scan
    BATCH_SIZE = 32
    pending_batch_frames = []
    pending_batch_indices = []

    logger.info(f"[Search] Starting GPU-accelerated scan on {os.path.basename(video_path)} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if f_cnt % sample_interval == 0:
            # Step 1: Fast YOLO person detection first (GPU)
            # This is much faster than running face detection on every empty frame
            dets = _detector.detect(frame)
            if dets:
                # We found people! Collect person crops for batch recognition
                person_boxes = [d[0] for d in dets]
                # To keep it simple but fast, we take the most prominent person if multiple
                # or we could batch ALL persons. Let's batch ALL persons in this frame.
                
                # However, to avoid exploding the batch, we limit to 1 per frame for search
                # and add this frame to the batch.
                bx, by, bw, bh = person_boxes[0]
                # Expand box slightly for face detection
                pad_w, pad_h = bw * 0.1, bh * 0.1
                face_box = [bx - pad_w, by - pad_h, bx + bw + pad_w, by + bh * 0.5 + pad_h]
                
                pending_batch_frames.append((frame.copy(), face_box))
                pending_batch_indices.append(f_cnt)

            # Step 2: Process batch if full
            if len(pending_batch_frames) >= BATCH_SIZE:
                # BUG-06 fix: _process_search_batch manages state via res_list directly
                _process_search_batch(pending_batch_frames, pending_batch_indices,
                                      target_encoding, fps, g_gap, res)
                pending_batch_frames.clear()
                pending_batch_indices.clear()

        f_cnt += 1

    # Process final partial batch
    if pending_batch_frames:
        _process_search_batch(pending_batch_frames, pending_batch_indices,
                              target_encoding, fps, g_gap, res)

    cap.release()
    logger.info(f"[Search] Scan complete. Found {len(res)} segments.")
    return res

def _process_search_batch(batch, indices, target_encoding, fps, g_gap, res_list):
    """
    Run TRUE batch recognition across multiple frames and update results list.
    SPEED: Now uses recognize_multi_frame_batch for 4x+ performance gain.
    """
    # Run entire batch in one GPU call
    batch_results = _recognizer.recognize_multi_frame_batch(batch)
    
    # Target encoding should be normalized for comparison
    target_v = target_encoding / np.linalg.norm(target_encoding)

    for i, (name, conf, enc) in enumerate(batch_results):
        f_idx = indices[i]
        if enc is None: continue

        match_found = False
        match_conf = 0.0

        # High-accuracy normalized L2 comparison
        dist = float(np.linalg.norm(target_v - enc))
        # 1.05 is the sweet spot for Forensic search accuracy
        if dist < 1.05:
            match_found = True
            match_conf = max(0.0, 1.0 - (dist / 1.15))

        if match_found:
            sec  = f_idx / fps
            tstr = f"{int(sec//60)}:{int(sec%60):02d}"

            # Extend the last segment if within gap, otherwise start a new one
            if res_list and (f_idx - res_list[-1]["end_frame"]) <= g_gap:
                res_list[-1]["end_seconds"]   = sec
                res_list[-1]["end_timestamp"] = tstr
                res_list[-1]["end_frame"]     = f_idx
                res_list[-1]["confidence"]    = max(res_list[-1]["confidence"], match_conf)
            else:
                res_list.append({
                    "start_seconds": sec, "start_timestamp": tstr,
                    "end_seconds":   sec, "end_timestamp":   tstr,
                    "confidence":    match_conf,
                    "start_frame":   f_idx, "end_frame": f_idx,
                })
