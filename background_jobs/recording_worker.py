"""
background_jobs/recording_worker.py — Crash‑safe Surveillance Recording
Adapted for master branch (no Redis dependency)
Design goals:
  1. Hourly chunks: rotate exactly on clock hour
  2. Crash-safe MKV: index flushed every 2s
  3. Restart-safe: partial files playable
"""
import os
import time
import threading
import subprocess
from datetime import datetime, timedelta
import logging
from core.state import get_ist_time, camera_results, results_lock, LOCAL_RECORDINGS_DIR


logger = logging.getLogger(__name__)

RECORD_FPS = 10
RECORDINGS_DIR = LOCAL_RECORDINGS_DIR if LOCAL_RECORDINGS_DIR else "./recordings"

def _chunk_path(camera_id: str, dt: datetime) -> str:
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = dt.strftime("%H")
    camera_dir = os.path.join(RECORDINGS_DIR, date_str, camera_id)
    os.makedirs(camera_dir, exist_ok=True)
    return os.path.join(camera_dir, f"{hour_str}.mkv")

def _next_hour(dt: datetime) -> datetime:
    return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

class CameraRecorder:
    def __init__(self, camera_id: str, db_manager):
        self.camera_id = camera_id
        self.db_manager = db_manager
        self._process = None
        self._db_id = None
        self._file_path = None
        self._chunk_end = None
        self._w = None
        self._h = None
        self._frame_count = 0
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, initial_frame):
        self._h, self._w = initial_frame.shape[:2]
        if not self._open_chunk():
            return False
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"Recorder-{self.camera_id}"
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _open_chunk(self):
        now = datetime.now()
        self._chunk_end = _next_hour(now)
        self._file_path = _chunk_path(self.camera_id, now)
        self._frame_count = 0

        if os.path.exists(self._file_path):
            crashed_path = self._file_path.replace(".mkv", "_recovered.mkv")
            try:
                os.rename(self._file_path, crashed_path)
                logger.info(f"[Recorder:{self.camera_id}] Renamed crash file → {crashed_path}")
            except Exception as e:
                logger.warning(f"[Recorder:{self.camera_id}] Rename failed: {e}")

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self._w}x{self._h}", "-pix_fmt", "bgr24", "-r", str(RECORD_FPS),
            "-i", "pipe:0",
            "-f", "matroska", "-vcodec", "libx264",
            "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-tune", "zerolatency",
            "-crf", "28", "-force_key_frames", "expr:gte(t,n_forced*2)",
            "-cluster_time_limit", "2000", "-flush_packets", "1",
            "-loglevel", "error",
            self._file_path
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            time.sleep(0.5)
            if self._process.poll() is not None:
                err = self._process.stderr.read(500).decode(errors="replace")
                logger.error(f"[Recorder:{self.camera_id}] FFmpeg died: {err}")
                return False
            self._db_id = self.db_manager.start_recording(self.camera_id, self._file_path)
            logger.info(f"[Recorder:{self.camera_id}] Opened chunk → {self._file_path} (until {self._chunk_end.strftime('%H:%M:%S')})")
            return True
        except Exception as e:
            logger.error(f"[Recorder:{self.camera_id}] FFmpeg start error: {e}")
            return False

    def _close_chunk(self, graceful: bool = True):
        if self._process is None:
            return
        if graceful:
            try:
                self._process.stdin.flush()
                self._process.stdin.close()
                self._process.wait(timeout=15)
                logger.info(f"[Recorder:{self.camera_id}] Chunk closed gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"[Recorder:{self.camera_id}] FFmpeg timeout — killing")
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.warning(f"[Recorder:{self.camera_id}] Graceful close error: {e}")
                try:
                    self._process.kill()
                    self._process.wait()
                except Exception:
                    pass
        else:
            try:
                self._process.kill()
                self._process.wait()
            except Exception:
                pass
        if self._db_id:
            try:
                self.db_manager.end_recording(self._db_id)
            except Exception:
                pass
            self._db_id = None
        self._process = None

    def _writer_loop(self):
        logger.info(f"[Recorder:{self.camera_id}] Started")
        frame_interval = 1.0 / RECORD_FPS
        next_frame_t = time.time()
        while not self._stop_event.is_set():
            try:
                now = datetime.now()
                if now >= self._chunk_end:
                    logger.info(f"[Recorder:{self.camera_id}] Rotating chunk")
                    self._close_chunk(graceful=True)
                    if not self._open_chunk():
                        logger.error(f"[Recorder:{self.camera_id}] Failed to open new chunk")
                        time.sleep(5)
                        continue
                if self._process and self._process.poll() is not None:
                    logger.warning(f"[Recorder:{self.camera_id}] FFmpeg died — reopening")
                    self._close_chunk(graceful=False)
                    time.sleep(1)
                    if not self._open_chunk():
                        time.sleep(5)
                    continue
                wait = next_frame_t - time.time()
                if wait > 0:
                    time.sleep(wait)
                next_frame_t += frame_interval
                if next_frame_t < time.time() - frame_interval*3:
                    next_frame_t = time.time() + frame_interval

                frame = None
                with results_lock:
                    data = camera_results.get(self.camera_id, {})
                    frame = data.get("rendered_frame")
                if frame is not None:
                    fh, fw = frame.shape[:2]
                    if fw != self._w or fh != self._h:
                        import cv2
                        frame = cv2.resize(frame, (self._w, self._h))
                    try:
                        self._process.stdin.write(frame.tobytes())
                        self._frame_count +=1
                    except (BrokenPipeError, OSError):
                        logger.warning(f"[Recorder:{self.camera_id}] Pipe error — FFmpeg died")
            except Exception as e:
                logger.error(f"[Recorder:{self.camera_id}] Error: {e}", exc_info=True)
                time.sleep(1)
        logger.info(f"[Recorder:{self.camera_id}] Stopping — killing FFmpeg")
        self._close_chunk(graceful=False)

_recorders = {}
_recorders_lock = threading.Lock()
_db_manager = None

def start_recorder(camera_id: str, initial_frame, db_manager):
    global _db_manager
    _db_manager = db_manager
    with _recorders_lock:
        if camera_id in _recorders:
            if _recorders[camera_id].is_alive:
                return
            del _recorders[camera_id]
        recorder = CameraRecorder(camera_id, db_manager)
        if recorder.start(initial_frame):
            _recorders[camera_id] = recorder

def stop_recorder(camera_id: str):
    with _recorders_lock:
        if camera_id in _recorders:
            _recorders[camera_id].stop()
            del _recorders[camera_id]
