"""
services/recording_worker.py — Standalone Recording Microservice

Runs in its own Docker container. Reads rendered frames from Redis
and writes them to MP4 files via FFmpeg subprocesses.

This replaces the RecordingService that was embedded in the monolith.
Instead of reading frames from a local Python dict, it now reads
JPEG frames from Redis, decodes them, and feeds raw pixels to FFmpeg.

Data Flow:
  1. Poll Redis for active cameras (camera:frame:{cam_id} keys)
  2. For each camera, start an FFmpeg subprocess
  3. Read rendered frames from Redis at 10 FPS
  4. Feed raw BGR24 pixels to FFmpeg stdin
  5. Rotate recordings every hour
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.redis_manager import get_redis_state
from database.postgres_manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("recording_worker")

_running = True
_redis = None
_db = None

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")
CHUNK_DURATION = 3600  # 1 hour


class CameraRecorder:
    """Manages FFmpeg recording for a single camera."""

    def __init__(self, camera_id: str, db, redis_state):
        self.camera_id = camera_id
        self.db = db
        self.redis_state = redis_state
        self.process = None
        self.db_id = None
        self.file_path = None
        self.start_time = None
        self.frame_count = 0
        self.w = None
        self.h = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start recording in a background thread."""
        # Get initial frame to determine dimensions
        frame = self.redis_state.get_rendered_frame(self.camera_id)
        if frame is None:
            logger.warning(f"[Recorder:{self.camera_id}] No frame available, cannot start.")
            return False

        self.h, self.w = frame.shape[:2]
        if not self._start_ffmpeg():
            return False

        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True,
            name=f"Recorder-{self.camera_id}"
        )
        self._thread.start()
        return True

    def _start_ffmpeg(self) -> bool:
        """Start a new FFmpeg subprocess."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")
        minute_str = now.strftime("%M")
        second_str = now.strftime("%S")

        camera_dir = os.path.join(RECORDINGS_DIR, date_str, self.camera_id)
        os.makedirs(camera_dir, exist_ok=True)

        filename = f"{hour_str}_{minute_str}{second_str}.mp4"
        self.file_path = os.path.join(camera_dir, filename)

        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.w}x{self.h}",
            "-pix_fmt", "bgr24",
            "-r", "10",
            "-i", "-",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "28",
            "-force_key_frames", "expr:gte(t,n_forced*2)",
            "-movflags", "+faststart",
            self.file_path
        ]

        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            time.sleep(0.1)
            if self.process.poll() is not None:
                logger.error(f"[Recorder:{self.camera_id}] FFmpeg failed to start.")
                return False

            self.db_id = self.db.start_recording(self.camera_id, self.file_path)
            self.start_time = time.time()
            self.frame_count = 0

            logger.info(f"[Recorder:{self.camera_id}] Started -> {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"[Recorder:{self.camera_id}] FFmpeg start error: {e}")
            return False

    def _writer_loop(self):
        """Feed frames from Redis to FFmpeg stdin at 10 FPS."""
        logger.info(f"[Recorder:{self.camera_id}] Writer thread started.")

        while not self._stop_event.is_set():
            try:
                # Check for hourly rotation
                if self.start_time and (time.time() - self.start_time) >= CHUNK_DURATION:
                    logger.info(f"[Recorder:{self.camera_id}] Hourly rotation.")
                    self._finalize()
                    if not self._start_ffmpeg():
                        break

                # Get frame from Redis
                frame = self.redis_state.get_rendered_frame(self.camera_id)

                if frame is not None and self.process and self.process.poll() is None:
                    try:
                        # Ensure frame matches expected dimensions
                        fh, fw = frame.shape[:2]
                        if fw != self.w or fh != self.h:
                            import cv2
                            frame = cv2.resize(frame, (self.w, self.h))

                        self.process.stdin.write(frame.tobytes())
                        self.frame_count += 1

                        if self.frame_count % 600 == 0:
                            elapsed = time.time() - self.start_time
                            logger.info(f"[Recorder:{self.camera_id}] {self.frame_count} frames ({elapsed/60:.1f} min)")

                    except (IOError, BrokenPipeError) as e:
                        logger.error(f"[Recorder:{self.camera_id}] Pipe error: {e}")
                        break

                elif self.process and self.process.poll() is not None:
                    logger.warning(f"[Recorder:{self.camera_id}] FFmpeg process died.")
                    break

                time.sleep(0.1)  # 10 FPS

            except Exception as e:
                logger.error(f"[Recorder:{self.camera_id}] Writer error: {e}")
                time.sleep(1)

        logger.info(f"[Recorder:{self.camera_id}] Writer stopped, {self.frame_count} frames total.")
        self._finalize()

    def _finalize(self):
        """Gracefully close FFmpeg and update DB."""
        if self.process:
            try:
                if self.process.stdin:
                    try: self.process.stdin.flush()
                    except Exception: pass
                    try: self.process.stdin.close()
                    except Exception: pass
                self.process.wait(timeout=15)
                logger.info(f"[Recorder:{self.camera_id}] FFmpeg finalized.")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"[Recorder:{self.camera_id}] Finalize error: {e}")
                if self.process:
                    self.process.kill()

        if self.db_id:
            self.db.end_recording(self.db_id)

        if self.file_path and os.path.exists(self.file_path):
            size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
            logger.info(f"[Recorder:{self.camera_id}] Saved: {self.file_path} ({size_mb:.2f} MB)")

    def stop(self):
        """Signal the writer thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=20)

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


def _management_loop():
    """
    Discover cameras and manage recordings.
    - Start recording for new cameras
    - Restart dead recorders
    - Stop recording for removed cameras
    """
    recorders = {}  # camera_id -> CameraRecorder

    while _running:
        try:
            # Get cameras with active frames in Redis
            active_cameras = _redis.get_active_camera_ids()

            # Start recorders for new cameras
            for cam_id in active_cameras:
                if cam_id not in recorders or not recorders[cam_id].is_alive:
                    logger.info(f"[RecordingWorker] Starting recorder for {cam_id}")
                    recorder = CameraRecorder(cam_id, _db, _redis)
                    if recorder.start():
                        recorders[cam_id] = recorder

            # Stop recorders for removed cameras
            removed = [k for k in recorders if k not in active_cameras]
            for cam_id in removed:
                logger.info(f"[RecordingWorker] Stopping recorder for {cam_id}")
                recorders[cam_id].stop()
                del recorders[cam_id]

        except Exception as e:
            logger.error(f"[RecordingWorker] Management error: {e}")

        time.sleep(10)

    # Cleanup all on shutdown
    logger.info(f"[RecordingWorker] Cleaning up {len(recorders)} active recorder(s)...")
    for cam_id, recorder in recorders.items():
        recorder.stop()


def _signal_handler(sig, frame):
    global _running
    logger.info("[RecordingWorker] Shutdown signal received.")
    _running = False


def main():
    global _running, _db, _redis

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 60)
    logger.info("  AI Vigilance — Recording Worker Starting")
    logger.info("=" * 60)

    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    # Wait for Redis
    for i in range(30):
        try:
            _redis = get_redis_state()
            if _redis.ping():
                break
        except Exception:
            pass
        logger.info(f"[RecordingWorker] Waiting for Redis... ({i+1}/30)")
        time.sleep(2)

    _db = DatabaseManager()
    _redis = get_redis_state()

    logger.info("[RecordingWorker] Entering management loop...")
    _management_loop()

    logger.info("[RecordingWorker] Exited.")


if __name__ == "__main__":
    main()
