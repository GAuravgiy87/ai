"""
services/recording_worker.py — Surveillance Recording Microservice

Design goals:
  1. Always-on: every active camera is recorded, no toggle needed.
  2. Hourly chunks: recordings rotate exactly on the clock hour
     (e.g. 13:00:00 → 14:00:00), named by their start hour.
  3. Crash-safe MKV: MKV writes its index incrementally every 2 s so
     a partial file is always playable — no finalization required.
  4. Restart-safe: on startup, any in-progress chunk from before the
     crash is left as-is (already playable). A new chunk starts from
     the current time and continues until the next hour boundary.
  5. Graceful rotation: on a clean hourly boundary, FFmpeg is closed
     via stdin EOF so the final cluster is flushed properly.

Data flow:
  camera_server → Redis (camera:frame:{cam_id}) → recording_worker → MKV file

File layout:
  recordings/
    YYYY-MM-DD/
      {camera_id}/
        HH.mkv          ← one file per clock-hour, named by hour (00–23)
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
from datetime import datetime, timedelta

import cv2
import numpy as np

# Add project root to path so imports work when run as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.redis_manager import get_redis_state
from data_access.manager import DatabaseManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("recording_worker")

# ── Config ────────────────────────────────────────────────────────────────────
RECORDINGS_DIR  = os.environ.get("RECORDINGS_DIR", "./database/recordings")
RECORD_FPS      = 10          # frames written to FFmpeg per second
MGMT_INTERVAL   = 10          # seconds between management loop ticks
REDIS_WAIT_SECS = 60          # max seconds to wait for Redis on startup

# ── Global state ──────────────────────────────────────────────────────────────
_running       = True
_redis         = None
_db            = None
_shutdown_event = threading.Event()   # wakes the management sleep immediately on shutdown


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_path(camera_id: str, dt: datetime) -> str:
    """
    Return the canonical file path for the chunk that covers datetime dt.
    Format: recordings/YYYY-MM-DD/{camera_id}/HH.mkv
    """
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = dt.strftime("%H")
    camera_dir = os.path.join(RECORDINGS_DIR, date_str, camera_id)
    os.makedirs(camera_dir, exist_ok=True)
    return os.path.join(camera_dir, f"{hour_str}.mkv")


def _next_hour(dt: datetime) -> datetime:
    """Return the datetime of the next exact clock-hour boundary."""
    return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def _seconds_until_next_hour(dt: datetime) -> float:
    """Seconds remaining until the next clock-hour boundary."""
    return (_next_hour(dt) - dt).total_seconds()


# ─────────────────────────────────────────────────────────────────────────────
# CameraRecorder
# ─────────────────────────────────────────────────────────────────────────────

class CameraRecorder:
    """
    Manages crash-safe MKV recording for a single camera.

    One FFmpeg subprocess per clock-hour chunk.  The subprocess reads raw
    BGR24 frames from stdin and writes an MKV file.

    Crash-safety flags:
      -f matroska          → MKV container (incremental index)
      -cluster_time_limit  → flush a new cluster every 2 s
      -flush_packets 1     → push every packet to the OS immediately
      -preset ultrafast    → minimum encoder latency / CPU
      -tune zerolatency    → no B-frames, no lookahead buffering
      -force_key_frames    → keyframe every 2 s for fine-grained seeking

    On a crash the file is valid up to the last 2-second cluster boundary.
    On a clean rotation stdin is closed so FFmpeg flushes its final cluster.
    """

    def __init__(self, camera_id: str, db, redis_state):
        self.camera_id   = camera_id
        self.db          = db
        self.redis_state = redis_state

        self._process    = None   # FFmpeg subprocess
        self._db_id      = None   # DB recording row id
        self._file_path  = None   # current chunk path
        self._chunk_end  = None   # datetime when current chunk must rotate
        self._w          = None
        self._h          = None
        self._frame_count = 0

        self._stop_event = threading.Event()
        self._thread     = None

    # ── Public interface ──────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Obtain frame dimensions from Redis, open the first FFmpeg chunk,
        and launch the writer thread.  Returns False if no frame is available.
        """
        frame = self._wait_for_frame(timeout=30)
        if frame is None:
            logger.warning(f"[Recorder:{self.camera_id}] No frame in Redis after 30 s — skipping.")
            return False

        self._h, self._w = frame.shape[:2]
        logger.info(f"[Recorder:{self.camera_id}] Frame size: {self._w}x{self._h}")

        if not self._open_chunk():
            return False

        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"Recorder-{self.camera_id}",
        )
        self._thread.start()
        return True

    def stop(self):
        """Signal the writer thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            # graceful=False kills FFmpeg instantly, so thread exits in ms.
            # 3 s is a generous safety margin.
            self._thread.join(timeout=3)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────

    def _wait_for_frame(self, timeout: int = 30):
        """Poll Redis until a frame arrives or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.redis_state.get_rendered_frame(self.camera_id)
            if frame is not None:
                return frame
            time.sleep(0.5)
        return None

    def _open_chunk(self) -> bool:
        """
        Start a new FFmpeg subprocess writing to the chunk file for the
        current clock-hour.  Sets self._chunk_end to the next hour boundary.
        """
        now            = datetime.now()
        self._chunk_end = _next_hour(now)
        self._file_path = _chunk_path(self.camera_id, now)
        self._frame_count = 0

        # If a partial file already exists from a previous crash, FFmpeg will
        # append to it — but MKV doesn't support append cleanly.  Instead we
        # rename the old file so the new chunk starts fresh and the old one
        # remains playable.
        if os.path.exists(self._file_path):
            crashed_path = self._file_path.replace(".mkv", "_recovered.mkv")
            try:
                os.rename(self._file_path, crashed_path)
                logger.info(
                    f"[Recorder:{self.camera_id}] Renamed crash-partial file "
                    f"→ {os.path.basename(crashed_path)}"
                )
            except Exception as e:
                logger.warning(f"[Recorder:{self.camera_id}] Could not rename partial file: {e}")

        cmd = [
            "ffmpeg",
            "-y",                          # overwrite if somehow still exists
            # ── Input: raw BGR24 frames from stdin ──────────────────────
            "-f",        "rawvideo",
            "-vcodec",   "rawvideo",
            "-s",        f"{self._w}x{self._h}",
            "-pix_fmt",  "bgr24",
            "-r",        str(RECORD_FPS),
            "-i",        "pipe:0",
            # ── Output: crash-safe MKV ───────────────────────────────────
            "-f",        "matroska",       # explicit MKV muxer
            "-vcodec",   "libx264",
            "-pix_fmt",  "yuv420p",
            "-preset",   "ultrafast",      # lowest CPU / latency
            "-tune",     "zerolatency",    # no B-frames, no lookahead
            "-crf",      "28",             # quality/size balance
            # Keyframe every 2 s → fine-grained seeking in partial files
            "-force_key_frames", "expr:gte(t,n_forced*2)",
            # MKV cluster every 2 s → data committed to file every 2 s
            "-cluster_time_limit", "2000",
            # Flush every packet to the OS immediately
            "-flush_packets", "1",
            # Suppress console spam
            "-loglevel",  "error",
            self._file_path,
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,          # unbuffered stdin — frames go to FFmpeg immediately
            )
            # Give FFmpeg 0.5 s to start; bail if it exits immediately
            time.sleep(0.5)
            if self._process.poll() is not None:
                err = self._process.stderr.read(500).decode(errors="replace")
                logger.error(f"[Recorder:{self.camera_id}] FFmpeg exited immediately: {err}")
                return False

            self._db_id = self.db.start_recording(self.camera_id, self._file_path)
            logger.info(
                f"[Recorder:{self.camera_id}] Chunk opened → {self._file_path} "
                f"(until {self._chunk_end.strftime('%H:%M:%S')})"
            )
            return True

        except FileNotFoundError:
            logger.error(
                f"[Recorder:{self.camera_id}] 'ffmpeg' not found. "
                "Install FFmpeg and ensure it is on PATH."
            )
            return False
        except Exception as e:
            logger.error(f"[Recorder:{self.camera_id}] FFmpeg start error: {e}")
            return False

    def _close_chunk(self, graceful: bool = True):
        """
        Close the current FFmpeg chunk.

        graceful=True  → close stdin so FFmpeg flushes its final cluster
                         cleanly (used on hourly rotation and clean shutdown).
        graceful=False → kill immediately (used when FFmpeg has already died
                         or on emergency stop).
        """
        if self._process is None:
            return

        if graceful:
            try:
                self._process.stdin.flush()
                self._process.stdin.close()
                self._process.wait(timeout=15)
                logger.info(f"[Recorder:{self.camera_id}] Chunk closed gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning(f"[Recorder:{self.camera_id}] FFmpeg did not exit in 15 s — killing.")
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

        # Update DB record
        if self._db_id:
            try:
                self.db.end_recording(self._db_id)
            except Exception:
                pass
            self._db_id = None

        # Log file size
        if self._file_path and os.path.exists(self._file_path):
            size_mb = os.path.getsize(self._file_path) / (1024 * 1024)
            logger.info(
                f"[Recorder:{self.camera_id}] Saved: "
                f"{os.path.basename(self._file_path)} ({size_mb:.1f} MB, "
                f"{self._frame_count} frames)"
            )

        self._process = None

    def _writer_loop(self):
        """
        Main recording loop.

        Reads rendered JPEG frames from Redis, decodes them to raw BGR24,
        and writes them to FFmpeg stdin at RECORD_FPS.

        Rotation logic:
          - Checks wall-clock time every frame.
          - When the clock crosses the hour boundary, closes the current
            chunk gracefully and opens a new one.
        """
        logger.info(f"[Recorder:{self.camera_id}] Writer thread started.")
        frame_interval = 1.0 / RECORD_FPS
        next_frame_t   = time.time()

        while not self._stop_event.is_set():
            try:
                # ── Hourly rotation ───────────────────────────────────────
                now = datetime.now()
                if now >= self._chunk_end:
                    logger.info(
                        f"[Recorder:{self.camera_id}] "
                        f"Hour boundary reached — rotating chunk."
                    )
                    self._close_chunk(graceful=True)
                    if not self._open_chunk():
                        logger.error(
                            f"[Recorder:{self.camera_id}] "
                            "Failed to open new chunk — retrying in 5 s."
                        )
                        time.sleep(5)
                        continue

                # ── FFmpeg died unexpectedly ──────────────────────────────
                if self._process and self._process.poll() is not None:
                    err = ""
                    try:
                        err = self._process.stderr.read(300).decode(errors="replace").strip()
                    except Exception:
                        pass
                    logger.warning(
                        f"[Recorder:{self.camera_id}] FFmpeg died unexpectedly. "
                        f"stderr: {err or '(none)'}. Reopening chunk."
                    )
                    self._close_chunk(graceful=False)
                    time.sleep(1)
                    if not self._open_chunk():
                        time.sleep(5)
                    continue

                # ── Frame timing ──────────────────────────────────────────
                wait = next_frame_t - time.time()
                if wait > 0:
                    time.sleep(wait)
                next_frame_t += frame_interval
                # Prevent spiral if we fall behind
                if next_frame_t < time.time() - frame_interval * 3:
                    next_frame_t = time.time() + frame_interval

                # ── Get frame from Redis ──────────────────────────────────
                jpeg = self.redis_state.get_rendered_jpeg(self.camera_id)
                if jpeg is None:
                    continue   # camera not producing frames yet

                # Decode JPEG → raw BGR24
                arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue

                # Resize if dimensions changed (e.g. camera reconnected)
                fh, fw = arr.shape[:2]
                if fw != self._w or fh != self._h:
                    arr = cv2.resize(arr, (self._w, self._h))

                # ── Write raw pixels to FFmpeg stdin ──────────────────────
                try:
                    self._process.stdin.write(arr.tobytes())
                    self._frame_count += 1
                except (BrokenPipeError, OSError) as e:
                    logger.warning(
                        f"[Recorder:{self.camera_id}] Pipe write error: {e} — "
                        "FFmpeg likely died, will reopen."
                    )
                    # Loop will detect process.poll() != None on next iteration

            except Exception as e:
                logger.error(f"[Recorder:{self.camera_id}] Writer error: {e}", exc_info=True)
                time.sleep(1)

        # ── Clean shutdown ────────────────────────────────────────────────
        # MKV is already safe: -cluster_time_limit 2000 + -flush_packets 1
        # guarantee the file is valid up to the last 2-second cluster.
        # There is NO need to gracefully drain FFmpeg's encoder on shutdown —
        # kill immediately so we don't wait up to 15 s for nothing.
        logger.info(
            f"[Recorder:{self.camera_id}] Stop requested — "
            "killing FFmpeg instantly (MKV already safe on disk)."
        )
        self._close_chunk(graceful=False)
        logger.info(f"[Recorder:{self.camera_id}] Writer thread exited.")


# ─────────────────────────────────────────────────────────────────────────────
# Management loop
# ─────────────────────────────────────────────────────────────────────────────

def _management_loop():
    """
    Discover active cameras from Redis and manage their recorders.

    - Starts a recorder for every camera that has frames in Redis.
    - Restarts recorders that have died unexpectedly.
    - Stops recorders for cameras that have gone offline.
    """
    recorders: dict = {}   # camera_id → CameraRecorder

    while _running:
        try:
            active_cameras = set(_redis.get_active_camera_ids())

            # ── Start / restart recorders ─────────────────────────────────
            for cam_id in active_cameras:
                existing = recorders.get(cam_id)
                if existing is None or not existing.is_alive:
                    if existing is not None:
                        logger.warning(
                            f"[RecordingWorker] Recorder for {cam_id} died — restarting."
                        )
                    recorder = CameraRecorder(cam_id, _db, _redis)
                    if recorder.start():
                        recorders[cam_id] = recorder
                    else:
                        logger.warning(
                            f"[RecordingWorker] Could not start recorder for {cam_id}."
                        )

            # ── Stop recorders for offline cameras ────────────────────────
            offline = [k for k in recorders if k not in active_cameras]
            for cam_id in offline:
                logger.info(f"[RecordingWorker] Camera {cam_id} offline — stopping recorder.")
                recorders[cam_id].stop()
                del recorders[cam_id]

        except Exception as e:
            logger.error(f"[RecordingWorker] Management loop error: {e}", exc_info=True)

        # Interruptible sleep: wakes immediately when _shutdown_event is set
        # instead of blocking for the full MGMT_INTERVAL on shutdown.
        _shutdown_event.wait(timeout=MGMT_INTERVAL)
        _shutdown_event.clear()

    # ── Shutdown: kill all FFmpeg processes in parallel ───────────────────
    # MKV files are already safe on disk (cluster flushed every 2 s).
    # No need to wait for graceful drain — kill all recorders at once.
    import concurrent.futures
    logger.info(f"[RecordingWorker] Shutdown — killing {len(recorders)} recorder(s) in parallel.")
    if recorders:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(recorders)) as pool:
            futures = [
                pool.submit(recorder.stop)
                for recorder in recorders.values()
            ]
            concurrent.futures.wait(futures, timeout=10)
    logger.info("[RecordingWorker] All recorders stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _signal_handler(sig, frame):
    global _running
    logger.info(f"[RecordingWorker] Signal {sig} received — shutting down.")
    _running = False
    _shutdown_event.set()   # wake management sleep immediately


def main():
    global _running, _db, _redis

    # Register signal handlers for clean Docker stop (SIGTERM) and Ctrl-C
    try:
        signal.signal(signal.SIGINT,  _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError):
        pass   # running in a thread — signals not supported

    logger.info("=" * 60)
    logger.info("  AI Vigilance — Recording Worker")
    logger.info(f"  Recordings dir : {RECORDINGS_DIR}")
    logger.info(f"  Record FPS     : {RECORD_FPS}")
    logger.info(f"  Chunk size     : 1 hour (clock-aligned)")
    logger.info(f"  Container      : MKV (crash-safe)")
    logger.info("=" * 60)

    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    # ── Wait for Redis ────────────────────────────────────────────────────
    logger.info("[RecordingWorker] Waiting for Redis...")
    deadline = time.time() + REDIS_WAIT_SECS
    while time.time() < deadline:
        try:
            _redis = get_redis_state()
            if _redis.ping():
                logger.info("[RecordingWorker] Redis OK.")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        logger.error("[RecordingWorker] Redis not available after timeout — exiting.")
        return

    # ── Connect to DB ─────────────────────────────────────────────────────
    _db = DatabaseManager()
    logger.info("[RecordingWorker] Database OK.")

    # ── Run ───────────────────────────────────────────────────────────────
    logger.info("[RecordingWorker] Entering management loop.")
    _management_loop()
    logger.info("[RecordingWorker] Exited.")


if __name__ == "__main__":
    main()
