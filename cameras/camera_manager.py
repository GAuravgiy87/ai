import cv2
import threading
import time
import os
import sys
import logging
import subprocess

from core.state import sanitize_rtsp_url

logger = logging.getLogger(__name__)

# Optimized for Windows: TCP reliability without over-aggressive buffer discarding that causes black screens
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyze_duration;100000|probesize;100000"

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("DISPLAY", ":0")

# Common RTSP stream paths to try when a bare IP is given (no path after port)
RTSP_PROBE_PATHS = [
    "",                                      # bare — try as-is first
    "/axis-media/media.amp",                 # Axis
    "/Streaming/Channels/101",               # Hikvision main
    "/Streaming/Channels/1",                 # Hikvision alt
    "/cam/realmonitor?channel=1&subtype=0",  # Dahua
    "/Streaming/Channels/102",               # Hikvision sub
    "/live/ch0",                             # Generic
    "/live/ch1",                             # Generic
    "/h264/ch1/main/av_stream",              # Generic Hikvision
    "/onvif-media/media.amp",                # ONVIF
    "/stream1",                              # Generic
    "/main",                                 # Generic
    "/live/main",                            # Generic
    "/11",                                   # AXIS/Generic
    "/12",                                   # AXIS/Generic
    "/cam/realmonitor?channel=1&subtype=1",  # Dahua sub
    "/live/ch01_0",                          # Reolink
    "/live/ch01_1",                          # Reolink sub
    "/0",                                    # Generic
    "/1",                                    # Generic
    "/video",                                # Generic
    "/vedio",                                # Common typo in some firmwares
    "/h264",                                 # Generic
]

def probe_rtsp_url(url: str) -> str:
    """
    Tries various common RTSP paths if only an IP/port is provided.
    Uses ffprobe with cv2 fallback if needed.
    """
    url = sanitize_rtsp_url(url)
    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url

    # If it already has a path (e.g. rtsp://ip:port/path), don't probe
    parts = url.rstrip('/').split('/')
    if len(parts) > 3:
        return url

    logger.info(f"[Prober] Starting automatic path detection for: {url.split('@')[-1]}")
    
    base_url = url.rstrip('/')
    for path in RTSP_PROBE_PATHS:
        test_url = base_url + path
        try:
            # 1. Try ffprobe (faster, cleaner)
            cmd = ['ffprobe', '-v', 'error', '-rtsp_transport', 'tcp', '-show_entries', 'format=format_name', test_url]
            subprocess.run(cmd, capture_output=True, timeout=2.0, check=True)
            logger.info(f"[Prober] Success (ffprobe)! Found working path: {path}")
            return test_url
        except Exception:
            try:
                # 2. Try cv2 fallback (if ffprobe missing or failing)
                cap = cv2.VideoCapture(test_url)
                is_ok = cap.isOpened()
                cap.release()
                if is_ok:
                    logger.info(f"[Prober] Success (cv2 fallback)! Found working path: {path}")
                    return test_url
            except Exception:
                continue
            
    logger.warning(f"[Prober] Auto-detection failed for {url.split('@')[-1]}. Please set path manually.")
    return url

class CameraHandler:
    def __init__(self, camera_id, source, vaapi_device=None):
        self.camera_id = camera_id
        # Ensure source is always sanitized
        self.source = sanitize_rtsp_url(source) if isinstance(source, str) else source
        self._vaapi = vaapi_device

        self.cap = self._open_capture()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        """Open capture with VAAPI hardware decode if Intel iGPU available."""
        if self._vaapi:
            # Try VAAPI-accelerated decode via GStreamer pipeline
            pipeline = (
                f"filesrc location={self.source} ! decodebin ! "
                f"vaapisink display=drm device={self._vaapi}"
                if not str(self.source).startswith("rtsp") else
                f"rtspsrc location={self.source} latency=0 ! "
                f"rtph264depay ! h264parse ! vaapih264dec ! "
                f"videoconvert ! appsink max-buffers=1 drop=true"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info(f"[Camera:{self.camera_id}] VAAPI decode active")
                    return cap
                cap.release()

        # Fallback: standard FFMPEG
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Try one more time without FFMPEG flag just in case (DSHOW for webcams on windows)
            cap = cv2.VideoCapture(self.source)
            
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def _update(self):
        """Drains the camera buffer as fast as possible to prevent lag and glitches."""
        fails = 0
        while self.running:
            try:
                # Read without manual sleep to keep the buffer empty (prevents "glitchy" lag)
                ret, frame = self.cap.read()
                if not ret:
                    fails += 1
                    if fails > 30: # Reconnect after ~5 seconds of silence
                        logger.warning(f"[Camera:{self.camera_id}] Signal lost at {self.source}. Reconnecting...")
                        self.cap.release()
                        time.sleep(3)
                        self.cap = self._open_capture()
                        fails = 0
                    else:
                        time.sleep(0.1)
                    continue
                
                # Update global state only on success
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                
                if fails == 0 and self.frame_id % 30 == 0: # Periodic health check
                    if frame.mean() < 0.1:
                        logger.error(f"[Camera:{self.camera_id}] ALERT: Stream is PITCH BLACK (Empty Data)")
                
                fails = 0
                time.sleep(0.02) # Cap capture at ~50 FPS to save CPU, still plenty for AI and smooth video
            except Exception as e:
                logger.error(f"[Camera:{self.camera_id}] Capture error: {e}")
                time.sleep(1)

    def get_frame(self):
        with self.lock:
            return self.frame if self.frame is not None else None

    def get_frame_with_id(self):
        with self.lock:
            return (self.frame, self.frame_id) if self.frame is not None else (None, 0)

    def stop(self):
        self.running = False
        # Wait for thread to finish before releasing capture
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

from typing import Dict, Any
from background_jobs.recording_worker import stop_recorder

class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, Any] = {}
        # Get VAAPI device from HardwareManager
        try:
            from utils.hw_manager import hw
            self._vaapi = hw.vaapi_device
        except Exception:
            self._vaapi = None

    def add_camera(self, camera_id, source):
        """Adds camera and returns (status, final_source): status 0=Success, 1=Duplicate ID, 2=Connection Failed."""
        if camera_id in self.cameras:
            return 1, source
        
        final_source = source
        if isinstance(source, str) and source.startswith("rtsp://"):
            final_source = probe_rtsp_url(source)
            
        handler = CameraHandler(camera_id, final_source, vaapi_device=self._vaapi)
        if not handler.is_opened():
            handler.stop()
            return 2, final_source
            
        self.cameras[camera_id] = handler
        return 0, final_source

    def remove_camera(self, camera_id):
        stop_recorder(camera_id)
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            self.cameras.pop(camera_id, None)
            return True
        return False

    def get_camera_frame(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame()
        return None
        
    def get_camera_frame_with_id(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame_with_id()
        return None, 0

    def get_active_cameras(self):
        return list(self.cameras.keys())
