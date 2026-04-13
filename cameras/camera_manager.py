import cv2
import threading
import time
import os
import sys
import logging

logger = logging.getLogger(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyze_duration;100000|probesize;100000|rtsp_flags;prefer_tcp|fflags;discardcorrupt"

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
    "/11",                                   # Generic
    "/12",                                   # Generic
    "/cam/realmonitor?channel=1&subtype=1",  # Dahua sub
]

def probe_rtsp_url(url: str) -> str:
    """
    If the RTSP URL has no path (just host:port), try common stream paths
    and return the first one that successfully produces a frame.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.path and parsed.path not in ("", "/"):
        return url

    base = url.rstrip("/")
    # Force TCP for probe
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"
    
    for path in RTSP_PROBE_PATHS:
        candidate = base + path
        cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            continue
            
        # Try to read at least one frame to confirm it actually works
        ret, img = cap.read()
        cap.release()
        if ret and img is not None:
            return candidate

    return url

class CameraHandler:
    def __init__(self, camera_id, source, vaapi_device=None):
        self.camera_id = camera_id
        self.source = source
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _update(self):
        """Capture frames at ~25 FPS, reconnect on failure."""
        _cap_interval = 1.0 / 25  # 25 FPS cap — avoids burning a full CPU core
        fails = 0
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails > 100:
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self._open_capture()
                    fails = 0
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            fails = 0
            elapsed = time.time() - t0
            sleep_t = _cap_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

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
        if camera_id not in self.cameras:
            if isinstance(source, str) and source.startswith("rtsp://"):
                source = probe_rtsp_url(source)
            handler = CameraHandler(camera_id, source, vaapi_device=self._vaapi)
            self.cameras[camera_id] = handler
            return True
        return False

    def remove_camera(self, camera_id):
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
