import cv2
import threading
import time
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Fixed pipeline FPS — all stages run at this rate
PIPELINE_FPS = 10
_FRAME_INTERVAL = 1.0 / PIPELINE_FPS

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    "|analyze_duration;100000|probesize;100000"
    "|rtsp_flags;prefer_tcp|fflags;discardcorrupt"
)

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("DISPLAY", ":0")

RTSP_PROBE_PATHS = [
    "",
    "/axis-media/media.amp",
    "/Streaming/Channels/101",
    "/Streaming/Channels/1",
    "/cam/realmonitor?channel=1&subtype=0",
    "/Streaming/Channels/102",
    "/live/ch0", "/live/ch1",
    "/h264/ch1/main/av_stream",
    "/onvif-media/media.amp",
    "/stream1", "/main", "/live/main",
    "/11", "/12",
    "/cam/realmonitor?channel=1&subtype=1",
]


def probe_rtsp_url(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.path and parsed.path not in ("", "/"):
        return url
    base = url.rstrip("/")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"
    for path in RTSP_PROBE_PATHS:
        candidate = base + path
        cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            continue
        ret, img = cap.read()
        cap.release()
        if ret and img is not None:
            return candidate
    return url


class CameraHandler:
    def __init__(self, camera_id, source, vaapi_device=None):
        self.camera_id = camera_id
        self.source    = source
        self._vaapi    = vaapi_device
        self.cap       = self._open_capture()
        self.frame     = None
        self.frame_id  = 0
        self.running   = True
        self.lock      = threading.Lock()
        self.thread    = threading.Thread(target=self._update, daemon=True,
                                          name=f"cam-{camera_id}")
        self.thread.start()

    def _open_capture(self):
        if self._vaapi:
            pipeline = (
                f"rtspsrc location={self.source} latency=0 ! "
                f"rtph264depay ! h264parse ! vaapih264dec ! "
                f"videoconvert ! appsink max-buffers=1 drop=true"
                if str(self.source).startswith("rtsp") else
                f"filesrc location={self.source} ! decodebin ! "
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

        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, PIPELINE_FPS)
        return cap

    def _update(self):
        """
        Capture at exactly PIPELINE_FPS (10 FPS).
        Stores only the latest frame — detection thread reads it on its own tick.
        """
        fails = 0
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails > 50:
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self._open_capture()
                    fails = 0
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame    = frame
                self.frame_id += 1
            fails = 0
            elapsed = time.time() - t0
            sleep_t = _FRAME_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_frame_with_id(self):
        with self.lock:
            return (self.frame, self.frame_id) if self.frame is not None else (None, 0)

    def stop(self):
        self.running = False
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
        try:
            from utils.hw_manager import hw
            self._vaapi = getattr(hw, "vaapi_device", None)
        except Exception:
            self._vaapi = None

    def add_camera(self, camera_id, source):
        if camera_id not in self.cameras:
            if isinstance(source, str) and source.startswith("rtsp://"):
                source = probe_rtsp_url(source)
            self.cameras[camera_id] = CameraHandler(
                camera_id, source, vaapi_device=self._vaapi)
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
