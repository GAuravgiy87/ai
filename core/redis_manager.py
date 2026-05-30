"""
core/redis_manager.py — Distributed State via Redis

Replaces the local in-memory dictionaries in core/state.py so that
multiple Docker containers can share video frames, detection results,
and event notifications over the network.

Channels:
  - camera:frames:{cam_id}   — JPEG-encoded rendered frames (Pub/Sub)
  - camera:results:{cam_id}  — JSON metadata (count, tracks, timestamp)
  - camera:raw:{cam_id}      — Raw video frames for inference worker (Pub/Sub)
  - detections:{cam_id}      — Detection results from inference worker
  - events:notifications      — SSE broadcast events
"""

import os
import json
import time
import logging
import threading
import numpy as np
import cv2
import redis

logger = logging.getLogger("redis_manager")

# Uses "redis" hostname which matches the Docker Compose service name.
# For local (non-Docker) runs, set REDIS_URL in your environment.
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")


class RedisStateManager:
    """
    Drop-in replacement for the local camera_results dictionary.

    Instead of storing frames in a Python dict protected by a threading.Lock,
    this class publishes/subscribes frames via Redis so that separate containers
    (camera_server, main_app, recording_service) can all access them.
    """

    def __init__(self, redis_url: str = None):
        url = redis_url or REDIS_URL
        self._client = redis.Redis.from_url(url, decode_responses=False)
        self._pubsub_client = redis.Redis.from_url(url, decode_responses=False)
        self._subscribers = {}  # channel -> pubsub object
        self._lock = threading.Lock()
        logger.info(f"[RedisState] Connected to {url}")

    # ──────────────────────────────────────────────────────────────────────────
    # Frame Publishing (used by camera_server)
    # ──────────────────────────────────────────────────────────────────────────

    def publish_rendered_frame(self, camera_id: str, frame: np.ndarray,
                                count: int = 0, tracks: list = None,
                                timestamp: str = None):
        """
        Publish a rendered (annotated) frame + metadata for a camera.

        The frame is JPEG-encoded before publishing to minimize bandwidth.
        Metadata (count, tracks) is stored separately as JSON in a Redis key
        for polling, and also broadcast on Pub/Sub for real-time subscribers.
        """
        try:
            # JPEG-encode the rendered frame
            _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpeg_bytes = jpeg_buf.tobytes()

            # Store the latest JPEG in a key (for polling by MJPEG endpoint)
            self._client.set(f"camera:frame:{camera_id}", jpeg_bytes)
            # Also publish for real-time subscribers
            self._client.publish(f"camera:frames:{camera_id}", jpeg_bytes)

            # Store metadata
            meta = {
                "count": count,
                "tracks": tracks or [],
                "timestamp": timestamp or "",
            }
            self._client.set(f"camera:meta:{camera_id}", json.dumps(meta))

            # Set expiry so stale data auto-cleans (30s)
            self._client.expire(f"camera:frame:{camera_id}", 30)
            self._client.expire(f"camera:meta:{camera_id}", 30)
        except Exception as e:
            logger.error(f"[RedisState] publish_rendered_frame error: {e}")

    def publish_raw_frame(self, camera_id: str, frame: np.ndarray):
        """
        Publish a raw (un-annotated) frame for the AI inference worker.

        Uses lossy JPEG to keep bandwidth manageable. The inference worker
        decodes it back to numpy for YOLO/FaceNet processing.
        """
        try:
            _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg_bytes = jpeg_buf.tobytes()
            # Store as latest (inference worker polls this)
            self._client.set(f"camera:raw:{camera_id}", jpeg_bytes)
            self._client.expire(f"camera:raw:{camera_id}", 10)
        except Exception as e:
            logger.error(f"[RedisState] publish_raw_frame error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Frame Retrieval (used by main_app, recording_service)
    # ──────────────────────────────────────────────────────────────────────────

    def get_rendered_jpeg(self, camera_id: str) -> bytes:
        """Get the latest JPEG-encoded rendered frame for a camera."""
        try:
            return self._client.get(f"camera:frame:{camera_id}")
        except Exception:
            return None

    def get_rendered_frame(self, camera_id: str) -> np.ndarray:
        """Get the latest rendered frame as a numpy array (decoded from JPEG)."""
        jpeg = self.get_rendered_jpeg(camera_id)
        if jpeg is None:
            return None
        try:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def get_raw_frame(self, camera_id: str) -> np.ndarray:
        """Get the latest raw frame for inference."""
        try:
            jpeg = self._client.get(f"camera:raw:{camera_id}")
            if jpeg is None:
                return None
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def get_camera_metadata(self, camera_id: str) -> dict:
        """Get the latest metadata (count, tracks, timestamp) for a camera."""
        try:
            data = self._client.get(f"camera:meta:{camera_id}")
            if data:
                return json.loads(data)
            return {"count": 0, "tracks": [], "timestamp": ""}
        except Exception:
            return {"count": 0, "tracks": [], "timestamp": ""}

    def get_active_camera_ids(self) -> list:
        """Get list of camera IDs that have recent frames."""
        try:
            keys = self._client.keys("camera:frame:*")
            return [k.decode().split(":")[-1] for k in keys]
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Detection Results (inference worker -> camera_server / main_app)
    # ──────────────────────────────────────────────────────────────────────────

    def publish_detections(self, camera_id: str, detections: list):
        """
        Publish detection results from the inference worker.
        Each detection: ([x, y, w, h], confidence, 'person')
        """
        try:
            serializable = [
                {"bbox": d[0], "conf": d[1], "label": d[2]}
                for d in detections
            ]
            self._client.set(f"detections:{camera_id}", json.dumps(serializable))
            self._client.expire(f"detections:{camera_id}", 10)
        except Exception as e:
            logger.error(f"[RedisState] publish_detections error: {e}")

    def get_detections(self, camera_id: str) -> list:
        """Get the latest detection results for a camera."""
        try:
            data = self._client.get(f"detections:{camera_id}")
            if data:
                parsed = json.loads(data)
                return [(d["bbox"], d["conf"], d["label"]) for d in parsed]
            return []
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # SSE Event Broadcasting
    # ──────────────────────────────────────────────────────────────────────────

    def broadcast_event(self, event_type: str, data: dict):
        """Broadcast an event (e.g., person recognized) to all subscribers."""
        try:
            payload = json.dumps({"type": event_type, "data": data})
            self._client.publish("events:notifications", payload.encode())
        except Exception as e:
            logger.error(f"[RedisState] broadcast_event error: {e}")

    def subscribe_events(self):
        """
        Subscribe to the events channel. Returns an iterator that yields
        event dicts as they arrive. Use in a background thread.
        """
        sub_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        pubsub = sub_client.pubsub()
        pubsub.subscribe("events:notifications")

        for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    yield json.loads(message["data"])
                except Exception:
                    continue

    def subscribe_frames(self, camera_id: str):
        """
        Subscribe to rendered frame updates for a specific camera.
        Yields JPEG bytes as they arrive (for MJPEG streaming).
        """
        sub_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        pubsub = sub_client.pubsub()
        pubsub.subscribe(f"camera:frames:{camera_id}")

        for message in pubsub.listen():
            if message["type"] == "message":
                yield message["data"]

    # ──────────────────────────────────────────────────────────────────────────
    # Recording Service Support
    # ──────────────────────────────────────────────────────────────────────────

    def get_recording_frame(self, camera_id: str) -> np.ndarray:
        """
        Get the latest rendered frame for recording purposes.
        Alias for get_rendered_frame() but semantically distinct.
        """
        return self.get_rendered_frame(camera_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Health & Utility
    # ──────────────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._client.ping()
        except Exception:
            return False

    def cleanup_camera(self, camera_id: str):
        """Remove all keys associated with a camera."""
        try:
            for prefix in ["camera:frame:", "camera:meta:", "camera:raw:", "detections:"]:
                self._client.delete(f"{prefix}{camera_id}")
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance = None
_instance_lock = threading.Lock()

def get_redis_state() -> RedisStateManager:
    """Get or create the global RedisStateManager singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = RedisStateManager()
    return _instance
