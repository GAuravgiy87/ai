"""
workers/inference_worker.py — Standalone AI Inference Microservice

This worker runs in its own Docker container and is responsible for
ALL heavy AI computation:
  - YOLOv8s person detection
  - FaceNet / MTCNN face recognition
  - Object tracking (Hungarian matching)

Data Flow:
  1. Subscribes to raw frames from Redis (camera:raw:{cam_id})
  2. Runs detection + recognition + tracking
  3. Publishes annotated frames + metadata back to Redis
  4. The main_app and recording_service read from Redis

This completely decouples the AI workload from the web server,
allowing independent scaling and GPU allocation.
"""

import os
import sys
import time
import json
import signal
import logging
import threading
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.redis_manager import get_redis_state
from data_access.manager import DatabaseManager
from ml_inference.detector import PersonDetector
from ml_inference.tracker import ObjectTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("inference_worker")

# ── Globals ───────────────────────────────────────────────────────────────────
_running = True
_redis = None
_db = None
_detector = None
_recognizer = None
_trackers = {}  # camera_id -> ObjectTracker


def _init():
    """Initialize all heavy singletons."""
    global _redis, _db, _detector, _recognizer

    logger.info("[InferenceWorker] Initializing...")

    _redis = get_redis_state()
    _db = DatabaseManager()
    _detector = PersonDetector(model_path='models/yolov8s.pt')

    try:
        from ml_inference.recognizer import FaceRecognizer
        _recognizer = FaceRecognizer()
        _recognizer.load_known_faces(_db)
        logger.info("[InferenceWorker] FaceRecognizer loaded.")
    except Exception as e:
        logger.warning(f"[InferenceWorker] FaceRecognizer init failed: {e}")
        _recognizer = None

    logger.info("[InferenceWorker] Initialization complete.")


def _get_tracker(camera_id: str) -> ObjectTracker:
    """Get or create a tracker for a camera."""
    if camera_id not in _trackers:
        _trackers[camera_id] = ObjectTracker()
    return _trackers[camera_id]


def _process_frame(camera_id: str, frame: np.ndarray):
    """
    Run the full AI pipeline on a single frame:
      1. Detect persons (YOLOv8s)
      2. Update tracker (Hungarian matching)
      3. Run face recognition on tracked persons
      4. Render bounding boxes
      5. Publish results to Redis
    """
    fh, fw = frame.shape[:2]

    # 1. Detection
    detections = _detector.detect(frame)

    # 2. Tracking
    tracker = _get_tracker(camera_id)
    active_tracks = tracker.update(detections, frame)

    # 3. Face Recognition (if available)
    recognized_names = {}
    if _recognizer and active_tracks:
        for track in active_tracks:
            track_id = track.get("id", 0)
            bbox = track.get("bbox", [0, 0, 0, 0])
            x, y, w, h = [int(v) for v in bbox]

            # Extract face region (upper 40% of person bbox)
            face_y_end = y + int(h * 0.4)
            face_crop = frame[max(0, y):min(fh, face_y_end), max(0, x):min(fw, x + w)]

            if face_crop.size > 0:
                try:
                    name, confidence = _recognizer.recognize(face_crop)
                    if name and confidence > 0.6:
                        recognized_names[track_id] = name
                except Exception:
                    pass

    # 4. Render bounding boxes on the frame
    rendered = frame.copy()
    tracks_data = []

    for track in active_tracks:
        track_id = track.get("id", 0)
        bbox = track.get("bbox", [0, 0, 0, 0])
        x, y, w, h = [int(v) for v in bbox]

        name = recognized_names.get(track_id, "")
        color = (0, 255, 0) if name else (0, 200, 255)

        # Draw bounding box
        cv2.rectangle(rendered, (x, y), (x + w, y + h), color, 2)

        # Draw label
        label = name if name else f"Person #{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(rendered, (x, y - label_size[1] - 8), (x + label_size[0] + 4, y), color, -1)
        cv2.putText(rendered, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        tracks_data.append({"id": track_id, "name": name, "bbox": bbox})

    # 5. Publish to Redis
    from core.state import get_ist_time
    _redis.publish_rendered_frame(
        camera_id=camera_id,
        frame=rendered,
        count=len(active_tracks),
        tracks=tracks_data,
        timestamp=get_ist_time().isoformat()
    )

    # Also publish raw detections for other services
    _redis.publish_detections(camera_id, detections)


def _camera_loop(camera_id: str):
    """
    Continuously process frames for a single camera.
    Runs in its own thread.
    """
    logger.info(f"[InferenceWorker] Starting loop for {camera_id}")
    frame_interval = 1.0 / 6  # 6 FPS target

    while _running:
        try:
            frame = _redis.get_raw_frame(camera_id)
            if frame is None:
                time.sleep(0.5)
                continue

            start = time.time()
            _process_frame(camera_id, frame)
            elapsed = time.time() - start

            # Throttle to target FPS
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"[InferenceWorker] Error processing {camera_id}: {e}")
            time.sleep(1)

    logger.info(f"[InferenceWorker] Stopped loop for {camera_id}")


def _discovery_loop():
    """
    Discover new cameras by polling Redis for raw frame keys.
    Starts a processing thread for each new camera.
    """
    active_threads = {}  # camera_id -> Thread

    while _running:
        try:
            camera_ids = []
            # Check which cameras have raw frames available
            keys = _redis._client.keys("camera:raw:*")
            for key in keys:
                cam_id = key.decode().split(":")[-1]
                camera_ids.append(cam_id)

            # Start threads for new cameras
            for cam_id in camera_ids:
                if cam_id not in active_threads or not active_threads[cam_id].is_alive():
                    t = threading.Thread(target=_camera_loop, args=(cam_id,), daemon=True,
                                         name=f"Inference-{cam_id}")
                    t.start()
                    active_threads[cam_id] = t
                    logger.info(f"[InferenceWorker] Started processing thread for {cam_id}")

            # Clean up dead threads
            dead = [k for k, v in active_threads.items() if not v.is_alive()]
            for k in dead:
                del active_threads[k]

        except Exception as e:
            logger.error(f"[InferenceWorker] Discovery error: {e}")

        time.sleep(5)  # Check for new cameras every 5 seconds


def _signal_handler(sig, frame):
    global _running
    logger.info("[InferenceWorker] Shutdown signal received.")
    _running = False


def main():
    global _running

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 60)
    logger.info("  AI Vigilance — Inference Worker Starting")
    logger.info("=" * 60)

    # Wait for Redis to be ready
    for i in range(30):
        try:
            redis_state = get_redis_state()
            if redis_state.ping():
                logger.info("[InferenceWorker] Redis connection OK.")
                break
        except Exception:
            pass
        logger.info(f"[InferenceWorker] Waiting for Redis... ({i+1}/30)")
        time.sleep(2)

    _init()

    logger.info("[InferenceWorker] Entering discovery loop...")
    _discovery_loop()

    logger.info("[InferenceWorker] Exited.")


if __name__ == "__main__":
    main()
