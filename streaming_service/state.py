import logging
import threading
import time
from typing import Optional

from device_management.camera_manager import CameraManager, probe_rtsp_url
from data_access.manager import DatabaseManager
from ml_inference.detector import PersonDetector
from ml_inference.recognizer import FaceRecognizer
from core.pipeline import init_pipeline, process_camera

logger = logging.getLogger("camera_server")

db_manager:     Optional[DatabaseManager]  = None
camera_manager: Optional[CameraManager]    = None
detector:       Optional[PersonDetector]   = None
recognizer      = None
reid_manager    = None

def build_singletons():
    global db_manager, camera_manager, detector, recognizer, reid_manager

    from core.diagnostics import install as _install_diag
    _install_diag(auto_restart=False, monitor_interval=0)

    from core.startup import GlobalReIDManager

    db_manager     = DatabaseManager()
    camera_manager = CameraManager()
    detector       = PersonDetector(model_path='models/yolov8s.pt')

    try:
        recognizer = FaceRecognizer()
        recognizer.load_known_faces(db_manager)
    except Exception as e:
        logger.critical(f"[CameraServer] FaceRecognizer init failed: {e}")
        recognizer = None

    reid_manager = GlobalReIDManager(db_manager)

    init_pipeline(db_manager, camera_manager, detector, recognizer, reid_manager)
    logger.info("[CameraServer] Models and pipeline ready.")

def restore_cameras():
    time.sleep(1)
    try:
        cameras = db_manager.get_cameras()
        logger.info(f"[CameraServer] Restoring {len(cameras)} camera(s)...")
        for cam_id, source in cameras:
            if cam_id in camera_manager.cameras:
                logger.info(f"[CameraServer] {cam_id} already active, skipping restore")
                continue

            if isinstance(source, str) and source.startswith("rtsp://"):
                new_source = probe_rtsp_url(source)
                if new_source != source:
                    db_manager.update_camera_source(cam_id, new_source)
                source = new_source

            parsed = int(source) if str(source).isdigit() else source
            status, final_source = camera_manager.add_camera(cam_id, parsed)
            if status == 0:
                logger.info(f"[CameraServer] Restored: {cam_id}")
                threading.Thread(
                    target=process_camera, args=(cam_id,), daemon=True
                ).start()
                time.sleep(2)
            else:
                logger.warning(f"[CameraServer] Could not restore {cam_id} (status={status})")
    except Exception as e:
        logger.error(f"[CameraServer] Restore error: {e}")
