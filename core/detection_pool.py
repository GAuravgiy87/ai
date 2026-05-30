import cv2
import time
import queue
import logging
import threading
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

    def __init__(self, detector, num_workers: int = 1, queue_size: int = 4):
        self.detector = detector
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

                dets   = self.detector.detect(proc) if self.detector else []
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