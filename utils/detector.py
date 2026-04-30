import cv2
import numpy as np
import logging
import os
import subprocess
import threading
from utils.hw_manager import hw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic threshold helpers
# ---------------------------------------------------------------------------

def _frame_brightness(frame: np.ndarray) -> float:
    """
    Returns mean luminance of the frame in [0, 255].
    Uses a small downsampled grayscale to keep it cheap.
    """
    small = cv2.resize(frame, (64, 64))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _dynamic_conf_threshold(brightness: float) -> float:
    """
    Raise the YOLO confidence threshold in dark scenes to suppress
    IR-noise / shadow false positives.

    Brightness zones (0-255 mean luminance):
      < 40  → very dark  (night / IR)  → 0.72
      40-80 → dim        (dusk / dawn) → 0.65
      80+   → normal                   → 0.58

    Values are linearly interpolated within each zone for smooth transitions.
    """
    if brightness < 40:
        # Very dark: interpolate 0.72 → 0.72 (flat)
        return 0.72
    elif brightness < 80:
        # Dim: interpolate 0.72 → 0.65
        t = (brightness - 40) / 40.0
        return 0.72 - t * (0.72 - 0.65)
    else:
        # Normal: interpolate 0.65 → 0.58 (capped at 160 lux)
        t = min(1.0, (brightness - 80) / 80.0)
        return 0.65 - t * (0.65 - 0.58)


def _dynamic_small_conf_threshold(brightness: float) -> float:
    """
    Small-person confidence threshold also scales with brightness.
    In dark scenes small blobs are almost always noise.
    """
    if brightness < 40:
        return 0.82
    elif brightness < 80:
        t = (brightness - 40) / 40.0
        return 0.82 - t * (0.82 - 0.72)
    else:
        t = min(1.0, (brightness - 80) / 80.0)
        return 0.72 - t * (0.72 - 0.65)


class PersonDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.use_gpu = False
        self.model   = None
        self.classes = [0]
        self.lock    = threading.Lock()  # Serializes YOLO calls across camera threads

        onnx_path = model_path.replace('.pt', '.onnx')

        if hw.dml_available:
            try:
                import onnxruntime as ort
                if not os.path.exists(onnx_path):
                    logger.info(f"[Detector] Exporting {model_path} to ONNX for GPU offloading...")
                    from ultralytics import YOLO
                    YOLO(model_path).export(format='onnx', imgsz=640, simplify=True)

                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(onnx_path, providers=providers)
                self.use_gpu = True
                logger.info(f"[Detector] {model_path.replace('.pt', '')} on GPU (DirectML ONNX)")
            except Exception as e:
                logger.error(f"[Detector] GPU initialization failed: {e}")
                self._init_cpu(model_path)
        else:
            self._init_cpu(model_path)

    def _init_cpu(self, model_path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to('cpu')
            logger.info(f"[Detector] {model_path.replace('.pt', '')} on CPU")
        except Exception:
            logger.error("[Detector] Failed to load YOLO on CPU")

    def detect(self, frame):
        with self.lock:
            if self.use_gpu:
                return self._detect_onnx(frame)
            elif self.model:
                return self._detect_yolo(frame)
        return []

    def _detect_onnx(self, frame):
        fh, fw = frame.shape[:2]
        input_size = 640

        # ── Dynamic threshold: raise confidence in dark/dim scenes ──────
        # Dark frames produce IR-noise blobs and shadow artefacts that YOLO
        # misclassifies as people.  Measuring brightness once per frame is
        # cheap (~0.2 ms) and lets us adapt without manual tuning.
        brightness     = _frame_brightness(frame)
        conf_threshold = _dynamic_conf_threshold(brightness)
        small_conf_thr = _dynamic_small_conf_threshold(brightness)

        # Letterbox (keeps aspect ratio) — prevents stretching artefacts
        r  = min(input_size / fw, input_size / fh)
        nw = int(fw * r)
        nh = int(fh * r)
        resized = cv2.resize(frame, (nw, nh))

        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_top  = (input_size - nh) // 2
        pad_left = (input_size - nw) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        blob = canvas.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        output  = outputs[0][0].transpose()  # [8400, 84]

        detections = []

        for row in output:
            # YOLOv8 ONNX: index 4 is the objectness × class-0 score
            conf = float(row[4])
            if conf < conf_threshold:
                continue

            x, y, w, h = row[:4]
            # De-letterbox: subtract padding then divide by scale factor
            cx = x - pad_left
            cy = y - pad_top
            x1 = (cx - w / 2) / r
            y1 = (cy - h / 2) / r
            bw = w / r
            bh = h / r

            # Tiered size filter with dynamic confidence gating.
            # In dark scenes small_conf_thr is raised further to suppress
            # noise blobs that are common in low-light / IR footage.
            if bh < (fh * 0.05):
                continue  # Too small — likely noise
            elif bh < (fh * 0.10):
                if conf < small_conf_thr:
                    continue
            elif bh > (fh * 0.95):
                continue  # Too large — likely camera artifact

            # Aspect ratio filter: people are taller than wide (1.2–5.0).
            # In dark scenes tighten the lower bound slightly (1.3) because
            # dark blobs tend to be more square/wide than real people.
            aspect = bh / max(bw, 1)
            ar_min = 1.3 if brightness < 60 else 1.2
            if aspect < ar_min or aspect > 5.0:
                continue

            detections.append(([float(x1), float(y1), float(bw), float(bh)], conf, 'person'))

        if not detections:
            return []

        boxes = [d[0] for d in detections]
        confs  = [d[1] for d in detections]

        # Tighter NMS IoU threshold to prevent duplicate detections.
        # 0.45 reduces cases where one person generates two boxes (head + body).
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.45)

        return [detections[i] for i in indices] if len(indices) > 0 else []

    def _detect_yolo(self, frame):
        # Dynamic threshold on CPU path too
        brightness     = _frame_brightness(frame)
        conf_threshold = _dynamic_conf_threshold(brightness)
        small_conf_thr = _dynamic_small_conf_threshold(brightness)

        results = self.model.predict(frame, classes=[0], conf=conf_threshold, imgsz=640, verbose=False)
        detections = []
        fh, fw = frame.shape[:2]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1

                if bh < (fh * 0.05):
                    continue
                elif bh < (fh * 0.10):
                    if conf < small_conf_thr:
                        continue
                elif bh > (fh * 0.95):
                    continue

                aspect = bh / max(bw, 1)
                ar_min = 1.3 if brightness < 60 else 1.2
                if aspect < ar_min or aspect > 5.0:
                    continue

                detections.append(([x1, y1, bw, bh], conf, 'person'))
        return detections
