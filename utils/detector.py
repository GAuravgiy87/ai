"""
detector.py — Person detection on CPU (YOLOv8n)
YOLO runs on CPU — YOLOv8n is fast enough and RX 550 ROCm support is unstable.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.use_yolo = False
        self.use_opencv_dnn = False

        try:
            from ultralytics import YOLO
            import torch
            # Force CPU — stable across all setups
            self.model = YOLO(model_path)
            self.model.to("cpu")
            self.classes = [0]
            self.use_yolo = True
            logger.info("[Detector] YOLOv8n on CPU")
        except Exception as e:
            logger.warning(f"[Detector] YOLO unavailable: {e} — falling back to HOG")
            self._init_hog()

    def _init_hog(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.use_opencv_dnn = True

    def detect(self, frame):
        if self.use_yolo:
            try:
                return self._detect_yolo(frame)
            except Exception as e:
                logger.warning(f"[Detector] YOLO failed: {e} — switching to HOG")
                self.use_yolo = False
                self._init_hog()
        return self._detect_hog(frame) if self.use_opencv_dnn else []

    def _detect_yolo(self, frame):
        fh, fw = frame.shape[:2]
        results = self.model.predict(
            frame, classes=self.classes, conf=0.35, imgsz=800, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1

                # --- Size gates ---
                # Too small: noise / very distant
                if bh < 25 or bw < 12:
                    continue
                # Too large: gate, wall, full-frame background
                if bw > fw * 0.90 or bh > fh * 0.95:
                    continue

                # --- Aspect ratio ---
                # Full standing person: ~1.8–3.5
                # Seated person:        ~0.8–2.0  (wider relative to height)
                # Half-visible (torso): ~0.6–2.5
                # We accept anything from 0.5 upward to catch seated/partial views.
                # Upper bound 5.0 still rejects thin poles.
                ar = bh / bw
                if ar < 0.5 or ar > 5.0:
                    continue

                # --- Minimum area ---
                if bw * bh < 400:
                    continue

                # --- Edge clipping: only reject boxes at the very bottom edge
                # that are tiny — likely a pole top, not a seated/partial person.
                # Raised threshold: must be < 15% of frame height (was 25%)
                if y2 >= fh * 0.97 and bh < fh * 0.15:
                    continue

                # Shrink box inward slightly to hug the body
                pad_x     = bw * 0.04
                pad_y_top = bh * 0.01
                pad_y_bot = bh * 0.01
                x1 += pad_x;  x2 -= pad_x
                y1 += pad_y_top; y2 -= pad_y_bot
                bw = x2 - x1;  bh = y2 - y1
                detections.append(([x1, y1, bw, bh], conf, 'person'))
        return detections

    def _detect_hog(self, frame):
        detections = []
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for i, (x, y, wr, hr) in enumerate(rects):
            conf = float(weights[i]) if i < len(weights) else 0.5
            if scale != 1.0:
                x, y, wr, hr = int(x/scale), int(y/scale), int(wr/scale), int(hr/scale)
            if hr < 40 or wr < 10 or hr/wr < 1.1 or hr/wr > 6.0:
                continue
            detections.append(([x, y, wr, hr], conf, 'person'))
        return detections
