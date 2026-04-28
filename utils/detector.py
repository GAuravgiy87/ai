import cv2
import numpy as np
import logging
import os
import subprocess
import threading
from utils.hw_manager import hw

logger = logging.getLogger(__name__)

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

        # Issue 1 Fix: Raised confidence threshold to reduce false positives
        # 0.58 significantly reduces tree/bike/shadow detections while maintaining
        # accuracy for actual people. YOLOv8n at 0.30-0.45 was too permissive.
        conf_threshold = 0.58

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

            # Issue 2 Fix: Tiered size filter with confidence gating
            # Small detections (5-10% frame height) require high confidence
            # to avoid false positives from small blobs (bushes, bike parts)
            if bh < (fh * 0.05):
                continue  # Too small — likely noise
            elif bh < (fh * 0.10):
                if conf < 0.65:  # Small person needs high confidence
                    continue
            elif bh > (fh * 0.95):
                continue  # Too large — likely camera artifact
            
            # Issue 4 Fix: Aspect ratio filter to eliminate bikes/trees
            # People are taller than wide (aspect 1.5-4.5)
            # Bikes/trees produce wide or square boxes (aspect 0.5-1.2)
            aspect = bh / max(bw, 1)
            if aspect < 1.2 or aspect > 5.0:
                continue

            detections.append(([float(x1), float(y1), float(bw), float(bh)], conf, 'person'))

        if not detections:
            return []

        boxes = [d[0] for d in detections]
        confs  = [d[1] for d in detections]

        # Issue 8 Fix: Tighter NMS IoU threshold to prevent duplicate detections
        # 0.45 reduces cases where one person generates two boxes (head + body)
        # Combined with higher confidence threshold, maintains crowd detection
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.45)

        return [detections[i] for i in indices] if len(indices) > 0 else []

    def _detect_yolo(self, frame):
        # Issue 1 Fix: Raised CPU path confidence to 0.45 (matching ONNX logic)
        results = self.model.predict(frame, classes=[0], conf=0.45, imgsz=640, verbose=False)
        detections = []
        fh, fw = frame.shape[:2]
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1
                
                # Issue 2 Fix: Apply same tiered size filter as ONNX path
                if bh < (fh * 0.05):
                    continue
                elif bh < (fh * 0.10):
                    if conf < 0.65:
                        continue
                elif bh > (fh * 0.95):
                    continue
                
                # Issue 4 Fix: Apply aspect ratio filter
                aspect = bh / max(bw, 1)
                if aspect < 1.2 or aspect > 5.0:
                    continue
                
                detections.append(([x1, y1, bw, bh], conf, 'person'))
        return detections
