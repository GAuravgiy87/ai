import cv2
import numpy as np
import logging
import os
import subprocess
import threading
from utils.hw_manager import hw

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.use_gpu = False
        self.model = None
        self.classes = [0]
        self.lock = threading.Lock() # Serializes YOLO calls across camera threads
        
        onnx_path = model_path.replace('.pt', '.onnx')
        
        if hw.dml_available:
            try:
                import onnxruntime as ort
                # Export to ONNX if needed
                if not os.path.exists(onnx_path):
                    logger.info("[Detector] Exporting YOLOv8n to ONNX for GPU offloading...")
                    from ultralytics import YOLO
                    YOLO(model_path).export(format='onnx', imgsz=640, simplify=True)
                
                # Load with DirectML
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(onnx_path, providers=providers)
                self.use_gpu = True
                logger.info(f"[Detector] YOLOv8n on GPU (DirectML ONNX)")
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
            logger.info("[Detector] YOLOv8n on CPU")
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
        
        # Letterbox resizing (keep aspect ratio by padding)
        # This prevents stretching that causes false positives (trees/shadows)
        r = min(input_size / fw, input_size / fh)
        nw, nh = int(fw * r), int(fh * r)
        resized = cv2.resize(frame, (nw, nh))
        
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[(input_size - nh) // 2 : (input_size - nh) // 2 + nh, (input_size - nw) // 2 : (input_size - nw) // 2 + nw, :] = resized
        
        blob = canvas.transpose(2, 0, 1) # HWC to CHW
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        
        # Inference
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        output = outputs[0][0].transpose() # [8400, 84]
        
        detections = []
        conf_threshold = 0.35 # Higher to avoid trees and air vehicles
        
        for row in output:
            conf = row[4] # YOLOv8n person score is at index 4 (class 0)
            if conf < conf_threshold: continue
            
            x, y, w, h = row[:4]
            
            # Map back from canvas (640x640) to original frame
            # 1. Remove padding
            cx = x - (input_size - nw) / 2
            cy = y - (input_size - nh) / 2
            # 2. Scale back
            x1 = (cx - w/2) / r
            y1 = (cy - h/2) / r
            bw = w / r
            bh = h / r
            
            # Ignore detections if they are extremely small or outside frame
            if bh < (fh * 0.05) or bh > (fh * 0.95): continue 
            
            detections.append(([float(x1), float(y1), float(bw), float(bh)], float(conf), 'person'))
        
        # NMS
        if not detections: return []
        boxes = [d[0] for d in detections]
        confs = [d[1] for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.45)
        
        return [detections[i] for i in indices] if len(indices) > 0 else []

    def _detect_yolo(self, frame):
        # Fallback YOLOv8 threshold also increased
        results = self.model.predict(frame, classes=[0], conf=0.35, imgsz=640, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        return detections
