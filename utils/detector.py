import cv2
import numpy as np
import logging
import os
import subprocess
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
        # Preprocess: resize to 640x640, BGR to RGB, normalize
        blob = cv2.resize(frame, (640, 640))
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        
        # Inference
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        output = outputs[0][0]
        
        # Post-process (Simplified YOLOv8 parsing)
        # output is [84, 8400] -> [x,y,w,h, cls0...cls79]
        output = output.transpose()
        detections = []
        for row in output:
            conf = row[4:].max()
            if conf < 0.45: continue
            cls = row[4:].argmax()
            if cls != 0: continue # person
            
            x, y, w, h = row[:4]
            # Map back to frame size
            x1 = (x - w/2) * fw / 640
            y1 = (y - h/2) * fh / 640
            x2 = (x + w/2) * fw / 640
            y2 = (y + h/2) * fh / 640
            
            # Size gates
            if (h * fh / 640) < 40: continue
            
            detections.append(([x1, y1, (x2-x1), (y2-y1)], float(conf), 'person'))
        
        # NMS
        if not detections: return []
        boxes = [d[0] for d in detections]
        confs = [d[1] for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.45, 0.45)
        return [detections[i] for i in indices] if len(indices) > 0 else []

    def _detect_yolo(self, frame):
        results = self.model.predict(frame, classes=[0], conf=0.45, imgsz=640, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        return detections
