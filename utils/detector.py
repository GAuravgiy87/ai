import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Try to use YOLO if available, fallback to OpenCV DNN
        self.use_yolo = False
        self.use_opencv_dnn = False
        
        try:
            from ultralytics import YOLO
            import torch
            self.device = '0' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(model_path).to(self.device)
            self.classes = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
            self.use_yolo = True
            print(f"[PersonDetector] Using YOLOv8 on {self.device} (Classes: {self.classes})")
        except Exception as e:
            print(f"[PersonDetector] YOLO not available: {e}")
            print("[PersonDetector] Falling back to OpenCV HOG+SVM detector")
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """Initialize OpenCV's HOG+SVM person detector as fallback"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.use_opencv_dnn = True

    def detect(self, frame):
        detections = []
        
        if self.use_yolo:
            try:
                return self._detect_yolo(frame)
            except Exception as e:
                print(f"[PersonDetector] YOLO detection failed: {e}, switching to fallback")
                self.use_yolo = False
                self._init_opencv_detector()
        
        if self.use_opencv_dnn:
            return self._detect_opencv(frame)
            
        return detections

    def _detect_yolo(self, frame):
        """YOLOv8 detection optimized for all person detection scenarios."""
        # Lower confidence to catch distant/small persons
        # Higher imgsz for better detection of small/distant objects
        results = self.model.predict(frame, classes=self.classes, conf=0.35, imgsz=800, verbose=False, device=self.device)
        detections = []
        h, w = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                bw, bh = x2-x1, y2-y1
                cls_id = int(box.cls[0])
                
                # Class mapping
                class_map = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                label = class_map.get(cls_id, 'person')

                # Allow very small detections for distant entities
                if bh < 20 or bw < 10:
                    continue
                    
                # Relaxed aspect ratio for vehicles vs persons
                aspect_ratio = bh / bw
                if label == 'person':
                    if aspect_ratio < 0.7 or aspect_ratio > 5.0:
                        continue
                else:
                    # Vehicles can be very wide (cars) or tall (trucks)
                    if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                        continue
                    
                detections.append(([x1, y1, bw, bh], conf, label))
        return detections

    def _detect_opencv(self, frame):
        """OpenCV HOG+SVM detection as fallback"""
        detections = []
        h, w = frame.shape[:2]
        
        # Resize large frames for faster processing
        scale = 1.0
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small_frame = frame
        
        # Detect people using HOG+SVM
        rects, weights = self.hog.detectMultiScale(
            small_frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
            useMeanshiftGrouping=False
        )
        
        for i, (x, y, w_rect, h_rect) in enumerate(rects):
            conf = float(weights[i]) if i < len(weights) else 0.5
            
            # Scale back to original frame size
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w_rect = int(w_rect / scale)
                h_rect = int(h_rect / scale)
            
            # Filter by size and aspect ratio
            if h_rect < 40 or w_rect < 10:
                continue
            if h_rect / w_rect < 1.1 or h_rect / w_rect > 6.0:
                continue
            
            detections.append(([x, y, w_rect, h_rect], conf, 'person'))
        
        return detections
