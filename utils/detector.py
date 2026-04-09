import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Try to use YOLO if available, fallback to OpenCV DNN
        self.use_yolo = False
        self.use_opencv_dnn = False
        self.ov_model_path = model_path.replace('.pt', '_openvino_model')
        
        try:
            from ultralytics import YOLO
            import torch
            import os
            
            # Device Discovery for Hardware Acceleration
            if torch.cuda.is_available():
                self.device = '0' # CUDA/ROCm
            else:
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                except ImportError:
                    self.device = 'cpu'

            # Use OpenVINO if available (Best for Intel CPU/iGPU and AMD GPUs)
            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices
                print(f"[PersonDetector] OpenVINO Discovery: Found devices {devices}")
                
                for dev in devices:
                    try:
                        name = core.get_property(dev, "FULL_DEVICE_NAME")
                        print(f"  -> {dev}: {name}")
                    except: pass

                # Priority: Environment Override > MULTI GPU > Single GPU > CPU
                ov_device = os.getenv("OPENVINO_DEVICE", "")
                
                if not ov_device:
                    gpu_devices = [d for d in devices if "GPU" in d]
                    # Sort GPUs to put Discrete usually at the top (GPU.1, GPU.0)
                    gpu_devices.sort(reverse=True)
                    
                    if len(gpu_devices) > 1:
                        # Use MULTI plugin to load balance across all found GPUs
                        ov_device = f"MULTI:{','.join(gpu_devices)}"
                    elif len(gpu_devices) == 1:
                        # Fallback to MULTI:GPU,CPU to leverage the i7-8700 CPU alongside iGPU
                        ov_device = f"MULTI:{gpu_devices[0]},CPU"
                    else:
                        ov_device = "CPU"
                
                if not os.path.exists(self.ov_model_path):
                    print(f"[PersonDetector] Exporting {model_path} to OpenVINO ({ov_device})...")
                    tmp_model = YOLO(model_path)
                    tmp_model.export(format='openvino', imgsz=800)
                
                self.model = YOLO(self.ov_model_path, task='detect')
                self.device = ov_device
                print(f"[PersonDetector] Final Inference Device: {self.device}")
                print(f"[PersonDetector] Using YOLOv8 with OpenVINO on {self.device} (Available: {devices})")
            except Exception as ov_err:
                print(f"[PersonDetector] OpenVINO acceleration failed or not found: {ov_err}")
                self.model = YOLO(model_path).to(self.device)
                print(f"[PersonDetector] Using YOLOv8 on {self.device}")

            self.classes = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
            self.use_yolo = True
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
        # LOWER confidence (0.20) to maintain tracks even in difficult conditions
        # Higher imgsz (800) for better detection of distant/small persons
        results = self.model.predict(frame, classes=self.classes, conf=0.20, imgsz=800, verbose=False, device=self.device)
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
                    # RELAXED aspect ratios to catch sitting/crouching persons
                    if aspect_ratio < 0.4 or aspect_ratio > 6.0:
                        continue
                else:
                    # Vehicles can be very wide (cars) or tall (trucks)
                    if aspect_ratio < 0.15 or aspect_ratio > 4.0:
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
