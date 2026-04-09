import cv2
import numpy as np
import os

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.use_yolo = False
        self.use_opencv_dnn = False
        self.ov_model_path = model_path.replace('.pt', '_openvino_model')
        
        try:
            from ultralytics import YOLO
            import torch

            # Device Discovery for Hardware Acceleration
            if torch.cuda.is_available():
                self.device = '0'  # CUDA/ROCm
            else:
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                except ImportError:
                    self.device = 'cpu'

            # Use OpenVINO if available
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

                # ── Auto device selection: dGPU first, then iGPU, then CPU ──
                # Runs automatically every time the app starts — no scripts needed.
                discrete_gpus, integrated_gpus = [], []
                for d in [x for x in devices if x.startswith("GPU")]:
                    try:
                        dtype = str(core.get_property(d, "DEVICE_TYPE")).upper()
                        name  = ""
                        try: name = core.get_property(d, "FULL_DEVICE_NAME").upper()
                        except: pass
                        is_discrete = ("DISCRETE" in dtype or
                                       "AMD" in name or "RADEON" in name or "RX " in name)
                        (discrete_gpus if is_discrete else integrated_gpus).append((d, name))
                    except:
                        integrated_gpus.append((d, ""))

                if discrete_gpus:
                    ov_device = discrete_gpus[0][0]
                    print(f"[PersonDetector] ✅ Dedicated GPU → {ov_device} ({discrete_gpus[0][1]})")
                elif integrated_gpus:
                    ov_device = integrated_gpus[0][0]
                    print(f"[PersonDetector] ⚠️  No dGPU found, using iGPU → {ov_device}")
                else:
                    ov_device = "CPU"
                    print("[PersonDetector] ⚠️  No GPU found, using CPU")

                # ── Model: load existing export, never re-export ──
                # Check for the actual .xml model file inside the export folder
                ov_xml = os.path.join(self.ov_model_path, 'yolov8n.xml')
                model_ready = os.path.isfile(ov_xml)

                if not model_ready:
                    print(f"[PersonDetector] OpenVINO model not found at {ov_xml}, exporting...")
                    tmp_model = YOLO(model_path)
                    tmp_model.export(format='openvino', imgsz=800)
                    print(f"[PersonDetector] Export complete.")
                else:
                    print(f"[PersonDetector] OpenVINO model already exists — skipping export.")

                self.model = YOLO(self.ov_model_path, task='detect')
                self.device = ov_device
                print(f"[PersonDetector] Final Inference Device: {self.device}")

            except Exception as ov_err:
                print(f"[PersonDetector] OpenVINO not available: {ov_err}")
                self.model = YOLO(model_path).to(self.device)
                print(f"[PersonDetector] Using YOLOv8 on {self.device}")

            self.classes = [0]  # person only
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
                
                # Class mapping — persons only
                class_map = {0: 'person'}
                label = class_map.get(cls_id, 'person')

                if bh < 20 or bw < 10:
                    continue

                # Person aspect ratio filter (sitting/crouching included)
                aspect_ratio = bh / bw
                if aspect_ratio < 0.4 or aspect_ratio > 6.0:
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
