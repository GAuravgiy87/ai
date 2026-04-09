import cv2
import numpy as np
import os


def _nms(detections, iou_thresh=0.45):
    """Non-max suppression to remove duplicate boxes for the same person."""
    if not detections:
        return detections
    boxes  = np.array([d[0] for d in detections], dtype=float)  # [x,y,w,h]
    scores = np.array([d[1] for d in detections], dtype=float)
    # convert to x1y1x2y2
    x1 = boxes[:, 0];  y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2];  y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return [detections[k] for k in keep]


class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.use_yolo = False
        self.use_opencv_dnn = False
        self.ov_model_path = model_path.replace('.pt', '_openvino_model')

        try:
            from ultralytics import YOLO
            import torch

            if torch.cuda.is_available():
                self.device = '0'
            else:
                try:
                    import platform
                    if platform.system() == 'Windows':
                        import torch_directml
                        self.device = torch_directml.device()
                    else:
                        self.device = 'cpu'
                except Exception:
                    self.device = 'cpu'

            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices
                print(f"[PersonDetector] OpenVINO devices: {devices}")

                for dev in devices:
                    try:
                        name = core.get_property(dev, "FULL_DEVICE_NAME")
                        print(f"  -> {dev}: {name}")
                    except: pass

                # Auto-select dGPU → iGPU → CPU
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
                    print(f"[PersonDetector] ⚠️  No dGPU, using iGPU → {ov_device}")
                else:
                    ov_device = "CPU"
                    print("[PersonDetector] ⚠️  No GPU, using CPU")

                # Load existing model — never re-export
                # Use glob so it works regardless of exact XML filename Ultralytics chose
                import glob as _glob
                xml_files = _glob.glob(os.path.join(self.ov_model_path, '*.xml'))
                if not xml_files:
                    print(f"[PersonDetector] Exporting to OpenVINO (one-time)...")
                    YOLO(model_path).export(format='openvino', imgsz=1280)
                    print(f"[PersonDetector] Export complete.")
                else:
                    print(f"[PersonDetector] OpenVINO model found ({os.path.basename(xml_files[0])}) — skipping export.")

                self.model  = YOLO(self.ov_model_path, task='detect')
                self.device = ov_device
                self.is_openvino = True
                print(f"[PersonDetector] Inference device: {self.device}")

            except Exception as ov_err:
                print(f"[PersonDetector] OpenVINO unavailable: {ov_err}")
                self.model = YOLO(model_path).to(self.device)
                self.is_openvino = False
                print(f"[PersonDetector] Using YOLOv8 on {self.device}")

            self.classes = [0]  # person only
            self.use_yolo = True

        except Exception as e:
            print(f"[PersonDetector] YOLO unavailable: {e} — falling back to HOG")
            self.is_openvino = False
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.use_opencv_dnn = True

    def detect(self, frame):
        if self.use_yolo:
            try:
                return self._detect_yolo(frame)
            except Exception as e:
                print(f"[PersonDetector] YOLO failed: {e} — switching to HOG")
                self.use_yolo = False
                self._init_opencv_detector()
        if self.use_opencv_dnn:
            return self._detect_opencv(frame)
        return []

    def _detect_yolo(self, frame):
        """
        Full-frame person detection.
        - imgsz=1280  : catches distant/small persons in wide CCTV frames
        - conf=0.15   : low threshold so no one is missed; NMS cleans duplicates
        - agnostic_nms: prevents same person getting two boxes from different anchors
        """
        results = self.model.predict(
            frame,
            classes=self.classes,
            conf=0.15,
            imgsz=1280,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
            # OpenVINO models must NOT receive a device arg — device is set at load time.
            # For plain PyTorch models pass the device normally.
            **({} if self.is_openvino else {"device": self.device})
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1

                # Minimum pixel size — allow tiny distant persons (15px tall)
                if bh < 15 or bw < 8:
                    continue

                # Aspect ratio — wide range to cover all poses/angles
                ar = bh / bw
                if ar < 0.3 or ar > 7.0:
                    continue

                detections.append(([x1, y1, bw, bh], conf, 'person'))

        # Post-YOLO NMS — removes any remaining duplicate boxes
        return _nms(detections, iou_thresh=0.45)

    def _detect_opencv(self, frame):
        """HOG+SVM fallback."""
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4),
            scale=1.05, useMeanshiftGrouping=False
        )

        detections = []
        for i, (x, y, wr, hr) in enumerate(rects):
            conf = float(weights[i]) if i < len(weights) else 0.5
            if scale != 1.0:
                x, y, wr, hr = int(x/scale), int(y/scale), int(wr/scale), int(hr/scale)
            if hr < 40 or wr < 10:
                continue
            if hr / wr < 1.1 or hr / wr > 6.0:
                continue
            detections.append(([x, y, wr, hr], conf, 'person'))

        return _nms(detections, iou_thresh=0.45)
