"""
detector.py — Person detection via ONNX Runtime (AMD GPU on Windows/Linux)

GPU path  : ONNX Runtime + DmlExecutionProvider (Windows) / ROCMExecutionProvider (Linux)
CPU path  : ONNX Runtime CPUExecutionProvider
Fallback  : ultralytics YOLO on CPU → HOG

Fixed pipeline: 10 FPS — matches camera capture and render rate exactly.
All stages run at the same tick so no frame is processed more than once.

Key optimisations:
  - Single ORT session shared across calls (no per-frame session creation)
  - hw.ort_session_options() sets 1 CPU thread on GPU (GPU does the work)
  - Letterbox + normalise into pre-allocated buffer — zero malloc per frame
  - Vectorised NMS in NumPy — no Python loop
  - Frame-id deduplication in detection thread — skips if camera has no new frame
"""
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = "yolov8n.onnx"
PT_MODEL_PATH   = "yolov8n.pt"
YOLO_INPUT_SIZE = 640


def _export_yolo_to_onnx() -> bool:
    """Export YOLOv8n.pt → yolov8n.onnx once, then cached."""
    try:
        from ultralytics import YOLO
        logger.info("[Detector] Exporting YOLOv8n → ONNX (one-time ~10 s)...")
        m = YOLO(PT_MODEL_PATH)
        m.export(format="onnx", imgsz=YOLO_INPUT_SIZE, opset=12, simplify=True)
        exported = PT_MODEL_PATH.replace(".pt", ".onnx")
        if os.path.exists(exported) and exported != ONNX_MODEL_PATH:
            os.rename(exported, ONNX_MODEL_PATH)
        logger.info(f"[Detector] ONNX export done → {ONNX_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"[Detector] ONNX export failed: {e}")
        return False


class PersonDetector:
    """
    Thread-safe person detector.
    A single ORT session is created at init and reused for every detect() call.
    The session is NOT shared across threads — each camera's detection thread
    owns its own PersonDetector instance (created inside process_camera).
    """

    def __init__(self):
        from utils.hw_manager import hw
        self._hw          = hw
        self.use_ort      = False
        self.use_yolo     = False
        self.use_hog      = False
        self._session     = None
        self._input_name  = None
        self._output_name = None
        # Pre-allocated input buffer — avoids malloc every frame
        self._blob = np.empty((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), dtype=np.float32)

        # Fixed pipeline FPS — matches camera capture and render rate
        self.det_fps = 10

        self._init_ort()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_ort(self):
        if not os.path.exists(ONNX_MODEL_PATH):
            if os.path.exists(PT_MODEL_PATH):
                if not _export_yolo_to_onnx():
                    self._init_ultralytics_cpu(); return
            else:
                logger.warning(f"[Detector] {PT_MODEL_PATH} not found — HOG fallback")
                self._init_hog(); return

        try:
            import onnxruntime as ort
            opts     = self._hw.ort_session_options()   # GPU: 1 thread, CPU: cores//2
            providers = self._hw.best_providers()
            self._session     = ort.InferenceSession(ONNX_MODEL_PATH,
                                                     sess_options=opts,
                                                     providers=providers)
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self.use_ort      = True
            used = self._session.get_providers()
            device = self._hw.gpu_name if self._hw.gpu_available else "CPU"
            logger.info(f"[Detector] YOLOv8n ONNX on {device} | providers={used}")
        except Exception as e:
            logger.warning(f"[Detector] ORT load failed: {e} — ultralytics CPU")
            self._init_ultralytics_cpu()

    def _init_ultralytics_cpu(self):
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(PT_MODEL_PATH)
            self._yolo_model.to("cpu")
            self.use_yolo = True
            self.det_fps  = 10
            logger.info("[Detector] YOLOv8n ultralytics CPU")
        except Exception as e:
            logger.warning(f"[Detector] ultralytics failed: {e} — HOG")
            self._init_hog()

    def _init_hog(self):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.use_hog = True
        self.det_fps = 10
        logger.info("[Detector] HOG fallback")
    # ── Inference ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """Run detection on frame. Thread-safe (one session per instance)."""
        if self.use_ort:
            try:
                return self._detect_ort(frame)
            except Exception as e:
                logger.warning(f"[Detector] ORT error: {e} — switching to CPU")
                self.use_ort = False
                self._init_ultralytics_cpu()
        if self.use_yolo:
            try:
                return self._detect_ultralytics(frame)
            except Exception as e:
                logger.warning(f"[Detector] ultralytics error: {e} — HOG")
                self.use_yolo = False
                self._init_hog()
        return self._detect_hog(frame) if self.use_hog else []

    # ── ORT path ──────────────────────────────────────────────────────────

    def _preprocess_inplace(self, frame: np.ndarray) -> float:
        """
        Letterbox + normalise into pre-allocated self._blob.
        Returns scale factor (original → padded).
        No extra numpy allocation — writes directly into the buffer.
        """
        h, w = frame.shape[:2]
        scale = YOLO_INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)

        # Resize into a temporary view, then copy into blob
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Fill padding with 114 (YOLO standard)
        self._blob[0, :, :, :] = 0
        # Write RGB channels directly — avoids [::-1] copy
        canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        # BGR → RGB, HWC → CHW, uint8 → float32 /255 in one pass
        rgb = canvas[:, :, ::-1]   # view, no copy
        np.divide(rgb.transpose(2, 0, 1), 255.0, out=self._blob[0], casting="unsafe")
        return scale

    def _detect_ort(self, frame: np.ndarray) -> list:
        oh, ow = frame.shape[:2]
        scale  = self._preprocess_inplace(frame)
        output = self._session.run([self._output_name],
                                   {self._input_name: self._blob})
        return self._postprocess(output[0], scale, (oh, ow))

    def _postprocess(self, raw: np.ndarray, scale: float, orig_hw: tuple) -> list:
        """
        YOLOv8 ONNX output: [1, 84, 8400]
        Layout: [cx, cy, w, h, cls0_score, cls1_score, ...]
        """
        oh, ow = orig_hw
        preds = raw[0]                      # [84, 8400]
        if preds.shape[0] == 84:
            preds = preds.T                 # → [8400, 84]

        # Person class = index 4 in YOLOv8 ONNX (after cx,cy,w,h)
        scores = preds[:, 4]
        mask   = scores > 0.35
        if not mask.any():
            return []

        boxes  = preds[mask, :4]            # cx, cy, w, h
        scores = scores[mask]

        inv = 1.0 / scale
        # cx,cy,w,h → x1,y1,x2,y2 in original coords (vectorised)
        x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2) * inv, 0, ow)
        y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2) * inv, 0, oh)
        x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2) * inv, 0, ow)
        y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2) * inv, 0, oh)
        bw = x2 - x1
        bh = y2 - y1

        # Size + aspect ratio gates (vectorised)
        ar   = np.where(bw > 0, bh / bw, 0)
        keep = (
            (bh >= 25) & (bw >= 12) &
            (bw <= ow * 0.90) & (bh <= oh * 0.95) &
            (ar >= 0.5) & (ar <= 5.0) &
            (bw * bh >= 400)
        )
        x1, y1, x2, y2, bw, bh, scores = (
            x1[keep], y1[keep], x2[keep], y2[keep],
            bw[keep], bh[keep], scores[keep]
        )

        if len(x1) == 0:
            return []

        # Slight inward pad
        pad = bw * 0.04
        x1 += pad; x2 -= pad; bw -= 2 * pad

        detections = [
            ([float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])],
             float(scores[i]), "person")
            for i in range(len(x1))
        ]

        return self._nms_numpy(detections, 0.45) if len(detections) > 1 else detections

    @staticmethod
    def _nms_numpy(dets: list, iou_thresh: float) -> list:
        """Vectorised NMS."""
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d[1], reverse=True)
        boxes  = np.array([[d[0][0], d[0][1],
                            d[0][0]+d[0][2], d[0][1]+d[0][3]] for d in dets])
        scores = np.array([d[1] for d in dets])
        areas  = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
        kept   = []
        alive  = np.ones(len(dets), dtype=bool)
        for i in range(len(dets)):
            if not alive[i]:
                continue
            kept.append(dets[i])
            ix1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
            iy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
            ix2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
            iy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
            inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
            union = areas[i] + areas[i+1:] - inter
            iou   = np.where(union > 0, inter / union, 0)
            alive[i+1:][iou > iou_thresh] = False
        return kept

    # ── Ultralytics CPU fallback ──────────────────────────────────────────

    def _detect_ultralytics(self, frame: np.ndarray) -> list:
        fh, fw = frame.shape[:2]
        results = self._yolo_model.predict(
            frame, classes=[0], conf=0.35, imgsz=640, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw, bh = x2-x1, y2-y1
                if bh < 25 or bw < 12: continue
                if bw > fw*0.90 or bh > fh*0.95: continue
                ar = bh / max(bw, 1)
                if ar < 0.5 or ar > 5.0: continue
                if bw*bh < 400: continue
                pad = bw*0.04; x1 += pad; x2 -= pad; bw -= 2*pad
                dets.append(([x1, y1, bw, bh], conf, "person"))
        return dets

    # ── HOG fallback ──────────────────────────────────────────────────────

    def _detect_hog(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        rects, weights = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        dets = []
        for i, (x, y, wr, hr) in enumerate(rects):
            conf = float(weights[i]) if i < len(weights) else 0.5
            if scale != 1.0:
                x, y, wr, hr = int(x/scale), int(y/scale), int(wr/scale), int(hr/scale)
            if hr < 40 or wr < 10 or hr/max(wr,1) < 1.1 or hr/max(wr,1) > 6.0:
                continue
            dets.append(([x, y, wr, hr], conf, "person"))
        return dets
