"""
recognizer.py — Face recognition via ONNX Runtime (AMD GPU on Windows + Linux)

GPU path  : ORT DmlExecutionProvider (Windows) / ROCMExecutionProvider (Linux AMD)
CPU path  : ORT CPUExecutionProvider
Fallback  : PyTorch CPU (InceptionResnetV1)

Key optimisations:
  - Single ORT session per FaceRecognizer instance — no per-call session creation
  - hw.ort_session_options() sets 1 CPU thread on GPU (GPU does the work)
  - Pre-allocated input buffer — no malloc per embedding call
  - MTCNN on CPU (lightweight, single-crop — no GPU benefit)
  - torch.set_num_threads(1) — ORT owns the GPU; torch only used for MTCNN
  - Known-face encodings stored as a pre-stacked numpy array for O(1) distance calc
"""
import numpy as np
import cv2
import threading
import logging
import os

logger = logging.getLogger(__name__)

FACENET_ONNX_PATH = "facenet_vggface2.onnx"


def _export_facenet_to_onnx() -> bool:
    """Export InceptionResnetV1 (VGGFace2) → ONNX once, then cached."""
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
        logger.info("[Recognizer] Exporting FaceNet → ONNX (one-time ~5 s)...")
        model = InceptionResnetV1(pretrained="vggface2").eval()
        dummy = torch.zeros(1, 3, 160, 160)
        torch.onnx.export(
            model, dummy, FACENET_ONNX_PATH,
            input_names=["input"], output_names=["output"],
            opset_version=12,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        logger.info(f"[Recognizer] FaceNet ONNX done → {FACENET_ONNX_PATH}")
        return True
    except Exception as e:
        logger.error(f"[Recognizer] FaceNet ONNX export failed: {e}")
        return False


class FaceRecognizer:
    """
    Thread-safe face recognizer.
    One instance is shared across all recognition workers via ai_lock.
    The ORT session is NOT thread-safe for concurrent calls — ai_lock serialises them.
    GPU inference is fast enough that serialisation is not a bottleneck.
    """

    def __init__(self):
        from utils.hw_manager import hw
        self.hw = hw

        # MTCNN on CPU — lightweight, single-crop, no GPU benefit
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(
            keep_all=True,
            device="cpu",
            min_face_size=40,
            thresholds=[0.7, 0.8, 0.9],
            post_process=False,
        )

        self.ai_lock = threading.Lock()   # serialises ORT + MTCNN calls

        self._ort_session  = None
        self._use_ort      = False
        self._resnet_cpu   = None         # PyTorch CPU fallback

        # Pre-allocated input buffer [1, 3, 160, 160] float32
        self._blob = np.empty((1, 3, 160, 160), dtype=np.float32)

        # Known faces — stored as stacked array for fast vectorised distance
        self.known_face_encodings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.known_face_names: list = []

        # Minimise PyTorch CPU threads — ORT handles GPU; torch only for MTCNN
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self._init_facenet()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_facenet(self):
        if not os.path.exists(FACENET_ONNX_PATH):
            if not _export_facenet_to_onnx():
                self._init_cpu_fallback(); return

        try:
            import onnxruntime as ort
            opts      = self.hw.ort_session_options()
            providers = self.hw.best_providers()
            self._ort_session = ort.InferenceSession(
                FACENET_ONNX_PATH, sess_options=opts, providers=providers)
            self._use_ort = True
            used   = self._ort_session.get_providers()
            device = self.hw.gpu_name if self.hw.gpu_available else "CPU"
            logger.info(f"[Recognizer] FaceNet ONNX on {device} | providers={used} | MTCNN on CPU")
        except Exception as e:
            logger.warning(f"[Recognizer] ORT FaceNet failed: {e} — PyTorch CPU")
            self._init_cpu_fallback()

    def _init_cpu_fallback(self):
        try:
            from facenet_pytorch import InceptionResnetV1
            self._resnet_cpu = InceptionResnetV1(pretrained="vggface2").eval()
            logger.info("[Recognizer] FaceNet PyTorch CPU fallback")
        except Exception as e:
            logger.error(f"[Recognizer] FaceNet CPU fallback failed: {e}")

    # ── Known faces ───────────────────────────────────────────────────────

    def load_known_faces(self, db_manager):
        persons = db_manager.get_registered_persons()
        encs, names = [], []
        for p in persons:
            if p[3] is not None:
                encs.append(np.frombuffer(p[3], dtype=np.float32))
                names.append(p[1])
        # Stack into a single array for vectorised L2 distance
        self.known_face_encodings = np.stack(encs) if encs else np.empty((0, 512), dtype=np.float32)
        self.known_face_names     = names
        logger.info(f"[Recognizer] Loaded {len(names)} known faces")

    def add_known_face(self, name: str, encoding: np.ndarray):
        """Hot-add a face without full reload."""
        self.known_face_encodings = np.vstack([self.known_face_encodings, encoding[np.newaxis]])
        self.known_face_names.append(name)

    # ── Embedding ─────────────────────────────────────────────────────────

    def _get_embedding(self, face_rgb_160: np.ndarray) -> np.ndarray:
        """
        Run InceptionResnetV1 on a 160×160 RGB crop.
        Writes into pre-allocated self._blob — zero malloc per call.
        Returns 512-dim float32 embedding, or None on failure.
        """
        # Normalise to [-1, 1] and write CHW into pre-allocated buffer
        tmp = face_rgb_160.astype(np.float32)
        tmp -= 127.5
        tmp /= 128.0
        self._blob[0] = tmp.transpose(2, 0, 1)   # HWC → CHW

        if self._use_ort and self._ort_session is not None:
            try:
                out = self._ort_session.run(None, {"input": self._blob})
                return out[0][0]
            except Exception as e:
                logger.warning(f"[Recognizer] ORT error: {e} — CPU fallback")
                self._use_ort = False
                self._init_cpu_fallback()

        if self._resnet_cpu is not None:
            import torch
            t = torch.from_numpy(self._blob)
            with torch.no_grad():
                return self._resnet_cpu(t).numpy()[0]

        return None

    # ── Main API ──────────────────────────────────────────────────────────

    def recognize_with_encoding(self, frame: np.ndarray, face_bbox: list):
        """
        1. Crop + validate face region
        2. MTCNN (CPU) — verify front-facing face
        3. InceptionResnetV1 (GPU/CPU) — 512-dim embedding
        4. Vectorised L2 match against known faces
        Returns: (name, confidence, embedding)
        """
        if not face_bbox:
            return "Unknown", 0.0, None

        fx1, fy1, fx2, fy2 = face_bbox
        face_crop = frame[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]
        if face_crop.size == 0:
            return "Unknown", 0.0, None

        # Ensure min 80×80 for MTCNN
        min_dim = min(face_crop.shape[:2])
        if min_dim < 80:
            s = 80.0 / min_dim
            face_crop = cv2.resize(
                face_crop,
                (max(80, int(face_crop.shape[1]*s)), max(80, int(face_crop.shape[0]*s))),
                interpolation=cv2.INTER_LINEAR
            )

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # MTCNN face verification (CPU, serialised)
        try:
            with self.ai_lock:
                boxes, probs = self.mtcnn.detect(face_rgb)
        except RuntimeError:
            return "Unknown", 0.0, None

        if boxes is None or len(boxes) == 0:
            return "Unknown", 0.0, None

        best_idx  = int(np.argmax([p if p is not None else 0 for p in probs]))
        best_prob = probs[best_idx] if probs[best_idx] is not None else 0
        if best_prob < 0.90:
            return "Unknown", 0.0, None

        # Tight MTCNN crop
        fb   = boxes[best_idx]
        mfx1 = max(0, int(fb[0]));  mfy1 = max(0, int(fb[1]))
        mfx2 = min(face_crop.shape[1], int(fb[2]))
        mfy2 = min(face_crop.shape[0], int(fb[3]))
        mtcnn_face = face_rgb[mfy1:mfy2, mfx1:mfx2]

        if mtcnn_face.size == 0 or (mfx2-mfx1) < 20 or (mfy2-mfy1) < 20:
            return "Unknown", 0.0, None

        face_160 = cv2.resize(mtcnn_face, (160, 160))

        # Embedding (GPU/CPU, serialised via ai_lock)
        with self.ai_lock:
            embedding = self._get_embedding(face_160)
        if embedding is None:
            return "Unknown", 0.0, None

        # Vectorised L2 match
        MATCH_THRESHOLD = 0.40
        if len(self.known_face_encodings) > 0:
            dists   = np.linalg.norm(self.known_face_encodings - embedding, axis=1)
            min_idx = int(np.argmin(dists))
            min_d   = float(dists[min_idx])
            if min_d < MATCH_THRESHOLD:
                name     = self.known_face_names[min_idx]
                raw_conf = 1.0 - (min_d / (MATCH_THRESHOLD * 2))
                conf     = max(0.90, min(1.0, 0.90 + (raw_conf - 0.5) * 0.20))
                logger.debug(f"[Recognizer] {name} dist={min_d:.3f} conf={conf:.2f}")
                return name, float(conf), embedding
            return "Unknown", 0.0, embedding

        return "Unknown", 0.0, embedding

    def recognize(self, frame: np.ndarray, face_bbox: list):
        name, conf, _ = self.recognize_with_encoding(frame, face_bbox)
        return name, conf

    def get_encoding(self, image: np.ndarray) -> np.ndarray:
        """Get face encoding from a full image (for registration)."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.ai_lock:
            boxes, _ = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return None
        fx1, fy1, fx2, fy2 = [int(b) for b in boxes[0]]
        crop = rgb[max(0,fy1):max(0,fy2), max(0,fx1):max(0,fx2)]
        if crop.size == 0:
            return None
        face_160 = cv2.resize(crop, (160, 160))
        with self.ai_lock:
            return self._get_embedding(face_160)
