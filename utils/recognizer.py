"""
recognizer.py — Face recognition on AMD dGPU (ROCm) when available, else CPU.
Uses HardwareManager for dynamic device routing.
"""
import numpy as np
import cv2
import torch
import threading
import logging

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self):
        from utils.hw_manager import hw
        self.hw = hw

        # MTCNN always on CPU — lightweight, no benefit from GPU for single crops
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.mtcnn = MTCNN(
            keep_all=True,
            device="cpu",
            min_face_size=40,             # ignore tiny/distant faces
            thresholds=[0.7, 0.8, 0.9],  # P-Net, R-Net, O-Net — tighter O-Net
            post_process=False,
        )

        # InceptionResnetV1 on AMD dGPU if available
        self._face_device = hw.face_device
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.resnet.to(torch.device(self._face_device))
        # Disable autograd globally — saves memory + speeds up inference
        torch.set_grad_enabled(False)
        # Limit PyTorch thread count — prevents over-subscription on CPU
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        logger.info(f"[Recognizer] FaceNet on {self._face_device} | MTCNN on cpu")

        self.ai_lock = threading.Lock()
        self.known_face_encodings = []
        self.known_face_names = []

    def _get_resnet_device(self) -> torch.device:
        """Dynamically pick best device based on current GPU load."""
        best = self.hw.best_face_device()
        if best != self._face_device:
            # Migrate model to new device
            try:
                self.resnet.to(torch.device(best))
                self._face_device = best
                logger.info(f"[Recognizer] Migrated FaceNet to {best}")
            except Exception as e:
                logger.warning(f"[Recognizer] Device migration failed: {e}")
        return torch.device(self._face_device)

    def load_known_faces(self, db_manager):
        persons = db_manager.get_registered_persons()
        self.known_face_encodings = []
        self.known_face_names = []
        for person in persons:
            if person[3] is not None:
                enc = np.frombuffer(person[3], dtype=np.float32)
                self.known_face_encodings.append(enc)
                self.known_face_names.append(person[1])
        logger.info(f"[Recognizer] Loaded {len(self.known_face_names)} known faces")

    def recognize_with_encoding(self, frame, face_bbox):
        """
        1. MTCNN (CPU) — verify front-facing face exists
        2. InceptionResnetV1 (AMD dGPU / CPU) — generate embedding
        3. L2 distance match against known faces
        """
        if not face_bbox:
            return "Unknown", 0.0, None

        fx1, fy1, fx2, fy2 = face_bbox
        face_crop = frame[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]
        if face_crop.size == 0:
            return "Unknown", 0.0, None

        # Step 1: MTCNN on CPU — verify real front-facing face
        # Ensure crop is large enough for MTCNN (min 80x80 to avoid torch.cat on empty list)
        min_dim = min(face_crop.shape[:2])
        if min_dim < 80:
            scale = 80.0 / min_dim
            new_w = max(80, int(face_crop.shape[1] * scale))
            new_h = max(80, int(face_crop.shape[0] * scale))
            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        try:
            with self.ai_lock:
                boxes, probs = self.mtcnn.detect(face_rgb)
        except RuntimeError:
            # torch.cat on empty list — no face candidates at any scale
            return "Unknown", 0.0, None

        if boxes is None or len(boxes) == 0:
            return "Unknown", 0.0, None

        best_idx = int(np.argmax([p if p is not None else 0 for p in probs]))
        best_prob = probs[best_idx] if probs[best_idx] is not None else 0
        if best_prob < 0.90:
            return "Unknown", 0.0, None

        # Step 2: Tight MTCNN crop
        fb = boxes[best_idx]
        mfx1 = max(0, int(fb[0]))
        mfy1 = max(0, int(fb[1]))
        mfx2 = min(face_crop.shape[1], int(fb[2]))
        mfy2 = min(face_crop.shape[0], int(fb[3]))
        mtcnn_face = face_rgb[mfy1:mfy2, mfx1:mfx2]

        if mtcnn_face.size == 0 or (mfx2 - mfx1) < 20 or (mfy2 - mfy1) < 20:
            return "Unknown", 0.0, None

        # Step 3: Embedding on AMD dGPU (dynamic device)
        device = self._get_resnet_device()
        face_resized = cv2.resize(mtcnn_face, (160, 160))
        # Normalize to [-1, 1] as expected by InceptionResnetV1
        face_np = face_resized.astype(np.float32) / 255.0
        face_np = (face_np - 0.5) / 0.5
        face_tensor = torch.tensor(
            np.transpose(face_np, (2, 0, 1))
        ).float().unsqueeze(0).to(device)

        with self.ai_lock:
            with torch.no_grad():
                embedding = self.resnet(face_tensor).cpu().numpy()[0]

        # Step 4: Match — tight threshold for high-confidence identification only
        # InceptionResnetV1/VGGFace2: dist < 0.40 → very high confidence (>90%)
        MATCH_THRESHOLD = 0.40   # Only accept strong matches
        CONF_SCALE = 0.80        # dist=0 → 100%, dist=0.40 → ~50% (scaled up below)
        if self.known_face_encodings:
            enc_arr = np.array(self.known_face_encodings)
            distances = np.linalg.norm(enc_arr - embedding, axis=1)
            min_idx = int(np.argmin(distances))
            min_dist = distances[min_idx]
            if min_dist < MATCH_THRESHOLD:
                name = self.known_face_names[min_idx]
                # Map [0, MATCH_THRESHOLD] → [1.0, 0.5] then scale to [1.0, 0.90]
                raw_conf = 1.0 - (min_dist / (MATCH_THRESHOLD * 2))
                conf = 0.90 + (raw_conf - 0.5) * 0.20  # clamp to [0.90, 1.0]
                conf = max(0.90, min(1.0, conf))
                logger.debug(f"[Recognizer] Match: {name} dist={min_dist:.3f} conf={conf:.2f}")
                return name, float(conf), embedding
            return "Unknown", 0.0, embedding

        return "Unknown", 0.0, embedding

    def recognize(self, frame, face_bbox):
        name, conf, _ = self.recognize_with_encoding(frame, face_bbox)
        return name, conf

    def get_encoding(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.ai_lock:
            boxes, _ = self.mtcnn.detect(image_rgb)
        if boxes is None or len(boxes) == 0:
            return None
        fx1, fy1, fx2, fy2 = [int(b) for b in boxes[0]]
        face_crop = image_rgb[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]
        if face_crop.size == 0:
            return None
        device = self._get_resnet_device()
        face_resized = cv2.resize(face_crop, (160, 160))
        face_np = face_resized.astype(np.float32) / 255.0
        face_np = (face_np - 0.5) / 0.5
        face_tensor = torch.tensor(
            np.transpose(face_np, (2, 0, 1))
        ).float().unsqueeze(0).to(device)
        with self.ai_lock:
            with torch.no_grad():
                embedding = self.resnet(face_tensor).cpu().numpy()[0]
        return embedding
