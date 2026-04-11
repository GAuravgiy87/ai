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
        self.mtcnn = MTCNN(keep_all=True, device="cpu")

        # InceptionResnetV1 on AMD dGPU if available
        self._face_device = hw.face_device
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.resnet.to(torch.device(self._face_device))
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
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        with self.ai_lock:
            boxes, probs = self.mtcnn.detect(face_rgb)

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
        face_tensor = torch.tensor(
            np.transpose(face_resized, (2, 0, 1))
        ).float().unsqueeze(0).to(device)
        face_tensor = (face_tensor - 127.5) / 128.0

        with self.ai_lock:
            with torch.no_grad():
                embedding = self.resnet(face_tensor).cpu().numpy()[0]

        # Step 4: Match
        if self.known_face_encodings:
            enc_arr = np.array(self.known_face_encodings)
            distances = np.linalg.norm(enc_arr - embedding, axis=1)
            min_idx = int(np.argmin(distances))
            min_dist = distances[min_idx]
            if min_dist < 0.65:
                name = self.known_face_names[min_idx]
                conf = 1.0 - (min_dist / 1.3)
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
        face_tensor = torch.tensor(
            np.transpose(face_resized, (2, 0, 1))
        ).float().unsqueeze(0).to(device)
        face_tensor = (face_tensor - 127.5) / 128.0
        with self.ai_lock:
            with torch.no_grad():
                embedding = self.resnet(face_tensor).cpu().numpy()[0]
        return embedding
