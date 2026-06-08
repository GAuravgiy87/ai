"""
recognizer.py — Face recognition on AMD dGPU (ROCm) when available, else CPU.
Uses HardwareManager for dynamic device routing.
"""
import numpy as np
import cv2
import torch
import threading
import logging
import os

# CRITICAL FIX for 100% CPU lockups in multi-worker environments:
# Prevent PyTorch and OpenCV from spawning (num_workers * num_cpu_cores) threads
torch.set_num_threads(1)
cv2.setNumThreads(1)
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self):
        from ml_inference.hw_manager import hw
        self.hw = hw

        # InceptionResnetV1 on hardware accelerator (ROCm/CUDA/DML)
        self._face_device = hw.best_face_device()

        # Use GPU for MTCNN if available (faster for forensic batch scans)
        # Fallback to CPU if device is DML (DirectML sometimes has PReLU issues with MTCNN)
        mtcnn_device = "cpu"
        if "cuda" in str(self._face_device):
            mtcnn_device = self._face_device
        
        from facenet_pytorch import MTCNN, InceptionResnetV1
        try:
            self.mtcnn = MTCNN(
                keep_all=True,
                device=mtcnn_device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
            )
        except Exception as e:
            logger.warning(f"[Recognizer] MTCNN GPU init failed ({e}), falling back to CPU")
            self.mtcnn = MTCNN(
                keep_all=True,
                device="cpu",
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
            )
        
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self._face_device)
        
        actual_mtcnn_dev = str(next(self.mtcnn.parameters()).device) if list(self.mtcnn.parameters()) else "cpu"
        logger.info(f"[Recognizer] FaceNet on {self._face_device} | MTCNN on {actual_mtcnn_dev}")

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
                # Normalize on load for speed
                norm = np.linalg.norm(enc)
                if norm > 0: enc /= norm
                self.known_face_encodings.append(enc)
                self.known_face_names.append(person[1])
        logger.info(f"[Recognizer] Loaded {len(self.known_face_names)} known faces (normalized)")

    def recognize_batch(self, frame, face_boxes):
        """
        Process multiple face boxes in a single GPU batch for high speed.
        Returns a list of (name, confidence, embedding).
        """
        if not face_boxes:
            return []

        results = []
        crops = []
        valid_indices = []

        # Step 1: Prepare crops
        for i, bbox in enumerate(face_boxes):
            fx1, fy1, fx2, fy2 = [int(v) for v in bbox]
            crop = frame[max(0, fy1):fy2, max(0, fx1):fx2]
            if crop.size == 0:
                results.append(("Unknown", 0.0, None))
                continue
            
            # Fast resize/MTCNN check on CPU (sequential but lightweight)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                with self.ai_lock:
                    boxes, probs = self.mtcnn.detect(rgb)
            except:
                results.append(("Unknown", 0.0, None))
                continue

            if boxes is None or len(boxes) == 0 or probs[0] < 0.90:
                results.append(("Unknown", 0.0, None))
                continue

            # Crop the MTCNN-aligned face
            fb = boxes[0]
            mtcnn_face = rgb[max(0,int(fb[1])):min(rgb.shape[0],int(fb[3])), 
                             max(0,int(fb[0])):min(rgb.shape[1],int(fb[2]))]
            
            if mtcnn_face.size == 0:
                results.append(("Unknown", 0.0, None))
                continue

            # Normalize and resize
            face_resized = cv2.resize(mtcnn_face, (160, 160))
            face_np = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            crops.append(np.transpose(face_np, (2, 0, 1)))
            valid_indices.append(i)
            results.append(None) # placeholder

        if not crops:
            return results

        # Step 2: Batch inference on GPU
        device = self._get_resnet_device()
        batch_tensor = torch.tensor(np.array(crops)).float().to(device)
        
        with self.ai_lock:
            with torch.no_grad():
                embeddings = self.resnet(batch_tensor).cpu().numpy()

        # Step 3: Match batch results
        MATCH_THRESHOLD = 1.05 # Normalized scale threshold
        for i, idx in enumerate(valid_indices):
            embedding = embeddings[i]
            # Normalize for consistent matching
            norm = np.linalg.norm(embedding)
            if norm > 0: embedding /= norm
            
            best_name, best_conf = "Unknown", 0.0
            
            if self.known_face_encodings:
                enc_arr = np.array(self.known_face_encodings)
                dists = np.linalg.norm(enc_arr - embedding, axis=1)
                min_idx = int(np.argmin(dists))
                min_dist = dists[min_idx]
                
                if min_dist < MATCH_THRESHOLD:
                    name = self.known_face_names[min_idx]
                    raw_conf = 1.0 - (min_dist / (MATCH_THRESHOLD * 2))
                    best_conf = 0.90 + (raw_conf - 0.5) * 0.20
                    best_conf = max(0.90, min(1.0, best_conf))
                    best_name = name
            
            results[idx] = (best_name, float(best_conf), embedding)

        return results

    def recognize_multi_frame_batch(self, frame_box_pairs):
        """
        True forensic batching: processes multiple (frame, box) pairs at once.
        Crucial for scanning video files at 100fps+.
        Returns a list of (name, confidence, embedding).
        """
        if not frame_box_pairs:
            return []

        results = [("Unknown", 0.0, None)] * len(frame_box_pairs)
        crops = []
        valid_indices = []

        # Step 1: Sequential MTCNN on CPU (alignment is key for accuracy)
        # Optimization: We only use MTCNN if the crop is large enough to matter
        for i, (frame, bbox) in enumerate(frame_box_pairs):
            fx1, fy1, fx2, fy2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            crop = frame[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
            if crop.size == 0: continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                with self.ai_lock:
                    boxes, probs = self.mtcnn.detect(rgb)
            except: continue

            if boxes is None or len(boxes) == 0 or probs[0] < 0.85:
                # Fallback: if MTCNN fails but person is clear, use center-top crop
                # this improves recall for forensic scans
                ch, cw = rgb.shape[:2]
                mtcnn_face = rgb[0:int(ch*0.6), int(cw*0.1):int(cw*0.9)]
            else:
                fb = boxes[0]
                mtcnn_face = rgb[max(0,int(fb[1])):min(rgb.shape[0],int(fb[3])), 
                                 max(0,int(fb[0])):min(rgb.shape[1],int(fb[2]))]

            if mtcnn_face.size == 0: continue

            face_resized = cv2.resize(mtcnn_face, (160, 160))
            face_np = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            crops.append(np.transpose(face_np, (2, 0, 1)))
            valid_indices.append(i)

        if not crops:
            return results

        # Step 2: GPU Batch Inference
        device = self._get_resnet_device()
        batch_tensor = torch.tensor(np.array(crops)).float().to(device)
        
        with self.ai_lock:
            with torch.no_grad():
                embeddings = self.resnet(batch_tensor).cpu().numpy()

        # Step 3: Map results
        MATCH_THRESHOLD = 0.42 # Slightly more permissive for forensic match
        for i, idx in enumerate(valid_indices):
            embedding = embeddings[i]
            # Normalize embedding for consistent similarity comparison
            embedding = embedding / np.linalg.norm(embedding)
            
            best_name, best_conf = "Unknown", 0.0
            if self.known_face_encodings:
                enc_arr = np.array(self.known_face_encodings)
                # Ensure known encodings are also normalized
                # (In practice we should normalize them on load once)
                dists = np.linalg.norm(enc_arr - embedding, axis=1)
                min_idx = int(np.argmin(dists))
                min_dist = dists[min_idx]
                
                if min_dist < MATCH_THRESHOLD:
                    best_name = self.known_face_names[min_idx]
                    best_conf = 1.0 - (min_dist / (MATCH_THRESHOLD * 2))
            
            results[idx] = (best_name, float(best_conf), embedding)

        return results

    def recognize_with_encoding(self, frame, face_bbox):
        res = self.recognize_batch(frame, [face_bbox])
        return res[0] if res else ("Unknown", 0.0, None)

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
        h_img, w_img = image_rgb.shape[:2]
        # BUG-12 fix: clamp all coords to valid image bounds
        face_crop = image_rgb[
            max(0, fy1):min(h_img, fy2),
            max(0, fx1):min(w_img, fx2)
        ]
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
        
        # Normalize for consistency
        norm = np.linalg.norm(embedding)
        if norm > 0: embedding /= norm
        return embedding
