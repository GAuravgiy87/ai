import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # MTCNN for face detection within the bbox just to be sure
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # Resnet for generating embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.ai_lock = threading.Lock()
        
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self, db_manager):
        persons = db_manager.get_registered_persons()
        self.known_face_encodings = []
        self.known_face_names = []
        for person in persons:
            if person[3] is not None:
                # encoding is stored as blob of floats
                encoding = np.frombuffer(person[3], dtype=np.float32)
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person[1])

    def recognize(self, frame, face_bbox):
        """
        frame: full color frame
        face_bbox: [x1, y1, x2, y2] tight face bounding box
        """
        name, conf, _ = self.recognize_with_encoding(frame, face_bbox)
        return name, conf

    def recognize_with_encoding(self, frame, face_bbox):
        """
        Recognize face using MTCNN to first verify a real front-facing face exists,
        then run InceptionResnetV1 for identification.
        Returns: (name, confidence, face_encoding)
        """
        if not face_bbox:
            return "Unknown", 0.0, None

        fx1, fy1, fx2, fy2 = face_bbox
        face_crop = frame[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]

        if face_crop.size == 0:
            return "Unknown", 0.0, None

        # Step 1: Use MTCNN to verify a real front-facing face is present
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        with self.ai_lock:
            boxes, probs = self.mtcnn.detect(face_rgb)

        if boxes is None or len(boxes) == 0:
            return "Unknown", 0.0, None  # No face detected — person is facing away

        # Pick best detection
        best_idx = int(np.argmax([p if p is not None else 0 for p in probs]))
        best_prob = probs[best_idx] if probs[best_idx] is not None else 0

        # Require high confidence that it's a real front-facing face
        if best_prob < 0.90:
            return "Unknown", 0.0, None

        # Step 2: Crop the MTCNN-detected face tightly for embedding
        fb = boxes[best_idx]
        mfx1 = max(0, int(fb[0]))
        mfy1 = max(0, int(fb[1]))
        mfx2 = min(face_crop.shape[1], int(fb[2]))
        mfy2 = min(face_crop.shape[0], int(fb[3]))
        mtcnn_face = face_rgb[mfy1:mfy2, mfx1:mfx2]

        if mtcnn_face.size == 0 or (mfx2 - mfx1) < 20 or (mfy2 - mfy1) < 20:
            return "Unknown", 0.0, None  # Face too small / too far away

        # Step 3: Generate embedding from the tight MTCNN face crop
        face_resized = cv2.resize(mtcnn_face, (160, 160))
        face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        face_tensor = (face_tensor - 127.5) / 128.0

        with self.ai_lock:
            with torch.no_grad():
                embedding = self.resnet(face_tensor).cpu().numpy()[0]

        # Step 4: Match against known faces with strict threshold
        if self.known_face_encodings:
            encodings_arr = np.array(self.known_face_encodings)
            distances = np.linalg.norm(encodings_arr - embedding, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            # Strict threshold — 0.65 avoids cross-gender/cross-person false matches
            if min_dist < 0.65:
                name = self.known_face_names[min_idx]
                confidence = 1.0 - (min_dist / 1.3)
                return name, float(confidence), embedding
            else:
                return "Unknown", 0.0, embedding
        else:
            return "Unknown", 0.0, embedding

    def get_encoding(self, image):
        """
        Get encoding for registration. Image should be BGR array (from cv2.imread)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.ai_lock:
            boxes, _ = self.mtcnn.detect(image_rgb)
        
        if boxes is not None and len(boxes) > 0:
            fx1, fy1, fx2, fy2 = [int(b) for b in boxes[0]]
            face_crop = image_rgb[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(self.device)
                face_tensor = (face_tensor - 127.5) / 128.0
                
                with self.ai_lock:
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor).cpu().numpy()[0]
                return embedding
        return None
