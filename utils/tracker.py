import numpy as np


class ObjectTracker:
    """
    IoU + velocity-prediction tracker.
    - Predicts next box position using exponential moving average velocity
    - Keeps tracks alive for max_age frames (boxes follow person, no flicker)
    - Merges duplicate tracks by face encoding
    """

    def __init__(self, max_age=3, n_init=1, iou_threshold=0.15):
        self.max_age = max_age          # frames to keep a lost track alive
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.face_encodings = {}
        self.merged_tracks = {}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def _center(b):
        return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    @staticmethod
    def _dist(b1, b2):
        c1, c2 = ObjectTracker._center(b1), ObjectTracker._center(b2)
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    @staticmethod
    def _area(b):
        return (b[2]-b[0]) * (b[3]-b[1])

    # ------------------------------------------------------------------
    # Velocity-based prediction
    # ------------------------------------------------------------------

    def _predict(self, track):
        """Shift bbox by current velocity estimate."""
        vx, vy = track.get('vx', 0.0), track.get('vy', 0.0)
        b = track['bbox']
        return [b[0]+vx, b[1]+vy, b[2]+vx, b[3]+vy]

    def _update_velocity(self, track, new_bbox):
        """Exponential moving average of displacement — fast response."""
        old_b = track['bbox']
        dx = ((new_bbox[0]+new_bbox[2])/2) - ((old_b[0]+old_b[2])/2)
        dy = ((new_bbox[1]+new_bbox[3])/2) - ((old_b[1]+old_b[3])/2)
        alpha = 0.9  # high alpha = very fast response to direction changes
        track['vx'] = alpha * dx + (1 - alpha) * track.get('vx', 0.0)
        track['vy'] = alpha * dy + (1 - alpha) * track.get('vy', 0.0)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, detections, frame=None):
        self.frame_count += 1

        # Convert [x,y,w,h] → [x1,y1,x2,y2]
        det_boxes = []
        for det in detections:
            bbox, conf, label = det
            x, y, w, h = bbox
            det_boxes.append({
                'bbox': [float(x), float(y), float(x+w), float(y+h)],
                'conf': float(conf),
                'label': label,
                'matched': False
            })

        matched_tracks = set()
        matched_dets   = set()

        # ── Pass 1: IoU match against predicted positions ──────────────
        for di, det in enumerate(det_boxes):
            best_iou, best_ti = self.iou_threshold, -1
            for ti, track in enumerate(self.tracks):
                if ti in matched_tracks:
                    continue
                predicted = self._predict(track)
                iou = self._iou(det['bbox'], predicted)
                if iou > best_iou:
                    best_iou, best_ti = iou, ti
            if best_ti >= 0:
                self._update_velocity(self.tracks[best_ti], det['bbox'])
                self.tracks[best_ti].update({
                    'bbox': det['bbox'], 'conf': det['conf'],
                    'age': 0, 'hits': self.tracks[best_ti]['hits'] + 1,
                    'last_seen': self.frame_count
                })
                matched_tracks.add(best_ti); matched_dets.add(di)
                det['matched'] = True

        # ── Pass 2: Center-distance match for fast movers ─────────────
        for di, det in enumerate(det_boxes):
            if di in matched_dets:
                continue
            best_dist, best_ti = 300, -1
            for ti, track in enumerate(self.tracks):
                if ti in matched_tracks:
                    continue
                predicted = self._predict(track)
                d = self._dist(det['bbox'], predicted)
                # Larger boxes = bigger person = can move further per frame
                max_d = 300 if self._area(track['bbox']) >= 5000 else 150
                if d < best_dist and d < max_d:
                    best_dist, best_ti = d, ti
            if best_ti >= 0:
                self._update_velocity(self.tracks[best_ti], det['bbox'])
                self.tracks[best_ti].update({
                    'bbox': det['bbox'], 'conf': det['conf'],
                    'age': 0, 'hits': self.tracks[best_ti]['hits'] + 1,
                    'last_seen': self.frame_count
                })
                matched_tracks.add(best_ti); matched_dets.add(di)
                det['matched'] = True

        # ── Age unmatched tracks — keep in memory for re-ID, NO box rendered ─
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track['age'] += 1
                # Keep velocity for re-ID matching but do NOT move the stored bbox.
                # The box must NOT be rendered when the person is not detected.
                track['vx'] = track.get('vx', 0.0) * 0.7
                track['vy'] = track.get('vy', 0.0) * 0.7

        # ── Create new tracks ─────────────────────────────────────────
        for det in det_boxes:
            if not det['matched']:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'label': det['label'],
                    'age': 0, 'hits': 1,
                    'last_seen': self.frame_count,
                    'created_at': self.frame_count,
                    'vx': 0.0, 'vy': 0.0,
                })
                self.next_id += 1

        # ── Prune dead tracks ─────────────────────────────────────────
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        # ── Return ONLY tracks detected in THIS frame (age == 0) ──────
        # Tracks with age > 0 stay in memory for re-ID but are NEVER rendered.
        # This guarantees zero ghosting — box disappears the instant YOLO
        # stops detecting the person.
        active = []
        for t in self.tracks:
            if t['hits'] >= self.n_init and t['age'] == 0:
                active.append({'id': t['id'], 'bbox': t['bbox']})
        return active

    def get_active_count(self):
        return len([t for t in self.tracks
                    if t['hits'] >= self.n_init and t['age'] == 0])

    def get_total_unique_count(self):
        return self.next_id - 1
