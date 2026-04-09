import numpy as np


def _iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center_dist(a, b):
    cx1, cy1 = (a[0]+a[2])/2, (a[1]+a[3])/2
    cx2, cy2 = (b[0]+b[2])/2, (b[1]+b[3])/2
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)


def _box_diag(box):
    """Diagonal of a box — used to normalise distance threshold."""
    return np.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)


def _hungarian(cost, max_cost):
    """
    Simple greedy Hungarian-style assignment on a cost matrix.
    Returns list of (det_idx, track_idx) pairs where cost < max_cost.
    Processes rows in ascending cost order so best matches win.
    """
    rows, cols = cost.shape
    flat = [(cost[r, c], r, c) for r in range(rows) for c in range(cols)]
    flat.sort()
    used_r, used_c = set(), set()
    matches = []
    for val, r, c in flat:
        if val >= max_cost:
            break
        if r in used_r or c in used_c:
            continue
        matches.append((r, c))
        used_r.add(r); used_c.add(c)
    return matches


class ObjectTracker:
    """
    Two-pass tracker:
      Pass 1 — IoU matching  (handles normal movement)
      Pass 2 — centre-distance matching  (handles fast movers / brief occlusion)
    Uses proper cost-matrix assignment so close-together people never steal
    each other's IDs.
    """
    def __init__(self, max_age=30, n_init=1, iou_threshold=0.25):
        self.max_age       = max_age
        self.n_init        = n_init
        self.iou_threshold = iou_threshold
        self.tracks        = []
        self.next_id       = 1
        self.frame_count   = 0
        self.face_encodings = {}
        self.merged_tracks  = {}

    def update(self, detections, frame=None):
        self.frame_count += 1

        # Convert [x,y,w,h] → [x1,y1,x2,y2]
        det_boxes = []
        for bbox, conf, label in detections:
            x, y, w, h = bbox
            det_boxes.append({
                'bbox':    [float(x), float(y), float(x+w), float(y+h)],
                'conf':    float(conf),
                'label':   label,
                'matched': False
            })

        matched_tracks = set()
        matched_dets   = set()

        # ── Pass 1: IoU matching ──────────────────────────────────────────────
        if det_boxes and self.tracks:
            nd, nt = len(det_boxes), len(self.tracks)
            iou_mat = np.zeros((nd, nt), dtype=float)
            for di, det in enumerate(det_boxes):
                for ti, trk in enumerate(self.tracks):
                    iou_mat[di, ti] = _iou(det['bbox'], trk['bbox'])

            # cost = 1 - IoU; only consider pairs above threshold
            cost_mat = 1.0 - iou_mat
            for di, ti in _hungarian(cost_mat, max_cost=1.0 - self.iou_threshold):
                trk = self.tracks[ti]
                trk['bbox']      = det_boxes[di]['bbox']
                trk['conf']      = det_boxes[di]['conf']
                trk['age']       = 0
                trk['hits']     += 1
                trk['last_seen'] = self.frame_count
                matched_tracks.add(ti)
                matched_dets.add(di)
                det_boxes[di]['matched'] = True

        # ── Pass 2: centre-distance for unmatched dets / fast movers ─────────
        unmatched_dets   = [i for i in range(len(det_boxes))   if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]

        if unmatched_dets and unmatched_tracks:
            nd2 = len(unmatched_dets)
            nt2 = len(unmatched_tracks)
            dist_mat = np.full((nd2, nt2), np.inf)

            for di2, di in enumerate(unmatched_dets):
                for ti2, ti in enumerate(unmatched_tracks):
                    trk = self.tracks[ti]
                    gap = self.frame_count - trk['last_seen']
                    if gap > self.max_age:
                        continue
                    dist = _center_dist(det_boxes[di]['bbox'], trk['bbox'])
                    # Max allowed distance = 1.5× box diagonal + small per-frame budget
                    # Tight enough that nearby people don't steal each other's IDs
                    max_d = _box_diag(trk['bbox']) * 1.5 + gap * 30
                    if dist < max_d:
                        dist_mat[di2, ti2] = dist

            for di2, ti2 in _hungarian(dist_mat, max_cost=np.inf):
                if dist_mat[di2, ti2] == np.inf:
                    continue
                di = unmatched_dets[di2]
                ti = unmatched_tracks[ti2]
                trk = self.tracks[ti]
                trk['bbox']      = det_boxes[di]['bbox']
                trk['conf']      = det_boxes[di]['conf']
                trk['age']       = 0
                trk['hits']     += 1
                trk['last_seen'] = self.frame_count
                matched_tracks.add(ti)
                matched_dets.add(di)
                det_boxes[di]['matched'] = True

        # ── Age unmatched tracks ──────────────────────────────────────────────
        for ti, trk in enumerate(self.tracks):
            if ti not in matched_tracks:
                trk['age'] += 1

        # ── Create new tracks for unmatched detections ────────────────────────
        for det in det_boxes:
            if not det['matched']:
                self.tracks.append({
                    'id':         self.next_id,
                    'bbox':       det['bbox'],
                    'conf':       det['conf'],
                    'label':      det['label'],
                    'age':        0,
                    'hits':       1,
                    'last_seen':  self.frame_count,
                    'created_at': self.frame_count,
                })
                self.next_id += 1

        # ── Prune dead tracks ─────────────────────────────────────────────────
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        # ── Return live and 'Grace Period' tracks ─────────────────────────────
        # GRACE_PERIOD: Keeps boxes visible for 5 frames (approx 1s) even if YOLO misses.
        # This prevents flickering and 'missing persons' in low-light/crowds.
        GRACE_PERIOD = 5
        active = []
        for t in self.tracks:
            # Show if it was recently seen (within grace period) OR if it is brand new
            if t['hits'] >= self.n_init and t['age'] <= GRACE_PERIOD:
                active.append({
                    'id':     t['id'],
                    'bbox':   t['bbox'],
                    'label':  t['label'],
                    'stable': t['age'] == 0, # stable means detected in THIS frame
                })
        return active

    def get_active_count(self):
        return len([t for t in self.tracks if t['hits'] >= self.n_init and t['age'] == 0])

    def get_total_unique_count(self):
        return self.next_id - 1
