"""
tracker.py — IoU + velocity + appearance tracker.

Fixes over previous version:
  1. Hungarian assignment (scipy) instead of greedy — globally optimal,
     prevents ID swaps when two people cross each other.
  2. Per-track HSV color histogram — breaks IoU ties using appearance.
     When people cross, their shirt colors are usually different.
  3. Re-entry buffer — remembers recently lost track appearances for
     ~10 seconds so a person re-entering frame keeps their original ID.
  4. velocity EMA alpha lowered 0.9 → 0.65 to reduce overreaction to
     noisy detections (was causing bbox to over-predict).
  5. Returns vx/vy in active list so pipeline can apply render-time
     lag compensation.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class ObjectTracker:
    def __init__(self, max_age: int = 20, n_init: int = 2, iou_threshold: float = 0.2, min_track_age: int = 3):
        self.max_age       = max_age
        self.n_init        = n_init
        self.iou_threshold = iou_threshold
        self.tracks: List[dict] = []
        self.next_id   = 1
        self.frame_count = 0
        
        # Issue 7 Fix: Minimum track age before counting
        # Prevents ghost detections (reflections, noise) from being counted
        # Track must be alive for N consecutive frames to be considered valid
        self.min_track_age = min_track_age

        # Legacy compat (used by pipeline for face-merge logic)
        self.face_encodings: dict = {}
        self.merged_tracks: dict  = {}

        # Re-entry buffer: id → {histogram, bbox, vx, vy, lost_at}
        self._lost_buffer: Dict[int, dict] = {}
        # Keep a lost track's appearance for up to N detection-frames (~10s at 6fps)
        self._LOST_BUFFER_AGE    = 60
        # Minimum appearance similarity to accept a re-entry match
        self._REENTRY_HIST_THRESH = 0.55
        # Cost gate: assignments with combined cost above this are rejected
        self._COST_GATE = 0.88

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(b) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    @staticmethod
    def _dist(b1, b2) -> float:
        c1, c2 = ObjectTracker._center(b1), ObjectTracker._center(b2)
        return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

    @staticmethod
    def _area(b) -> float:
        return (b[2] - b[0]) * (b[3] - b[1])

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------
    def _predict(self, track) -> list:
        vx, vy = track.get('vx', 0.0), track.get('vy', 0.0)
        b = track['bbox']
        return [b[0] + vx, b[1] + vy, b[2] + vx, b[3] + vy]

    def _update_velocity(self, track, new_bbox, frames_since_last: int = 1):
        old_b = track['bbox']
        dx = ((new_bbox[0] + new_bbox[2]) / 2) - ((old_b[0] + old_b[2]) / 2)
        dy = ((new_bbox[1] + new_bbox[3]) / 2) - ((old_b[1] + old_b[3]) / 2)
        if frames_since_last > 1:
            dx /= frames_since_last
            dy /= frames_since_last
        # Lowered from 0.9 → 0.65: smoother velocity, less over-prediction
        alpha = 0.65
        track['vx'] = alpha * dx + (1 - alpha) * track.get('vx', 0.0)
        track['vy'] = alpha * dy + (1 - alpha) * track.get('vy', 0.0)

    # ------------------------------------------------------------------
    # Appearance model  (lightweight HSV histogram)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_histogram(frame, bbox) -> Optional[np.ndarray]:
        """
        24-dim HSV histogram: H(16 bins) + S(8 bins).
        Uses only the torso region (middle 60% height, inner 70% width)
        to avoid background contamination.
        """
        if frame is None:
            return None
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h_frame, w_frame = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 12 or bh < 12:
                return None

            # Torso region: skip top 20% (head) and bottom 25% (legs)
            ty1 = y1 + int(bh * 0.20)
            ty2 = y1 + int(bh * 0.75)
            tx1 = x1 + int(bw * 0.15)
            tx2 = x2 - int(bw * 0.15)
            if ty2 - ty1 < 8 or tx2 - tx1 < 8:
                return None

            crop = frame[ty1:ty2, tx1:tx2]
            hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
            hist   = np.concatenate([h_hist, s_hist]).astype(np.float32)

            norm = hist.sum()
            if norm > 0:
                hist /= norm
            return hist
        except Exception:
            return None

    @staticmethod
    def _hist_similarity(h1, h2) -> float:
        """Bhattacharyya-based similarity → [0, 1], 1 = perfect match."""
        if h1 is None or h2 is None:
            return 0.0
        try:
            dist = cv2.compareHist(
                h1.astype(np.float32),
                h2.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA
            )
            return float(max(0.0, 1.0 - dist))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Check if a track is in a crowded region
    # ------------------------------------------------------------------
    def _is_crowded(self, track_idx: int) -> bool:
        """
        True when the track's bbox center is within ~1.5 body-widths of
        another active track — i.e. people are close / crossing.
        """
        t = self.tracks[track_idx]
        side = float(self._area(t['bbox'])) ** 0.5  # approx body size
        for i, other in enumerate(self.tracks):
            if i == track_idx:
                continue
            if self._dist(t['bbox'], other['bbox']) < side * 1.5:
                return True
        return False

    # ------------------------------------------------------------------
    # Cost matrix for Hungarian assignment
    # ------------------------------------------------------------------
    def _build_cost_matrix(self, det_boxes: list, frame) -> np.ndarray:
        """
        Cost[det_i, track_j] combines:
          - IoU cost        (primary, position-based)
          - Distance cost   (secondary, for fast movers with low IoU)
          - Appearance cost (tertiary, dominant during crossing/occlusion)

        Lower cost = better match. Unreachable pairs → 1e6.
        """
        n_d = len(det_boxes)
        n_t = len(self.tracks)
        cost = np.full((n_d, n_t), fill_value=1e6, dtype=np.float32)

        # Precompute detection histograms once
        det_hists = [
            self._compute_histogram(frame, d['bbox']) for d in det_boxes
        ]

        frame_diag = 1.0
        if frame is not None:
            frame_diag = float(np.sqrt(frame.shape[1] ** 2 + frame.shape[0] ** 2))
        frame_diag = max(frame_diag, 1.0)

        for ti, track in enumerate(self.tracks):
            predicted = self._predict(track)
            is_crowded = self._is_crowded(ti)
            track_hist = track.get('histogram')

            # Max allowable center-distance for this track (px, in 640-space)
            max_d = 350.0 if self._area(track['bbox']) >= 5000 else 200.0

            for di, det in enumerate(det_boxes):
                iou  = self._iou(det['bbox'], predicted)
                dist = self._dist(det['bbox'], predicted)

                # Hard gate: too far away and low IoU → impossible match
                if dist > max_d and iou < 0.05:
                    continue

                iou_cost  = 1.0 - iou
                dist_cost = min(1.0, dist / (frame_diag * 0.25))

                # Appearance cost
                app_cost = 1.0
                if track_hist is not None and det_hists[di] is not None:
                    sim      = self._hist_similarity(track_hist, det_hists[di])
                    app_cost = 1.0 - sim

                # In crowded/crossing zones: heavily upweight appearance
                if is_crowded:
                    combined = 0.20 * iou_cost + 0.20 * dist_cost + 0.60 * app_cost
                else:
                    combined = 0.50 * iou_cost + 0.25 * dist_cost + 0.25 * app_cost

                cost[di, ti] = combined

        return cost

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
                'bbox':    [float(x), float(y), float(x + w), float(y + h)],
                'conf':    float(conf),
                'label':   label,
                'matched': False,
            })

        matched_track_idx = set()
        matched_det_idx   = set()

        if det_boxes and self.tracks:
            if _HAS_SCIPY:
                # ── Hungarian assignment (globally optimal) ─────────────
                cost = self._build_cost_matrix(det_boxes, frame)
                row_ind, col_ind = linear_sum_assignment(cost)

                for di, ti in zip(row_ind, col_ind):
                    if cost[di, ti] >= self._COST_GATE:
                        continue  # Too costly — don't force a bad match

                    track = self.tracks[ti]
                    gap   = track.get('age', 0) + 1
                    self._update_velocity(track, det_boxes[di]['bbox'], gap)

                    # Update appearance with EMA (80% keep old, 20% new)
                    new_hist = self._compute_histogram(frame, det_boxes[di]['bbox'])
                    if new_hist is not None:
                        old_hist = track.get('histogram')
                        track['histogram'] = (
                            0.80 * old_hist + 0.20 * new_hist
                            if old_hist is not None else new_hist
                        )

                    track.update({
                        'bbox':      det_boxes[di]['bbox'],
                        'conf':      det_boxes[di]['conf'],
                        'age':       0,
                        'hits':      track['hits'] + 1,
                        'last_seen': self.frame_count,
                    })
                    matched_track_idx.add(ti)
                    matched_det_idx.add(di)
                    det_boxes[di]['matched'] = True

            else:
                # ── Greedy fallback (no scipy) ──────────────────────────
                # Pass 1: IoU
                for di, det in enumerate(det_boxes):
                    best_iou, best_ti = self.iou_threshold, -1
                    for ti, track in enumerate(self.tracks):
                        if ti in matched_track_idx:
                            continue
                        iou = self._iou(det['bbox'], self._predict(track))
                        if iou > best_iou:
                            best_iou, best_ti = iou, ti
                    if best_ti >= 0:
                        gap = self.tracks[best_ti].get('age', 0) + 1
                        self._update_velocity(self.tracks[best_ti], det['bbox'], gap)
                        self.tracks[best_ti].update({
                            'bbox': det['bbox'], 'conf': det['conf'],
                            'age': 0, 'hits': self.tracks[best_ti]['hits'] + 1,
                            'last_seen': self.frame_count,
                        })
                        matched_track_idx.add(best_ti)
                        matched_det_idx.add(di)
                        det['matched'] = True

                # Pass 2: Center-distance for fast movers
                for di, det in enumerate(det_boxes):
                    if di in matched_det_idx:
                        continue
                    best_dist, best_ti = 300, -1
                    for ti, track in enumerate(self.tracks):
                        if ti in matched_track_idx:
                            continue
                        d     = self._dist(det['bbox'], self._predict(track))
                        max_d = 300 if self._area(track['bbox']) >= 5000 else 150
                        if d < best_dist and d < max_d:
                            best_dist, best_ti = d, ti
                    if best_ti >= 0:
                        gap = self.tracks[best_ti].get('age', 0) + 1
                        self._update_velocity(self.tracks[best_ti], det['bbox'], gap)
                        self.tracks[best_ti].update({
                            'bbox': det['bbox'], 'conf': det['conf'],
                            'age': 0, 'hits': self.tracks[best_ti]['hits'] + 1,
                            'last_seen': self.frame_count,
                        })
                        matched_track_idx.add(best_ti)
                        matched_det_idx.add(di)
                        det['matched'] = True

        # ── Age unmatched tracks ──────────────────────────────────────────
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idx:
                track['age'] += 1
                track['bbox'] = self._predict(track)
                track['vx']   = track.get('vx', 0.0) * 0.80
                track['vy']   = track.get('vy', 0.0) * 0.80

                # Save to lost buffer just before the track is pruned
                if track['age'] == self.max_age - 1:
                    hist = track.get('histogram')
                    if hist is not None:
                        self._lost_buffer[track['id']] = {
                            'histogram': hist.copy(),
                            'bbox':      track['bbox'][:],
                            'vx':        track.get('vx', 0.0),
                            'vy':        track.get('vy', 0.0),
                            'lost_at':   self.frame_count,
                        }

        # ── Create new tracks (or restore from re-entry buffer) ──────────
        for di, det in enumerate(det_boxes):
            if det['matched']:
                continue

            new_hist    = self._compute_histogram(frame, det['bbox'])
            reentry_id  = self._try_reentry(det['bbox'], new_hist)

            if reentry_id is not None:
                # Restore old ID with appearance memory
                lost = self._lost_buffer.pop(reentry_id)
                self.tracks.append({
                    'id':        reentry_id,
                    'bbox':      det['bbox'],
                    'conf':      det['conf'],
                    'label':     det['label'],
                    'age':       0,
                    'hits':      self.n_init + 1,  # Skip confirmation delay
                    'last_seen': self.frame_count,
                    'created_at': self.frame_count,
                    'vx':        lost['vx'],
                    'vy':        lost['vy'],
                    'histogram': lost['histogram'],
                })
            else:
                self.tracks.append({
                    'id':        self.next_id,
                    'bbox':      det['bbox'],
                    'conf':      det['conf'],
                    'label':     det['label'],
                    'age':       0,
                    'hits':      1,
                    'last_seen': self.frame_count,
                    'created_at': self.frame_count,
                    'vx':        0.0,
                    'vy':        0.0,
                    'histogram': new_hist,
                })
                self.next_id += 1

        # ── Prune dead tracks ─────────────────────────────────────────────
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        # ── Prune stale lost-buffer entries ───────────────────────────────
        stale_ids = [
            k for k, v in self._lost_buffer.items()
            if (self.frame_count - v['lost_at']) > self._LOST_BUFFER_AGE
        ]
        for k in stale_ids:
            del self._lost_buffer[k]

        # ── Return active tracks (include vx/vy for render-time lag fix) ──
        active = []
        for t in self.tracks:
            # Issue 7 Fix: Only count tracks that have been alive for min_track_age frames
            # This eliminates flickering counts (3→4→3→4) from transient detections
            # Render up to 8-frame-old tracks to prevent flicker
            if t['hits'] >= self.n_init and t['age'] < 8 and t['hits'] >= self.min_track_age:
                active.append({
                    'id':   t['id'],
                    'bbox': t['bbox'],
                    'vx':   t.get('vx', 0.0),   # NEW: needed for lag compensation
                    'vy':   t.get('vy', 0.0),
                })
        return active

    # ------------------------------------------------------------------
    # Re-entry matching
    # ------------------------------------------------------------------
    def _try_reentry(self, bbox, new_hist) -> Optional[int]:
        """
        Try to match a new (unmatched) detection against the lost-track buffer
        using appearance similarity.  Returns lost track id or None.
        """
        if not self._lost_buffer or new_hist is None:
            return None

        best_id  = None
        best_sim = self._REENTRY_HIST_THRESH

        for track_id, info in self._lost_buffer.items():
            sim = self._hist_similarity(info['histogram'], new_hist)
            if sim > best_sim:
                # Also do a loose position sanity check:
                # the new detection shouldn't be too far from where
                # the lost track was last seen + its carried velocity
                frames_lost = self.frame_count - info['lost_at']
                pred_bbox = [
                    info['bbox'][0] + info['vx'] * frames_lost,
                    info['bbox'][1] + info['vy'] * frames_lost,
                    info['bbox'][2] + info['vx'] * frames_lost,
                    info['bbox'][3] + info['vy'] * frames_lost,
                ]
                pos_dist = self._dist(bbox, pred_bbox)
                side     = float(self._area(info['bbox'])) ** 0.5
                max_pos_d = max(200.0, side * 4.0)

                if pos_dist < max_pos_d:
                    best_sim = sim
                    best_id  = track_id

        return best_id

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_active_count(self) -> int:
        # Issue 7 Fix: Apply min_track_age filter to count
        return len([t for t in self.tracks if t['hits'] >= self.n_init and t['age'] == 0 and t['hits'] >= self.min_track_age])

    def get_total_unique_count(self) -> int:
        return self.next_id - 1
