"""
tracker.py — Robust IoU + appearance tracker.

Core design:
  - Hungarian assignment (scipy) — globally optimal matching
  - HSV histogram appearance — survives occlusion / crossing
  - Re-entry buffer — person re-entering frame keeps their ID
  - Dynamic max_age — established tracks survive longer occlusion
  - Dynamic render gate — speed-aware: fast movers shown only when
    detected this frame; stationary allowed 2 missed frames
  - NO bbox drift on unmatched tracks (zero ghost boxes)
  - Center-only smoothing, raw size — no stretching
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Speed thresholds (detection-space px/frame @ 6fps) ───────────────────────
_SPD_FAST       = 18.0   # px/frame — fast walker / runner
_SPD_SLOW       = 5.0    # px/frame — slow / stationary


class ObjectTracker:
    def __init__(self,
                 max_age: int = 6,
                 n_init: int = 2,
                 iou_threshold: float = 0.15):
        """
        max_age  : base frames a track survives without detection.
                   Established tracks (hits > 8) get up to 2× this value
                   to survive occlusion by a passing person.
        n_init   : 2 hits to confirm. High-conf (>0.75) shown immediately.
        """
        self.max_age       = max_age
        self.n_init        = n_init
        self.iou_threshold = iou_threshold
        self.tracks: List[dict] = []
        self.next_id     = 1
        self.frame_count = 0

        # Legacy compat
        self.face_encodings: dict = {}
        self.merged_tracks:  dict = {}

        # Re-entry buffer: id → {histogram, bbox, vx, vy, lost_at, hits}
        self._lost_buffer: Dict[int, dict] = {}
        self._LOST_BUFFER_AGE     = 48   # 8s @ 6fps — enough for slow walkers
        self._REENTRY_HIST_THRESH = 0.58  # slightly looser for re-entry
        self._COST_GATE           = 0.80

    # ── Geometry ──────────────────────────────────────────────────────────
    @staticmethod
    def _iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(b) -> Tuple[float, float]:
        return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    @staticmethod
    def _dist(b1, b2) -> float:
        c1, c2 = ObjectTracker._center(b1), ObjectTracker._center(b2)
        return float(np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2))

    @staticmethod
    def _area(b) -> float:
        return max(0.0, (b[2]-b[0]) * (b[3]-b[1]))

    # ── Dynamic max_age ───────────────────────────────────────────────────
    def _effective_max_age(self, track: dict) -> int:
        """
        Established tracks (many hits) survive longer occlusion.
        A person who has been tracked for 10+ frames is real — give them
        up to 2× base max_age to survive someone walking in front.
        Stationary tracks get even more patience.
        """
        hits = track.get('hits', 1)
        spd  = float(np.sqrt(track.get('vx',0)**2 + track.get('vy',0)**2))

        if hits >= 12:
            # Well-established track
            if spd < _SPD_SLOW:
                return self.max_age * 3   # stationary: very patient
            elif spd < _SPD_FAST:
                return self.max_age * 2   # walking: patient
            else:
                return self.max_age       # fast: normal
        elif hits >= 4:
            return self.max_age + 2       # confirmed but young
        else:
            return self.max_age           # new track: normal

    # ── Velocity ──────────────────────────────────────────────────────────
    def _predict(self, track) -> list:
        """Predict next position — for matching only, never for rendering."""
        vx, vy = track.get('vx', 0.0), track.get('vy', 0.0)
        b = track['bbox']
        return [b[0]+vx, b[1]+vy, b[2]+vx, b[3]+vy]

    def _update_velocity(self, track, new_bbox, frames_gap: int = 1,
                         conf: float = 1.0):
        old_b = track['bbox']
        dx = ((new_bbox[0]+new_bbox[2])/2) - ((old_b[0]+old_b[2])/2)
        dy = ((new_bbox[1]+new_bbox[3])/2) - ((old_b[1]+old_b[3])/2)
        if frames_gap > 1:
            dx /= frames_gap
            dy /= frames_gap
        alpha = 0.35 + 0.30 * min(1.0, max(0.0, (conf - 0.45) / 0.55))
        track['vx'] = alpha * dx + (1 - alpha) * track.get('vx', 0.0)
        track['vy'] = alpha * dy + (1 - alpha) * track.get('vy', 0.0)

    # ── Bbox smoothing — center only, raw size ────────────────────────────
    @staticmethod
    def _smooth_bbox(old_b: list, new_b: list, alpha: float) -> list:
        """
        Smooth center position only — raw detection size is always used.
        This prevents stretching when person moves toward/away from camera.
        alpha=1.0 means no smoothing (use raw detection directly).
        """
        if alpha >= 1.0:
            return list(new_b)
        old_cx = (old_b[0] + old_b[2]) / 2
        old_cy = (old_b[1] + old_b[3]) / 2
        new_cx = (new_b[0] + new_b[2]) / 2
        new_cy = (new_b[1] + new_b[3]) / 2
        cx = alpha * new_cx + (1 - alpha) * old_cx
        cy = alpha * new_cy + (1 - alpha) * old_cy
        hw_ = (new_b[2] - new_b[0]) / 2
        hh_ = (new_b[3] - new_b[1]) / 2
        return [cx - hw_, cy - hh_, cx + hw_, cy + hh_]

    # ── Appearance model ──────────────────────────────────────────────────
    @staticmethod
    def _compute_histogram(frame, bbox) -> Optional[np.ndarray]:
        """32-dim HSV histogram on torso region."""
        if frame is None:
            return None
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            bw, bh = x2-x1, y2-y1
            if bw < 10 or bh < 10:
                return None
            ty1 = y1 + int(bh * 0.20)
            ty2 = y1 + int(bh * 0.70)
            tx1 = x1 + int(bw * 0.10)
            tx2 = x2 - int(bw * 0.10)
            if ty2-ty1 < 6 or tx2-tx1 < 6:
                return None
            crop = frame[ty1:ty2, tx1:tx2]
            hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
            hist   = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
            norm   = hist.sum()
            if norm > 0:
                hist /= norm
            return hist
        except Exception:
            return None

    @staticmethod
    def _hist_sim(h1, h2) -> float:
        if h1 is None or h2 is None:
            return 0.0
        try:
            d = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                                cv2.HISTCMP_BHATTACHARYYA)
            return float(max(0.0, 1.0 - d))
        except Exception:
            return 0.0

    # ── Crowding check ────────────────────────────────────────────────────
    def _is_crowded(self, ti: int) -> bool:
        t    = self.tracks[ti]
        side = float(self._area(t['bbox'])) ** 0.5
        for i, other in enumerate(self.tracks):
            if i != ti and self._dist(t['bbox'], other['bbox']) < side * 1.5:
                return True
        return False

    # ── Cost matrix ───────────────────────────────────────────────────────
    def _build_cost_matrix(self, det_boxes: list, frame) -> np.ndarray:
        n_d, n_t = len(det_boxes), len(self.tracks)
        cost = np.full((n_d, n_t), 1e6, dtype=np.float32)

        det_hists = [self._compute_histogram(frame, d['bbox']) for d in det_boxes]

        frame_diag = 1.0
        if frame is not None:
            frame_diag = float(np.sqrt(frame.shape[1]**2 + frame.shape[0]**2))
        frame_diag = max(frame_diag, 1.0)

        for ti, track in enumerate(self.tracks):
            predicted  = self._predict(track)
            is_crowded = self._is_crowded(ti)
            t_hist     = track.get('histogram')

            spd    = float(np.sqrt(track.get('vx',0)**2 + track.get('vy',0)**2))
            base_d = 320.0 if self._area(track['bbox']) >= 4000 else 200.0
            # Fast movers get larger search radius
            max_d  = base_d + min(180.0, spd * 8.0)

            for di, det in enumerate(det_boxes):
                iou  = self._iou(det['bbox'], predicted)
                dist = self._dist(det['bbox'], predicted)

                if dist > max_d and iou < 0.05:
                    continue

                iou_cost  = 1.0 - iou
                dist_cost = min(1.0, dist / (frame_diag * 0.25))

                app_cost = 1.0
                if t_hist is not None and det_hists[di] is not None:
                    app_cost = 1.0 - self._hist_sim(t_hist, det_hists[di])

                # Crowded: rely heavily on appearance to avoid ID swaps
                if is_crowded:
                    combined = 0.10*iou_cost + 0.10*dist_cost + 0.80*app_cost
                else:
                    combined = 0.55*iou_cost + 0.20*dist_cost + 0.25*app_cost

                cost[di, ti] = combined

        return cost

    # ── Match and update helper ───────────────────────────────────────────
    def _apply_match(self, track: dict, det: dict, frame,
                     matched_track_idx: set, matched_det_idx: set,
                     ti: int, di: int):
        """Apply a confirmed match between track ti and detection di."""
        gap      = track.get('age', 0) + 1
        det_conf = det['conf']
        self._update_velocity(track, det['bbox'], gap, det_conf)

        # Smoothing alpha:
        #   - Fast movers: alpha=1.0 (no smoothing — box must follow person instantly)
        #   - Slow movers: alpha=0.80 (slight smoothing to reduce jitter)
        spd = float(np.sqrt(track.get('vx',0)**2 + track.get('vy',0)**2))
        bb_alpha = 1.0 if spd >= _SPD_FAST else (0.80 + 0.20*(spd/_SPD_FAST))
        smoothed = self._smooth_bbox(track['bbox'], det['bbox'], bb_alpha)

        # Appearance EMA — weight new detection more when gap > 1
        new_hist = self._compute_histogram(frame, det['bbox'])
        if new_hist is not None:
            old_hist = track.get('histogram')
            w_new = min(0.50, 0.25 * gap)   # more weight if gap was large
            track['histogram'] = (
                (1-w_new)*old_hist + w_new*new_hist
                if old_hist is not None else new_hist
            )

        track.update({
            'bbox':      smoothed,
            'conf':      det_conf,
            'age':       0,
            'hits':      track['hits'] + 1,
            'last_seen': self.frame_count,
        })
        matched_track_idx.add(ti)
        matched_det_idx.add(di)
        det['matched'] = True

    # ── Main update ───────────────────────────────────────────────────────
    def update(self, detections, frame=None):
        self.frame_count += 1

        det_boxes = []
        for det in detections:
            bbox, conf, label = det
            x, y, w, h = bbox
            det_boxes.append({
                'bbox':    [float(x), float(y), float(x+w), float(y+h)],
                'conf':    float(conf),
                'label':   label,
                'matched': False,
            })

        matched_track_idx = set()
        matched_det_idx   = set()

        if det_boxes and self.tracks:
            if _HAS_SCIPY:
                cost = self._build_cost_matrix(det_boxes, frame)
                row_ind, col_ind = linear_sum_assignment(cost)
                for di, ti in zip(row_ind, col_ind):
                    if cost[di, ti] >= self._COST_GATE:
                        continue
                    self._apply_match(self.tracks[ti], det_boxes[di], frame,
                                      matched_track_idx, matched_det_idx, ti, di)
            else:
                # Greedy fallback — IoU pass
                for di, det in enumerate(det_boxes):
                    best_iou, best_ti = self.iou_threshold, -1
                    for ti, track in enumerate(self.tracks):
                        if ti in matched_track_idx:
                            continue
                        if self._iou(det['bbox'], self._predict(track)) > best_iou:
                            best_iou = self._iou(det['bbox'], self._predict(track))
                            best_ti  = ti
                    if best_ti >= 0:
                        self._apply_match(self.tracks[best_ti], det, frame,
                                          matched_track_idx, matched_det_idx,
                                          best_ti, di)

                # Greedy fallback — distance pass (fast movers)
                for di, det in enumerate(det_boxes):
                    if di in matched_det_idx:
                        continue
                    best_dist, best_ti = 9999, -1
                    for ti, track in enumerate(self.tracks):
                        if ti in matched_track_idx:
                            continue
                        d   = self._dist(det['bbox'], self._predict(track))
                        spd = float(np.sqrt(track.get('vx',0)**2 + track.get('vy',0)**2))
                        max_d = (320.0 if self._area(track['bbox']) >= 4000 else 200.0) \
                                + min(180.0, spd*8.0)
                        if d < best_dist and d < max_d:
                            best_dist, best_ti = d, ti
                    if best_ti >= 0:
                        self._apply_match(self.tracks[best_ti], det, frame,
                                          matched_track_idx, matched_det_idx,
                                          best_ti, di)

        # ── Age unmatched tracks ──────────────────────────────────────────
        # CRITICAL: bbox stays at last DETECTED position — no drift.
        # Velocity decays so re-entry prediction stays near last position.
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idx:
                track['age'] += 1
                track['vx'] = track.get('vx', 0.0) * 0.60
                track['vy'] = track.get('vy', 0.0) * 0.60

                # Save to re-entry buffer when track is about to die
                eff_age = self._effective_max_age(track)
                if track['age'] == eff_age - 1:
                    hist = track.get('histogram')
                    if hist is not None:
                        self._lost_buffer[track['id']] = {
                            'histogram': hist.copy(),
                            'bbox':      track['bbox'][:],
                            'vx':        track.get('vx', 0.0),
                            'vy':        track.get('vy', 0.0),
                            'hits':      track.get('hits', 1),
                            'lost_at':   self.frame_count,
                        }

        # ── Prune dead tracks (dynamic max_age per track) ─────────────────
        self.tracks = [
            t for t in self.tracks
            if t['age'] < self._effective_max_age(t)
        ]

        # ── Create new tracks ─────────────────────────────────────────────
        for di, det in enumerate(det_boxes):
            if det['matched']:
                continue
            new_hist   = self._compute_histogram(frame, det['bbox'])
            reentry_id = self._try_reentry(det['bbox'], new_hist)

            if reentry_id is not None:
                lost = self._lost_buffer.pop(reentry_id)
                # Restore with previous hits count so render gate works correctly
                self.tracks.append({
                    'id':         reentry_id,
                    'bbox':       det['bbox'],
                    'conf':       det['conf'],
                    'label':      det['label'],
                    'age':        0,
                    'hits':       max(self.n_init + 1, lost.get('hits', self.n_init+1)),
                    'last_seen':  self.frame_count,
                    'created_at': self.frame_count,
                    'vx':         lost.get('vx', 0.0) * 0.5,  # carry some velocity
                    'vy':         lost.get('vy', 0.0) * 0.5,
                    'histogram':  lost['histogram'],
                })
            else:
                self.tracks.append({
                    'id':         self.next_id,
                    'bbox':       det['bbox'],
                    'conf':       det['conf'],
                    'label':      det['label'],
                    'age':        0,
                    'hits':       1,
                    'last_seen':  self.frame_count,
                    'created_at': self.frame_count,
                    'vx': 0.0, 'vy': 0.0,
                    'histogram':  new_hist,
                })
                self.next_id += 1

        # ── Prune stale re-entry buffer ───────────────────────────────────
        stale = [k for k, v in self._lost_buffer.items()
                 if (self.frame_count - v['lost_at']) > self._LOST_BUFFER_AGE]
        for k in stale:
            del self._lost_buffer[k]

        # ── Return active tracks ──────────────────────────────────────────
        # Dynamic render gate based on speed:
        #
        #   Fast  (spd >= 18px/f): age == 0 only — box must be detected THIS frame
        #   Walk  (5 < spd < 18) : age <= 1     — 1 missed frame allowed
        #   Slow  (spd <= 5)     : age <= 2     — 2 missed frames (stationary)
        #
        # High-confidence first detection (hits=1, conf>=0.75): show immediately.
        # This catches fast walkers who may only appear in 1-2 frames.
        active = []
        for t in self.tracks:
            conf = t.get('conf', 0.0)
            spd  = float(np.sqrt(t.get('vx',0)**2 + t.get('vy',0)**2))

            # Immediate show for high-confidence first detection
            if t['hits'] == 1 and t['age'] == 0 and conf >= 0.75:
                active.append({
                    'id': t['id'], 'bbox': t['bbox'],
                    'vx': 0.0, 'vy': 0.0,
                })
                continue

            if t['hits'] < self.n_init:
                continue

            # Dynamic render age gate
            if spd >= _SPD_FAST:
                max_render_age = 0   # fast: detected this frame only
            elif spd > _SPD_SLOW:
                max_render_age = 1   # walking: 1 missed frame
            else:
                max_render_age = 2   # stationary: 2 missed frames

            if t['age'] <= max_render_age:
                active.append({
                    'id':   t['id'],
                    'bbox': t['bbox'],
                    'vx':   t.get('vx', 0.0),
                    'vy':   t.get('vy', 0.0),
                })
        return active

    # ── Re-entry matching ─────────────────────────────────────────────────
    def _try_reentry(self, bbox, new_hist) -> Optional[int]:
        """
        Match a new detection against the lost-track buffer.
        Uses appearance similarity + position sanity check.
        Established tracks (many hits) get a larger position tolerance
        because they may have moved during occlusion.
        """
        if not self._lost_buffer or new_hist is None:
            return None
        best_id, best_sim = None, self._REENTRY_HIST_THRESH
        for tid, info in self._lost_buffer.items():
            sim = self._hist_sim(info['histogram'], new_hist)
            if sim <= best_sim:
                continue
            # Position tolerance scales with how established the track was
            hits     = info.get('hits', 1)
            side     = float(self._area(info['bbox'])) ** 0.5
            # More established tracks get larger position tolerance
            pos_tol  = max(120.0, side * (3.0 + min(3.0, hits / 6.0)))
            pos_dist = self._dist(bbox, info['bbox'])
            if pos_dist < pos_tol:
                best_sim, best_id = sim, tid
        return best_id

    # ── Utility ───────────────────────────────────────────────────────────
    def get_active_count(self) -> int:
        count = 0
        for t in self.tracks:
            conf = t.get('conf', 0.0)
            spd  = float(np.sqrt(t.get('vx',0)**2 + t.get('vy',0)**2))
            if t['hits'] == 1 and t['age'] == 0 and conf >= 0.75:
                count += 1
                continue
            if t['hits'] < self.n_init:
                continue
            if spd >= _SPD_FAST:
                max_render_age = 0
            elif spd > _SPD_SLOW:
                max_render_age = 1
            else:
                max_render_age = 2
            if t['age'] <= max_render_age:
                count += 1
        return count

    def get_total_unique_count(self) -> int:
        return self.next_id - 1
