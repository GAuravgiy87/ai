# 🔍 AI Vigilance — Accuracy & Counting Report

---

## 1. ROOT CAUSES: FALSE POSITIVES (Trees, Bikes Detected as Persons)

### 🔴 Issue 1 — Confidence Thresholds Are Too Low

**File:** `detector.py`

```python
# ONNX/GPU path
conf_threshold = 0.45   # Line ~50

# CPU/YOLO path  
results = self.model.predict(..., conf=0.30, ...)  # Line ~100
```

**Problem:** 0.30–0.45 is dangerously low for a surveillance system. YOLOv8n (the "nano" model) at these thresholds will fire on tree silhouettes, parked bikes, mannequins, and shadows — especially at oblique camera angles or in wind (moving foliage looks like a walking person to YOLO).

**Fix:**

```python
# ONNX path — raise to 0.55–0.60
conf_threshold = 0.58

# CPU/YOLO path — raise to 0.45
results = self.model.predict(..., conf=0.45, ...)
```

---

### 🔴 Issue 2 — Size Filter Is Too Permissive

**File:** `detector.py`

```python
if bh < (fh * 0.05) or bh > (fh * 0.98):
    continue
```

**Problem:** 5% of frame height means a ~27-pixel tall blob on a 540p stream passes. Trees, bushes, and bike seats routinely produce blobs that size. The comment says this was *intentionally lowered* from 10% to catch distant people — but that trades accuracy for sensitivity.

**Fix (tiered approach):**

```python
# Accept small detections only if confidence is very high
if bh < (fh * 0.05):
    continue  # Too small — remove entirely
elif bh < (fh * 0.10):
    if conf < 0.65:   # Small person needs high confidence
        continue
elif bh > (fh * 0.95):
    continue  # Too large — likely camera artifact
```

---

### 🔴 Issue 3 — Wrong Model for Outdoor Surveillance

**File:** `detector.py`, `startup.py`

```python
detector = PersonDetector()  # loads yolov8n.pt
```

**Problem:** `yolov8n` (nano) is the smallest, fastest, and *least accurate* YOLOv8 variant. It was designed for edge devices with <1W power. For a surveillance system where false positives cause real-world problems, this is the wrong trade-off.

**Fix — upgrade the model:**

| Model | Size | FP Rate | Speed (CPU) |
|-------|------|---------|-------------|
| `yolov8n` | 6 MB | High ❌ | Fastest |
| `yolov8s` | 22 MB | Medium ✅ | Fast |
| `yolov8m` | 52 MB | Low ✅✅ | Moderate |

```python
# In startup.py / load_models()
detector = PersonDetector(model_path='yolov8s.pt')  # Minimum recommended
# OR
detector = PersonDetector(model_path='yolov8m.pt')  # Better accuracy
```

On your i7-8700 system `yolov8s` is a free upgrade — roughly the same latency since you have QuickSync/DirectML for encoding offload.

---

### 🔴 Issue 4 — No Aspect Ratio Filter

**File:** `detector.py`

A person's bounding box has a characteristic aspect ratio: taller than wide. Trees and bikes often produce wide, square, or irregular boxes.

**Fix — add aspect ratio validation:**

```python
aspect = bh / max(bw, 1)  # height / width

# A standing/walking person: aspect ratio 1.5 to 4.5
# A bike: ~0.8–1.2 (wide box)
# A tree canopy: ~0.5–1.0 (wide box)
if aspect < 1.2 or aspect > 5.0:
    continue
```

This single filter eliminates a large class of bike and tree false positives at zero performance cost.

---

### 🔴 Issue 5 — No Region of Interest (ROI) / Exclusion Zones

**Problem:** If a camera has a tree in the corner or a bike rack in frame, it will always generate noise detections regardless of threshold tuning, because the model will periodically score those regions above threshold.

**Fix — add per-camera exclusion zones to your DB schema:**

```python
# In detector.py detect() — filter out boxes overlapping exclusion zones
for zone in camera_exclusion_zones:
    if box_overlaps(detection_box, zone) > 0.5:
        skip detection
```

This is the single highest-impact fix for static false-positive sources.

---

## 2. ROOT CAUSES: INACCURATE COUNTING

### 🔴 Issue 6 — Count = Current Tracks, Not Unique Entries

**File:** `cameras.py`

```python
l_cnt = data.get("count", 0) or occupancy_last_count.get(cam_id, 0)
```

**Problem:** `count` is the number of active bounding boxes in the current frame. This creates two problems:
- **Overcounting:** One person tracked as two IDs (ID switch) = counted twice
- **Undercounting:** Person momentarily occluded, track dropped, reappears as new ID = counted again on re-entry

Without seeing `tracker.py` (not provided), the tracker is almost certainly using a simple IoU-based tracker like SORT. SORT is known for ID switching in crowds and re-occlusion scenarios.

**Fix — use ByteTrack or BoT-SORT:**

```python
# Replace SORT with ByteTrack in your pipeline
# pip install lapx
from ultralytics import YOLO
# ByteTrack is built into ultralytics:
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
```

ByteTrack uses low-confidence detections as "tentative" tracks — it keeps IDs stable through occlusion instead of killing them.

---

### 🔴 Issue 7 — No Minimum Track Age Before Counting

**Problem:** A detection that appears for 1 frame (ghost, reflection, noise) immediately increments the count. This is a classic source of "flickering" counts going 3→4→3→4.

**Fix — add minimum confirmation frames:**

```python
# Only count a track if it has been alive >= N consecutive frames
MIN_TRACK_AGE_FRAMES = 3

confirmed_tracks = [t for t in tracks if t.age >= MIN_TRACK_AGE_FRAMES]
count = len(confirmed_tracks)
```

---

### 🔴 Issue 8 — NMS IoU Threshold Causes Duplicate Detections in Crowds

**File:** `detector.py`

```python
indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.50)
```

**Problem:** 0.50 IoU for NMS suppression is *too permissive for crowd scenarios*. When two people stand side-by-side at 40–45% overlap, NMS keeps both — correct. But when one person produces two detections (head + body from different scales), NMS at 0.50 may keep both, inflating count by 1.

**Fix:**

```python
# Use 0.45 NMS IoU (tighter suppression, fewer duplicates)
# Combined with higher confidence threshold, this is net-positive
indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, 0.45)
```

---

### 🔴 Issue 9 — Re-ID Threshold Is Too Loose

**File:** `startup.py`

```python
def match(self, encoding, threshold=0.75):
```

**Problem:** 0.75 L2 distance for face re-ID is very loose. This can cause two different people to be merged into one global ID, undercounting unique visitors. Meanwhile `recognizer.py` uses 0.40 for *named person* matching — the inconsistency means unknowns are merged more aggressively than knowns.

**Fix:**

```python
def match(self, encoding, threshold=0.55):  # Tighten re-ID matching
```

---

## 3. SUMMARY TABLE

| # | File | Issue | Impact | Fix Effort |
|---|------|-------|--------|------------|
| 1 | `detector.py` | Confidence too low (0.30–0.45) | Trees/bikes detected | Low — 2 lines |
| 2 | `detector.py` | Size filter too permissive (5%) | Small blobs detected | Low — 5 lines |
| 3 | `startup.py` | Using YOLOv8n (nano) | High FP rate | Low — 1 line |
| 4 | `detector.py` | No aspect ratio filter | Bikes/trees pass | Low — 4 lines |
| 5 | Architecture | No exclusion zones | Permanent static FP | Medium |
| 6 | `pipeline.py` | SORT tracker ID switching | Overcounting | Medium — swap tracker |
| 7 | `pipeline.py` | No min track age | Ghost detections counted | Low — 3 lines |
| 8 | `detector.py` | NMS IoU 0.50 too loose | Duplicate boxes | Low — 1 line |
| 9 | `startup.py` | Re-ID threshold 0.75 too loose | Undercounts uniques | Low — 1 line |

---

## 4. RECOMMENDED PRIORITY ORDER

1. **Add aspect ratio filter** (Issue 4) — eliminates most bike/tree FPs immediately, zero side effects
2. **Raise confidence thresholds** (Issue 1) — 0.58 ONNX, 0.45 CPU
3. **Upgrade to YOLOv8s** (Issue 3) — biggest accuracy jump for minimal cost
4. **Add min track age = 3 frames** (Issue 7) — eliminates count flickering
5. **Swap to ByteTrack** (Issue 6) — fixes crowd ID stability
6. **Add exclusion zones UI** (Issue 5) — for cameras with fixed foliage/bike racks

Issues 1, 2, 4, and 7 alone should reduce your false positives by an estimated **60–70%** and stabilize counts noticeably.