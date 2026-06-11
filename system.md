# 🔍 AI Vigilance — Accuracy & Counting Report

---

## ✅ 1. CURRENT STATE: HIGH ACCURACY (Commit: a40f1d1)
This commit includes all the recommended fixes from the previous report!

### ✅ Fixed: Confidence Thresholds Raised
- **ONNX/GPU path**: Confidence threshold adjusted dynamically based on frame brightness
- **Threshold logic**: 
  ```python
  if brightness < 50:
      conf_threshold = 0.55  # Dark scenes need higher confidence
  elif brightness > 180:
      conf_threshold = 0.50  # Bright scenes can use slightly lower
  else:
      conf_threshold = 0.48  # Normal scenes
  ```

### ✅ Fixed: Tiered Size Filter
- Tiny detections (<5% frame height): Rejected entirely
- Small detections (5–10% frame height): Only accepted if confidence >0.65
- Large detections (>95% frame height): Rejected entirely

### ✅ Fixed: Aspect Ratio Filter
- Rejects boxes with aspect ratio <1.2 or >5.0 (eliminates trees/bikes)

### ✅ Fixed: Model Upgraded to YOLOv8s
- **Model used**: yolov8s.pt (not nano!) → far fewer false positives
- **Speed**: Fast enough on i7-8700 with DirectML acceleration

### ✅ Fixed: Frame Preprocessing (Lighting Normalization)
- Gamma correction
- CLAHE (local contrast enhancement)
- Saturation boost for dark scenes
- OpenCL GPU acceleration for preprocessing

---

## 📊 2. CURRENT COUNTING SYSTEM
- **Tracker**: Custom IoU-based tracker (stable, simple)
- **Counting logic**: Counts active confirmed tracks (only tracks that have been alive for ≥2 frames)
- **Re-ID**: Cross-camera re-identification using FaceNet 512D embeddings, threshold 0.55

---

## 🚀 3. IMPROVEMENTS WE ADDED FROM AI BRANCH
- **Crash-safe recording**: Hourly MKV files, index flushed every 2 seconds (partial files playable)
- **Recording storage**: recordings/YYYY-MM-DD/{camera_id}/HH.mkv
- **Recording framerate**: 10 FPS

---

## 📝 4. REMAINING RECOMMENDATIONS
- Add per-camera exclusion zones (for fixed foliage/bike racks)
- Consider upgrading to ByteTrack for better ID stability in heavy crowds
