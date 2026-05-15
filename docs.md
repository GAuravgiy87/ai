# AI Vigilance: Technical Reference & System Documentation

## 1. Abstract
AI Vigilance is a distributed, real-time intelligent surveillance system with dual-server architecture. It integrates YOLOv8s detection, Hungarian tracking with HSV appearance modeling, and FaceNet recognition with hardware acceleration (DirectML/ROCm/VAAPI). The system features dynamic resource management, automatic recording, and cross-camera re-identification. This document serves as a comprehensive technical reference for research, engineering audits, and future development.

---

## 2. System Architecture & Concurrency

### 2.1 Dual-Server Architecture
The system runs two FastAPI servers in a single Python process:
- **Main Application (Port 9000)**: Handles web UI, authentication, analytics, and database queries. Lightweight business logic only.
- **Camera Server (Port 9001)**: Owns all AI models (YOLOv8s, FaceNet, Re-ID), camera management, detection pipeline, and recording. Runs in a daemon thread started by `core/startup.py`.

**Rationale**: Separating AI workload from web traffic prevents GIL contention. The camera server can saturate CPU with detection while the main app remains responsive for dashboard queries.

### 2.2 Concurrency Model
- **Per-Camera Pipeline Threads**: Each camera runs `process_camera()` in a dedicated thread with its own tracker and recognition cache.
- **Shared Detection Pool**: Single worker thread (`DetectionWorkerPool`) processes frames from all cameras sequentially. The detector has a global lock, so multiple workers would just block each other.
- **Recording Writer Threads**: Each active recording has a dedicated thread (`recording_writer_thread`) that writes frames to FFmpeg stdin at 15 FPS.
- **Shared State**: `core/state.py` provides thread-safe access to `camera_results`, `camera_writers`, `occupancy_last_count` via `threading.Lock()`.
- **Resource Guard Thread**: Monitors CPU usage every second and dynamically adjusts detection FPS, CLAHE, and JPEG quality.

---

## 3. Algorithmic Deep-Dive

### 3.1 Object Detection (YOLOv8s + Dynamic Preprocessing)
- **Model**: YOLOv8s (22MB) — upgraded from nano for 60-70% fewer false positives
- **Acceleration**: ONNX Runtime with DirectML (AMD/Intel GPU) or PyTorch CPU fallback
- **Dynamic Preprocessing** (`detector.py`):
  - **Lighting Analysis**: 64×64 downsample measures brightness (0-255) and contrast
  - **Gamma Correction**: LUT-based gamma (0.4-2.5) applied on GPU via OpenCL UMat
  - **CLAHE**: Adaptive histogram equalization on L channel (clip 1.5-3.0)
  - **Saturation Boost**: 1.4× in dark scenes to enhance person visibility
- **Dynamic Thresholds**: Post-normalization brightness determines confidence (0.48-0.60)
- **Validation Filters**:
  - Size: 6-96% of frame height (small detections need 0.60-0.72 confidence)
  - Aspect ratio: 1.1-6.0 (rejects bikes, trees, vehicles)
  - Width cap: <55% frame width (rejects groups, vehicles)

### 3.2 Object Tracking (Hungarian + HSV + Re-Entry)
Custom tracker (`utils/tracker.py`) with:
- **Hungarian Algorithm**: Globally optimal assignment via `scipy.optimize.linear_sum_assignment`
- **Hybrid Cost Matrix**:
  - IoU cost: `1.0 - IoU(predicted, detection)`
  - Distance cost: Euclidean distance / frame diagonal
  - Appearance cost: `1.0 - HSV_similarity` (32-bin histogram on torso)
  - Crowded scenes: 80% appearance weight to prevent ID swaps
- **Dynamic Max Age**: Established tracks (12+ hits) survive 2-3× longer occlusion
- **Re-Entry Buffer**: Lost tracks stored for 48 frames (8s @ 6fps) with histogram + velocity
- **Speed-Aware Rendering**:
  - Fast (≥18px/f): shown only when detected this frame
  - Walking (5-18px/f): 1 missed frame allowed
  - Stationary (<5px/f): 2 missed frames allowed
- **Velocity Smoothing**: EMA with alpha 0.35-0.65 based on detection confidence
- **Bbox Smoothing**: Center-only (alpha 0.80-1.0), raw size to prevent stretching

### 3.3 Face Recognition (MTCNN + FaceNet + Batch Processing)
- **MTCNN**: Face detection with 0.90 confidence threshold, runs on CPU (GPU has PReLU issues with DirectML)
- **InceptionResnetV1**: Pre-trained on VGGFace2, runs on best available device (ROCm/CUDA/DirectML/CPU)
- **Batch Processing**: `recognize_batch()` processes multiple faces in one GPU call for forensic scans
- **Matching**:
  - Known persons: L2 distance < 1.05 (normalized embeddings)
  - Confidence: 0.90-1.0 scaled from distance
- **Global Re-ID Manager** (`core/startup.py`):
  - Tracks unknown persons across cameras with 0.55 threshold
  - Monotonic U-ID counter (U-1000, U-1001...) prevents collisions
  - 24-hour active identity buffer

---

## 4. Data Persistence & Schema

### 4.1 Database Configuration (`database/sqlite_manager.py`)
- **Engine**: SQLite3 with integrity checks on startup
- **WAL Mode**: Write-Ahead Logging for concurrent read/write
- **Auto-Checkpoint**: Every 1000 pages to prevent unbounded WAL growth
- **Synchronous**: `NORMAL` for optimized disk I/O
- **Corruption Handling**: Automatic backup to `.bak` file and fresh start

### 4.2 Core Schemas (11 Tables)
| Table | Key Fields | Purpose |
|---|---|---|
| **`cameras`** | `camera_id`, `source`, `updated_at` | Camera registry with RTSP URLs |
| **`camera_settings`** | `camera_id`, `recording_enabled`, `tracking_area` | Per-camera configuration |
| **`persons`** | `name`, `encoding (BLOB)`, `image_path`, `last_seen` | Registered persons with face embeddings |
| **`registered_detections`** | `person_name`, `camera_id`, `timestamp`, `snapshot_path` | Detection history for known persons |
| **`detection_snapshots`** | `camera_id`, `person_count`, `bbox_data`, `face_encodings` | All detections with metadata |
| **`occupancy_logs`** | `camera_id`, `timestamp`, `count` | Time-series occupancy data |
| **`video_recordings`** | `camera_id`, `file_path`, `start_time`, `end_time` | Recording metadata |
| **`global_identities`** | `global_id`, `encoding (BLOB)`, `thumbnail`, `type` | Cross-camera Re-ID (U-1000, U-1001...) |
| **`journeys`** | `global_id`, `camera_id`, `timestamp`, `snapshot_path` | Person movement across cameras |
| **`alerts`** | `camera_id`, `person_id`, `snapshot_path`, `type` | Real-time alert log |
| **`analytics_snapshots`** | `metric_type`, `camera_id`, `value`, `metadata` | Dashboard metrics cache |

---

## 5. Performance & Resource Management

### 5.1 Resource Guard (`core/resource_guard.py`)
Dynamic CPU-based throttling with state-change-only logging:
- **Monitoring**: `psutil.cpu_percent()` sampled every 1 second
- **Thresholds**:
  - **75-85% (Warning)**: Sustained 4s → 4 FPS, CLAHE on, JPEG 65
  - **85-92% (High)**: Sustained 5s → 3 FPS, CLAHE off, JPEG 60
  - **>92% (Critical)**: Sustained 5s → Detection paused 8s, then 2 FPS, JPEG 55
- **Cooldown**: 15s after returning to normal before restoring full 6 FPS
- **State Tracking**: Logs only on level transitions (ok → warn → high → crit)

### 5.2 Hardware Acceleration (`utils/hw_manager.py`)
- **GPU Detection**: Probes for AMD (ROCm), NVIDIA (CUDA), Intel/AMD (DirectML)
- **Video Decode**: VAAPI on Intel iGPU via GStreamer pipeline
- **Video Encode**: Auto-selects h264_qsv (Intel) > h264_amf (AMD) > libx264 (CPU)
- **OpenCV Preprocessing**: OpenCL UMat for GPU-accelerated resize, LUT, CLAHE

### 5.3 Recording Pipeline (`core/pipeline.py`)
Hourly MP4 chunks with automatic rotation:
```bash
ffmpeg -y -f rawvideo -s {w}x{h} -pix_fmt bgr24 -r 15 -i - \
  -vf scale={scale_w}:{scale_h} \
  -vcodec h264_qsv -global_quality 25 -preset veryfast \
  -pix_fmt yuv420p -movflags +faststart+frag_keyframe \
  {recordings/YYYY-MM-DD/camera/HH.mp4}
```
- **Writer Thread**: Dedicated thread per camera writes frames at 15 FPS
- **Rotation**: Closes and starts new file every 3600 seconds
- **Graceful Shutdown**: `cleanup_all_recordings()` closes all FFmpeg processes on exit

---

## 6. Full Logic Flow (Sequential)

1. **Initialization** (`app.py`):
   - Load `SqliteManager` with integrity check
   - Install diagnostics (crash handler, auto-restart)
   - Start camera server thread (port 9001)
   - Mount static file directories (snapshots, recordings, dataset)
   - Include API routers (auth, cameras, people, recordings, search, analytics)

2. **Camera Server Startup** (`camera_server/server.py`):
   - Build singletons: `CameraManager`, `PersonDetector` (YOLOv8s), `FaceRecognizer`, `GlobalReIDManager`
   - Initialize pipeline with `init_pipeline()` — wires models into shared state
   - Start resource guard thread
   - Restore cameras from database with automatic recording enabled

3. **Camera Ingestion** (`cameras/camera_manager.py`):
   - `CameraHandler` opens RTSP stream with TCP transport + hardware decode (VAAPI)
   - Background thread drains buffer at 30 FPS to prevent lag
   - Reconnects automatically after 30 failed reads (5 seconds)

4. **AI Pipeline** (`core/pipeline.py` → `process_camera()`):
   - **Frame Submit**: Submit frame to `DetectionWorkerPool` at controlled rate (6 FPS default)
   - **Detection**: Worker applies CLAHE + gamma → YOLOv8s ONNX → NMS (0.40 IoU)
   - **Tracking**: `ObjectTracker.update()` with Hungarian assignment + HSV matching
   - **Recognition**: Submit unidentified tracks to `recognition_executor` (ThreadPoolExecutor)
   - **Rendering**: Overlay bboxes + names on normalized display frame
   - **Recording**: Write rendered frame to FFmpeg stdin (15 FPS, hourly rotation)
   - **State Update**: Store results in `camera_results` with `results_lock`

5. **Output Channels**:
   - **MJPEG Stream**: `/video_feed/{camera_id}` serves JPEG frames at 4 FPS
   - **Occupancy API**: `/occupancy` returns live count + total unique today
   - **SSE Notifications**: `NotificationManager.broadcast()` pushes alerts to dashboard
   - **Database Logs**: Detection snapshots, occupancy logs, registered detections

6. **Resource Management**:
   - **Resource Guard**: Monitors CPU every 1s, adjusts FPS/CLAHE/JPEG on sustained load
   - **Recording Rotation**: Closes FFmpeg and starts new file every 3600s
   - **Cleanup**: `cleanup_all_recordings()` on shutdown closes all FFmpeg processes gracefully

---

## 7. Future Research Directions
- **Edge Deployment**: Offload camera server to Jetson Nano/Raspberry Pi 5 with gRPC communication
- **Behavioral Analytics**: LSTM/Transformer models for loitering, fall detection, crowd anomaly
- **Privacy-Preserving**: Differential privacy on face embeddings before storage
- **Multi-Modal Fusion**: Combine face + gait + clothing for robust re-identification
- **Active Learning**: User feedback loop to improve detection thresholds per camera
- **Distributed Storage**: MinIO/S3 for recordings with automatic tiering (hot/cold)
- **WebRTC Streaming**: Replace MJPEG with WebRTC for lower latency and better mobile support

---

## 8. Known Issues & Mitigations

### 8.1 False Positives (Trees, Bikes)
**Root Causes** (documented in `im.md`):
- YOLOv8n (nano) was too small — **fixed by upgrading to YOLOv8s**
- Low confidence thresholds (0.30-0.45) — **fixed with dynamic 0.48-0.60**
- No aspect ratio filter — **fixed with 1.1-6.0 validation**
- Permissive size filter (5%) — **fixed with 6% minimum + high-conf requirement**

**Remaining Work**:
- Per-camera exclusion zones (ROI masking) for static objects
- NMS IoU tuning (currently 0.40, may need 0.35 for dense crowds)

### 8.2 ID Switching in Crowds
**Mitigations**:
- Hungarian algorithm ensures globally optimal assignment
- HSV appearance model weighted 80% in crowded scenes
- Re-entry buffer preserves IDs for 8 seconds after occlusion

**Remaining Work**:
- Upgrade to ByteTrack or BoT-SORT for better occlusion handling
- Add minimum track age (3 frames) before counting to reduce flicker

### 8.3 Recording Gaps
**Causes**:
- FFmpeg process dies (fixed with automatic restart on `poll() != None`)
- Camera offline (fixed with 10s timeout before closing recording)
- Writer thread blocked (fixed with dedicated thread per camera)

**Monitoring**: Check `app.log` for `[Recording]` errors and `[FFmpeg]` stderr output

---
*Technical Documentation v4.0 | AI Vigilance Project | Updated 2026-05-15*
