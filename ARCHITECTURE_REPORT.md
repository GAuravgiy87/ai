# 🏛️ AI Vigilance: System Architecture Deep-Dive
**A Technical Reference for the Dual-Server AI Surveillance Ecosystem**

---

## 1. Executive Summary
AI Vigilance is built on a **Dual-Server Distributed Architecture** running in a single Python process. By decoupling the **AI Processing Engine** (port 9001) from the **Web Interface** (port 9000), the system ensures that heavy AI computations never interfere with user experience or system stability. The camera server runs in a daemon thread, owns all AI models, and processes frames from multiple cameras concurrently while the main app handles authentication, analytics, and dashboard queries.

**Key Architectural Decisions**:
- **Process Isolation via Threading**: Camera server runs in daemon thread, not subprocess
- **Shared Detection Pool**: Single worker thread (detector has global lock anyway)
- **Per-Camera Pipeline Threads**: Each camera has dedicated tracking and recognition cache
- **Dynamic Resource Management**: CPU-based throttling adjusts FPS, CLAHE, JPEG quality
- **Hardware Acceleration**: DirectML (GPU), VAAPI (decode), QSV/AMF (encode)

---

## 2. Layer 1: Presentation (The User Interface)
The frontend is a modern, responsive dashboard that communicates with the backend via three distinct protocols.

### Communication Channels
- **HTTP/REST (Port 9000)**: Configuration (add cameras, manage users), historical data (logs, analytics)
- **SSE (Server-Sent Events)**: Persistent unidirectional pipe for real-time person detection alerts
- **MJPEG Stream (Port 9001)**: Live video feed at 4 FPS with JPEG quality 55-75 (CPU-adaptive)

### Key Features
- **Live View**: Grid layout with per-camera MJPEG streams
- **Occupancy Overlay**: Live count + total unique today displayed on each feed
- **Real-Time Alerts**: SSE notifications for registered person detections
- **Recordings Browser**: Date/camera/hour selector with MP4 playback
- **Analytics Dashboard**: Hourly/daily/weekly charts with camera breakdowns
- **Search Interface**: Forensic search by person name, date range, camera

---

## 3. Layer 2: Main Application (The Control Plane - Port 9000)
Built on **FastAPI and Uvicorn**, this layer manages application state and user access.

### Core Components

#### 3.1 Authentication & Authorization (`routes/auth.py`)
- **JWT Tokens**: Secure login with expiring tokens
- **Role-Based Access**: Admin vs viewer permissions
- **Session Management**: Token refresh and logout

#### 3.2 Database Manager (`database/sqlite_manager.py`)
- **SQLite3 with WAL Mode**: Concurrent read/write without locking
- **Auto-Checkpoint**: Every 1000 pages to prevent unbounded WAL growth
- **Integrity Checks**: On startup, corrupted DB moved to `.bak` and reset
- **11 Tables**: cameras, camera_settings, persons, registered_detections, detection_snapshots, occupancy_logs, video_recordings, global_identities, journeys, alerts, analytics_snapshots

#### 3.3 Analytics Engine (`routes/analytics.py`)
- **Hourly Analytics**: Max occupancy per hour (last 24h) with camera breakdown
- **Daily Stats**: AM/PM/total counts per camera
- **Weekly/Monthly Trends**: Total detection counts with period comparison
- **Cached Snapshots**: `analytics_snapshots` table stores pre-computed metrics

#### 3.4 API Routers
- **`routes/cameras.py`**: Add/remove cameras, list active cameras, get settings
- **`routes/people.py`**: Register persons, upload face images, rename/delete
- **`routes/recordings.py`**: List recordings by date/camera, serve MP4 files
- **`routes/search.py`**: Forensic search in detection history
- **`routes/detections.py`**: View detection snapshots with bbox data
- **`routes/journey.py`**: Track person movement across cameras (Re-ID)

---

## 4. Layer 3: Camera Server (The Processing Engine - Port 9001)
This is the "heavy-lifting" layer running in a **daemon thread** started by `core/startup.py`.

### 4.1 Singleton Initialization (`camera_server/server.py`)
Built once when camera server starts:
- **`CameraManager`**: Manages RTSP connections, auto-discovery, reconnection
- **`PersonDetector`**: YOLOv8s ONNX with DirectML or PyTorch CPU fallback
- **`FaceRecognizer`**: FaceNet + MTCNN with batch processing
- **`GlobalReIDManager`**: Cross-camera unknown person tracking (U-1000, U-1001...)

### 4.2 Camera Management (`cameras/camera_manager.py`)
- **RTSP Auto-Discovery**: Probes 20+ common paths (Hikvision, Dahua, Axis, ONVIF)
- **Hardware Decode**: VAAPI on Intel iGPU via GStreamer pipeline
- **Auto-Reconnect**: After 30 failed reads (5 seconds), releases and reopens capture
- **Buffer Draining**: Background thread reads at 30 FPS to prevent lag

### 4.3 AI Pipeline (`core/pipeline.py`)

#### Detection Worker Pool
- **Single Worker Thread**: Detector has global lock, multiple workers just block each other
- **Queue Size 4**: Only keep 4 most recent frames, drop old ones
- **OpenCL Preprocessing**: GPU-accelerated resize, LUT, CLAHE on AMD/Intel
- **Result Consumption**: `get_result()` pops (not gets) — stale detections never reused

#### Per-Camera Pipeline Thread (`process_camera()`)
Each camera runs in a dedicated thread with:
- **Warmup**: Wait for 5 valid frames before starting (max 30 attempts)
- **Automatic Recording**: Always enabled on camera add/restore
- **Frame Submit**: Controlled by resource guard (6 FPS default)
- **Tracking**: `ObjectTracker` with Hungarian + HSV appearance
- **Recognition**: Submit unidentified tracks to `recognition_executor`
- **Rendering**: Overlay bbox, ID, name on normalized display frame
- **Recording**: Dedicated writer thread at 15 FPS with hourly rotation

#### Recording Writer Thread (`recording_writer_thread()`)
- **Dedicated Thread per Camera**: Reads `camera_results` every 66ms (15 FPS)
- **Frame Reuse**: If current frame is None, reuse last frame (prevents gaps)
- **Dimension Check**: Resize if frame size doesn't match FFmpeg input
- **Graceful Shutdown**: Stop event + stdin close + wait(5s) + kill if timeout

### 4.4 Resource Guard (`core/resource_guard.py`)
- **Monitoring**: `psutil.cpu_percent()` sampled every 1 second
- **Sustained Thresholds**: Must stay above threshold for 4-5 seconds before action
- **State-Change Logging**: Only logs on level transitions (ok → warn → high → crit)
- **Cooldown**: 15 seconds after returning to normal before restoring full FPS

### 4.5 Hardware Manager (`utils/hw_manager.py`)
- **GPU Detection**: Probes for AMD (ROCm), NVIDIA (CUDA), Intel/AMD (DirectML)
- **Encoder Selection**: h264_qsv (Intel) > h264_amf (AMD) > libx264 (CPU)
- **VAAPI Device**: `/dev/dri/renderD129` for Intel iGPU decode

---

## 5. Data Flow: Life of a Frame

```
1. RTSP Stream (30 FPS)
   ↓
2. CameraHandler Thread (drains buffer)
   ↓
3. process_camera() (6 FPS controlled)
   ↓
4. DetectionWorkerPool.submit_frame()
   ↓
5. Worker: CLAHE + Gamma → YOLOv8s ONNX → NMS
   ↓
6. DetectionWorkerPool.get_result() [consume-once]
   ↓
7. ObjectTracker.update() [Hungarian + HSV]
   ↓
8. recognition_executor.submit() [FaceNet batch]
   ↓
9. Render: overlay bbox + name on display frame
   ↓
10. JPEG encode (quality 55-75, CPU-adaptive)
    ↓
11. Store in camera_results with results_lock
    ↓
12. ┌─ MJPEG Stream (4 FPS) → Browser
    └─ Recording Writer (15 FPS) → FFmpeg → MP4
```

---

## 6. Performance Optimization Summary

### 6.1 CPU Optimization
- **Dynamic FPS Throttling**: 6 → 4 → 3 → pause based on sustained CPU load
- **CLAHE Skip**: Disabled at 85%+ CPU (saves 5ms/frame)
- **JPEG Quality**: 75 → 65 → 60 → 55 based on CPU load

### 6.2 GPU Acceleration
- **DirectML**: YOLOv8s ONNX inference on AMD/Intel GPU
- **OpenCL**: Resize, LUT, CLAHE on GPU via UMat (15-25% CPU reduction)
- **VAAPI**: H.264 decode on Intel iGPU (offloads CPU)
- **QSV/AMF**: Hardware encoding saves 70% CPU vs libx264

### 6.3 Memory Management
- **Detection Queue**: Size 4 (only keep freshest frames)
- **Result Cleanup**: Pop (not get) — stale detections never reused
- **Re-Entry Buffer**: Limited to 48 frames per track, pruned every frame
- **Recognition Cache**: 18 frames (3 seconds) per track

### 6.4 Concurrency
- **Single Detection Worker**: Detector has global lock, multiple workers waste threads
- **Per-Camera Pipelines**: Each camera has dedicated thread with own tracker
- **Shared State Locks**: `results_lock`, `writer_lock`, `cooldown_lock` for thread safety
- **ThreadPoolExecutor**: Recognition jobs queued (max_workers=1)

---

## 7. Deployment Considerations

### 7.1 Hardware Requirements
- **CPU**: 4+ cores (i5-8400 or Ryzen 5 2600 minimum)
- **RAM**: 4GB minimum, 8GB recommended for 4+ cameras
- **GPU**: Optional but recommended (AMD RX 550+, Intel UHD 630+, NVIDIA GTX 1050+)
- **Storage**: 100GB+ for recordings (1 camera = ~2GB/day at 15 FPS)

### 7.2 Docker Deployment
- **GPU Passthrough**: `/dev/dri` for AMD/Intel, `/dev/kfd` for ROCm
- **Resource Limits**: 4 CPU cores, 4.5GB RAM (adjust per camera count)
- **Volumes**: Persist `snapshots/`, `recordings/`, `dataset/`, `db.sqlite3`
- **Environment**: `HSA_OVERRIDE_GFX_VERSION=8.0.3` for AMD RX 550 (Polaris)

### 7.3 Scaling Guidelines
- **1-4 Cameras**: Single machine, CPU-only viable
- **5-10 Cameras**: GPU acceleration recommended
- **10+ Cameras**: Multiple machines with load balancer, or edge deployment

---

## 8. Security & Privacy

### 8.1 Data Protection
- **Local Processing**: No cloud dependency, all data stays on-premises
- **Encrypted Storage**: SQLite database can be encrypted at rest (SQLCipher)
- **RTSP Credentials**: Percent-encoded in URLs, never logged in plaintext
- **Face Embeddings**: 512-d vectors cannot reconstruct original face

### 8.2 Access Control
- **JWT Authentication**: Secure token-based login with expiration
- **Role-Based Permissions**: Admin vs viewer roles
- **Audit Trail**: All detections logged with timestamps and snapshots

### 8.3 Compliance
- **GDPR**: Configurable retention policies, right to deletion
- **CCPA**: Data export and deletion APIs
- **HIPAA**: Can be deployed in air-gapped environments

---

*Architecture Documentation v4.0 | AI Vigilance Project | Updated: 2026-05-15*
