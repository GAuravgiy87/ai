# AI Vigilance: Smart Multi-Camera Surveillance System - Comprehensive Technical Guide

## 1. Professional System Overview
AI Vigilance is a production-grade, distributed AI surveillance ecosystem with dual-server architecture. It bridges the gap between simple video recording and high-level behavioral intelligence through YOLOv8s detection, Hungarian tracking with HSV appearance modeling, and FaceNet recognition. By leveraging hardware acceleration (DirectML, VAAPI, QSV/AMF) and dynamic resource management, it provides real-time insights with minimal latency.

The system is built on the philosophy of **Edge Intelligence** and **Process Isolation**:
- All AI processing happens locally (no cloud dependency)
- Camera server (port 9001) isolates heavy AI workload from web UI (port 9000)
- Automatic recording with hourly rotation and hardware encoding
- Dynamic FPS throttling based on CPU load

---

## 2. Detailed Dual-Server Architecture

The system follows a strict separation between presentation and processing:

### Architecture Visual Map (Mermaid)
```mermaid
graph TD
    subgraph "Layer 1: Presentation (Browser)"
        UI[Web Dashboard - JS/CSS]
        SSE[SSE Listener - Real-time Alerts]
        VLC[MJPEG Player - Live Feed]
    end

    subgraph "Layer 2: Main App (FastAPI - Port 9000)"
        AUTH[Auth Router - JWT]
        DASH[Dashboard Router]
        REC[Recordings Manager]
        ANA[Analytics Engine]
        DBM[SQLite Manager - WAL Mode]
    end

    subgraph "Layer 3: Camera Server (Port 9001 - Daemon Thread)"
        CS[Camera Server API]
        CM[Camera Manager - RTSP/Webcam]
        PIPE[AI Pipeline Thread per Camera]
        POOL[Detection Worker Pool - Single Thread]
        DET[YOLOv8s ONNX/DirectML]
        TRK[Hungarian + HSV Tracker]
        REC_AI[FaceNet Batch Recognizer]
        FFM[FFmpeg HW Encoder - QSV/AMF]
        RG[Resource Guard - CPU Monitor]
    end

    %% Connections
    UI <-->|HTTP REST| DASH
    SSE <==|SSE Events| PIPE
    VLC <==|MJPEG Stream| CS
    
    DASH <-->|Internal HTTP| CS
    REC <-->|File Access| FFM
    ANA <-->|SQL Queries| DBM
    
    CS <-->|Shared State| PIPE
    PIPE -->|Submit Frame| POOL
    POOL -->|Run Detection| DET
    DET -->|Detections| TRK
    TRK -->|Track IDs| REC_AI
    PIPE -->|Rendered Frame| FFM
    CM -->|Raw Frames| PIPE
    RG -->|Adjust FPS| PIPE
    
    DBM <-->|Storage| DB[(SQLite3 WAL)]
    FFM -->|Files| DISK[(recordings/YYYY-MM-DD/camera/HH.mp4)]
```

---

## 3. Detailed Component & Connection Analysis

### Layer-to-Layer Connectivity
1. **Layer 1 ↔ Layer 2 (User Interaction)**:
   - **HTTP/REST**: Browser sends requests (Add Camera, Search People, View Analytics)
   - **SSE (Server-Sent Events)**: Persistent uni-directional pipe for instant person detection alerts
   - **Static Files**: Snapshots, recordings, dataset served via FastAPI `StaticFiles`

2. **Layer 2 ↔ Layer 3 (System Control)**:
   - **Internal HTTP API**: Main app (9000) calls camera server (9001) via `camera_server/client.py`
   - **Shared Memory State**: Both layers access `core/state.py` for live counts and results
   - **Database**: Main app owns `SqliteManager`, camera server reads/writes via same instance

3. **Layer 3 ↔ External World (Data Ingest/Output)**:
   - **RTSP/TCP**: `CameraManager` establishes stable connections with auto-reconnect
   - **RTSP Auto-Discovery**: Probes 20+ common paths (Hikvision, Dahua, Axis, ONVIF)
   - **FFmpeg Subprocess**: Rendered frames piped to stdin, MP4 written to disk
   - **Hardware Decode**: VAAPI (Intel iGPU) via GStreamer for RTSP decode offload

---

## 4. Full Lifecycle of a Detection Event

Let's follow a single person walking past a camera:

1. **Ingestion** (30 FPS):
   - `CameraHandler` thread drains RTSP stream continuously
   - Latest frame stored in `self.frame` with `threading.Lock()`

2. **Frame Submit** (6 FPS controlled):
   - `process_camera()` submits frame to `DetectionWorkerPool` at resource-guard-controlled rate
   - Old frames dropped if queue full (always process freshest data)

3. **Detection** (GPU-accelerated):
   - Worker applies CLAHE + gamma correction on GPU (OpenCL UMat)
   - YOLOv8s ONNX inference on DirectML (AMD/Intel GPU)
   - Dynamic confidence threshold (0.48-0.60) based on brightness
   - Aspect ratio (1.1-6.0) and size (6-96% height) validation

4. **Tracking** (Hungarian + HSV):
   - `ObjectTracker.update()` builds cost matrix (IoU + distance + appearance)
   - Hungarian algorithm assigns detections to tracks globally
   - HSV histogram updated with EMA (25-50% weight for new detection)
   - Velocity smoothed with alpha 0.35-0.65 based on confidence

5. **Recognition** (Batch FaceNet):
   - Unidentified tracks submitted to `recognition_executor`
   - MTCNN crops face, FaceNet generates 512-d embedding
   - L2 distance matching against known persons (threshold 1.05)
   - Global Re-ID manager assigns U-ID for unknowns (U-1000, U-1001...)

6. **Rendering**:
   - Overlay bbox, ID, name, confidence on normalized display frame
   - JPEG encode at dynamic quality (55-75 based on CPU load)
   - Store in `camera_results` with `results_lock`

7. **Recording** (15 FPS):
   - Dedicated writer thread reads `camera_results` every 66ms
   - Writes frame to FFmpeg stdin (h264_qsv/h264_amf hardware encoding)
   - Hourly rotation: closes FFmpeg and starts new file every 3600s

8. **Alerting**:
   - If known person detected: `NotificationManager.broadcast()` sends SSE event
   - Dashboard receives alert within milliseconds
   - Snapshot saved to `snapshots/YYYY-MM-DD/camera/logs/`

---

## 5. Security, Privacy & Ethics

- **Local Processing**: 100% on-site, no cloud dependency, no data leaves network
- **Biometric Security**: Face embeddings are 512-d normalized vectors (cannot reconstruct face)
- **Access Control**: JWT authentication with role-based permissions
- **Audit Trail**: All detections logged to SQLite with timestamps and snapshots
- **GDPR Compliance**: Configurable retention policies, right to deletion
- **Encryption**: RTSP credentials sanitized (percent-encoded), database can be encrypted at rest

---

## 6. Performance Optimization: The "Resource Guard"

Surveillance is resource-intensive. To ensure the system never freezes:

### Dynamic Throttling (`core/resource_guard.py`)
| CPU Usage | State | Detection FPS | CLAHE | JPEG Quality | Action |
|-----------|-------|---------------|-------|--------------|--------|
| < 75% | OK | 6 FPS | Enabled | 75 | Full performance |
| 75-85% (4s) | Warning | 4 FPS | Enabled | 65 | Reduce FPS |
| 85-92% (5s) | High | 3 FPS | Disabled | 60 | Skip CLAHE |
| > 92% (5s) | Critical | Paused 8s | Disabled | 55 | Pause detection |

### Memory Management
- **Circular Buffer**: Detection pool queue size = 4 (only keep 4 most recent frames)
- **Result Cleanup**: `get_result()` pops (not gets) — stale detections never reused
- **Re-Entry Buffer**: Limited to 48 frames per track, pruned every frame

### Hardware Acceleration
- **GPU Preprocessing**: OpenCL UMat for resize, LUT, CLAHE (15-25% CPU reduction)
- **Video Decode**: VAAPI on Intel iGPU offloads H.264 decode from CPU
- **Video Encode**: QSV (Intel) or AMF (AMD) saves 70% CPU vs libx264

---

## 7. Non-Technical Glossary

- **RTSP**: Real-Time Streaming Protocol — how IP cameras send video over network
- **YOLO (You Only Look Once)**: AI model that finds objects in images in milliseconds
- **FPS (Frames Per Second)**: How many images processed per second (6 FPS = every 166ms)
- **Embedding**: Mathematical "fingerprint" of a face (512 numbers) used for matching
- **SSE (Server-Sent Events)**: Technology that lets server push updates to browser instantly
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization — makes dark images brighter
- **Hungarian Algorithm**: Optimal way to match detections to existing tracks
- **HSV**: Hue-Saturation-Value color space — better for tracking than RGB
- **WAL (Write-Ahead Logging)**: Database mode that allows reading while writing
- **DirectML**: Microsoft's GPU acceleration for AI on AMD/Intel/NVIDIA

---
*Documentation Version: 4.0 | Status: Production | Updated: 2026-05-15*
