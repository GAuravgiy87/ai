# AI Vigilance: Technical Reference & System Documentation

## 1. Abstract
AI Vigilance is a distributed, real-time intelligent surveillance system designed for heterogeneous hardware environments. It integrates state-of-the-art computer vision models (YOLOv8, FaceNet) with a robust multi-process architecture to provide low-latency monitoring, person tracking, and biometric identification. This document serves as a comprehensive technical reference for research, engineering audits, and future development.

---

## 2. System Architecture & Concurrency

### 2.1 Multi-Server Isolation
The system is bifurcated into two primary processes to ensure performance isolation:
- **Main Application (Web Server - Port 9000)**: Built on FastAPI/Uvicorn, it handles high-level business logic, database orchestration, and user interaction.
- **Camera Server (Processing Engine - Port 9001)**: A dedicated high-load process that manages camera I/O and the AI inference pipeline. This separation prevents the Python Global Interpreter Lock (GIL) from bottlenecking inference during high web traffic.

### 2.2 Concurrency Model
- **Threaded Pipelines**: Each camera runs in a dedicated `process_camera` thread.
- **Shared State Architecture**: Uses a centralized `core/state.py` with `threading.Lock()` and `threading.Event()` to manage cross-thread data access (e.g., `results_lock`, `writer_lock`).
- **Detection Worker Pool**: A shared pool of worker threads processes detections for all cameras, ensuring that a single slow camera doesn't block others.

---

## 3. Algorithmic Deep-Dive

### 3.1 Object Detection (YOLOv8)
- **Model**: YOLOv8s (Small) restricted to the `person` class (Class ID 0).
- **Optimization**: Deployed via **ONNX Runtime** for CPU-bound environments or PyTorch for GPU-enabled systems.
- **Inference Strategy**: Frames are letterboxed to 640x640 before inference to maintain aspect ratio integrity.

### 3.2 Object Tracking (IoU + HSV Appearance)
The system uses a custom-built tracker (`utils/tracker.py`) utilizing:
- **Hungarian Algorithm**: Global optimal assignment via `scipy.optimize.linear_sum_assignment`.
- **Cost Matrix**: A hybrid cost function combining:
    - **IoU (Intersection over Union)**: $1.0 - \text{IoU}(Box_A, Box_B)$
    - **Euclidean Distance**: Distance between bounding box centers.
    - **HSV Histograms**: 32-bin HSV color signature of the person's torso for identity persistence during occlusions.
- **Dynamic Age Management**: Established tracks survive up to $2 \times max\_age$ frames during missed detections.

### 3.3 Face Recognition (MTCNN + FaceNet)
- **MTCNN**: Multi-task Cascaded Convolutional Networks used for high-fidelity face localization and alignment.
- **InceptionResnetV1**: Pre-trained on VGGFace2, generating 512-dimensional biometric embeddings.
- **Distance Metric**: L2 (Euclidean) distance with a tight threshold ($d < 0.40$) for identification.
- **Identity Re-ID**: A global re-identification manager tracks "unknown" individuals across different cameras by comparing their embeddings against a temporary session buffer.

---

## 4. Data Persistence & Schema

### 4.1 Database Configuration
- **Engine**: SQLite3.
- **Mode**: **WAL (Write-Ahead Logging)** enabled to allow concurrent read/write operations without locking the database.
- **Synchronous**: Set to `NORMAL` to optimize disk I/O performance.

### 4.2 Core Schemas
| Table | Key Fields | Purpose |
|---|---|---|
| **`cameras`** | `camera_id`, `source`, `updated_at` | Global camera registry. |
| **`persons`** | `name`, `encoding (BLOB)`, `image_path` | Authorized personnel biometrics. |
| **`video_recordings`**| `file_path`, `start_time`, `end_time` | Metadata for H.264 MP4 files. |
| **`global_identities`**| `global_id`, `encoding (BLOB)`, `type` | Re-ID identities for transient tracking. |
| **`occupancy_logs`** | `camera_id`, `timestamp`, `count` | Time-series data for analytics. |

---

## 5. Performance & Resource Management

### 5.1 Resource Guard Logic
The `ResourceGuard` (`core/resource_guard.py`) performs active monitoring:
- **Metrics**: CPU Usage (%), RAM Usage (%), and System Temperature.
- **Throttling Policy**: 
    - **CPU > 85%**: Throttles detection FPS by 50%.
    - **CPU > 95% (Critical)**: Suspends non-essential AI tasks and pauses MJPEG encoding.
- **FPS Control**: Detection FPS is dynamically scaled per camera based on total system throughput.

### 5.2 Video Encoding (FFmpeg Subprocess)
Video recording is handled by a separate FFmpeg subprocess to offload encoding from Python:
```bash
ffmpeg -y -f rawvideo -vcodec rawvideo -s {w}x{h} -pix_fmt bgr24 -r 2 \
-i - -vcodec h264_qsv -pix_fmt yuv420p -movflags +faststart {output_path}
```
The system automatically probes for hardware encoders like **h264_qsv** (Intel), **h264_amf** (AMD), or **h264_nvenc** (NVIDIA).

---

## 6. Full Logic Flow (Sequential)

1.  **Initialization**: `app.py` loads `SqliteManager` and starts the `Camera Server` thread.
2.  **Model Loading**: `startup.py` loads YOLOv8 and FaceNet models into VRAM/RAM.
3.  **Ingestion Loop**: `CameraManager` pulls frames via OpenCV with a `TCP` transport to avoid UDP frame drops.
4.  **AI Pipeline**:
    -   `DetectionWorkerPool` provides a 640px detection result.
    -   `ObjectTracker` updates track states and handles re-entry logic.
    -   `FaceRecognizer` triggers on new/unidentified tracks.
5.  **Rendering**: OpenCV overlays bboxes and text on the raw 1080p frame.
6.  **Output**:
    -   **Web**: MJPEG stream served via `StreamingResponse`.
    -   **Disk**: Rendered frames written to FFmpeg `stdin` pipe.
    -   **Notification**: Real-time alerts sent via `NotificationManager` (SSE).

---

## 7. Future Research Directions
- **Distributed AI Nodes**: Offloading the Camera Server to Edge devices (Raspberry Pi/Jetson Nano) using gRPC.
- **Behavioral Analytics**: Integrating LSTM or Transformer models to detect suspicious activities (e.g., loitering, falling).
- **Privacy-Preserving Computation**: Implementing differential privacy on face embeddings before storage.

---
*Technical Documentation v3.5 | AI Vigilance Project*
