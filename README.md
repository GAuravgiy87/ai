<div align="center">

# AI Vigilance

### Smart Multi-Camera Surveillance System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)](https://github.com)

Real-time AI surveillance with YOLOv8s detection, Hungarian-algorithm tracking, FaceNet recognition, cross-camera Re-ID, and crash-safe MKV recording — all in a single deployable process or a full Docker stack.

</div>

---

## Features

| Area | Details |
|---|---|
| **Detection** | YOLOv8s via ONNX (DirectML/CUDA) or PyTorch CPU; dynamic confidence 0.48–0.60 based on scene brightness |
| **Preprocessing** | CLAHE + gamma correction + saturation boost; GPU-accelerated via OpenCL UMat |
| **Tracking** | Hungarian algorithm + 32-bin HSV appearance model; 48-frame re-entry buffer; speed-aware render gate |
| **Recognition** | FaceNet (InceptionResnetV1 / VGGFace2) + MTCNN; batch GPU inference; L2 threshold 1.05 |
| **Cross-camera Re-ID** | Global U-ID system (U-1000, U-1001 …) persisted in PostgreSQL |
| **Recording** | Always-on crash-safe MKV; hourly clock-aligned chunks; FFmpeg stdin pipe |
| **Resource guard** | Auto-throttles FPS (6→4→3→pause) and JPEG quality based on sustained CPU load |
| **Diagnostics** | Live ANSI resource table; crash forensics log with full traceback + system snapshot |
| **UI** | MJPEG live feeds, SSE push notifications, forensic video search, journey timeline |

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Backend** | FastAPI, Uvicorn, Python 3.11 |
| **AI / ML** | PyTorch, ONNX Runtime, Ultralytics YOLOv8s, facenet-pytorch |
| **Computer Vision** | OpenCV (OpenCL), FFmpeg |
| **Database** | PostgreSQL (psycopg2 pool) with SQLite fallback |
| **Messaging** | Redis Pub/Sub (frame transport for recording worker) |
| **Acceleration** | DirectML (AMD/Intel), CUDA (NVIDIA), ROCm, OpenCL |
| **Deployment** | Docker Compose, Nginx reverse proxy |
| **Frontend** | Jinja2 templates, vanilla JS, SSE |

---

## Architecture

The system runs as **two co-located servers** inside one process (local mode) or as **separate Docker containers** (production mode).

```
Browser / Client
      │
   Nginx :80
      │
  ┌───┴────────────────────────────────────┐
  │  main_app  :9000  (FastAPI)            │
  │  Routes: auth, dashboard, cameras,     │
  │          people, recordings, search,   │
  │          detections, journey,          │
  │          analytics                     │
  └───────────────┬────────────────────────┘
                  │ HTTP (httpx)
  ┌───────────────▼────────────────────────┐
  │  streaming_service  :9001  (FastAPI)   │
  │  - YOLOv8s detection                   │
  │  - FaceNet + MTCNN recognition         │
  │  - Hungarian tracker per camera        │
  │  - Global Re-ID manager                │
  │  - MJPEG stream endpoint               │
  │  - RTSP auto-discovery & reconnect     │
  └───────────────┬────────────────────────┘
                  │ Redis (rendered JPEG frames)
  ┌───────────────▼────────────────────────┐
  │  background_jobs  (thread / container) │
  │  - Reads frames from Redis             │
  │  - Writes crash-safe MKV via FFmpeg    │
  │  - Hourly clock-aligned rotation       │
  └────────────────────────────────────────┘
                  │
         PostgreSQL  +  Redis
```

### Data flow

1. **Camera** → `CameraHandler` drains RTSP buffer into memory at native FPS
2. **DetectionWorkerPool** → shared ONNX/YOLO worker processes frames at controlled FPS
3. **Pipeline** → tracker assigns IDs; recognizer runs in thread pool; results written to `camera_results`
4. **streaming_service** → serves MJPEG stream and REST results to `main_app`
5. **background_jobs** → reads rendered JPEG from Redis → decodes → pipes raw BGR24 to FFmpeg stdin → MKV file
6. **main_app** → serves UI, proxies video feed, exposes all API routes

---

## Project Structure

```
ai-vigilance/
├── run.py                        # Main application entry point
├── ml_inference/                 # AI models (YOLOv8, FaceNet, tracker)
├── scripts/                      # Bash/PS1 helper scripts)
├── core/                         # Core pipelines, state, and app definition
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
│
├── streaming_service/
│   ├── server.py                 # Camera FastAPI server (port 9001)
│   │                             #   owns: models, pipeline, MJPEG stream
│   ├── state.py                  # Singleton state and DB connection
│   ├── client.py                 # Async HTTP client used by main_app
│   └── routes/                   # Segmented API endpoints
│
├── device_management/
│   └── camera_manager.py         # CameraHandler (RTSP/webcam), RTSP prober
│
├── core/
│   ├── pipeline.py               # Detection loop, recognition, Re-ID, snapshots
│   ├── detection_pool.py         # Shared DetectionWorkerPool (OpenCL GPU resize)
│   ├── search_pipeline.py        # Forensic video scan (batch GPU recognition)
│   ├── startup.py                # GlobalReIDManager, analytics task, lifespan
│   ├── state.py                  # Shared globals, IST helpers, sanitize_rtsp_url
│   ├── notifications.py          # SSE NotificationManager
│   ├── resource_guard.py         # CPU-based FPS / quality throttle
│   ├── diagnostics.py            # Live ANSI table, crash forensics log
│   ├── auth.py                   # Session-cookie authentication
│   └── logging_config.py         # File + terminal log handlers
│
├── common/
│   ├── network.py                # get_local_ip() utility
│   └── logging.py                # Extracted logging utilities
│
├── data_access/
│   ├── manager.py                # Facade
│   ├── connection.py             # DB pooling
│   └── crud/                     # Domain-specific CRUD modules
│
├── background_jobs/
│   ├── recording_worker.py       # CameraRecorder + management loop
│   ├── inference_worker.py       # Detection processing tasks
│   └── analytics_worker.py       # Background aggregation
│
├── api/
│   ├── auth.py                   # /login  /logout
│   ├── dashboard.py              # /  /dashboard  /api/dashboard_metrics  SSE
│   ├── cameras.py                # /cameras  /api/cameras  video proxy
│   ├── people.py                 # /people  /api/persons  register / edit / delete
│   ├── recordings.py             # /recordings_page  /api/recordings  video stream
│   ├── detections.py             # /detection_logs  /api/detection_snapshots
│   ├── search.py                 # /search  /api/search  video scan by name/image
│   ├── journey.py                # /journey  /api/journey/*
│   └── analytics.py             # /analytics  /api/analytics/hourly|daily
│
├── templates/                    # Jinja2 HTML templates
├── static/                       # CSS + JS assets
├── models/                       # yolov8s.pt  yolov8s.onnx  facenet.onnx
├── docs/                         # DEPLOYMENT.md  docs.md  spreadsheet
│
├── database/
│   ├── dataset/                  # Registered person face images (runtime)
│   ├── snapshots/                # Detection snapshot JPEGs (runtime)
│   ├── logs/                     # System logs
│   └── recordings/               # MKV video chunks (runtime)
```

---

## Installation

### Prerequisites

- Python 3.11+
- FFmpeg on PATH
- 8 GB RAM minimum (16 GB recommended)
- GPU optional — AMD (DirectML/OpenCL), NVIDIA (CUDA), Intel (VAAPI)

### Local (Windows / Linux)

```bash
# 1. Clone
git clone https://github.com/yourorg/ai-vigilance.git
cd ai-vigilance

# 2. Virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# 3. PyTorch — pick one:
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# AMD DirectML (Windows):
pip install torch-directml

# 4. All other dependencies
pip install -r requirements.txt

# 5. The system automatically creates placeholder directories (database/dataset, database/snapshots, database/recordings, database/logs) on startup.

# 6. Run
# Run with dynamic auto-scaling (default: 2-8 workers, auto-scales at 70% CPU)
python run.py

# Or run with custom limits:
# python run.py --max-workers 16 --cpu-threshold 60

# Or run without auto-scaling (fixed worker count):
# python run.py --disable-autoscale
```

Open `http://127.0.0.1:9000` — the camera server starts automatically on port 9001.

### Docker (Production)

```bash
# 1. Copy and edit environment config
cp .env.example .env
# Edit .env: set DB_PASSWORD, REDIS_PASSWORD, etc.

# 2. Build and start all services
docker compose up -d --build

# 3. Follow logs
docker compose logs -f

# 4. Stop
docker compose down
```

Services started by Docker Compose:

| Container | Port | Role |
|---|---|---|
| `nginx` | 80 | Reverse proxy / gateway |
| `main_app` | 9000 | UI + API server |
| `streaming_service` | 9001 | AI pipeline + MJPEG |
| `background_jobs` | — | MKV recording worker |
| `postgres` | 5432 (internal) | Persistent storage |
| `redis` | 6379 (internal) | Frame transport |

---

## Quick Start

### 1. Login

Navigate to `http://localhost` (Docker) or `http://localhost:9000` (local).

Default credentials — **change immediately**:
- Username: `admin`
- Password: `deiadmin@789`

### 2. Add a camera

Go to **Cameras → Add Camera** and enter:

| Field | Example |
|---|---|
| Camera ID | `entrance` |
| Type | `rtsp` / `webcam` / `droidcam` / `ipwebcam` |
| Source | `rtsp://admin:pass@192.168.1.100:554` |

The system auto-probes 20+ common RTSP paths (Hikvision, Dahua, Axis, Reolink …) if no path is given. Recording starts automatically.

### 3. Register a person

Go to **People → Register** — upload a clear face photo and enter a name. The encoding is extracted and synced to the camera server immediately.

### 4. Forensic search

Go to **Search** — search detection history by name, time range, or upload an image to find visual matches across all recorded video.

---

## Recording

Recordings are stored as crash-safe MKV files, rotated on the clock hour:

```
database/recordings/
└── 2026-05-30/
    └── entrance/
        ├── 09.mkv      ← 09:00:00 → 10:00:00
        ├── 10.mkv      ← 10:00:00 → 11:00:00
        └── 14.mkv      ← 14:00:00 → 15:00:00
```

- **Filename** = hour (00–23, 24-hour clock)
- **Crash-safe**: MKV index flushed every 2 seconds — partial files are always playable
- **Restart-safe**: crashed partial file is renamed `HH_recovered.mkv`; new chunk starts fresh
- **FPS**: 10 fps written to FFmpeg; encoded with `libx264 -preset ultrafast -crf 28`

Storage estimate at 10 fps:

| Resolution | Per camera / hour | Per camera / day |
|---|---|---|
| 1080p | ~360 MB | ~8.6 GB |
| 720p | ~180 MB | ~4.3 GB |
| 480p | ~60 MB | ~1.4 GB |

---

## Resource Guard

The system automatically throttles when CPU is sustained above thresholds:

| CPU (sustained) | Level | Detection FPS | CLAHE | JPEG quality |
|---|---|---|---|---|
| < 75% | Normal | 6 | on | 75 |
| ≥ 75% for 4 s | Warn | 4 | on | 65 |
| ≥ 85% for 5 s | High | 3 | off | 60 |
| ≥ 92% for 5 s | Critical | paused 8 s | off | 55 |

Restores full performance after 15 s cooldown below 75%.

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
# PostgreSQL
DB_USER=aiv_user
DB_PASSWORD=aiv_pass
DB_NAME=aiv_db
DATABASE_URL=postgresql://aiv_user:aiv_pass@postgres:5432/aiv_db

# Redis
REDIS_PASSWORD=aiv_redis_pass
REDIS_URL=redis://:aiv_redis_pass@redis:6379/0

# Service URLs (Docker — use service names)
STREAMING_SERVICE_URL=http://streaming_service:9001

# Uvicorn workers (local mode)
UVICORN_WORKERS=2

# Directories
RECORDINGS_DIR=./database/recordings
```

---

## API Reference

All endpoints require a valid session cookie (set by `POST /api/login`).

### Cameras

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/cameras` | List active cameras |
| `POST` | `/api/add_camera` | Add camera (form or JSON) |
| `DELETE` | `/api/remove_camera/{id}` | Remove camera |
| `GET` | `/video_feed/{id}` | MJPEG stream |
| `GET` | `/api/capture_frame/{id}` | Single JPEG snapshot |
| `GET` | `/api/occupancy` | Live person counts |
| `GET` | `/api/live_results/{id}` | Tracked persons for camera |

### People

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/persons` | List registered persons |
| `POST` | `/api/register_person` | Register with face photo |
| `PUT` | `/api/edit_person/{id}` | Rename / update photo |
| `DELETE` | `/api/delete_person/{id}` | Remove person |

### Recordings

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/recordings` | List recordings |
| `GET` | `/api/recording_video?path=…` | Stream video file |
| `DELETE` | `/api/recordings/{id}` | Delete recording |

### Search

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/search` | Search detection history |
| `POST` | `/api/search_by_image` | Match by uploaded image |
| `POST` | `/api/search_video_by_name` | Scan videos for person |
| `POST` | `/api/search_video_by_image` | Scan videos by image |

### Dashboard & Analytics

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/dashboard_metrics` | Counts + recent detections |
| `GET` | `/api/notifications/stream` | SSE push stream |
| `GET` | `/api/analytics/hourly` | 24-hour occupancy chart |
| `GET` | `/api/analytics/daily` | N-day occupancy chart |
| `GET` | `/api/hw_status` | CPU / GPU / RAM stats |

---

## Troubleshooting

**Camera not connecting**
```bash
# Test the RTSP URL directly
ffprobe -rtsp_transport tcp "rtsp://user:pass@ip:554/path"
```

**High CPU / slow detection**
- Check the live resource table in the terminal — the resource guard level is shown
- Reduce camera count or lower source resolution
- Enable hardware acceleration (DirectML / CUDA)

**Recording not starting**
```bash
# Verify FFmpeg is on PATH
ffmpeg -version

# Check recording worker logs
grep "RecordingWorker\|Recorder" database/logs/app.log
```

**Face recognition not working**
```bash
# Confirm model cache exists
python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"
```

**Crash forensics**
```bash
cat crash_forensics.log
```

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — FaceNet / MTCNN
- [FastAPI](https://fastapi.tiangolo.com/) — async web framework
- [OpenCV](https://opencv.org/) — computer vision
- [FFmpeg](https://ffmpeg.org/) — video encoding
