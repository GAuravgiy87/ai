# AI Vigilance — Smart Multi-Camera Surveillance System

A production-ready, real-time AI surveillance system for multi-camera RTSP deployments. Detects, tracks, and identifies individuals across cameras using a fully threaded GPU-accelerated pipeline.

---

## Features

| Feature | Details |
|---|---|
| Person Detection | YOLOv8n via ONNX Runtime — GPU accelerated (AMD/NVIDIA/Intel) |
| Zero-Ghosting Tracking | Custom IoU tracker — boxes disappear instantly when person leaves |
| Face Recognition | FaceNet (InceptionResnetV1) + MTCNN — 512D biometric embeddings |
| Global Re-ID | Cross-camera person tracking with journey logging |
| Live Head Count | Per-camera + total count via SSE real-time push |
| H.264 Recordings | Browser-compatible MP4 via FFmpeg, auto-split every hour |
| Active Search | Scan live feeds and recordings for a specific person |
| RTSP Auto-Discovery | Probes 15+ common stream paths for any camera brand |
| Fixed 10 FPS Pipeline | Capture → Detection → Render all locked at 10 FPS |
| Minimal Logging | Only critical events logged — startup, shutdown, crash, camera events |

---

## GPU Support

| Platform | GPU Path | How |
|---|---|---|
| Windows (AMD/Intel/NVIDIA) | DirectML via DirectX 12 | `pip install onnxruntime-directml` |
| Linux (AMD) | ROCm | `pip install onnxruntime-gpu` |
| Linux (NVIDIA) | CUDA | `pip install onnxruntime-gpu` |
| Any (fallback) | CPU | `pip install onnxruntime` |

Both YOLO and FaceNet run on GPU via ONNX Runtime. No CUDA required for AMD.

---

## AI Stack

**YOLOv8n** — exported to ONNX on first run, then loaded via ORT with GPU provider.

**FaceNet (InceptionResnetV1/VGGFace2)** — exported to ONNX on first run, GPU inference via ORT. MTCNN face detection stays on CPU (lightweight, no benefit from GPU for single crops).

**Custom IoU Tracker** — velocity-prediction, EMA smoothing, zero ghosting.

**Global Re-ID** — vectorised L2 matching with pre-stacked numpy matrix across cameras.

---

## Setup

### Windows (AMD GPU — recommended)

```bat
setup_windows.bat
.venv\Scripts\activate
python app.py
```

The setup script installs `onnxruntime-directml` and removes any conflicting `onnxruntime` packages.

### Linux (AMD ROCm / NVIDIA CUDA)

```bash
chmod +x setup_linux.sh && ./setup_linux.sh
source .venv/bin/activate
python app.py
```

The setup script auto-detects your GPU (NVIDIA/AMD/none) and installs the correct PyTorch and ONNX Runtime builds.

### Docker

```bash
docker compose up -d
```

For GPU passthrough, uncomment the relevant section in `docker-compose.yml` (NVIDIA or AMD ROCm).

### Access

```
http://<server-ip>:8000
```

Default credentials: `admin` / `deiadmin@789`

---

## Project Structure

```
ai-vigilance/
├── app.py                      # Entry point (~90 lines) — wires modules, starts uvicorn
│
├── core/
│   ├── state.py                # All shared state, locks, queues, thread helpers
│   ├── startup.py              # DB init, model loading, GlobalReIDManager, lifespan
│   ├── pipeline.py             # Camera pipeline: detection, tracking, recognition, recording
│   ├── auth.py                 # Session management, credential helpers
│   └── logging_config.py       # Minimal logging: critical-only file, coloured terminal
│
├── routes/
│   ├── auth.py                 # /login  /logout
│   ├── cameras.py              # Camera CRUD, live feed, occupancy, settings
│   ├── people.py               # Register / edit / delete persons
│   ├── detections.py           # Detection snapshots, detection logs
│   ├── recordings.py           # Recording toggle, list, video timeline
│   ├── search.py               # Face search (live + video)
│   ├── dashboard.py            # Metrics, analytics, system logs, hw status
│   └── reid.py                 # Re-ID targets, journeys, SSE notifications
│
├── utils/
│   ├── detector.py             # YOLOv8n ONNX — GPU via ORT, HOG fallback
│   ├── recognizer.py           # FaceNet ONNX — GPU via ORT, PyTorch CPU fallback
│   ├── tracker.py              # IoU + velocity tracker (zero ghosting)
│   └── hw_manager.py           # GPU detection, ORT provider selection, load monitoring
│
├── cameras/
│   └── camera_manager.py       # RTSP capture at 10 FPS, VAAPI decode (Linux), auto-probe
│
├── database/
│   └── sqlite_manager.py       # SQLite3 — WAL mode, all queries
│
├── templates/                  # Jinja2 HTML pages
├── static/                     # CSS, JS, icons
├── dataset/                    # Registered person face images
├── snapshots/                  # Detection snapshots: YYYY-MM-DD/camera/logs|identities/
├── recordings/                 # MP4 recordings: YYYY-MM-DD/camera/
│
├── requirements.txt
├── setup_windows.bat           # Windows one-time setup (installs onnxruntime-directml)
├── setup_linux.sh              # Linux one-time setup (auto-detects GPU)
├── start.sh                    # Linux launcher
├── Dockerfile
└── docker-compose.yml
```

---

## Thread Budget (per camera)

| Thread | Priority | Task |
|---|---|---|
| `cam-{id}` | NORMAL | RTSP capture at 10 FPS |
| `det-{id}` | HIGH | YOLO inference on GPU |
| render (pipeline body) | NORMAL | Track + draw + encode at 10 FPS |
| `rec-{id}` | LOW | Write frames to FFmpeg at 2 FPS |

Global threads (shared): `recog-1/2`, `transfer-0/1`, `model-init`, `db-log-drain`, `cleanup`, `hw-monitor`, uvicorn event loop.

---

## Logging

**`app.log`** — critical events only, rotating 5 MB × 3 backups:
- System startup with CPU/RAM/GPU/temp snapshot
- System shutdown / SIGTERM / SIGINT with system snapshot
- Crash with full traceback + system state at time of crash
- Camera added / removed
- All ERROR-level events

**Terminal** — real-time coloured output for ERROR+:
```
[10:23:45] [ERROR] Camera DEI_Gate_5 restore failed: connection timeout

============================================================
  CRITICAL CRASH — RuntimeError
  Cause: camera pipe broken
============================================================
System state at crash: cpu=92% | ram=78% | gpu=AMD Radeon | uptime=7200s
```

**`system_logs` DB table** — WARNING+ events, visible in the dashboard `/system_logs` page.

---

## File Naming

| Type | Pattern | Example |
|---|---|---|
| Recording | `{camera}_{date}_{time}.mp4` | `DEI_Gate_5_2026-04-10_143500.mp4` |
| Detection snapshot | `{camera}_{date}_{time}.jpg` | `DEI_Gate_5_2026-04-10_143500.jpg` |
| Identity snapshot | `id_{name}_{time}.jpg` | `id_Gaurav_143500123.jpg` |

---

## Requirements

- Python 3.10+
- FFmpeg (in PATH)
- AMD GPU: Adrenalin driver 23.7.2+ (Windows) or ROCm 6.0+ (Linux)
- NVIDIA GPU: CUDA 12.x + cuDNN (Linux/Windows)
- psutil (`pip install psutil`) for system monitoring
