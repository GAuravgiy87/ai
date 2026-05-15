# AI Vigilance: Smart Multi-Camera Surveillance System

A production-ready, real-time AI surveillance system with distributed architecture. AI Vigilance detects, tracks, and identifies individuals across multiple cameras using YOLOv8s detection, custom IoU tracking, and FaceNet recognition with hardware acceleration support.

---

## 🚀 Key Features

| Feature | Details |
|---|---|
| **Dual-Server Architecture** | Main app (port 9000) + Camera server (port 9001) for process isolation |
| **YOLOv8s Detection** | Upgraded from nano to small model with dynamic confidence thresholds (0.48-0.60) |
| **Advanced Tracking** | Hungarian algorithm + HSV appearance model with re-entry buffer (48 frames) |
| **Dynamic Lighting** | CLAHE + gamma correction adapts to any lighting condition |
| **Hardware Acceleration** | DirectML (AMD/Intel), VAAPI decode, QSV/AMF encoding |
| **Face Recognition** | FaceNet + MTCNN with batch processing and GPU acceleration |
| **Automatic Recording** | Hourly MP4 chunks with hardware encoding at 15 FPS |
| **Resource Guard** | Dynamic FPS throttling based on CPU load (6fps → 4fps → 3fps → pause) |
| **RTSP Auto-Discovery** | Probes 20+ common paths for Hikvision, Dahua, Axis cameras |
| **Global Re-ID** | Cross-camera person tracking with face embeddings |

---

## 🧠 AI Stack

### 1. YOLOv8s (Ultralytics)
- Small model (22MB) for better accuracy vs nano (6MB)
- ONNX Runtime with DirectML for AMD/Intel GPU acceleration
- Dynamic confidence thresholds (0.48-0.60) based on post-normalization brightness
- Aspect ratio filter (1.1-6.0) and size validation (6-96% frame height)

### 2. Custom IoU Tracker (`utils/tracker.py`)
- Hungarian algorithm for globally optimal assignment
- HSV histogram appearance model (32-dim) for occlusion handling
- Re-entry buffer (48 frames / 8 seconds) preserves IDs
- Dynamic max_age: established tracks survive 2-3× longer
- Speed-aware rendering: fast movers (≥18px/f) shown only when detected

### 3. FaceNet + MTCNN (Recognition)
- InceptionResnetV1 on ROCm/CUDA/DirectML
- MTCNN face detection with 0.90 confidence threshold
- Batch processing for forensic video scans
- L2 distance matching with 1.05 normalized threshold
- Thread-safe with global lock for concurrent cameras

---

## 💻 Tech Stack

- **FastAPI + Uvicorn** — Dual-server async architecture (main + camera server)
- **OpenCV (headless)** — RTSP/TCP capture with OpenCL GPU preprocessing
- **SQLite3 (WAL mode)** — Concurrent read/write with auto-checkpoint
- **PyTorch + ONNX Runtime** — DirectML/ROCm acceleration
- **FFmpeg** — Hardware encoding (QSV/AMF/NVENC) with faststart

---

## 🛠️ Setup & Deployment

### Linux (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd ai-vigilance

# 2. Run the one-time setup script
chmod +x setup_linux.sh && ./setup_linux.sh

# 3. Start the system
chmod +x start.sh && ./start.sh
```

### Windows (Development)

```powershell
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch (CPU or CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the system
python app.py
```

### Docker Deployment

```bash
# Build and run with GPU passthrough
docker-compose up -d

# View logs
docker logs -f ai_vigilance
```

### Access the Dashboard
- **Main App**: `http://<server-ip>:9000`
- **Camera Server**: `http://<server-ip>:9001` (internal API)
- **Network Access**: Available on LAN from any browser

---

## 📂 Repository Structure

```
ai-vigilance/
├── app.py                      # Main FastAPI app (port 9000)
├── camera_server/
│   ├── server.py               # Camera processing server (port 9001)
│   └── client.py               # Client for camera server API
├── cameras/
│   └── camera_manager.py       # RTSP handler, auto-discovery, CameraHandler threads
├── core/
│   ├── pipeline.py             # AI pipeline, detection pool, recording threads
│   ├── startup.py              # Lifespan, camera server launcher, Re-ID manager
│   ├── state.py                # Shared global state, locks, directories
│   ├── resource_guard.py       # Dynamic CPU throttling
│   ├── diagnostics.py          # Crash handler, auto-restart
│   └── auth.py                 # JWT authentication
├── utils/
│   ├── detector.py             # YOLOv8s with dynamic thresholds & CLAHE
│   ├── tracker.py              # Hungarian + HSV tracker with re-entry
│   ├── recognizer.py           # FaceNet + MTCNN batch recognition
│   └── hw_manager.py           # Hardware detection (GPU, encoders)
├── database/
│   └── sqlite_manager.py       # SQLite3 WAL mode, 11 tables
├── routes/                     # API route modules
│   ├── cameras.py              # Camera management
│   ├── people.py               # Person registration
│   ├── recordings.py           # Video playback
│   ├── search.py               # Forensic search
│   ├── analytics.py            # Dashboard metrics
│   └── ...
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS, JS, assets
├── dataset/                    # Registered person images
├── snapshots/                  # Detection snapshots (YYYY-MM-DD/camera/)
├── recordings/                 # Hourly MP4 files (YYYY-MM-DD/camera/HH.mp4)
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker deployment with GPU
└── Dockerfile                  # Container image
```

---

## 📋 Monitoring & Logs

All application logs are written to `app.log` and `crash_forensics.log`.

```bash
# Watch live logs
tail -f app.log

# Check for errors
grep -i "error" app.log

# View crash forensics
cat crash_forensics.log
```

### Resource Guard Throttling

The system automatically adjusts performance based on CPU load:

| CPU Usage | Action | Detection FPS | CLAHE | JPEG Quality |
|-----------|--------|---------------|-------|--------------|
| < 75% | Normal | 6 FPS | Enabled | 75 |
| 75-85% | Warning | 4 FPS | Enabled | 65 |
| 85-92% | High | 3 FPS | Disabled | 60 |
| > 92% | Critical | Paused 8s | Disabled | 55 |

---

## 🔧 File Organization

All recordings and snapshots are organized by date and camera:

| Type | Path Pattern | Example |
|------|-------------|---------|
| Hourly Recording | `recordings/YYYY-MM-DD/camera/HH.mp4` | `recordings/2026-05-15/gate/14.mp4` |
| Detection Snapshot | `snapshots/YYYY-MM-DD/camera/logs/camera_YYYY-MM-DD_HHMMSS.jpg` | `snapshots/2026-05-15/gate/logs/gate_2026-05-15_143022.jpg` |
| Person Dataset | `dataset/PersonName.jpg` | `dataset/John_Doe.jpg` |
