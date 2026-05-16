<div align="center">

# 🎯 AI Vigilance
### Smart Multi-Camera Surveillance System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)](https://github.com)

**A production-ready, real-time AI surveillance system with distributed architecture**

Detects, tracks, and identifies individuals across multiple cameras using YOLOv8s detection, custom IoU tracking, and FaceNet recognition with hardware acceleration support.

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-system-architecture)

</div>

---

## 🌟 Key Features

<table>
<tr>
<td width="50%">

### 🏗️ **Architecture**
- **Dual-Server Design**: Main app (9000) + Camera server (9001)
- **Process Isolation**: Separate AI workload from web traffic
- **Async Processing**: FastAPI + Uvicorn for high concurrency
- **Thread-Safe**: Shared state with proper locking mechanisms

### 🤖 **AI & Detection**
- **YOLOv8s Detection**: 22MB model with 60-70% fewer false positives
- **Dynamic Thresholds**: Adaptive confidence (0.48-0.60) based on lighting
- **CLAHE + Gamma**: Automatic lighting correction for any condition
- **Hardware Acceleration**: DirectML (AMD/Intel), CUDA (NVIDIA), ROCm

</td>
<td width="50%">

### 👁️ **Tracking & Recognition**
- **Hungarian Algorithm**: Globally optimal track assignment
- **HSV Appearance Model**: 32-dim histogram for occlusion handling
- **Re-Entry Buffer**: 48-frame (8s) ID preservation
- **FaceNet + MTCNN**: Face recognition with batch processing
- **Cross-Camera Re-ID**: Global person tracking across all cameras

### 📹 **Recording & Storage**
- **Automatic Recording**: Starts on camera add, runs 24/7
- **Timestamp-Based Files**: `HH_MMSS.mp4` format prevents overwrites
- **Hourly Rotation**: Seamless 3600s chunks with no frame loss
- **Crash Recovery**: Auto-restart with preserved recordings
- **Hardware Encoding**: QSV/AMF/NVENC support

</td>
</tr>
</table>

### 🎛️ **Resource Management**
- **Dynamic FPS Throttling**: 6fps → 4fps → 3fps → pause based on CPU
- **Adaptive Quality**: CLAHE, JPEG quality adjust automatically
- **Memory Efficient**: Shared frame buffers, optimized caching
- **Crash Protection**: Auto-restart with forensic logging

### 🌐 **Network & Cameras**
- **RTSP Auto-Discovery**: Probes 20+ common paths (Hikvision, Dahua, Axis)
- **TCP Transport**: Reliable streaming with automatic reconnection
- **VAAPI Decode**: Hardware video decoding on Intel iGPU
- **Multi-Camera**: Unlimited cameras (limited by hardware)

### 📊 **Analytics & UI**
- **Real-Time Dashboard**: Live occupancy, detection counts, alerts
- **MJPEG Streaming**: 4 FPS video feeds in browser
- **SSE Notifications**: Push alerts for registered persons
- **Forensic Search**: Search recordings by person, time, camera
- **Journey Tracking**: Cross-camera movement visualization

---

## 🧠 AI Technology Stack

### 1. 🎯 YOLOv8s Object Detection
```
Model Size: 22MB | Accuracy: High | Speed: Real-time
```
- **ONNX Runtime** with DirectML for AMD/Intel GPU acceleration
- **Dynamic Confidence**: 0.48-0.60 based on scene brightness
- **Smart Filtering**: Aspect ratio (1.1-6.0), size validation (6-96% height)
- **False Positive Reduction**: 60-70% improvement over YOLOv8n

### 2. 🎭 Custom IoU Tracker
```
Algorithm: Hungarian | Features: HSV Appearance + Re-Entry Buffer
```
- **Globally Optimal Assignment**: Hungarian algorithm via scipy
- **Hybrid Cost Matrix**:
  - IoU cost: Intersection over Union
  - Distance cost: Euclidean / frame diagonal
  - Appearance cost: 32-bin HSV histogram similarity
- **Dynamic Max Age**: Established tracks survive 2-3× longer
- **Re-Entry Buffer**: 48 frames (8 seconds) ID preservation
- **Speed-Aware Rendering**: Fast movers shown only when detected

### 3. 👤 FaceNet + MTCNN Recognition
```
Model: InceptionResnetV1 | Dataset: VGGFace2 | Threshold: 1.05
```
- **MTCNN Face Detection**: 0.90 confidence threshold
- **GPU Acceleration**: ROCm/CUDA/DirectML/CPU fallback
- **Batch Processing**: Multiple faces in one GPU call
- **L2 Distance Matching**: Normalized embeddings with 1.05 threshold
- **Global Re-ID**: Cross-camera tracking with U-ID system (U-1000, U-1001...)

### 4. 🎨 Dynamic Preprocessing
```
Techniques: CLAHE + Gamma Correction + Saturation Boost
```
- **Lighting Analysis**: 64×64 downsample for brightness/contrast
- **GPU-Accelerated**: OpenCL UMat for LUT, CLAHE operations
- **Adaptive Gamma**: 0.4-2.5 range based on scene analysis
- **CLAHE**: Clip limit 1.5-3.0 on L channel
- **Saturation Boost**: 1.4× in dark scenes

---

## 💻 Technology Stack

<div align="center">

| Category | Technologies |
|:--------:|:------------|
| **Backend** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-2C5BB4?style=flat) ![Python](https://img.shields.io/badge/Python_3.8+-3776AB?style=flat&logo=python&logoColor=white) |
| **AI/ML** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat&logo=onnx&logoColor=white) ![Ultralytics](https://img.shields.io/badge/Ultralytics-00C9FF?style=flat) |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=ffmpeg&logoColor=white) |
| **Database** | ![SQLite](https://img.shields.io/badge/SQLite_3-003B57?style=flat&logo=sqlite&logoColor=white) (WAL Mode) |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) |
| **Acceleration** | ![DirectML](https://img.shields.io/badge/DirectML-0078D4?style=flat&logo=microsoft&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) |
| **Deployment** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white) |

</div>

---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **FFmpeg**: Required for video recording
- **Git**: For cloning the repository
- **Hardware**: 
  - CPU: 4+ cores recommended
  - RAM: 8GB minimum, 16GB recommended
  - GPU: Optional (AMD/NVIDIA/Intel for acceleration)

### 🐧 Linux Installation (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-vigilance.git
cd ai-vigilance

# 2. Run the automated setup script
chmod +x setup_linux.sh
./setup_linux.sh

# The script will:
# - Create Python virtual environment
# - Install system dependencies (FFmpeg, build tools)
# - Install Python packages
# - Download YOLOv8s model
# - Set up directory structure

# 3. Start the system
chmod +x start.sh
./start.sh
```

### 🪟 Windows Installation

```powershell
# 1. Clone the repository
git clone https://github.com/yourusername/ai-vigilance.git
cd ai-vigilance

# 2. Install FFmpeg
# Download from: https://ffmpeg.org/download.html
# Add to PATH environment variable

# 3. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install PyTorch (choose CPU or CUDA)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start the system
python app.py
```

### 🐳 Docker Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-vigilance.git
cd ai-vigilance

# 2. Build and run with Docker Compose
docker-compose up -d

# 3. View logs
docker logs -f ai_vigilance

# 4. Stop the system
docker-compose down
```

### 📝 Post-Installation

After installation, the system will be available at:
- **Main Dashboard**: `http://localhost:9000`
- **Camera Server API**: `http://localhost:9001` (internal)

Default credentials:
- **Username**: `admin`
- **Password**: `admin` (change immediately after first login)

---

## 🚀 Quick Start

### 1. Add Your First Camera

```bash
# Via Web UI:
1. Navigate to http://localhost:9000
2. Click "Add Camera" in the sidebar
3. Enter Camera ID (e.g., "gate", "entrance")
4. Enter RTSP URL: rtsp://username:password@camera-ip:554/path
5. Click "Add Camera"

# The system will:
# - Auto-discover the correct RTSP path
# - Start video processing
# - Begin recording automatically
# - Display live feed in dashboard
```

### 2. Register Known Persons

```bash
# Via Web UI:
1. Go to "People" section
2. Click "Register New Person"
3. Upload a clear face photo
4. Enter person's name
5. Click "Register"

# The system will:
# - Extract face encoding
# - Store in database
# - Start recognizing in all cameras
# - Send alerts when detected
```

### 3. View Recordings

```bash
# Via Web UI:
1. Go to "Recordings" section
2. Select camera and date
3. Browse hourly video files
4. Click to play in browser

# File format: HH_MMSS.mp4
# Example: 14_3045.mp4 = Started at 2:30:45 PM
```

### 4. Search & Analytics

```bash
# Forensic Search:
1. Go to "Search" section
2. Select person, camera, time range
3. View all detections with snapshots
4. Export results

# Journey Tracking:
1. Go to "Journey" section
2. Select person
3. View movement across cameras
4. Timeline visualization
```

---

## 📖 Documentation

### Core Documentation
- **[README.md](README.md)** - This file (overview, installation, quick start)
- **[docs.md](docs.md)** - Technical reference (architecture, algorithms, API)

### Configuration Files
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[docker-compose.yml](docker-compose.yml)** - Docker deployment config
- **[Dockerfile](Dockerfile)** - Container image definition

### Key Modules
- **[app.py](app.py)** - Main application entry point
- **[camera_server/server.py](camera_server/server.py)** - Camera processing server
- **[core/pipeline.py](core/pipeline.py)** - AI detection pipeline
- **[utils/detector.py](utils/detector.py)** - YOLOv8s detection
- **[utils/tracker.py](utils/tracker.py)** - Object tracking
- **[utils/recognizer.py](utils/recognizer.py)** - Face recognition
- **[services/recording.py](services/recording.py)** - Video recording service

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Application (Port 9000)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Web UI     │  │     API      │  │  Database    │         │
│  │  (FastAPI)   │  │   Routes     │  │  (SQLite)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Camera Server (Port 9001)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AI Pipeline                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │  │
│  │  │ YOLOv8s  │→ │ Tracker  │→ │ FaceNet  │→ │ Re-ID   │ │  │
│  │  │ Detector │  │ (IoU+HSV)│  │ (MTCNN)  │  │ Manager │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌──────────────────────────┼────────────────────────────────┐ │
│  │  Camera Manager          │    Recording Service           │ │
│  │  ┌────────┐ ┌────────┐  │    ┌────────┐  ┌────────┐    │ │
│  │  │Camera 1│ │Camera 2│  │    │FFmpeg 1│  │FFmpeg 2│    │ │
│  │  │ Thread │ │ Thread │  │    │ Writer │  │ Writer │    │ │
│  │  └────────┘ └────────┘  │    └────────┘  └────────┘    │ │
│  └──────────────────────────┴────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ RTSP/TCP
                              ▼
                    ┌──────────────────┐
                    │  IP Cameras      │
                    │  (Hikvision,     │
                    │   Dahua, Axis)   │
                    └──────────────────┘
```

### Data Flow

```
Camera → RTSP Stream → CameraHandler → Frame Buffer
                                            │
                                            ▼
                                    Detection Worker
                                            │
                                            ▼
                                    YOLOv8s Detector
                                            │
                                            ▼
                                    Object Tracker
                                            │
                                            ▼
                                    Face Recognizer
                                            │
                                            ├─→ Recording Writer → MP4 Files
                                            ├─→ Database Logger → SQLite
                                            ├─→ MJPEG Stream → Web UI
                                            └─→ SSE Notifications → Dashboard
```

---

## 📂 Project Structure

```
ai-vigilance/
├── 📄 app.py                          # Main FastAPI application (port 9000)
├── 📄 requirements.txt                # Python dependencies
├── 📄 docker-compose.yml              # Docker deployment configuration
├── 📄 Dockerfile                      # Container image definition
├── 📄 README.md                       # Project overview (this file)
├── 📄 docs.md                         # Technical documentation
│
├── 📁 camera_server/                  # Camera processing server (port 9001)
│   ├── server.py                      # FastAPI server for AI pipeline
│   └── client.py                      # HTTP client for camera server API
│
├── 📁 cameras/                        # Camera management
│   └── camera_manager.py              # RTSP handler, auto-discovery, threads
│
├── 📁 core/                           # Core system modules
│   ├── pipeline.py                    # AI detection pipeline
│   ├── startup.py                     # System initialization
│   ├── state.py                       # Shared global state
│   ├── resource_guard.py              # Dynamic CPU throttling
│   ├── diagnostics.py                 # Crash handler & auto-restart
│   ├── auth.py                        # JWT authentication
│   └── logging_config.py              # Logging configuration
│
├── 📁 utils/                          # AI utilities
│   ├── detector.py                    # YOLOv8s detection + preprocessing
│   ├── tracker.py                     # Hungarian + HSV tracker
│   ├── recognizer.py                  # FaceNet + MTCNN recognition
│   └── hw_manager.py                  # Hardware detection (GPU, encoders)
│
├── 📁 services/                       # Business services
│   └── recording.py                   # Video recording service
│
├── 📁 database/                       # Data persistence
│   └── sqlite_manager.py              # SQLite3 with WAL mode (11 tables)
│
├── 📁 routes/                         # API endpoints
│   ├── cameras.py                     # Camera CRUD operations
│   ├── people.py                      # Person registration
│   ├── recordings.py                  # Video playback
│   ├── search.py                      # Forensic search
│   ├── analytics.py                   # Dashboard metrics
│   ├── auth.py                        # Authentication
│   └── ...
│
├── 📁 templates/                      # Jinja2 HTML templates
│   ├── index.html                     # Landing page
│   ├── dashboard.html                 # Main dashboard
│   ├── cameras.html                   # Camera management
│   ├── people.html                    # Person registry
│   ├── recordings.html                # Video browser
│   ├── search.html                    # Forensic search
│   └── ...
│
├── 📁 static/                         # Frontend assets
│   ├── style.css                      # Main stylesheet
│   ├── script.js                      # Dashboard JavaScript
│   ├── shared.css                     # Shared styles
│   └── shared.js                      # Shared utilities
│
├── 📁 dataset/                        # Registered person images
│   └── PersonName.jpg                 # Face photos for recognition
│
├── 📁 snapshots/                      # Detection snapshots
│   └── YYYY-MM-DD/
│       └── camera_id/
│           └── logs/
│               └── camera_YYYY-MM-DD_HHMMSS.jpg
│
├── 📁 recordings/                     # Video recordings
│   └── YYYY-MM-DD/
│       └── camera_id/
│           ├── HH_MMSS.mp4           # Timestamp-based files
│           └── ...
│
└── 📁 venv/                           # Python virtual environment
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `app.py` | Main entry point, initializes both servers |
| `camera_server/server.py` | AI processing server with models |
| `core/pipeline.py` | Detection → Tracking → Recognition flow |
| `services/recording.py` | Automatic video recording with rotation |
| `utils/detector.py` | YOLOv8s with dynamic preprocessing |
| `utils/tracker.py` | Custom IoU tracker with re-entry buffer |
| `utils/recognizer.py` | FaceNet face recognition |
| `database/sqlite_manager.py` | Database operations (11 tables) |

---

## 📊 Monitoring & Performance

### System Logs

All application logs are written to `app.log` and `crash_forensics.log`:

```bash
# Watch live logs
tail -f app.log

# Filter by component
grep "[RecordingService]" app.log
grep "[ResourceGuard]" app.log
grep "[CameraServer]" app.log

# Check for errors
grep -i "error" app.log | tail -20

# View crash forensics
cat crash_forensics.log
```

### Resource Guard Throttling

The system automatically adjusts performance based on CPU load:

| CPU Usage | Level | Detection FPS | CLAHE | JPEG Quality | Action |
|-----------|-------|---------------|-------|--------------|--------|
| < 75% | ✅ Normal | 6 FPS | ✅ Enabled | 75 | Full performance |
| 75-85% | ⚠️ Warning | 4 FPS | ✅ Enabled | 65 | Light throttle |
| 85-92% | 🔶 High | 3 FPS | ❌ Disabled | 60 | Heavy throttle |
| > 92% | 🔴 Critical | Paused 8s | ❌ Disabled | 55 | Emergency pause |

**Cooldown**: 15 seconds after returning to normal before restoring full 6 FPS

### Performance Metrics

```bash
# Check system status
curl http://localhost:9001/health

# Get camera list
curl http://localhost:9001/cameras

# View occupancy
curl http://localhost:9000/occupancy

# Check recording status
ls -lh recordings/$(date +%Y-%m-%d)/*/
```

### Storage Requirements

| Resolution | FPS | Bitrate | Per Hour | Per Day | Per Week |
|------------|-----|---------|----------|---------|----------|
| 1920x1080 | 10 | ~6 MB/min | ~360 MB | ~8.6 GB | ~60 GB |
| 1280x720 | 10 | ~3 MB/min | ~180 MB | ~4.3 GB | ~30 GB |
| 640x480 | 10 | ~1 MB/min | ~60 MB | ~1.4 GB | ~10 GB |

**Multiple Cameras**: Multiply by number of cameras
**Example**: 4 cameras @ 1080p = ~34 GB/day = ~240 GB/week

---

## 🗂️ File Organization

### Recordings Structure
```
recordings/
└── 2026-05-16/                    # Date folder (YYYY-MM-DD)
    ├── gate/                      # Camera ID
    │   ├── 12_4530.mp4           # Started at 12:45:30 PM
    │   ├── 13_0000.mp4           # Hourly rotation at 1:00:00 PM
    │   ├── 14_0000.mp4           # Next hour
    │   └── ...
    └── entrance/
        ├── 09_1520.mp4
        └── ...
```

**Filename Format**: `HH_MMSS.mp4`
- `HH` = Hour (00-23, 24-hour format)
- `MM` = Minute (00-59)
- `SS` = Second (00-59)

**Benefits**:
- ✅ No overwrites (unique timestamps)
- ✅ Chronological sorting
- ✅ Easy gap detection
- ✅ Crash-safe (preserves all recordings)

### Snapshots Structure
```
snapshots/
└── 2026-05-16/
    └── gate/
        └── logs/
            ├── gate_2026-05-16_143022.jpg
            ├── gate_2026-05-16_143045.jpg
            └── ...
```

### Dataset Structure
```
dataset/
├── John_Doe.jpg
├── Jane_Smith.jpg
└── ...
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Server Configuration
MAIN_PORT=9000
CAMERA_SERVER_PORT=9001

# Database
DATABASE_PATH=db.sqlite3

# Recording
RECORDINGS_DIR=recordings
CHUNK_DURATION=3600  # seconds (1 hour)
RECORDING_FPS=10

# Detection
DETECTION_FPS=6
CONFIDENCE_THRESHOLD=0.48
NMS_IOU_THRESHOLD=0.40

# Recognition
FACE_RECOGNITION_THRESHOLD=1.05
MTCNN_CONFIDENCE=0.90

# Resource Management
CPU_WARN_THRESHOLD=75
CPU_HIGH_THRESHOLD=85
CPU_CRITICAL_THRESHOLD=92

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

### Camera Configuration

Cameras are stored in the database. Add via web UI or API:

```python
# Example RTSP URLs
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101  # Hikvision
rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0  # Dahua
rtsp://admin:password@192.168.1.102:554/axis-media/media.amp  # Axis
```

### Hardware Acceleration

The system auto-detects available hardware:

```bash
# Check detected hardware
grep "Hardware" app.log

# Expected output:
[HardwareManager] GPU: AMD Radeon RX 6800 (DirectML)
[HardwareManager] Video Encoder: h264_amf
[HardwareManager] Video Decoder: VAAPI
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Connecting
```bash
# Check RTSP URL
ffprobe -rtsp_transport tcp rtsp://user:pass@ip:port/path

# Test with VLC
vlc rtsp://user:pass@ip:port/path

# Check firewall
sudo ufw allow 554/tcp  # RTSP port
```

#### 2. High CPU Usage
```bash
# Reduce camera count
# Lower resolution in camera settings
# Enable hardware acceleration
# Reduce detection FPS in config
```

#### 3. Recording Not Starting
```bash
# Check logs
grep "RecordingService" app.log

# Verify FFmpeg
ffmpeg -version

# Check disk space
df -h
```

#### 4. Face Recognition Not Working
```bash
# Check model files
ls -lh ~/.cache/torch/hub/checkpoints/

# Test MTCNN
python -c "from facenet_pytorch import MTCNN; MTCNN()"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 5. Database Locked
```bash
# Check WAL mode
sqlite3 db.sqlite3 "PRAGMA journal_mode;"

# Should output: wal

# If not, enable it:
sqlite3 db.sqlite3 "PRAGMA journal_mode=WAL;"
```

### Debug Mode

Enable debug logging:

```python
# In app.py, change:
logging.basicConfig(level=logging.DEBUG)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linter
flake8 .

# Format code
black .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - YOLOv8 object detection
- **[facenet-pytorch](https://github.com/timesler/facenet-pytorch)** - Face recognition
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[FFmpeg](https://ffmpeg.org/)** - Video processing

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-vigilance/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-vigilance/discussions)

---

<div align="center">

**Made with ❤️ by the AI Vigilance Team**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-vigilance?style=social)](https://github.com/yourusername/ai-vigilance/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-vigilance?style=social)](https://github.com/yourusername/ai-vigilance/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/yourusername/ai-vigilance?style=social)](https://github.com/yourusername/ai-vigilance/watchers)

</div>
