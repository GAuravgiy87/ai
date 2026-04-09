# AI Vigilance — Smart Multi-Camera Surveillance System

A production-ready, real-time surveillance platform built for multi-camera environments. Tracks, identifies, and logs individuals across live feeds and recorded video using a fully local AI stack — no cloud dependency.

---

## Features

**Live Surveillance**
- **Adaptive Multi-Camera Streaming**: Supports up to **20 FPS** person and vehicle tracking on capable hardware.
- **HUD Overlays**: Live head counts and per-camera bounding box visualization.
- **Smart Status**: Pause/resume any feed with instant frozen-frame snapshots for investigation.
- **Fullscreen Dashboard**: Immersive monitoring with live count and camera controls.

**Person Detection & Tracking**
- **High-Frequency Detection**: YOLOv8n optimized for up to **20 FPS** (50ms interval).
- **Persistent Tracking**: Custom IoU + center-distance tracker with 50-frame occlusion tolerance.
- **Deduplication**: Deep track merging prevents double-counting during overlaps or re-entry.
- **Color-Coded Bounding Boxes**: Visual identifiers persistent across frames.

**Biometric Identification (Face Recognition)**
- **MTCNN Face Extraction**: Locates faces within person boxes for extreme precision.
- **InceptionResnetV1 (FaceNet)**: Generates 512-d biometric embeddings.
- **Global Re-ID**: Automatically assigns unique IDs (U-XXXX) to unknown persons and tracks them across multiple cameras.
- **Face Registry**: Register known individuals with photos for name-based alerting.

**Vehicle Monitoring & ALPR**
- **Multi-Class Support**: Tracks Cars, Motorcycles, Buses, and Trucks.
- **On-the-Fly ALPR**: License plate recognition via EasyOCR.
- **Safety Compliance**: Automatic occupancy counting and helmet detection heuristics for two-wheelers.
- **Archival**: Side-by-side storage of full investigation frames and plate crops.

**Recording & Archival**
- **H.264 MP4 Recording**: FFmpeg-powered background recording (2 FPS for storage optimization).
- **Auto-Cleanup**: Automated retention policies (Snapshots: 24h, Recordings: 2d, Vehicles: 7d).
- **Searchable Timeline**: Jump to specific detections in historical recordings.

---

## AI Stack

| Component | Model | Purpose |
|---|---|---|
| **Object Detection** | YOLOv8n | People and vehicle classification |
| **Face Detection** | MTCNN | Biometric localization |
| **Face Embedding** | InceptionResnetV1 | 512-d feature extraction |
| **OCR / ALPR** | EasyOCR | License plate text recognition |
| **Tracking** | Custom IoU/Centroid | Temporal object persistence |

---

## Hardware Requirements

| Setup | Graphics Card | Performance Goal |
|---|---|---|
| **Basic** | CPU Only (Fast i7+) | 1-2 Cam @ 2-5 FPS |
| **Standard** | RTX 3060 (12GB) | 2-3 Cam @ 15-20 FPS |
| **Professional** | RTX 4070+ | 4-6 Cam @ 20 FPS |

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed on system
- SQLite3

### 2. Setup (Linux / Windows Bash)
```bash
# Automated setup (installs system deps, venv, and python packages)
bash setup_linux.sh
```

### 3. Start
```bash
# Runs the FastAPI server on http://0.0.0.0:8000
bash start.sh
```

---

## Repository Map

- `app.py`: Central logic, Adaptive FPS processing loops, and API routes.
- `cameras/`: RTSP/Webcam stream management.
- `database/`: SQLite schema and persistence layer.
- `utils/`: Core AI modules (Detector, Recognizer, Tracker, VehicleProcessor).
- `templates/`: Modern glassmorphism UI templates.
- `recordings/`: Local MP4 storage.
- `snapshots/`: Real-time detection snapshots.

---

## API Summary
Full Swagger documentation available at `/docs` when the server is running.

- `POST /api/add_camera`: Initialize a new camera source.
- `GET /api/occupancy`: Current people count metrics.
- `GET /api/vehicle_logs`: History of detected vehicles and plates.
- `GET /api/target_journey/{id}`: Timeline of where a specific ID was seen.
- `GET /api/notifications/stream`: SSE endpoint for real-time dashboard events.
