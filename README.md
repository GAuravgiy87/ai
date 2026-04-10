# AI Vigilance: Smart Multi-Camera Surveillance System

A production-ready, real-time AI surveillance dashboard designed for multi-camera RTSP deployments. AI Vigilance detects, tracks, and optionally identifies individuals across multiple cameras simultaneously using a fully threaded AI pipeline.

---

## 🚀 Key Features

| Feature | Details |
|---|---|
| **Real-Time Person Detection** | YOLOv8-based detection at 2 FPS per camera, optimized for CPU deployments |
| **Zero-Ghosting Tracking** | Custom IoU tracker that removes bounding boxes instantly when a person leaves |
| **Live HEAD COUNT** | Live per-camera person count visible on the dashboard overlay |
| **TOTAL COUNT** | Cumulative 24-hour unique visitor count per camera from the database |
| **Face Recognition** | Optional FaceNet-based identity matching with registered-person alerts |
| **H.264 Recordings** | Browser-compatible MP4 recordings with fast-start and HTTP range seeking |
| **Organized Storage** | All files auto-sorted by `Day → Camera → Type` |
| **Silent Terminal** | All logs redirected to `app.log`; terminal stays clean |
| **RTSP Auto-Discovery** | Automatically probes 15+ common RTSP stream paths for any camera brand |
| **Active Search Missions** | Scan live feeds and historical recordings for a specific registered person |

---

## 🧠 AI Stack

### 1. YOLOv8 (Ultralytics)
- Real-time person detection on raw camera frames
- Restricted to `person` class only to minimize CPU load

### 2. Custom IoU Tracker (`utils/tracker.py`)
- Assigns unique IDs to each person per camera
- `age < 1` visibility policy: bounding boxes disappear the **instant** a person leaves the frame (zero ghosting)
- 10-frame ID memory for re-identification after brief occlusion

### 3. FaceNet + MTCNN (Optional Recognition)
- MTCNN crops face regions from within YOLO bounding boxes
- FaceNet converts face crops to 512D biometric embeddings
- Matching uses Euclidean distance with a configurable confidence threshold
- Thread-safe with `threading.Lock()` for multi-camera concurrent recognition

---

## 💻 Tech Stack

- **FastAPI + Uvicorn** — Async Python web backend; handles concurrent MJPEG streams
- **OpenCV (headless)** — RTSP capture with TCP transport and low-latency FFMPEG flags
- **SQLite3** — Local database for cameras, persons, recordings, detections, and occupancy
- **Jinja2 + Vanilla CSS** — Glassmorphism UI with live overlays
- **FFmpeg** — H.264 MP4 recording pipeline at 2 FPS with `+faststart` for web playback

---

## 🛠️ Setup & Deployment

### Linux VM (Recommended — Headless)

```bash
# 1. Clone the repository
git clone https://github.com/GAuravgiy87/ai.git -b ai
cd ai

# 2. Run the one-time setup script
chmod +x setup_linux.sh && ./setup_linux.sh

# 3. Start the system
chmod +x start.sh && ./start.sh
```

### Windows (Development)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

### Access the Dashboard
Navigate to `http://<server-ip>:8000` from any browser on the same network.

---

## 📂 Repository Structure

```
ai-vigilance/
├── app.py                  # Main FastAPI app, AI pipeline, all API routes
├── cameras/
│   └── camera_manager.py   # RTSP handler, auto-path discovery, CameraHandler threads
├── utils/
│   ├── tracker.py          # Custom IoU-based person tracker (zero-ghosting)
│   ├── detector.py         # YOLOv8 person detector wrapper
│   └── recognizer.py       # FaceNet + MTCNN face recognition
├── database/
│   └── db_manager.py       # SQLite3 schema, queries, and managers
├── templates/
│   └── index.html          # Main dashboard (live view, recordings, search)
├── static/                 # CSS, JS, icons
├── dataset/                # Registered person face images (auto-created)
├── snapshots/              # Detection snapshots: snapshots/YYYY-MM-DD/cam/
├── recordings/             # MP4 recordings: recordings/YYYY-MM-DD/cam/
├── requirements.txt        # Python dependencies
├── setup_linux.sh          # One-time Linux VM setup script
└── start.sh                # Application launcher script
```

---

## 📋 Monitoring

All application logs are written to `app.log`. The terminal stays **completely silent**.

```bash
# Watch live logs
tail -f app.log

# Check for errors only
grep -i "error" app.log
```

---

## 🔧 Filename Convention

All recordings and snapshots follow a clear, consistent naming pattern:

| File Type | Format | Example |
|---|---|---|
| Recording | `{Camera}_{Date}_{Time}.mp4` | `DEI_Gate_5_2026-04-10_143500.mp4` |
| Detection Snapshot | `{Camera}_{Date}_{Time}.jpg` | `DigitalLab_2026-04-10_143500.jpg` |
| Identity Snapshot | `{Camera}_{Date}_{Time}_ID.jpg` | `Gate5_2026-04-10_143500_ID.jpg` |
