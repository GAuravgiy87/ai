# AI Vigilance: Smart Multi-Camera Surveillance System

A production-ready, real-time AI surveillance dashboard designed for multi-camera RTSP deployments. AI Vigilance detects, tracks, and optionally identifies individuals across multiple cameras simultaneously using a fully threaded AI pipeline.

---

## ⚠️ Prerequisites (CRITICAL)
- **Python 3.11 or 3.12 is REQUIRED.** 
- **Do not use Python 3.13+**, as many AI libraries (like Numpy and ONNX) do not have pre-built binaries for it yet, which will cause massive C++ compilation errors during installation.

---

## 🚀 Quick Start (One-Click!)
### Windows
```powershell
.\run.ps1
```

### Linux/Mac
```bash
chmod +x run.sh
./run.sh
```

That's IT! The script will automatically create a virtual environment, install dependencies, and start the system.

---

## 🛠️ Manual Installation (If One-Click Fails)
If you prefer to set up the environment manually or encounter errors, run these commands:

### 1. Create & Activate Virtual Environment
```powershell
# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```
```bash
# Linux/Mac
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Start the Application
```powershell
python app.py
```

---

## 🚀 Key Features
| Feature | Details |
|---|---|
| **High-Accuracy Person Detection** | YOLOv8s-based detection with dynamic lighting thresholds, tiered size and aspect ratio filters, and frame normalization |
| **Crash-Safe Hourly Recording** | H.264 MKV recordings with index flushed every 2 seconds; partial files are playable |
| **Zero-Ghosting Tracking** | Custom IoU tracker that removes bounding boxes instantly when a person leaves |
| **Live HEAD COUNT** | Live per-camera person count visible on the dashboard overlay |
| **Total COUNT** | Cumulative 24-hour unique visitor count per camera from the database |
| **Face Recognition** | Optional FaceNet-based identity matching with registered-person alerts |
| **Organized Storage** | All files auto-sorted by `Day → Camera → Type` |
| **Silent Terminal** | All logs redirected to `app.log`; terminal stays clean |
| **RTSP Auto-Discovery** | Automatically probes 15+ common RTSP stream paths for any camera brand |
| **Active Search Missions** | Scan live feeds and historical recordings for a specific registered person |

---

## 🧠 AI Stack
### 1. YOLOv8 (Ultralytics)
- Real-time person detection on raw camera frames
- Restricted to `person` class only to minimize CPU load
- Uses yolov8s.pt (small, high-accuracy model; not nano)
- Dynamic confidence thresholds based on frame brightness

### 2. Custom IoU Tracker (`utils/tracker.py`)
- Assigns unique IDs to each person per camera
- `age < 1` visibility policy: bounding boxes disappear the **instant** a person leaves the frame (zero ghosting)
- 10-frame ID memory for re-identification after brief occlusion

### 3. FaceNet + MTCNN (Optional Recognition)
- MTCNN crops face regions from within YOLO bounding boxes
- FaceNet converts face crops to 512D biometric embeddings
- Matching uses Euclidean distance with a configurable confidence threshold
- Thread-safe with `threading.Lock()` for multi-camera concurrent recognition

### 4. Crash-Safe Recording Worker (`background_jobs/recording_worker.py`)
- Hourly chunk rotation (exactly on clock hour)
- MKV container with incremental index flush every 2 seconds
- Partial files playable in case of crash
- 10 FPS recording for smooth video

---

## 💻 Tech Stack
- **FastAPI + Uvicorn** — Async Python web backend; handles concurrent MJPEG streams
- **OpenCV (headless)** — RTSP capture with TCP transport and low-latency FFMPEG flags
- **SQLite3** — Local database for cameras, persons, recordings, detections, and occupancy
- **Jinja2 + Vanilla CSS** — Glassmorphism UI with live overlays
- **FFmpeg** — H.264 MKV recording pipeline at 10 FPS

---

## 📂 Repository Structure
```
ai-vigilance/
├── app.py                      # Main FastAPI app, all API routes
├── run.ps1                     # Windows ONE-CLICK setup & run
├── run.sh                      # Linux/Mac ONE-CLICK setup & run
├── cameras/
│   └── camera_manager.py       # RTSP handler, auto-path discovery, CameraHandler threads
├── core/
│   ├── pipeline.py             # Detection, tracking, and face recognition pipeline
│   ├── state.py                # Global application state management
│   ├── resource_guard.py       # System health monitor (CPU/RAM)
│   └── startup.py              # App startup helpers (camera server, lifespan)
├── utils/
│   ├── tracker.py              # Custom IoU-based person tracker (zero-ghosting)
│   ├── detector.py             # YOLOv8s person detector with preprocessing and filters
│   ├── recognizer.py           # FaceNet + MTCNN face recognition
│   └── hw_manager.py           # Hardware detection (GPU, FFmpeg encoders)
├── background_jobs/
│   ├── __init__.py
│   └── recording_worker.py     # Crash-safe hourly MKV recording worker
├── database/
│   └── sqlite_manager.py       # SQLite3 schema, queries, and managers
├── routes/
│   └── (all FastAPI route modules)
├── templates/
│   └── (all Jinja2 HTML templates)
├── static/                     # CSS, JS, icons
├── dataset/                    # Registered person face images (auto-created)
├── snapshots/                  # Detection snapshots: snapshots/YYYY-MM-DD/cam/
├── recordings/                 # MKV recordings: recordings/YYYY-MM-DD/cam/HH.mkv
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── im.md                       # Accuracy & counting report
```

---

## 📋 Monitoring
All application logs are written to `app.log`. The terminal stays **completely silent**.
```bash
# Watch live logs (Linux/Mac)
tail -f app.log

# Watch live logs (Windows PowerShell)
Get-Content app.log -Wait

# Check for errors only
grep -i "error" app.log
```

---

## 🔧 Filename Convention
All recordings and snapshots follow a clear, consistent naming pattern:
| File Type | Format | Example |
|---|---|---|
| Recording | `YYYY-MM-DD/{camera_id}/HH.mkv` | `2026-06-10/DEI_Gate_5/14.mkv` |
| Detection Snapshot | `YYYY-MM-DD/{camera_id}/logs/{camera_id}_{timestamp}.jpg` | `2026-06-10/DEI_Gate_5/logs/DEI_Gate_5_2026-06-10_143500.jpg` |

---

## Access
After running the one-click script, open your browser to:
`http://localhost:8000`
