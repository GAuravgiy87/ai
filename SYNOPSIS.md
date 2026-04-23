# Synopsis

## Project Title
**AI Vigilance: Smart Multi-Camera Surveillance System**

---

## Submitted By

| Field | Details |
|---|---|
| Degree | Bachelor of Technology (B.Tech) |
| Branch | Computer Science & Engineering |
| Academic Year | 2025–2026 |

---

## Guide

| Field | Details |
|---|---|
| Project Guide | [Guide Name] |
| Department | Computer Science & Engineering |
| Institution | [Institution Name] |

---

## 1. Introduction

Surveillance systems in educational institutions and public infrastructure have traditionally been passive — cameras record footage, but no real-time analysis is performed. Security personnel are required to manually review hours of video after an incident, which is slow, error-prone, and often too late to be useful.

With the rapid advancement of deep learning and computer vision, it is now feasible to deploy intelligent surveillance that detects, tracks, and identifies individuals in real time — on commodity hardware, without any cloud dependency. AI Vigilance is a self-hosted, privacy-preserving intelligent surveillance platform built to address this gap.

---

## 2. Problem Statement

Existing CCTV infrastructure at most institutions suffers from the following limitations:

- Cameras record passively with no real-time analysis or alerting.
- There is no automated person counting or occupancy monitoring.
- Individuals cannot be identified or tracked across multiple cameras.
- Finding a specific person in historical recordings requires manual review.
- Commercial AI surveillance solutions are expensive and require cloud connectivity.
- Generic object detectors produce false positives on static objects like gate poles, signboards, and shadows.

AI Vigilance is designed to solve all of the above within a single, deployable system.

---

## 3. Objectives

1. Detect persons in real time from multiple simultaneous RTSP camera streams using YOLOv8.
2. Track each detected person with a persistent unique ID using a custom zero-ghosting IoU tracker.
3. Recognize registered persons by face using FaceNet with ≥90% confidence.
4. Assign a consistent cross-camera identity to every person via a Global Re-ID engine.
5. Record all camera feeds continuously as browser-compatible H.264 MP4 files.
6. Enable forensic search of historical recordings by person name or uploaded photo.
7. Provide a live web dashboard with MJPEG streams, person counts, alerts, and analytics.
8. Filter false positives (gate poles, walls, small objects) using multi-stage detection gates.
9. Deploy the entire system as a Docker container for reproducible, one-command setup.

---

## 4. Proposed System

AI Vigilance is a FastAPI-based web application that runs a fully threaded AI pipeline for each connected camera. The system is structured as follows:

### 4.1 Three-Thread Pipeline (per camera)

- **Thread A — Detection**: Grabs the latest camera frame and runs YOLOv8n inference at up to 15 FPS. Results are written atomically to a shared pipe.
- **Thread B — Render**: Reads the latest detections, runs the IoU tracker, draws bounding boxes and identity labels, encodes the frame as JPEG, and publishes it for streaming. Runs at 4 FPS.
- **Thread C — Recognition**: An async executor worker that runs MTCNN face detection followed by FaceNet embedding generation. Results are cached per track ID to avoid redundant inference.

This decoupled design ensures that slow AI inference never blocks the video render loop.

### 4.2 AI Components

| Component | Model | Task |
|---|---|---|
| Person Detection | YOLOv8n (Ultralytics) | Detect persons in each frame |
| Face Detection | MTCNN (facenet-pytorch) | Verify and crop front-facing faces |
| Face Recognition | InceptionResnetV1 / VGGFace2 | Generate 512D biometric embeddings |
| Object Tracking | Custom IoU + Velocity Tracker | Assign persistent IDs, zero ghosting |
| Cross-Camera Re-ID | L2 embedding matching | Consistent identity across cameras |

### 4.3 Key Features

- **Live person count** per camera, visible on the dashboard overlay.
- **24-hour unique visitor count** per camera from the database.
- **Journey tracking** — chronological camera path for every person seen.
- **Forensic video search** — scan recordings for a specific person by name or photo.
- **Auto-split recordings** — hourly MP4 segments with `+faststart` for web playback.
- **Real-time SSE alerts** — browser notifications when a registered person is detected.
- **RTSP auto-discovery** — probes 15+ common stream paths to auto-configure any camera brand.
- **Hardware acceleration** — AMD ROCm for FaceNet, Intel VAAPI for video decode.

---

## 5. Methodology

The development of AI Vigilance follows a structured pipeline-based methodology, divided into the following phases:

### Phase 1 — Requirement Analysis
Identified the limitations of existing passive CCTV systems at the institution. Defined functional requirements (real-time detection, face recognition, recording, search) and non-functional requirements (low latency, CPU-capable, self-hosted, no cloud dependency).

### Phase 2 — System Design
Designed the 3-thread decoupled pipeline architecture to separate detection, rendering, and recognition concerns. Designed the SQLite3 schema with 10 tables to support all data requirements. Defined the file storage convention (`YYYY-MM-DD/{camera_id}/type/`).

### Phase 3 — Camera Integration
Implemented `CameraHandler` with OpenCV + FFmpeg backend for RTSP capture at 25 FPS. Built `probe_rtsp_url()` to auto-discover stream paths across 15+ camera brands. Added Intel VAAPI hardware decode support via GStreamer pipeline.

### Phase 4 — Detection Module
Integrated YOLOv8n restricted to the `person` class. Implemented a multi-stage filtering pipeline:
- Minimum/maximum size gates
- Aspect ratio gate (1.2 – 4.5) to reject non-human shapes
- Minimum area gate (800 px²)
- Bottom-edge clipping filter to reject gate poles and fence tops at frame boundaries

### Phase 5 — Tracking Module
Built a custom IoU + velocity-prediction tracker from scratch:
- Exponential moving average velocity (α = 0.9) for smooth motion prediction
- Two-pass matching: IoU first, then center-distance for fast movers
- Zero-ghosting output policy — only age-0 tracks are rendered
- Latency compensation to shift boxes forward by `velocity × pipeline_lag`

### Phase 6 — Face Recognition Module
Integrated MTCNN for face detection and verification (O-Net threshold 0.90). Integrated InceptionResnetV1 pretrained on VGGFace2 for 512D embedding generation. Set match threshold at Euclidean distance 0.40 for high-confidence identification only. Added thread-safe locking for concurrent multi-camera recognition.

### Phase 7 — Global Re-ID Engine
Built a session-level global identity registry. For each tracked person:
- If recognized → global ID = registered name
- If unknown → match against global embedding registry (threshold 0.75)
- If no match → register as new `U-XXXX` identity

All sightings are logged to the `journeys` table for cross-camera movement tracking.

### Phase 8 — Recording & Storage
Implemented FFmpeg-based H.264 MP4 recording via stdin pipe at 2 FPS. Auto-splits recordings every hour. Implemented async file I/O via a `transfer_queue` to avoid blocking the render thread. Implemented hourly storage cleanup (snapshots > 24h, recordings > 2 days).

### Phase 9 — Web Dashboard & API
Built a FastAPI application with MJPEG live stream endpoints, REST API for camera management, person registration, search and analytics, SSE for real-time browser notifications, session-cookie authentication, and Jinja2-rendered pages for dashboard, recordings, journey tracker, and forensic search.

### Phase 10 — Containerization & Deployment
Wrote a `Dockerfile` (Python 3.11-slim, FFmpeg, VAAPI, GStreamer, ROCm OpenCL) and `docker-compose.yml` with GPU passthrough, persistent volumes, and resource limits. Tested deployment on Linux with AMD RX 550 + Intel iGPU hardware.

### Phase 11 — Testing & Optimization
Tested at DEI Gate 5 and Digital Lab camera locations. Identified and fixed false detections (gate pole), MTCNN empty tensor crashes, SQLite write contention, and pipeline lag. Tuned detection thresholds, tracker parameters, and recognition cooldowns based on real-world results.

---

## 6. Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Object Detection | YOLOv8n (Ultralytics) |
| Face Recognition | FaceNet / InceptionResnetV1 (VGGFace2) |
| Face Detection | MTCNN (facenet-pytorch) |
| Deep Learning | PyTorch ≥ 2.0 |
| Computer Vision | OpenCV (headless) |
| Video Encoding | FFmpeg (H.264, CRF 32) |
| Database | SQLite3 |
| Frontend | Jinja2 + Vanilla CSS (Glassmorphism UI) |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11 |

---

## 7. System Architecture

```
IP Cameras (RTSP)
       │
       ▼
CameraManager (per-camera capture threads @ 25 FPS)
       │
       ▼
3-Thread AI Pipeline (per camera)
  ├── Thread A: YOLOv8 Detection @ 15 FPS
  ├── Thread B: Tracker + Render @ 4 FPS
  └── Thread C: FaceNet + Re-ID (async)
       │
  ┌────┴────────────────┐
  ▼                     ▼
SQLite3 DB         FFmpeg MP4
(events, persons,  (H.264, 2 FPS,
 journeys)          +faststart)
       │
       ▼
FastAPI MJPEG Stream → Browser Dashboard
```

---

## 8. Database Design

The system uses SQLite3 with 10 tables:

| Table | Purpose |
|---|---|
| `cameras` | Registered camera IDs and RTSP sources |
| `persons` | Registered persons with FaceNet embeddings (BLOB) |
| `global_identities` | Cross-camera Re-ID registry |
| `journeys` | Chronological sighting log per person |
| `detection_snapshots` | Per-frame snapshots with bounding box data |
| `occupancy_logs` | Time-series person count per camera |
| `video_recordings` | Recording metadata |
| `registered_detections` | History of registered person sightings |
| `alerts` | Detection and intrusion alerts |
| `camera_settings` | Per-camera configuration |

---

## 9. Expected Outcomes

- Real-time detection and tracking of persons across multiple RTSP camera feeds simultaneously.
- Accurate face recognition of registered individuals with ≥90% confidence.
- Consistent cross-camera identity assignment for both registered and unknown persons.
- Continuous H.264 MP4 recordings organized by date and camera.
- A searchable history of all detections, with forensic video search capability.
- A live web dashboard accessible from any browser on the local network.
- Elimination of false positives from static objects (gate poles, walls) via multi-stage filtering.

---

## 10. Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel Core i5 (4 cores) | Intel Core i7 (8 cores) |
| RAM | 4 GB | 8 GB |
| Storage | 20 GB | 100 GB (for recordings) |
| GPU (optional) | — | AMD RX 550 (ROCm) or NVIDIA GTX 1050 |
| OS | Ubuntu 20.04+ / Windows 10+ | Ubuntu 22.04 LTS |
| Network | 100 Mbps LAN | Gigabit LAN |

---

## 11. Software Requirements

- Python 3.11
- PyTorch ≥ 2.0
- Ultralytics YOLOv8
- facenet-pytorch ≥ 2.5.3
- OpenCV (headless) ≥ 4.8
- FastAPI ≥ 0.100
- FFmpeg (system)
- Docker + Docker Compose (for containerized deployment)

---

## 12. References

### Phase 1 — Requirement Analysis
Identified the limitations of existing passive CCTV systems at the institution. Defined functional requirements (real-time detection, face recognition, recording, search) and non-functional requirements (low latency, CPU-capable, self-hosted, no cloud dependency).

### Phase 2 — System Design
Designed the 3-thread decoupled pipeline architecture to separate detection, rendering, and recognition concerns. Designed the SQLite3 schema with 10 tables to support all data requirements. Defined the file storage convention (`YYYY-MM-DD/{camera_id}/type/`).

### Phase 3 — Camera Integration
Implemented `CameraHandler` with OpenCV + FFmpeg backend for RTSP capture at 25 FPS. Built `probe_rtsp_url()` to auto-discover stream paths across 15+ camera brands. Added Intel VAAPI hardware decode support via GStreamer pipeline.

### Phase 4 — Detection Module
Integrated YOLOv8n restricted to the `person` class. Implemented a multi-stage filtering pipeline:
- Minimum/maximum size gates
- Aspect ratio gate (1.2 – 4.5) to reject non-human shapes
- Minimum area gate (800 px²)
- Bottom-edge clipping filter to reject gate poles and fence tops at frame boundaries

### Phase 5 — Tracking Module
Built a custom IoU + velocity-prediction tracker from scratch:
- Exponential moving average velocity (α = 0.9) for smooth motion prediction
- Two-pass matching: IoU first, then center-distance for fast movers
- Zero-ghosting output policy — only age-0 tracks are rendered
- Latency compensation to shift boxes forward by `velocity × pipeline_lag`

### Phase 6 — Face Recognition Module
Integrated MTCNN for face detection and verification (O-Net threshold 0.90). Integrated InceptionResnetV1 pretrained on VGGFace2 for 512D embedding generation. Set match threshold at Euclidean distance 0.40 for high-confidence identification only. Added thread-safe locking for concurrent multi-camera recognition.

### Phase 7 — Global Re-ID Engine
Built a session-level global identity registry. For each tracked person:
- If recognized → global ID = registered name
- If unknown → match against global embedding registry (threshold 0.75)
- If no match → register as new `U-XXXX` identity
All sightings are logged to the `journeys` table for cross-camera movement tracking.

### Phase 8 — Recording & Storage
Implemented FFmpeg-based H.264 MP4 recording via stdin pipe at 2 FPS. Auto-splits recordings every hour. Implemented async file I/O via a `transfer_queue` to avoid blocking the render thread. Implemented hourly storage cleanup (snapshots > 24h, recordings > 2 days).

### Phase 9 — Web Dashboard & API
Built a FastAPI application with:
- MJPEG live stream endpoints per camera
- REST API for camera management, person registration, search, and analytics
- SSE (Server-Sent Events) endpoint for real-time browser notifications
- Session-cookie authentication
- Jinja2-rendered pages for dashboard, recordings, journey tracker, and forensic search

### Phase 10 — Containerization & Deployment
Wrote a `Dockerfile` (Python 3.11-slim, FFmpeg, VAAPI, GStreamer, ROCm OpenCL) and `docker-compose.yml` with GPU passthrough, persistent volumes, and resource limits. Tested deployment on Linux with AMD RX 550 + Intel iGPU hardware.

### Phase 11 — Testing & Optimization
Tested at DEI Gate 5 and Digital Lab camera locations. Identified and fixed false detections (gate pole), MTCNN empty tensor crashes, SQLite write contention, and pipeline lag. Tuned detection thresholds, tracker parameters, and recognition cooldowns based on real-world results.

---

## 12. References

1. Redmon et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.
2. Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
3. Bewley et al. (2016). *Simple Online and Realtime Tracking*. ICIP 2016.
4. Schroff et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR 2015.
5. Zhang et al. (2016). *Joint Face Detection and Alignment Using MTCNN*. IEEE Signal Processing Letters.
6. Cao et al. (2018). *VGGFace2: A Dataset for Recognising Faces across Pose and Age*. FG 2018.
7. FastAPI Documentation. https://fastapi.tiangolo.com/
8. PyTorch Documentation. https://pytorch.org/docs/

---

*Synopsis submitted for approval prior to project commencement.*
