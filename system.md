# AI Vigilance: Smart Multi-Camera Surveillance System - Comprehensive Technical Guide

## 1. Professional System Overview
AI Vigilance is a production-grade, distributed AI surveillance ecosystem. It is designed to bridge the gap between simple video recording and high-level behavioral intelligence. By leveraging a multi-process architecture and a strictly threaded AI pipeline, it provides real-time insights into camera feeds with minimal latency.

The system is built on the philosophy of **Edge Intelligence**, meaning all AI processing happens locally on your machine. No video data is sent to the cloud, ensuring maximum privacy and speed.

---

## 2. Detailed 3-Layer System Architecture
The system follows a strict layered architecture where each layer communicates with its neighbors through defined interfaces (APIs and Shared States).

### Architecture Visual Map (Mermaid)
```mermaid
graph TD
    subgraph "Layer 1: Presentation (Browser)"
        UI[Web Dashboard - JS/CSS]
        SSE[SSE Listener - Real-time Alerts]
        VLC[MJPEG Player - Live Feed]
    end

    subgraph "Layer 2: Application (FastAPI - Port 9000)"
        AUTH[Auth Router]
        DASH[Dashboard Router]
        REC[Recordings Manager]
        ANA[Analytics Engine]
        DBM[SQLite Manager]
    end

    subgraph "Layer 3: Infrastructure & AI (Core Engine - Port 9001)"
        CS[Camera Server API]
        PIPE[AI Pipeline Thread]
        DET[YOLOv8 Detection Pool]
        TRK[IoU Object Tracker]
        REC_AI[FaceNet Recognizer]
        FFM[FFmpeg MP4 Writer]
        CM[Camera Manager - RTSP/Webcam]
    end

    %% Connections
    UI <-->|HTTP REST| DASH
    SSE <==|SSE Events| PIPE
    VLC <==|MJPEG Stream| CS
    
    DASH <-->|Local API Call| CS
    REC <-->|File Access| FFM
    ANA <-->|SQL Queries| DBM
    
    CS <-->|Shared State| PIPE
    PIPE -->|Submit Frame| DET
    DET -->|Detections| TRK
    TRK -->|Track IDs| REC_AI
    PIPE -->|Rendered Frame| FFM
    CM -->|Raw Frames| PIPE
    
    DBM <-->|Storage| DB[(SQLite Database)]
    FFM -->|Files| DISK[(Storage: MP4/JPG)]
```

---

## 3. Detailed Component & Connection Analysis

### Layer-to-Layer Connectivity
1.  **Layer 1 ↔ Layer 2 (User Interaction)**:
    *   **HTTP/REST**: The Browser sends requests (e.g., "Add Camera", "Search People") to the Application Layer.
    *   **SSE (Server-Sent Events)**: A persistent uni-directional pipe where Layer 2 pushes instant notifications (like a person being detected) to Layer 1.
2.  **Layer 2 ↔ Layer 3 (System Control)**:
    *   **Internal API Calls**: The Main App (Port 9000) acts as a client to the Camera Server (Port 9001). When you toggle a setting on the dashboard, Layer 2 sends a command to Layer 3.
    *   **Shared Data Memory**: Both layers share a "State" object in memory for fast access to current occupancy counts and system health stats.
3.  **Layer 3 ↔ External World (Data Ingest/Output)**:
    *   **RTSP/TCP**: The Camera Manager establishes stable connections to physical IP cameras.
    *   **Subprocess Pipes**: The AI Pipeline feeds raw video data into FFmpeg via standard input pipes for high-speed encoding.

---

## 4. Full Lifecycle of a Detection Event
To understand how the system works "properly," let's follow a single person walking past a camera:

1.  **Ingestion**: The `CameraManager` receives a compressed H.264 stream from the camera. It decodes it into a raw image (frame).
2.  **Detection**: The frame is sent to the `DetectionPool`. **YOLOv8** identifies a "person" object and provides coordinates (a bounding box).
3.  **Tracking**: The `ObjectTracker` compares this box to previous frames. It realizes this is the same person seen 0.5 seconds ago and maintains their **ID #102**.
4.  **Recognition**: If the person's face is clear, the `FaceRecognizer` crops the face, turns it into a mathematical signature (Embedding), and compares it against known faces in the database.
5.  **Alerting**: If a match is found (e.g., "John Doe"), the `NotificationManager` broadcasts an **SSE Event**. Within milliseconds, the browser dashboard flashes a "John Doe Detected" alert.
6.  **Recording**: Simultaneously, the frame is watermarked with the name and ID and sent to **FFmpeg**, which saves it into a permanent MP4 file for later review.

---

## 5. Security, Privacy & Ethics
*   **Local Processing**: Unlike many commercial systems, AI Vigilance processes 100% of the video on-site. No data ever leaves your local network.
*   **Biometric Security**: Face signatures are stored as 512-dimensional numbers (Embeddings). Even if the database is stolen, the original face images cannot be reconstructed from these numbers.
*   **Access Control**: The system includes a multi-user authentication layer to ensure only authorized personnel can view live feeds or historical recordings.

---

## 6. Performance Optimization: The "Resource Guard"
Surveillance is resource-intensive. To ensure the system never freezes your computer:
*   **Dynamic Throttling**: If the CPU usage exceeds 90%, the `ResourceGuard` automatically tells the AI to skip every other frame, reducing load instantly.
*   **Memory Management**: The system uses a "circular buffer" for frames, ensuring that old data is cleared out and never causes "Out of Memory" crashes.
*   **Hardware Acceleration**: The system automatically detects if you have an Intel, AMD, or NVIDIA chip and uses specialized hardware to encode video, saving up to 70% of CPU power.

---

## 7. Non-Technical Glossary
*   **RTSP**: The "language" cameras use to send video over a network.
*   **YOLO (You Only Look Once)**: A world-class AI model that can find objects in a fraction of a second.
*   **FPS (Frames Per Second)**: How "smooth" the video is. The system typically runs at 2-6 FPS for AI, which is perfect for security.
*   **Embedding**: A mathematical "fingerprint" of a face used for recognition.
*   **SSE**: A technology that lets the server "talk" to your browser without you having to click anything.

---
*Documentation Version: 3.0 | Status: Final Review Complete*
