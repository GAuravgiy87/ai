# 🏛️ AI Vigilance: System Architecture Deep-Dive
**A Technical Reference for the Multi-Layered AI Surveillance Ecosystem**

---

## 1. Executive Summary
AI Vigilance is built on a **3-Layer Distributed Architecture** designed for high-throughput video processing and real-time behavioral intelligence. By decoupling the **Processing Engine** from the **Web Interface**, the system ensures that heavy AI computations never interfere with user experience or system stability.

---

## 2. Layer 1: Presentation (The User Interface)
The frontend is a modern, responsive dashboard that communicates with the backend via three distinct protocols.

*   **Web Dashboard (HTTPS/REST):** Used for configuration (adding cameras, managing users) and historical data retrieval (viewing logs and analytics).
*   **SSE Listener (Server-Sent Events):** A persistent, unidirectional pipe that allows the server to "push" real-time person-detection alerts to the user within milliseconds.
*   **MJPEG Player (Video Stream):** Leverages the **Proxy Pattern**. Instead of connecting directly to the camera engine, the dashboard fetches streams from the Web Server (Port 9000), which proxies the data from the Camera Engine. This simplifies network security and prevents CORS (Cross-Origin Resource Sharing) errors.

---

## 3. Layer 2: Web Server (The Control Plane)
Built on **FastAPI and Uvicorn**, this layer manages the application state and user access.

*   **Security & Auth Router:** Implements JWT (JSON Web Token) authentication for secure login and permission-based access to camera feeds.
*   **Analytics Engine:** Aggregates raw detection data into meaningful trends, such as occupancy reports and peak activity times.
*   **SQLite3 Database (WAL Mode):** 
    *   **Architecture Choice:** The system uses **Write-Ahead Logging (WAL)**.
    *   **Rationale:** WAL allows the Camera Engine to write detection logs at high frequency while simultaneously allowing the Analytics Engine to read those logs without causing database "locked" errors.
*   **Business Services Layer:** The central orchestration point that validates inputs and coordinates data flow between the database and the UI routers.

---

## 4. Layer 3: Camera Engine (The Processing Server)
This is the "heavy-lifting" layer running on **Port 9001**. It handles raw data ingestion and high-speed AI inference.

*   **AI Pipeline (Accelerated Processing):**
    *   **Inference:** Uses **YOLOv8** for real-time person detection.
    *   **Biometrics:** Implements **FaceNet (512-d embeddings)** for facial recognition, converting faces into mathematical vectors.
    *   **Acceleration:** Utilizes **OpenCL/ROCm** for GPU-accelerated frame preprocessing (resizing and normalization).
*   **Internal Detection Worker Pool:** A thread-based pool that prevents the camera stream from "stuttering" during heavy AI load. It ensures frames are processed in parallel.
*   **FFmpeg HW Encoder (Infrastructure):** 
    *   Detects available hardware (Intel **QSV** or AMD **AMF**).
    *   Compresses the raw AI-annotated frames into efficient H.264 video files for recording, saving up to 70% of CPU resources.
*   **Event Sender:** Automatically generates events when a person enters or exits a frame, broadcasting these to the Web Server to trigger user alerts.

---

## 5. Sequential Data Flow (The Life of a Frame)
1.  **Ingestion:** The `RTSP Ingestion` module pulls raw video from an IP camera.
2.  **AI Analysis:** The frame is sent to the `Detection Worker Pool`. YOLOv8 finds a person; FaceNet identifies them.
3.  **State Management:** The result is stored in `Shared State` and written to the `SQLite3 DB`.
4.  **Encoding:** FFmpeg encodes the frame with a visual bounding box and saves it to disk.
5.  **Alerting:** The `Event Sender` notifies Layer 2, which then pushes an alert to Layer 1 via **SSE**.
6.  **Viewing:** The user sees the person on the dashboard and receives an instant notification.

---

## 6. Performance Optimization Summary
*   **Resource Guard:** Monitors system health and dynamically throttles AI FPS if CPU/RAM usage is too high.
*   **Shared Memory:** Uses shared memory structures for fast communication between the recording threads and the MJPEG stream output.
*   **Edge Computing:** 100% of processing is local, ensuring zero latency from cloud round-trips and maximum data privacy.
