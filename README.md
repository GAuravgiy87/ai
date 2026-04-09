# AI-VIGILANCE

**Advanced AI Surveillance & Identification Platform**

AI-VIGILANCE is a high-performance surveillance system for real-time person identification and automated event logging. Optimized for edge deployment with full GPU hardware acceleration at 10 FPS for maximum tracking accuracy.

---

## 🚀 Key Features

### 🧠 High-Memory Identity Tracking
- **Lifetime Recognition**: Maintains a **90-day memory**. Returning visitors (including unknowns) are identified by their unique historical ID (`U-XXXX`).
- **One-Snapshot Policy**: Reference photo taken only on first-ever encounter. Subsequent visits logged silently.
- **Zero Miss Tracking**: Every frame is processed — no frame skipping. 3-frame grace window prevents ID loss during brief occlusion.

### 📅 Advanced Day-wise History
- **Structured Archives**: Recordings, snapshots, and logs auto-organized into daily folders (`YYYY-MM-DD`).
- **History UI**: Date-based browsing for arrivals, discovery events, and frequent visitors.

### ⚡ Performance & Hardware Acceleration
- **10 FPS Processing**: Optimized for tracking accuracy over raw throughput.
- **Multi-GPU Offloading**: Dynamic hardware discovery via **OpenVINO** and **DirectML** (Intel/AMD/NVIDIA).

---

## 🛠️ Installation (Linux / Ubuntu)

```bash
git clone https://github.com/GAuravgiy87/ai.git
cd ai
bash setup_linux.sh
```

### Launch
```bash
bash start.sh
```
Access the dashboard at `http://localhost:8000`.

---

## ⚖️ Security & Privacy
- **Basic Auth**: Secured via admin credentials.
- **Local-First**: All data stored locally in `db.sqlite3`. No cloud dependence.

---

**Developed for Advanced Agentic Coding Projects**
