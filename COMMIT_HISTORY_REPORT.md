# AI Vigilance: Complete Commit History Report
---

## **Commit Order (Chronological, Oldest to Newest)**

---

---

## **1. Commit: `a40f1d19b46508fdaec6485fcf47062aff8f3117`**
### **Author:** ownai63-star <ownai63@gmail.com>
### **Date:** Fri May 8 11:54:25 2026 +0530
### **Message:** terminal log

---

---

## **2. Commit: `2ea8f0ac0fddf6233101bd8f62c5b7f49f2e38d0` (origin/ai)**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Sun May 10 20:20:03 2026 +0530
### **Message:** update with all latest - ai2 code to ai

---

### **Files Changed (18 files):**
- **ARCHITECTURE_REPORT.md** (+60 lines)
- **app.py** (+2, -2)
- **camera_server/server.py** (+5 lines)
- **cameras/camera_manager.py** (+5, -13)
- **core/pipeline.py** (+135, -71)
- **core/resource_guard.py** (+5, -4)
- **core/startup.py** (+11, -7)
- **database/sqlite_manager.py** (+92, -34)
- **db.sqlite3-shm** (Binary)
- **db.sqlite3-wal** (Binary)
- **docs.md** (+107 lines)
- **routes/dashboard.py** (+2, -20)
- **routes/detections.py** (+2, -2)
- **routes/people.py** (+11, -7)
- **routes/recordings.py** (+23, -12)
- **routes/search.py** (+33 lines)
- **system.md** (+112 lines)
- **utils/recognizer.py** (+183, -108)

---

### **Key Changes & Logic:**
- Added extensive documentation: ARCHITECTURE_REPORT.md, docs.md, system.md
- Refactored core pipeline logic
- Updated database manager (sqlite_manager.py) with improvements
- Enhanced face recognizer with better functionality
- Added search functionality in routes/search.py

---

---

## **3. Commit: `d7c82fa0e34b42515983790d68ae8d56605656d9`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Sun May 10 22:07:11 2026 +0530
### **Message:** chore: core stability fixes, forensic search optimization, and hardware acceleration

---

### **Full Commit Description:**
```
Stability & Bug Fixes:
- Resolved 23 critical bugs including FFmpeg streaming memory leaks and thread deadlocks.
- Fixed camera server startup conflicts and improved RTSP probe reliability.
- Implemented robust database WAL checkpointing to prevent unbounded growth.
- Fixed malformed database issues with emergency recovery for cameras and persons.

Performance & Search Optimization:
- Vectorized similarity search using GPU-accelerated NumPy/Torch operations.
- Implemented TRUE GPU batching (batch size 32) for high-speed forensic video scans.
- Added global L2 embedding normalization for consistent multi-camera recognition.
- Optimized MTCNN face alignment with GPU support and forensic fallback cropping.
```

---

### **Files Changed (8 files):**
- **camera_server/server.py** (+25, -21)
- **database/sqlite_manager.py** (+28, -3)
- **routes/recordings.py** (+22, -27)
- **templates/detection_logs.html** (+5, -7)
- **templates/registered_detections.html** (+7, -11)
- **templates/search.html** (+19, -16)
- **utils/detect_gpu.ps1** (+23 lines)
- **utils/hw_manager.py** (+31, -39)

---

### **Key Changes & Logic:**
- **Stability & Bug Fixes**:
  - Resolved FFmpeg streaming memory leaks
  - Fixed thread deadlocks
  - Improved camera server startup reliability
  - Enhanced RTSP probe reliability
  - Added WAL checkpointing to database manager for better growth control
  - Added emergency recovery for corrupted camera/person records
- **Performance & Search**:
  - Vectorized similarity search with GPU acceleration (NumPy/Torch)
  - Implemented GPU batching (size 32) for faster forensic video processing
  - Added L2 normalization of embeddings for consistent multi-camera recognition
  - Optimized MTCNN face alignment with GPU support
  - Added fallback face cropping for forensic use cases
- **Hardware Management**: Updated hw_manager.py and detect_gpu.ps1

---

---

## **4. Commit: `cc42a3551b1299fd73e50333fa11bc88e0385d9c`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Mon May 11 15:09:44 2026 +0530
### **Message:** Fix recognition model availability by falling back to global pipeline recognizer

---

### **Files Changed (2 files):**
- **routes/people.py** (+15, -8)
- **routes/search.py** (+15, -10)

---

### **Key Changes & Logic:**
- Modified routes/people.py and routes/search.py to add fallback logic
- When recognition model isn't available directly, now falls back to the global pipeline recognizer
- Prevents errors when model isn't initialized in the route context

---

---

## **5. Commit: `a5ea062367072391ecfd5a8f4cd411ad209d5a44`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Mon May 11 15:58:51 2026 +0530
### **Message:** Optimize CPU usage: fix thread leaks, improve RTSP stability, and implement async camera client

---

### **Files Changed (6 files):**
- **camera_server/client.py** (+36, -55)
- **cameras/camera_manager.py** (+6, -3)
- **core/pipeline.py** (+10, -73)
- **core/startup.py** (+3, -3)
- **routes/cameras.py** (+6, -10)
- **routes/dashboard.py** (+1, -3)

---

### **Key Changes & Logic:**
- **Thread Leaks Fixed**: Removed unnecessary threads that weren't being properly joined/cleaned up
- **RTSP Stability**: Improved RTSP connection handling and error recovery in camera_manager.py
- **Async Camera Client**: Rewrote camera_server/client.py to be async, reducing blocking calls and improving performance
- **Core Pipeline**: Simplified core/pipeline.py by removing redundant logic, cutting down CPU usage
- **Route Optimizations**: Updated routes/cameras.py and routes/dashboard.py for efficiency

---

---

## **6. Commit: `09e015cb8bbf6ca800fd453025a6bb1e9c44dbdd`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Mon May 11 16:02:38 2026 +0530
### **Message:** Fix startup RuntimeError by converting camera server initialization to async

---

### **Files Changed (1 file):**
- **core/startup.py** (+6, -7)

---

### **Key Changes & Logic:**
- Modified core/startup.py to make camera server initialization async
- This fixed a RuntimeError that occurred during startup due to synchronous initialization of async components
- Changed initialization flow to properly await async setup tasks

---

---

## **7. Commit: `21a870d1f4c14b2e3fdbbf60eb29c00417906c24`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Thu May 14 21:34:16 2026 +0530
### **Message:** update the video issues 1

---

### **Files Changed (6 files):**
- **camera_server/server.py** (+47 lines)
- **cameras/camera_manager.py** (+2, -2)
- **core/pipeline.py** (+106, -57)
- **db.sqlite3-shm** (Binary)
- **db.sqlite3-wal** (0 bytes)
- **utils/detector.py** (+18, -24)

---

### **Key Changes & Logic:**
- Major updates to core/pipeline.py video processing logic
- Enhanced camera_server/server.py with new video handling capabilities
- Refinements to person detector (utils/detector.py)
- Improvements to camera manager for more stable video acquisition

---

---

## **8. Commit: `00eb3370227571e019cb9f8f4b5f416834829cdd`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Fri May 15 16:07:41 2026 +0530
### **Message:** update the video issues 2

---

### **Files Changed (14 files):**
- **ARCHITECTURE_REPORT.md** (+180, -48)
- **README.md** (+104, -65)
- **camera_server/server.py** (+8, -6)
- **core/pipeline.py** (+241, -59)
- **core/startup.py** (+2 lines)
- **core/state.py** (+3, -3)
- **db.sqlite3-shm** (Binary)
- **db.sqlite3-wal** (Binary)
- **docs.md** (+173, -69)
- **im.md** (+262, -1)
- **routes/recordings.py** (+22, -23)
- **scratch/test_recording.py** (+107 lines)
- **system.md** (+115, -53)

---

### **Key Changes & Logic:**
- Comprehensive updates to core/pipeline.py's video pipeline
- Added test script scratch/test_recording.py for recording validation
- Updated all documentation files with new architecture and usage info
- Added im.md with new content
- Minor tweaks to camera server, startup, and state management
- Improvements to recordings route logic

---

---

## **9. Commit: `cdfb433d61924271dbfeafef07d1e46c179e9274`**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Fri May 15 16:39:31 2026 +0530
### **Message:** update the recording 3

---

### **Files Changed (4 files):**
- **core/pipeline.py** (+20, -14)
- **core/state.py** (+2, -1)
- **db.sqlite3-shm** (Binary)
- **db.sqlite3-wal** (Binary)

---

### **Key Changes & Logic:**
- Minor but important refinements to core/pipeline.py's recording logic
- Small update to core/state.py for recording state management
- Fixes for recording reliability issues

---

---

## **10. Commit: `33b1588d17ea044255ae2d50824187b6c4e81804` (origin/ai2, ai2)**
### **Author:** Tarun Kumar Singh <tarunkumarsingh295@gmail.com>
### **Date:** Sat May 16 14:31:01 2026 +0530
### **Message:** update the video issues 4

---

### **Files Changed (17 files):**
- **ARCHITECTURE_REPORT.md** (-220 lines, DELETED)
- **README.md** (+704, -128)
- **app.py** (+15, -7)
- **camera_server/server.py** (+23, -77)
- **core/pipeline.py** (+58, -280)
- **core/startup.py** (+2, -5)
- **core/state.py** (+7, -8)
- **docs.md** (-217 lines, DELETED)
- **im.md** (-495 lines, DELETED)
- **routes/recordings.py** (+24, -52)
- **scratch/add_test_camera.py** (-24 lines, DELETED)
- **scratch/check_cameras.py** (-18 lines, DELETED)
- **scratch/enable_all_recordings.py** (-52 lines, DELETED)
- **scratch/test_recording.py** (-107 lines, DELETED)
- **services/__init__.py** (+5 lines)
- **services/recording.py** (+505 lines, NEW)
- **system.md** (-188 lines, DELETED)

---

### **Key Changes & Logic:**
- **Major Refactoring**:
  - Deleted redundant documentation files (ARCHITECTURE_REPORT.md, docs.md, im.md, system.md)
  - Deleted scratch files (add_test_camera.py, check_cameras.py, enable_all_recordings.py, test_recording.py)
- **New Recording Service**:
  - Created new services/ directory
  - Added services/__init__.py
  - Implemented brand new services/recording.py (505 lines) - full refactoring of recording logic into dedicated service
- **Core Code Simplification**:
  - Massively simplified core/pipeline.py by moving recording logic out to dedicated service
  - Refactored camera_server/server.py
  - Updated app.py for new architecture
  - Updated core/startup.py and core/state.py
- **Updated README.md** with full new documentation (704 lines added!)

---

---

## **Summary of Overall Evolution**

The commit history shows a clear path of:
1. **Documentation & Initial Updates** (a40f1d1 → 2ea8f0a)
2. **Stability & Performance Overhaul** (d7c82fa)
3. **Model Availability Fixes** (cc42a35)
4. **CPU & Async Optimizations** (a5ea062 → 09e015c)
5. **Video Issue Fixes** (21a870d → 00eb337 → cdfb433)
6. **Final Major Refactoring & Cleanup** (33b1588)

The final commit (33b1588) is particularly significant, as it:
- Consolidates recording logic into a dedicated service
- Removes redundant/obsolete files
- Provides comprehensive, up-to-date README.md documentation
- Simplifies core pipeline for better maintainability
