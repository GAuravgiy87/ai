# AI Vigilance Code Changes Summary

## Commit Comparison: `c81f978704fd0af487f973f24e8b866a0dab5d14` → Current

This document summarizes all changes made between the specified commit and the current state of the codebase.

---

## Table of Contents
1. [Directory & File Renaming](#directory--file-renaming)
2. [Docker & Deployment Improvements](#docker--deployment-improvements)
3. [Application Architecture Changes](#application-architecture-changes)
4. [Data Access Layer Refactoring](#data-access-layer-refactoring)
5. [Configuration & Environment Changes](#configuration--environment-changes)
6. [Documentation Additions](#documentation-additions)

---

## 1. Directory & File Renaming

### Major Directory Structure Changes:

| Old Path | New Path | Description |
|----------|----------|-------------|
| `routes/` | `api/` | API routes moved to `api/` directory for clearer naming |
| `workers/` | `background_jobs/` | Background worker processes renamed to `background_jobs/` |
| `services/recording_worker.py` | `background_jobs/recording_worker.py` | Recording worker moved to `background_jobs/` |
| `cameras/` | `device_management/` | Camera management utilities moved to `device_management/` |
| `utils/` | `ml_inference/` & `common/` | Utilities split: ML-related code → `ml_inference/`, common utilities → `common/` |
| `camera_server/` | `streaming_service/` | Camera server renamed to `streaming_service/` |
| `database/` | `data_access/` | Database code refactored and moved to `data_access/` |
| `recordings/`, `snapshots/`, `dataset/` | `database/recordings/`, `database/snapshots/`, `database/dataset/` | Runtime data consolidated under `database/` directory |

### Files Deleted:
- `app.py` (replaced by `run.py`)
- `camera_server/server.py` (replaced by modularized code in `streaming_service/`)

### Files Created:
- `run.py`
- `core/api_server.py`
- `data_access/connection.py`
- `data_access/manager.py`
- `data_access/crud/*.py` (multiple CRUD modules)
- `docker/Dockerfile.api`
- `docker/Dockerfile.ml`
- `streaming_service/routes/*.py` (multiple route modules)
- `streaming_service/server.py`
- `streaming_service/state.py`
- `requirements-api.txt`
- `requirements-ml.txt`
- `docs/AI_Vigilance_Research_Paper.md`
- `docs/vigilance_paper.md`
- `docs/generate_figs.py`
- `docs/system_architecture/`
- `docs/system_architecture.png`

---

## 2. Docker & Deployment Improvements

### Dockerfile Changes:
- Modified to use `run.py` instead of `app.py` as entry point
- Updated directory creation to use `database/` subdirectories

### New Dockerfiles in `docker/` directory:
- **`Dockerfile.api`**: Lightweight image for API services (no PyTorch or FFmpeg)
- **`Dockerfile.ml`**: Full image for ML/streaming services (includes GPU support)

### docker-compose.yml Changes:
- Updated service names to reflect new architecture
- Uses the new separate Dockerfiles for different services
- Updated volume mounts for the new directory structure

---

## 3. Application Architecture Changes

### Entry Point:
- **Old**: `app.py` was the main entry point
- **New**: `run.py` is now the main entry point

### Streaming Service (formerly Camera Server):
- Refactored into modular structure with separate routes:
  - `streaming_service/routes/cameras.py`
  - `streaming_service/routes/faces.py`
  - `streaming_service/routes/stats.py`
  - `streaming_service/routes/stream.py`
- Added `streaming_service/state.py` to manage singleton state and DB connections

### API Server:
- New `core/api_server.py` defines the main FastAPI application

### Background Jobs:
- All worker processes consolidated in `background_jobs/` directory
- Renamed for clarity

---

## 4. Data Access Layer Refactoring

### Major Changes:
- **Old**: All database logic in `database/postgres_manager.py` (1431 lines)
- **New**:
  - `data_access/connection.py`: Handles DB pooling
  - `data_access/manager.py`: Facade layer
  - `data_access/crud/`: Domain-specific CRUD modules:
    - `alerts.py`
    - `analytics.py`
    - `cameras.py`
    - `detections.py`
    - `journeys.py`
    - `persons.py`
    - `recordings.py`

This refactoring improves maintainability and separation of concerns.

---

## 5. Configuration & Environment Changes

### .dockerignore:
- Updated to ignore `database/` instead of separate `recordings/`, `snapshots/`, `dataset/`

### .gitignore:
- Updated to reflect new directory structure
- Added `database/logs/`
- Changed media file ignore patterns to better support static UI assets

### requirements.txt:
- Split into two files:
  - **`requirements-api.txt`**: API dependencies (no PyTorch)
  - **`requirements-ml.txt`**: ML and video dependencies
- Original `requirements.txt` remains but is now a smaller subset

### Environment Variables:
- `CAMERA_SERVER_URL` → `STREAMING_SERVICE_URL`
- `RECORDINGS_DIR` default changed from `./recordings` → `./database/recordings`

---

## 6. Documentation Additions

### New Documentation Files:
- `docs/AI_Vigilance_Research_Paper.md`
- `docs/vigilance_paper.md`
- `docs/generate_figs.py` (script to generate figures)

### New Assets:
- `docs/accuracy_graph.png`
- `docs/ai_vigilance_dashboard.png`
- `docs/pipeline_workflow.png`
- `docs/system_architecture.png`
- `docs/system_architecture/` (directory with architecture files)

### README.md Updates:
- Updated directory structure diagrams
- Updated installation instructions to use new requirements files
- Updated Docker deployment documentation to explain the new microservices architecture
- Updated troubleshooting commands

---

## Summary Statistics

- **Total Files Changed**: 75
- **Lines Added**: ~3034
- **Lines Removed**: ~2228

### Key Improvements:
1. **Better Modularity**: Code split into logical modules with clear responsibilities
2. **Improved Docker Deployment**: Separate optimized images for API and ML services
3. **Maintainable Data Layer**: Database logic refactored into manageable CRUD modules
4. **Clearer Naming**: Directories and services renamed for better understandability
5. **Consolidated Runtime Data**: All runtime data now under `database/` directory
