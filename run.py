#!/usr/bin/env python
"""
run.py - Main entry point for the AI Vigilance application.
"""

import os
import sys
import argparse
import threading
import time
import uvicorn
import multiprocessing

if os.path.exists(".env"):
    with open(".env", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                if k not in os.environ:
                    os.environ[k] = v

from core.api_server import app, db_manager
from common.logging import setup_logging
from common.network import get_local_ip

logger = setup_logging()

def main():
    for folder in ["data", "models", "database/dataset", "database/snapshots", "database/recordings", "database/logs"]:
        os.makedirs(folder, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Launch AI Vigilance with optional Dynamic Auto-Scaling"
    )
    parser.add_argument("--disable-autoscale", action="store_true",
                        help="Disable auto-scaling, use fixed worker count")
    parser.add_argument("--min-workers", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-threshold", type=int, default=None)
    parser.add_argument("--camera-worker-ratio", type=float, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind")
    
    args = parser.parse_args()
    
    if not args.disable_autoscale:
        # Launch with Dynamic Auto-Scaling (runs core/autoscale.py logic)
        from core.autoscale import DynamicAutoScaler
        
        scaler_kwargs = {}
        if args.min_workers is not None:
            scaler_kwargs["min_workers"] = args.min_workers
        if args.max_workers is not None:
            scaler_kwargs["max_workers"] = args.max_workers
        if args.cpu_threshold is not None:
            scaler_kwargs["cpu_threshold"] = args.cpu_threshold
        if args.camera_worker_ratio is not None:
            scaler_kwargs["camera_worker_ratio"] = args.camera_worker_ratio
            
        scaler = DynamicAutoScaler(**scaler_kwargs)
        # Pass run.py so scaler respawns using the new entry point
        scaler.run(app_module="run.py", host=args.host, port=args.port)
        sys.exit(0)

    # ── If autoscale is disabled, run the app directly ───────────────────────
    local_ip = get_local_ip()
    print("\n" + "=" * 50)
    print("[OK] AI Vigilance System Starting...")
    print(f"[OK] Main App     : http://127.0.0.1:{args.port}")
    print(f"[OK] Camera Server: http://127.0.0.1:9001")
    print(f"[OK] Network      : http://{local_ip}:{args.port}")
    print("=" * 50 + "\n")
    
    # ── Start Background Servers (Master Process Only) ───────────────────────
    # Only the master process executes this block, preventing Uvicorn workers
    # from spawning multiple conflicting background threads.
    if multiprocessing.current_process().name == "MainProcess":
        def _run_camera_server():
            from streaming_service.server import start
            try:
                start()
            except Exception as e:
                logger.error(f"[CameraServer] Thread crashed: {e}")
                
        _cam_server_thread = threading.Thread(target=_run_camera_server, name="camera-server-master", daemon=True)
        _cam_server_thread.start()
        
        def _run_recording_worker():
            from background_jobs.recording_worker import main as rw_main
            try:
                rw_main()
            except Exception as e:
                import logging
                logging.error(f"[RecordingWorker] Thread crashed: {e}")

        _rec_worker_thread = threading.Thread(target=_run_recording_worker, name="recording-worker-master", daemon=True)
        _rec_worker_thread.start()
    
    time.sleep(1)

    try:
        # FastAPI is fully async — 1 worker handles all concurrent requests.
        # workers > 1 on Windows uses multiprocessing spawn which is slow to
        # kill on shutdown (+3-8 s per worker). Set UVICORN_WORKERS env var
        # or use --workers flag to override for CPU-bound scaling.
        workers = int(os.environ.get("UVICORN_WORKERS", "1"))
        uvicorn.run(
            "core.api_server:app",  # Still load FastAPI instance from core/app.py
            host=args.host,
            port=args.port,
            workers=workers,
            log_level="warning",
            access_log=False,
        )
    except (KeyboardInterrupt, InterruptedError):
        # Normal shutdown interruption on Windows
        pass
    except Exception as e:
        logger.error(f"[App] Uvicorn exited with error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
