#!/usr/bin/env python
"""
autoscale.py — Auto-scale Uvicorn workers based on CPU/memory load and camera count.

Usage:
    python autoscale.py  # Auto-scaling enabled by default
    python autoscale.py --min-workers 2 --max-workers 16 --cpu-threshold 70

Features:
    - Auto-scales main app workers (port 9000) based on CPU/memory
    - Auto-scales camera server workers (port 9001) based on camera count
    - Monitors system load every 30 seconds
    - Restarts dead processes automatically
    - Configurable via CLI or environment variables

Environment Variables (optional):
    AUTOSCALE_MIN_WORKERS=2
    AUTOSCALE_MAX_WORKERS=8
    AUTOSCALE_CPU_THRESHOLD=70
    AUTOSCALE_MEMORY_THRESHOLD=80
    AUTOSCALE_CAMERA_WORKER_RATIO=0.5  (workers per camera, default 0.5)
    AUTOSCALE_CHECK_INTERVAL=30
"""

import os
import sys
import time
import argparse
import subprocess
import psutil
import logging
import json
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [AUTOSCALER] %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicAutoScaler:
    def __init__(self, min_workers=2, max_workers=8, cpu_threshold=70, memory_threshold=80, 
                 camera_worker_ratio=0.5, check_interval=30):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.camera_worker_ratio = camera_worker_ratio  # Scale workers per camera
        self.check_interval = check_interval
        self.current_workers = min_workers
        self.current_camera_workers = 1
        self.process = None
        self.last_worker_change = time.time()

    def get_system_load(self):
        """Get current CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return cpu_percent, memory.percent

    def get_active_cameras(self):
        """Query the camera server for active camera count."""
        if not httpx:
            return 0  # Fallback if httpx not available
        try:
            client = httpx.Client(timeout=2)
            camera_url = os.environ.get("CAMERA_SERVER_URL", "http://localhost:9001").rstrip("/")
            response = client.get(f"{camera_url}/cameras")
            if response.status_code == 200:
                cameras = response.json()
                return len(cameras) if isinstance(cameras, list) else 0
        except Exception:
            pass
        return 0

    def decide_worker_count(self, cpu_percent, memory_percent, camera_count):
        """Decide how many workers we need based on load and cameras."""
        base_workers = self.min_workers
        
        # Scale based on camera count: +0.5 workers per camera
        camera_workers = max(1, int(camera_count * self.camera_worker_ratio))
        desired_workers = base_workers + camera_workers
        
        # Scale based on CPU/memory load
        if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
            desired_workers = min(desired_workers + 1, self.max_workers)
        elif cpu_percent < (self.cpu_threshold - 20) and memory_percent < (self.memory_threshold - 20):
            desired_workers = max(desired_workers - 1, self.min_workers)
        
        return min(desired_workers, self.max_workers)

    def decide_camera_workers(self, camera_count):
        """Decide camera server workers based on camera count."""
        # 1 worker per 1-2 cameras
        return max(1, min(camera_count // 2 + 1, 4))

    def run(self, app_module, host="0.0.0.0", port=9000, camera_port=9001):
        """Run Uvicorn with dynamic auto-scaling."""
        try:
            logger.info(f"🚀 Starting AI Vigilance with Dynamic Auto-Scaling")
            logger.info(f"   Main App: {host}:{port} | Camera Server: {host}:{camera_port}")
            logger.info(f"   Main Workers: {self.min_workers}-{self.max_workers}")
            logger.info(f"   Camera Worker Ratio: {self.camera_worker_ratio} per camera")
            logger.info(f"   Thresholds: CPU {self.cpu_threshold}% | Memory {self.memory_threshold}%")
            logger.info(f"   Check Interval: {self.check_interval}s\n")
            
            scale_log_cooldown = 0
            
            while True:
                cpu_percent, memory_percent = self.get_system_load()
                camera_count = self.get_active_cameras()
                
                # Calculate new worker counts
                new_workers = self.decide_worker_count(cpu_percent, memory_percent, camera_count)
                new_camera_workers = self.decide_camera_workers(camera_count)
                
                # Log scaling changes with cooldown (avoid spam)
                now = time.time()
                if (new_workers != self.current_workers or new_camera_workers != self.current_camera_workers) \
                   and (now - scale_log_cooldown) > 5:
                    logger.info(f"📊 Load: CPU {cpu_percent:.1f}% | Memory {memory_percent:.1f}% | Cameras {camera_count}")
                    if new_workers != self.current_workers:
                        logger.info(f"   🔄 Main App: {self.current_workers} → {new_workers} workers")
                    if new_camera_workers != self.current_camera_workers:
                        logger.info(f"   📹 Camera Server: {self.current_camera_workers} → {new_camera_workers} workers")
                    scale_log_cooldown = now
                
                self.current_workers = new_workers
                self.current_camera_workers = new_camera_workers
                
                # Set environment variables
                os.environ["UVICORN_WORKERS"] = str(self.current_workers)
                os.environ["CAMERA_SERVER_WORKERS"] = str(self.current_camera_workers)
                
                # If process died or workers changed, restart it
                if self.process is None or self.process.poll() is not None:
                    logger.info(f"▶️  Starting Uvicorn ({self.current_workers} workers)...")
                    
                    base_cmd = [sys.executable]
                    if app_module.startswith("-m "):
                        base_cmd.extend(["-m", app_module[3:]])
                    else:
                        base_cmd.append(app_module)
                        
                    self.process = subprocess.Popen(
                        base_cmd + ["--host", host, "--port", str(port)],
                        env=os.environ.copy(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    self.last_worker_change = time.time()
                
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                logger.info("✅ Shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-scale AI Vigilance workers based on system load and camera count",
        epilog="Example: python autoscale.py --min-workers 2 --max-workers 16 --cpu-threshold 65"
    )
    parser.add_argument("--app", default="app.py", help="FastAPI app entry point (default: app.py)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Main app port (default: 9000)")
    parser.add_argument("--camera-port", type=int, default=9001, help="Camera server port (default: 9001)")
    parser.add_argument("--min-workers", type=int, 
                        default=int(os.environ.get("AUTOSCALE_MIN_WORKERS", "2")),
                        help="Minimum workers (default: 2)")
    parser.add_argument("--max-workers", type=int,
                        default=int(os.environ.get("AUTOSCALE_MAX_WORKERS", "8")),
                        help="Maximum workers (default: 8)")
    parser.add_argument("--cpu-threshold", type=int,
                        default=int(os.environ.get("AUTOSCALE_CPU_THRESHOLD", "70")),
                        help="CPU % to scale up (default: 70)")
    parser.add_argument("--memory-threshold", type=int,
                        default=int(os.environ.get("AUTOSCALE_MEMORY_THRESHOLD", "80")),
                        help="Memory % to scale up (default: 80)")
    parser.add_argument("--camera-worker-ratio", type=float,
                        default=float(os.environ.get("AUTOSCALE_CAMERA_WORKER_RATIO", "0.5")),
                        help="Workers per camera (default: 0.5)")
    parser.add_argument("--check-interval", type=int,
                        default=int(os.environ.get("AUTOSCALE_CHECK_INTERVAL", "30")),
                        help="Check interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    scaler = DynamicAutoScaler(
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        cpu_threshold=args.cpu_threshold,
        memory_threshold=args.memory_threshold,
        camera_worker_ratio=args.camera_worker_ratio,
        check_interval=args.check_interval
    )
    
    scaler.run(args.app, args.host, args.port, args.camera_port)


if __name__ == "__main__":
    main()
