"""
hw_manager.py — Hardware detection and task routing
Detects Intel iGPU (VAAPI), AMD dGPU (ROCm/OpenCL), and CPU cores.
Provides a HardwareManager singleton that routes tasks to the best device.
"""
import os
import threading
import time
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Detects available hardware and exposes the best torch device
    for each task type:
      - face_device  : AMD dGPU (ROCm) > CPU
      - yolo_device  : CPU  (YOLOv8n is fast enough; RX 550 ROCm support is limited)
      - vaapi_device : Intel iGPU render node for OpenCV VAAPI decode
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.cpu_cores = os.cpu_count() or 4

        # ── Detect AMD dGPU via ROCm ──────────────────────────────────────
        self.amd_rocm_available = False
        self.face_device = "cpu"
        self._detect_amd_rocm()

        # ── Detect Intel iGPU VAAPI node ─────────────────────────────────
        self.vaapi_device: Optional[str] = None
        self._detect_vaapi()

        # ── YOLO always on CPU (YOLOv8n is fast; RX 550 ROCm is unstable) ─
        self.yolo_device = "cpu"

        # ── Load tracking ─────────────────────────────────────────────────
        self._cpu_load   = 0.0   # 0.0 – 1.0
        self._gpu_load   = 0.0
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self._log_summary()

    # ── Detection helpers ─────────────────────────────────────────────────

    def _detect_amd_rocm(self):
        """Try to import torch with ROCm and find an AMD GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                # On ROCm, cuda.is_available() returns True for AMD GPUs
                dev_name = torch.cuda.get_device_name(0).lower()
                if any(k in dev_name for k in ("radeon", "amd", "rx", "vega", "navi")):
                    self.amd_rocm_available = True
                    self.face_device = "cuda:0"
                    logger.info(f"[HW] AMD dGPU (ROCm): {torch.cuda.get_device_name(0)}")
                    return
                # Could be NVIDIA — still use it for face
                self.face_device = "cuda:0"
                logger.info(f"[HW] CUDA GPU: {torch.cuda.get_device_name(0)}")
                return
        except Exception:
            pass

        # Fallback: check OpenCL via pyopencl
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for p in platforms:
                for d in p.get_devices():
                    if cl.device_type.to_string(d.type) == "GPU":
                        name = d.name.lower()
                        if any(k in name for k in ("radeon", "amd", "rx", "polaris")):
                            # ROCm not available but OpenCL is — note it
                            logger.info(f"[HW] AMD GPU via OpenCL: {d.name} (ROCm not active)")
                            break
        except Exception:
            pass

        logger.info("[HW] AMD dGPU not available via ROCm — face recognition on CPU")

    def _detect_vaapi(self):
        """Find Intel iGPU render node for VAAPI-accelerated video decode."""
        try:
            import glob
            nodes = sorted(glob.glob("/dev/dri/renderD*"))
            for node in nodes:
                # Check if it's Intel via libva-info
                result = subprocess.run(
                    ["vainfo", "--display", "drm", "--device", node],
                    capture_output=True, text=True, timeout=3
                )
                if "Intel" in result.stdout or "iHD" in result.stdout:
                    self.vaapi_device = node
                    logger.info(f"[HW] Intel iGPU VAAPI: {node}")
                    return
        except Exception:
            pass
        logger.info("[HW] Intel iGPU VAAPI not detected — using CPU decode")

    # ── Load monitoring ───────────────────────────────────────────────────

    def _monitor_loop(self):
        """Sample CPU and GPU utilization every 2 seconds."""
        while True:
            try:
                # CPU load via /proc/stat
                with open("/proc/stat") as f:
                    line = f.readline()
                vals = list(map(int, line.split()[1:]))
                idle = vals[3]
                total = sum(vals)
                time.sleep(2)
                with open("/proc/stat") as f:
                    line2 = f.readline()
                vals2 = list(map(int, line2.split()[1:]))
                idle2 = vals2[3]
                total2 = sum(vals2)
                cpu_use = 1.0 - (idle2 - idle) / max(1, total2 - total)
                with self._lock:
                    self._cpu_load = round(cpu_use, 2)
            except Exception:
                time.sleep(2)

            # AMD GPU load via sysfs (ROCm exposes this)
            try:
                busy_path = "/sys/class/drm/card0/device/gpu_busy_percent"
                if os.path.exists(busy_path):
                    with open(busy_path) as f:
                        with self._lock:
                            self._gpu_load = int(f.read().strip()) / 100.0
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def cpu_load(self) -> float:
        with self._lock:
            return self._cpu_load

    @property
    def gpu_load(self) -> float:
        with self._lock:
            return self._gpu_load

    def best_face_device(self) -> str:
        """
        Dynamic routing: use AMD dGPU if available and not overloaded,
        otherwise fall back to CPU.
        """
        if self.amd_rocm_available and self.gpu_load < 0.85:
            return self.face_device
        return "cpu"

    def best_yolo_device(self) -> str:
        return self.yolo_device   # always CPU

    def get_status(self) -> dict:
        return {
            "cpu_cores":          self.cpu_cores,
            "cpu_load_pct":       round(self.cpu_load * 100, 1),
            "gpu_load_pct":       round(self.gpu_load * 100, 1),
            "face_device":        self.face_device,
            "yolo_device":        self.yolo_device,
            "vaapi_device":       self.vaapi_device,
            "amd_rocm":           self.amd_rocm_available,
        }

    def _log_summary(self):
        logger.info(
            f"[HW] Summary — CPU cores: {self.cpu_cores} | "
            f"YOLO: {self.yolo_device} | "
            f"Face: {self.face_device} | "
            f"VAAPI: {self.vaapi_device or 'none'}"
        )


# Singleton
hw = HardwareManager()
