"""
hw_manager.py — Hardware detection and GPU monitoring.

GPU monitoring on Windows AMD (Radeon RX 550 / DirectML):
  Uses Windows Performance Counters via psutil's win32 bindings.
  Counter: \\GPU Engine(*)\\Utilization Percentage
  Sums all 'engtype_compute' + 'engtype_3d' instances for the dGPU LUID.

GPU memory on Windows AMD:
  Uses \\GPU Adapter Memory(*)\\Dedicated Usage counter.
"""

import os
import threading
import time
import logging
import subprocess
import platform
import psutil
from typing import Optional

logger = logging.getLogger(__name__)


# ── GPU LUID detection ────────────────────────────────────────────────────────

def _get_active_gpu_name() -> Optional[str]:
    """
    Find the name of the primary GPU from WMI dynamically.
    Returns the name string like 'Intel(R) HD Graphics 520' or None.
    """
    try:
        import subprocess, json, tempfile, os
        # Use PowerShell to get GPU info — write to temp file to avoid quoting
        ps = (
            'Get-CimInstance Win32_VideoController | '
            'Select-Object Name | ConvertTo-Json'
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1',
                                         delete=False, encoding='utf-8') as f:
            f.write(ps); fname = f.name
        r = subprocess.run(
            ['powershell', '-NoProfile', '-File', fname],
            capture_output=True, text=True, timeout=6
        )
        os.unlink(fname)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        data = json.loads(r.stdout)
        if isinstance(data, list) and len(data) > 0:
            name = data[0].get('Name', '')
        elif isinstance(data, dict):
            name = data.get('Name', '')
        else:
            return None
            
        if name:
            logger.info(f"[HW] GPU detected dynamically: {name}")
            return name
        return None
    except Exception as e:
        logger.debug(f"[HW] GPU name detection failed: {e}")
        return None


# ── Windows GPU counter reader ────────────────────────────────────────────────

class _WinGpuMonitor:
    """
    Reads AMD GPU utilization and memory from Windows Performance Counters.
    Runs in a background thread, caches the latest values.
    """

    # Counter paths
    _UTIL_COUNTER = r'\GPU Engine(*)\Utilization Percentage'
    _MEM_COUNTER  = r'\GPU Adapter Memory(*)\Dedicated Usage'

    def __init__(self, gpu_name: str):
        self.gpu_name   = gpu_name
        self._util_pct  = 0.0
        self._mem_mb    = 0.0
        self._lock      = threading.Lock()
        self._available = False
        self._luid      = None   # filled on first successful read

        # Try to import win32pdh (pywin32) for fast counter access
        try:
            import win32pdh
            self._use_win32pdh = True
        except ImportError:
            self._use_win32pdh = False

        # Start background thread
        t = threading.Thread(target=self._loop, daemon=True, name='gpu-monitor')
        t.start()

    def _read_via_psutil(self) -> tuple:
        """
        Read GPU utilization % and memory from Windows Performance Counters dynamically.
        Returns (util_pct, mem_mb).
        Sums utilization across all engtype_3D engines across all GPUs.
        """
        try:
            import subprocess, tempfile, os, json

            if not hasattr(self, '_ps1_path') or not os.path.exists(self._ps1_path):
                ps = (
                    "$util = (Get-Counter '\\GPU Engine(*engtype_3D*)\\Utilization Percentage'"
                    " -ErrorAction SilentlyContinue).CounterSamples |"
                    " Measure-Object -Property CookedValue -Sum;"
                    "$memD = (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage'"
                    " -ErrorAction SilentlyContinue).CounterSamples |"
                    " Measure-Object -Property CookedValue -Sum;"
                    "$memS = (Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage'"
                    " -ErrorAction SilentlyContinue).CounterSamples |"
                    " Measure-Object -Property CookedValue -Sum;"
                    "$mem = 0;"
                    "if ($null -ne $memD) { $mem += $memD.Sum };"
                    "if ($null -ne $memS) { $mem += $memS.Sum };"
                    "Write-Output ('{\"util\":' + [math]::Round($util.Sum,2) +"
                    " ',\"mem\":' + [math]::Round($mem,0) + '}')"
                )
                f = tempfile.NamedTemporaryFile(mode='w', suffix='.ps1',
                                                delete=False, encoding='utf-8')
                f.write(ps); f.close()
                self._ps1_path = f.name

            r = subprocess.run(
                ['powershell', '-NoProfile', '-File', self._ps1_path],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return 0.0, 0.0

            out = r.stdout.strip()
            data = json.loads(out)
            util = float(data.get('util') or 0.0)
            mem  = float(data.get('mem')  or 0.0) / 1048576   # bytes → MB
            return round(min(util, 100.0), 1), round(mem, 1)

        except Exception as e:
            logger.debug(f"[HW] GPU counter read failed: {e}")
            return 0.0, 0.0

    def _loop(self):
        """Background loop — reads counters every 2 seconds."""
        # First read to check availability
        util, mem = self._read_via_psutil()
        with self._lock:
            self._available = True   # counter exists even if value is 0
            self._util_pct  = util
            self._mem_mb    = mem

        while True:
            time.sleep(2)
            try:
                util, mem = self._read_via_psutil()
                with self._lock:
                    self._util_pct = util
                    self._mem_mb   = mem
            except Exception:
                pass

    @property
    def util_pct(self) -> float:
        with self._lock:
            return self._util_pct

    @property
    def mem_mb(self) -> float:
        with self._lock:
            return self._mem_mb

    @property
    def available(self) -> bool:
        return self._available


# ── Main HardwareManager ──────────────────────────────────────────────────────

class HardwareManager:
    """
    Detects available hardware and exposes the best torch device.
    Cross-platform: Windows (DirectML/AMD) and Linux (ROCm/VAAPI).
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self.cpu_cores  = os.cpu_count() or 4
        self.is_windows = platform.system() == "Windows"

        self.dml_available = False
        self.face_device   = "cpu"
        self.yolo_device   = "cpu"
        self._gpu_name     = "N/A"
        self._gpu_monitor: Optional[_WinGpuMonitor] = None

        self._detect_gpu()

        self.encoder_codec = "libx264"
        self._detect_encoder()

        self.vaapi_device: Optional[str] = None
        if not self.is_windows:
            self._detect_vaapi()

        # CPU load via psutil
        self._cpu_load = 0.0
        threading.Thread(target=self._cpu_loop, daemon=True,
                         name='cpu-monitor').start()

        if self.cpu_cores <= 2:
            logger.warning(f"[HW] Low resource system ({self.cpu_cores} cores).")

        self._log_summary()

    # ── GPU detection ─────────────────────────────────────────────────────────

    def _detect_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self.face_device = "cuda:0"
                self.yolo_device = "cuda:0"
                self._gpu_name   = name
                logger.info(f"[HW] NVIDIA GPU (CUDA): {name}")
                return
        except Exception:
            pass

        if self.is_windows:
            # DirectML for YOLO (ONNX Runtime)
            try:
                import onnxruntime as ort
                if "DmlExecutionProvider" in ort.get_available_providers():
                    self.dml_available = True
                    self.yolo_device   = "dml"
                    logger.info("[HW] AMD GPU (DirectML via ONNX): Detected")
            except Exception:
                pass

            # torch-directml for FaceNet
            try:
                import torch_directml
                self.face_device = "dml"
                logger.info("[HW] AMD GPU (torch-directml): Detected for Face Recognition")
            except Exception:
                self.face_device = "cpu"

            # Identify GPU name and start monitor
            name = _get_active_gpu_name() or "Intel/AMD GPU"
            self._gpu_name  = name
            self._gpu_monitor = _WinGpuMonitor(name)
            return

        logger.info("[HW] GPU not available for AI — using CPU")

    # ── Encoder detection ─────────────────────────────────────────────────────

    def _detect_encoder(self):
        if self.is_windows:
            try:
                out = subprocess.check_output(
                    ["ffmpeg", "-encoders"], stderr=subprocess.STDOUT, text=True
                )
                if "h264_qsv" in out:
                    self.encoder_codec = "h264_qsv"
                    logger.info("[HW] Encoder: Intel QuickSync (h264_qsv)")
                elif "h264_amf" in out:
                    self.encoder_codec = "h264_amf"
                    logger.info("[HW] Encoder: AMD AMF (h264_amf)")
            except Exception:
                pass
        else:
            self.encoder_codec = "h264_vaapi" if self.vaapi_device else "libx264"

    # ── VAAPI (Linux) ─────────────────────────────────────────────────────────

    def _detect_vaapi(self):
        try:
            import glob
            for node in sorted(glob.glob("/dev/dri/renderD*")):
                r = subprocess.run(
                    ["vainfo", "--display", "drm", "--device", node],
                    capture_output=True, text=True, timeout=3
                )
                if "Intel" in r.stdout or "iHD" in r.stdout:
                    self.vaapi_device = node
                    logger.info(f"[HW] Intel VAAPI: {node}")
                    return
        except Exception:
            pass

    # ── CPU monitor ───────────────────────────────────────────────────────────

    def _cpu_loop(self):
        while True:
            try:
                self._cpu_load = psutil.cpu_percent(interval=2) / 100.0
            except Exception:
                time.sleep(2)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def cpu_load(self) -> float:
        return self._cpu_load

    @property
    def gpu_load(self) -> float:
        if self._gpu_monitor:
            return self._gpu_monitor.util_pct / 100.0
        return 0.0

    def best_face_device(self):
        if self.face_device == "dml":
            try:
                import torch_directml
                return torch_directml.device()
            except Exception:
                return "cpu"
        return self.face_device

    def best_yolo_device(self):
        if self.yolo_device == "dml":
            try:
                import torch_directml
                return torch_directml.device()
            except Exception:
                return "cpu"
        return self.yolo_device

    def get_status(self) -> dict:
        """Return a dict suitable for the diagnostics table."""
        gpu_info = None
        if self._gpu_monitor:
            gpu_info = {
                "name":        self._gpu_name,
                "load":        f"{self._gpu_monitor.util_pct:.1f}%",
                "memory_used": f"{self._gpu_monitor.mem_mb:.0f}",
            }
        return {
            "cpu":    {"usage_percent": round(self._cpu_load * 100, 1)},
            "memory": {"percent": psutil.virtual_memory().percent},
            "gpu":    gpu_info,
            "platform": platform.platform(),
            "encoder":  self.encoder_codec,
        }

    def _log_summary(self):
        logger.info(
            f"[HW] Summary - Platform: {platform.system()} | "
            f"CPU: {self.cpu_cores} cores | "
            f"Encoder: {self.encoder_codec} | "
            f"Face: {self.face_device} | "
            f"YOLO: {self.yolo_device} | "
            f"GPU: {self._gpu_name}"
        )


hw = HardwareManager()
