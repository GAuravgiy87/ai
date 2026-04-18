"""
hw_manager.py — Cross-platform hardware detection (Windows + Linux)

GPU acceleration:
  Windows : onnxruntime-directml  → DmlExecutionProvider  (AMD/Intel/NVIDIA via DX12)
  Linux   : onnxruntime-gpu       → ROCMExecutionProvider (AMD) | CUDAExecutionProvider (NVIDIA)
  Fallback: onnxruntime (CPU)     → CPUExecutionProvider

Load monitoring:
  psutil  — CPU (cross-platform)
  Windows — GPU via PowerShell Get-Counter (AMD Adrenalin / NVIDIA)
  Linux   — GPU via /sys/class/drm sysfs (AMD ROCm) or nvidia-smi (NVIDIA)
"""
import os
import sys
import threading
import time
import logging

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Detects GPU and exposes ORT provider lists for detector + recognizer.

    Public:
      gpu_available : bool
      gpu_name      : str
      ort_providers : list  — GPU-first provider list
      cpu_cores     : int
      vaapi_device  : str | None  — Linux Intel iGPU VAAPI node
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self.cpu_cores  = os.cpu_count() or 4
        self.gpu_available  = False
        self.gpu_name       = "CPU"
        self.ort_providers  = ["CPUExecutionProvider"]
        self.vaapi_device   = None   # Linux Intel iGPU VAAPI (camera decode)
        self._cpu_load  = 0.0
        self._gpu_load  = 0.0
        # Cached GPU load poll — avoid subprocess every 3 s on Windows
        self._last_gpu_poll = 0.0
        self._GPU_POLL_INTERVAL = 5.0   # poll GPU load every 5 s

        self._detect_gpu()
        if sys.platform.startswith("linux"):
            self._detect_vaapi()
        self._start_monitor()
        self._log_summary()

    # ── GPU detection ─────────────────────────────────────────────────────

    def _detect_gpu(self):
        """Query ORT for available providers — works on Windows + Linux."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info(f"[HW] ORT providers available: {available}")

            # Priority: DirectML (Windows DX12) > CUDA (NVIDIA) > ROCm (AMD Linux) > CPU
            if "DmlExecutionProvider" in available:
                self.gpu_available = True
                self.gpu_name      = self._get_gpu_name_windows()
                # enable_mem_reuse=1 reduces GPU memory fragmentation on AMD
                self.ort_providers = [
                    ("DmlExecutionProvider", {
                        "device_id":        0,
                        "enable_mem_reuse": 1,
                    }),
                    "CPUExecutionProvider",
                ]
                logger.info(f"[HW] DirectML GPU: {self.gpu_name}")

            elif "CUDAExecutionProvider" in available:
                self.gpu_available = True
                self.gpu_name      = self._get_gpu_name_cuda()
                self.ort_providers = [
                    ("CUDAExecutionProvider", {
                        "device_id":                0,
                        "arena_extend_strategy":    "kNextPowerOfTwo",
                        "cudnn_conv_algo_search":   "HEURISTIC",
                        "do_copy_in_default_stream": True,
                    }),
                    "CPUExecutionProvider",
                ]
                logger.info(f"[HW] CUDA GPU: {self.gpu_name}")

            elif "ROCMExecutionProvider" in available:
                self.gpu_available = True
                self.gpu_name      = self._get_gpu_name_rocm()
                self.ort_providers = [
                    ("ROCMExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ]
                logger.info(f"[HW] ROCm GPU: {self.gpu_name}")

            else:
                logger.info("[HW] No GPU provider found — CPU inference")

        except ImportError:
            logger.warning(
                "[HW] onnxruntime not installed. "
                "Windows: pip install onnxruntime-directml | "
                "Linux:   pip install onnxruntime-gpu"
            )
        except Exception as e:
            logger.warning(f"[HW] GPU detection error: {e} — CPU fallback")

    def _get_gpu_name_windows(self) -> str:
        try:
            import subprocess
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-WmiObject Win32_VideoController | Select-Object -First 1).Name"],
                capture_output=True, text=True, timeout=5
            )
            name = r.stdout.strip()
            return name if name else "DirectML GPU"
        except Exception:
            return "DirectML GPU"

    def _get_gpu_name_cuda(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "CUDA GPU"

    def _get_gpu_name_rocm(self) -> str:
        try:
            import subprocess
            r = subprocess.run(["rocm-smi", "--showproductname"],
                               capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                if "Card" in line or "GPU" in line:
                    return line.strip()
        except Exception:
            pass
        return "ROCm GPU"

    # ── VAAPI (Linux Intel iGPU — hardware video decode) ──────────────────

    def _detect_vaapi(self):
        try:
            import glob, subprocess
            for node in sorted(glob.glob("/dev/dri/renderD*")):
                r = subprocess.run(
                    ["vainfo", "--display", "drm", "--device", node],
                    capture_output=True, text=True, timeout=3
                )
                if "Intel" in r.stdout or "iHD" in r.stdout:
                    self.vaapi_device = node
                    logger.info(f"[HW] Intel iGPU VAAPI: {node}")
                    return
        except Exception:
            pass

    # ── Load monitoring ───────────────────────────────────────────────────

    def _start_monitor(self):
        t = threading.Thread(target=self._monitor_loop, daemon=True, name="hw-monitor")
        t.start()

    def _monitor_loop(self):
        """Sample CPU every 3 s, GPU every 5 s. LOW priority."""
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadPriority(
                    ctypes.windll.kernel32.GetCurrentThread(), -2)
            else:
                os.nice(10)
        except Exception:
            pass

        _psutil = None
        try:
            import psutil as _psutil
        except ImportError:
            logger.warning("[HW] psutil not installed — CPU monitoring disabled")

        while True:
            try:
                if _psutil:
                    cpu = _psutil.cpu_percent(interval=3) / 100.0
                    with self._lock:
                        self._cpu_load = round(cpu, 2)
                else:
                    time.sleep(3)

                now = time.time()
                if self.gpu_available and (now - self._last_gpu_poll) >= self._GPU_POLL_INTERVAL:
                    self._last_gpu_poll = now
                    self._poll_gpu_load()

            except Exception:
                time.sleep(3)

    def _poll_gpu_load(self):
        """Best-effort GPU utilization — never blocks pipeline."""
        try:
            if sys.platform == "win32":
                import subprocess
                ps = (
                    "Get-Counter '\\GPU Engine(*)\\Utilization Percentage' "
                    "-ErrorAction SilentlyContinue | "
                    "Select-Object -ExpandProperty CounterSamples | "
                    "Measure-Object -Property CookedValue -Maximum | "
                    "Select-Object -ExpandProperty Maximum"
                )
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps],
                    capture_output=True, text=True, timeout=4
                )
                val = r.stdout.strip()
                if val:
                    with self._lock:
                        self._gpu_load = round(min(float(val) / 100.0, 1.0), 2)
            else:
                # AMD ROCm sysfs (fastest — no subprocess)
                busy = "/sys/class/drm/card0/device/gpu_busy_percent"
                if os.path.exists(busy):
                    with open(busy) as f:
                        with self._lock:
                            self._gpu_load = int(f.read().strip()) / 100.0
                    return
                # NVIDIA
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3
                )
                val = r.stdout.strip()
                if val.isdigit():
                    with self._lock:
                        self._gpu_load = int(val) / 100.0
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

    def best_providers(self) -> list:
        """
        GPU-first providers. Falls back to CPU only if GPU is saturated (>92%).
        At <92% load, always prefer GPU — that's the whole point.
        """
        if self.gpu_available and self._gpu_load < 0.92:
            return self.ort_providers
        return ["CPUExecutionProvider"]

    def ort_session_options(self, intra_threads: int = None):
        """
        Return tuned SessionOptions.
        intra_threads: CPU threads for ORT ops. Defaults to min(cpu_cores//2, 4).
        On GPU sessions, set to 1 — GPU does the work, CPU threads just overhead.
        """
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        # On GPU: 1 CPU thread is enough (GPU does inference)
        # On CPU: use half the cores to leave room for render/recognition threads
        if intra_threads is None:
            intra_threads = 1 if self.gpu_available else max(1, self.cpu_cores // 2)
        opts.intra_op_num_threads  = intra_threads
        opts.inter_op_num_threads  = 1
        # Enable memory pattern optimization — reuses GPU memory buffers
        opts.enable_mem_pattern    = True
        opts.enable_cpu_mem_arena  = not self.gpu_available  # only useful on CPU
        return opts

    def get_status(self) -> dict:
        return {
            "cpu_cores":     self.cpu_cores,
            "cpu_load_pct":  round(self.cpu_load * 100, 1),
            "gpu_load_pct":  round(self.gpu_load * 100, 1),
            "gpu_available": self.gpu_available,
            "gpu_name":      self.gpu_name,
            "ort_providers": [
                p[0] if isinstance(p, tuple) else p
                for p in self.ort_providers
            ],
        }

    def _log_summary(self):
        p = self.ort_providers[0]
        pname = p[0] if isinstance(p, tuple) else p
        logger.info(
            f"[HW] {self.cpu_cores} cores | "
            f"GPU: {self.gpu_name if self.gpu_available else 'none'} | "
            f"ORT: {pname} | "
            f"VAAPI: {self.vaapi_device or 'none'}"
        )


# Singleton
hw = HardwareManager()
