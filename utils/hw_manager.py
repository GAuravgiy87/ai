import os
import threading
import time
import logging
import subprocess
import platform
import psutil
from typing import Optional

logger = logging.getLogger(__name__)

class HardwareManager:
    """
    Detects available hardware and exposes the best torch device.
    Cross-platform support for Linux (ROCm/VAAPI) and Windows.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.cpu_cores = os.cpu_count() or 4
        self.is_windows = platform.system() == "Windows"

        # ── Detect GPU (DirectML for AMD Windows / CUDA for NVIDIA) ──────
        self.dml_available = False
        self.face_device = "cpu"
        self.yolo_device = "cpu"
        self._detect_gpu()

        # ── Detect Video Hardware Encoding (Intel QuickSync / AMD AMF) ──
        self.encoder_codec = "libx264" # Default
        self._detect_encoder()

        # ── Detect Intel iGPU VAAPI (Linux only) ──────────────────────────
        self.vaapi_device: Optional[str] = None
        if not self.is_windows:
            self._detect_vaapi()

        # ── Load tracking ─────────────────────────────────────────────────
        self._cpu_load = 0.0
        self._gpu_load = 0.0
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        if self.cpu_cores <= 2:
            logger.warning(f"[HW] Low resource system detected ({self.cpu_cores} cores). Optimization: Throttled FPS modes active.")

        self._log_summary()

    def _detect_gpu(self):
        """Detect GPU devices for AI Inference (DirectML/CUDA)."""
        try:
            import torch
            if torch.cuda.is_available():
                self.face_device = "cuda:0"
                self.yolo_device = "cuda:0"
                logger.info(f"[HW] NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
                return
            
            # Check for DirectML (AMD Windows) via ONNX Runtime / torch-directml
            if self.is_windows:
                try:
                    import onnxruntime as ort
                    if "DmlExecutionProvider" in ort.get_available_providers():
                        self.dml_available = True
                        self.yolo_device = "dml"
                        logger.info("[HW] AMD GPU (DirectML via ONNX): Detected")
                except: pass

                try:
                    import torch_directml
                    self.face_device = "dml"
                    logger.info("[HW] AMD GPU (torch-directml): Detected for Face Recognition")
                except:
                    self.face_device = "cpu"
                
                if self.dml_available or self.face_device == "dml":
                    return
        except Exception: pass
        logger.info("[HW] GPU not available for AI - using CPU")

    def _detect_encoder(self):
        """Find the best hardware video encoder."""
        if self.is_windows:
            # Check for Intel QuickSync (i7 8700 has this)
            try:
                out = subprocess.check_output(["ffmpeg", "-encoders"], stderr=subprocess.STDOUT, text=True)
                if "h264_qsv" in out:
                    self.encoder_codec = "h264_qsv"
                    logger.info("[HW] Encoder: Intel QuickSync (h264_qsv)")
                elif "h264_amf" in out:
                    self.encoder_codec = "h264_amf"
                    logger.info("[HW] Encoder: AMD AMF (h264_amf)")
            except: pass
        else:
            # Linux VAAPI logic...
            self.encoder_codec = "h264_vaapi" if self.vaapi_device else "libx264"

    def _detect_vaapi(self):
        """Find Intel iGPU render node (Linux-specific)."""
        try:
            import glob
            nodes = sorted(glob.glob("/dev/dri/renderD*"))
            for node in nodes:
                result = subprocess.run(["vainfo", "--display", "drm", "--device", node], capture_output=True, text=True, timeout=3)
                if "Intel" in result.stdout or "iHD" in result.stdout:
                    self.vaapi_device = node
                    logger.info(f"[HW] Intel VAAPI: {node}")
                    return
        except Exception: pass

    def _monitor_loop(self):
        """Sample CPU and GPU utilization."""
        while True:
            try:
                self._cpu_load = psutil.cpu_percent(interval=2) / 100.0
                if self.is_windows and self.dml_available:
                    # Simplified GPU monitoring for Windows AMD
                    pass
                elif not self.is_windows:
                    busy_path = "/sys/class/drm/card0/device/gpu_busy_percent"
                    if os.path.exists(busy_path):
                        with open(busy_path) as f:
                            self._gpu_load = int(f.read().strip()) / 100.0
            except Exception:
                time.sleep(2)

    @property
    def cpu_load(self) -> float:
        return self._cpu_load

    @property
    def gpu_load(self) -> float:
        return self._gpu_load

    def best_face_device(self):
        """Returns string 'cpu', 'cuda', or the Actual DirectML Device Object."""
        if self.face_device == "dml":
            import torch_directml
            return torch_directml.device()
        return self.face_device

    def best_yolo_device(self):
        if self.yolo_device == "dml":
            import torch_directml
            return torch_directml.device()
        return self.yolo_device

    def get_status(self) -> dict:
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_load_pct": round(self.cpu_load * 100, 1),
            "gpu_load_pct": round(self.gpu_load * 100, 1),
            "face_device": str(self.face_device),
            "yolo_device": str(self.yolo_device),
            "encoder": self.encoder_codec,
            "platform": platform.platform()
        }

    def _log_summary(self):
        logger.info(f"[HW] Summary - Platform: {platform.system()} | CPU: {self.cpu_cores} cores | Encoder: {self.encoder_codec} | Face: {self.face_device} | YOLO: {self.yolo_device}")

hw = HardwareManager()
