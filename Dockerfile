# ── AI Vigilance — Linux container (CPU inference) ───────────────────────────
# For full GPU acceleration, run directly on the host (not in Docker):
#   Windows: setup_windows.bat then python app.py
#   Linux:   ./setup_linux.sh then python app.py
#
# Docker containers on Linux can use GPU if the host has nvidia-docker or
# ROCm passthrough configured. This Dockerfile installs onnxruntime-gpu
# so CUDA/ROCm providers are available when the container has GPU access.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# PyTorch CPU (GPU inference goes through onnxruntime-gpu, not torch.cuda)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ONNX Runtime GPU — includes CUDAExecutionProvider + ROCMExecutionProvider
# Falls back to CPU automatically if no GPU is available in the container
RUN pip install --no-cache-dir onnxruntime-gpu

# Remaining deps (torch + torchvision already installed above)
RUN grep -vE "^torch|^torchvision" requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt

COPY . .
RUN mkdir -p snapshots dataset recordings

EXPOSE 8000
CMD ["python", "app.py"]
