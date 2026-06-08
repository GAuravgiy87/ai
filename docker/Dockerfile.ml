FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Video encoding/decoding
    ffmpeg \
    # OpenCV runtime libs
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    # VAAPI for Intel iGPU hardware decode
    libva-drm2 \
    libva2 \
    vainfo \
    intel-media-va-driver \
    # GStreamer for VAAPI OpenCV backend
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-vaapi \
    python3-gst-1.0 \
    # OpenCL runtime (AMD ROCm userspace)
    ocl-icd-libopencl1 \
    clinfo \
    # PostgreSQL client libs
    libpq-dev \
    # curl: used by Docker healthcheck and debugging
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements-api.txt .
COPY requirements-ml.txt .

# Install PyTorch CPU-only first (large wheel, cached separately)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (skip torch lines already installed above)
RUN grep -v "^torch" requirements-ml.txt > /tmp/req_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/req_no_torch.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Ensure runtime directories exist (volumes will overlay these on container start)
RUN mkdir -p database/snapshots database/dataset database/recordings data database/logs models

# ── Ports ─────────────────────────────────────────────────────────────────────
# camera_server: 9001
EXPOSE 9001

# ── Default command (overridden per-service in docker-compose.yml) ────────────
CMD ["python3", "-m", "streaming_service.server", "--host", "0.0.0.0", "--port", "9001"]
