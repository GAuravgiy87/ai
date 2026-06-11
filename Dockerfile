FROM python:3.11-slim

# System deps: ffmpeg, VAAPI (Intel iGPU), libGL, ROCm OpenCL runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    # VAAPI for Intel iGPU hardware decode
    libva-drm2 \
    libva2 \
    vainfo \
    intel-media-va-driver-non-free \
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
    # PostgreSQL bindings
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch — ROCm build if AMD GPU present, else CPU
# ROCm 5.7 supports RX 550 (gfx803 / Fiji / Polaris)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7 \
    || pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements (torch already installed above)
RUN grep -v "^torch" requirements.txt > /tmp/req_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/req_no_torch.txt

COPY . .

RUN mkdir -p snapshots dataset recordings

EXPOSE 8000

CMD ["python3", "app.py"]
