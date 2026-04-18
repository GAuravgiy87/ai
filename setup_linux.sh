#!/bin/bash
# =============================================================================
# setup_linux.sh — AI Vigilance System Setup (Linux)
# Supports: AMD GPU (ROCm), NVIDIA GPU (CUDA), Intel iGPU (VAAPI), CPU-only
# Run once on a fresh machine.
# =============================================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   AI Vigilance — Linux Setup                 ║${NC}"
echo -e "${BOLD}${CYAN}║   AMD / NVIDIA / Intel / CPU auto-detect      ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
info "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    sqlite3 curl wget git \
    libva-drm2 libva2 vainfo \
    intel-media-va-driver-non-free \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-vaapi \
    ocl-icd-libopencl1 clinfo
ok "System packages installed."

# ── 2. Detect GPU ─────────────────────────────────────────────────────────────
info "[2/7] Detecting GPU..."
HAS_NVIDIA=false
HAS_AMD=false
HAS_VAAPI=false

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    ok "NVIDIA GPU: $GPU_NAME"
fi

if [ -e /dev/kfd ]; then
    HAS_AMD=true
    ok "AMD GPU (ROCm kernel driver present)"
elif lspci 2>/dev/null | grep -qi "amd\|radeon\|advanced micro"; then
    HAS_AMD=true
    warn "AMD GPU detected but /dev/kfd missing — ROCm may not be installed"
fi

if vainfo 2>/dev/null | grep -qi "iHD\|Intel"; then
    HAS_VAAPI=true
    ok "Intel iGPU VAAPI active"
fi

if [ "$HAS_NVIDIA" = false ] && [ "$HAS_AMD" = false ]; then
    warn "No GPU detected — will use CPU inference"
fi

# ── 3. AMD ROCm setup (if AMD GPU present) ───────────────────────────────────
if [ "$HAS_AMD" = true ] && [ ! -e /dev/kfd ]; then
    info "[3/7] Installing AMD ROCm..."
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
        | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
        https://repo.radeon.com/rocm/apt/6.0 $(lsb_release -cs) main" \
        | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y rocm-hip-runtime rocm-opencl-runtime
    ok "ROCm installed."
else
    info "[3/7] ROCm check skipped (not needed or already present)"
fi

# Add user to GPU groups
for grp in video render; do
    if getent group "$grp" &>/dev/null && ! groups "$USER" | grep -q "$grp"; then
        sudo usermod -aG "$grp" "$USER"
        warn "Added $USER to group '$grp' — re-login required"
    fi
done

# ── 4. Python venv ────────────────────────────────────────────────────────────
info "[4/7] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
ok "Virtual environment ready."

# ── 5. PyTorch — right build for detected GPU ────────────────────────────────
info "[5/7] Installing PyTorch..."
if [ "$HAS_NVIDIA" = true ]; then
    pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121 \
        && ok "PyTorch CUDA installed." \
        || { warn "CUDA PyTorch failed — falling back to CPU."; \
             pip install --quiet torch torchvision \
                 --index-url https://download.pytorch.org/whl/cpu; }
elif [ "$HAS_AMD" = true ]; then
    # ROCm 6.0 supports RX 6000/7000; for older RX 550 (gfx803) use rocm5.7
    pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/rocm6.0 \
        && ok "PyTorch ROCm 6.0 installed." \
        || {
            warn "ROCm 6.0 failed — trying ROCm 5.7 (for older AMD GPUs like RX 550)..."
            pip install --quiet torch torchvision \
                --index-url https://download.pytorch.org/whl/rocm5.7 \
                && ok "PyTorch ROCm 5.7 installed." \
                || { warn "ROCm PyTorch failed — falling back to CPU."; \
                     pip install --quiet torch torchvision \
                         --index-url https://download.pytorch.org/whl/cpu; }
        }
else
    pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    ok "PyTorch CPU installed."
fi

# ── 6. ONNX Runtime — right variant for detected GPU ─────────────────────────
info "[6/7] Installing ONNX Runtime..."
# Remove any conflicting onnxruntime packages first
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml 2>/dev/null || true

if [ "$HAS_NVIDIA" = true ]; then
    pip install --quiet onnxruntime-gpu \
        && ok "onnxruntime-gpu (CUDA) installed." \
        || { warn "onnxruntime-gpu failed — falling back to CPU."; \
             pip install --quiet onnxruntime; }
elif [ "$HAS_AMD" = true ]; then
    # onnxruntime-gpu includes ROCMExecutionProvider on Linux
    pip install --quiet onnxruntime-gpu \
        && ok "onnxruntime-gpu (ROCm) installed." \
        || { warn "onnxruntime-gpu failed — falling back to CPU."; \
             pip install --quiet onnxruntime; }
else
    pip install --quiet onnxruntime
    ok "onnxruntime (CPU) installed."
fi

# ── 7. Python dependencies ────────────────────────────────────────────────────
info "[7/7] Installing Python application dependencies..."
# Skip torch/torchvision — already installed above
grep -vE "^torch|^torchvision" requirements.txt > /tmp/req_no_torch.txt
pip install --quiet -r /tmp/req_no_torch.txt
ok "Python dependencies installed."

# Workspace directories
mkdir -p snapshots dataset recordings
touch db.sqlite3 app.log

# ── Environment variables ─────────────────────────────────────────────────────
ENV_FILE="$HOME/.ai_vigilance_env"
cat > "$ENV_FILE" << 'EOF'
# AI Vigilance environment
export OPENCV_LOG_LEVEL=OFF
export FFMPEG_LOG_LEVEL=quiet
export PYTHONWARNINGS=ignore
# AMD ROCm (uncomment and adjust if needed for older GPUs)
# export HSA_OVERRIDE_GFX_VERSION=8.0.3   # RX 550 / gfx803
# export ROCR_VISIBLE_DEVICES=0
# Intel iGPU VAAPI
export LIBVA_DRIVER_NAME=iHD
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
EOF
if ! grep -q "ai_vigilance_env" "$HOME/.bashrc" 2>/dev/null; then
    echo "[ -f $ENV_FILE ] && source $ENV_FILE" >> "$HOME/.bashrc"
fi
ok "Environment config written to $ENV_FILE"

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Setup Complete!                             ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Activate   : ${YELLOW}source .venv/bin/activate${NC}"
echo -e "  Start app  : ${YELLOW}python app.py${NC}"
echo -e "  Access     : ${YELLOW}http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo localhost):8000${NC}"
echo ""
echo -e "${YELLOW}NOTE: Re-login or run 'newgrp render' for GPU group changes.${NC}"
echo ""
