#!/bin/bash
# =============================================================================
# setup_linux.sh — AI Vigilance System Setup
# Ubuntu | Intel iGPU (VAAPI) + AMD RX 550 dGPU (ROCm) + 4-core CPU
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
echo -e "${BOLD}${CYAN}║   Intel iGPU + AMD RX 550 + 4-core CPU       ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
info "[1/8] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    sqlite3 curl wget git \
    # VAAPI — Intel iGPU hardware video decode
    libva-drm2 libva2 vainfo \
    intel-media-va-driver-non-free \
    # GStreamer + VAAPI plugin for OpenCV
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-vaapi \
    # OpenCL runtime (needed by ROCm userspace)
    ocl-icd-libopencl1 clinfo
ok "System packages installed."

# ── 2. AMD ROCm (dGPU — RX 550 / Polaris gfx803) ─────────────────────────────
info "[2/8] Checking AMD ROCm..."
if [ -e /dev/kfd ]; then
    ok "AMD ROCm kernel driver already present (/dev/kfd)"
else
    warn "AMD ROCm kernel driver not found. Installing ROCm..."
    # Add AMD ROCm repo
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
        | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
        https://repo.radeon.com/rocm/apt/6.0 $(lsb_release -cs) main" \
        | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y rocm-hip-runtime rocm-opencl-runtime
    ok "ROCm installed."
fi

# Add user to required groups for GPU access
for grp in video render; do
    if ! groups "$USER" | grep -q "$grp"; then
        sudo usermod -aG "$grp" "$USER"
        warn "Added $USER to group '$grp' — re-login required for GPU access."
    fi
done

# ── 3. Intel VAAPI driver check ───────────────────────────────────────────────
info "[3/8] Checking Intel iGPU VAAPI..."
if vainfo 2>/dev/null | grep -q "iHD\|Intel"; then
    ok "Intel iGPU VAAPI active"
else
    warn "Intel iHD VAAPI driver not detected. Trying to install..."
    sudo apt-get install -y intel-media-va-driver-non-free libigdgmm12 2>/dev/null || true
    warn "If VAAPI still fails, video decode will fall back to CPU (still works)."
fi

# ── 4. Python venv ────────────────────────────────────────────────────────────
info "[4/8] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
ok "Virtual environment ready."

# ── 5. PyTorch — ROCm build for AMD RX 550 ───────────────────────────────────
info "[5/8] Installing PyTorch (ROCm build for AMD RX 550)..."
# RX 550 = Polaris = gfx803. ROCm 5.7 is the last version supporting gfx803.
pip install --quiet \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm5.7 \
    && ok "PyTorch ROCm installed." \
    || {
        warn "ROCm PyTorch failed — falling back to CPU build."
        pip install --quiet torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu
        ok "PyTorch CPU installed."
    }

# ── 6. Python dependencies ────────────────────────────────────────────────────
info "[6/8] Installing Python application dependencies..."
# Skip torch/torchvision — already installed above
grep -v "^torch" requirements.txt > /tmp/req_no_torch.txt
pip install --quiet -r /tmp/req_no_torch.txt
ok "Python dependencies installed."

# ── 7. Workspace directories ──────────────────────────────────────────────────
info "[7/8] Creating workspace directories..."
mkdir -p snapshots dataset recordings
touch db.sqlite3 app.log
ok "Directories ready."

# ── 8. ROCm environment variables ────────────────────────────────────────────
info "[8/8] Writing ROCm environment config..."
ROCM_ENV_FILE="$HOME/.rocm_env"
cat > "$ROCM_ENV_FILE" << 'EOF'
# AMD RX 550 (Polaris / gfx803) ROCm settings
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
# Intel iGPU VAAPI
export LIBVA_DRIVER_NAME=iHD
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
# OpenCV / FFmpeg silence
export OPENCV_LOG_LEVEL=OFF
export FFMPEG_LOG_LEVEL=quiet
export PYTHONWARNINGS=ignore
EOF
# Source in .bashrc if not already there
if ! grep -q "rocm_env" "$HOME/.bashrc" 2>/dev/null; then
    echo "[ -f $ROCM_ENV_FILE ] && source $ROCM_ENV_FILE" >> "$HOME/.bashrc"
fi
ok "ROCm env written to $ROCM_ENV_FILE"

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Setup Complete!                             ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Start app  : ${YELLOW}./start.sh${NC}"
echo -e "  Docker     : ${YELLOW}./docker.sh${NC}"
echo -e "  Access     : ${YELLOW}http://$(hostname -I | awk '{print $1}'):8000${NC}"
echo ""
echo -e "${YELLOW}NOTE: Re-login or run 'newgrp render' for GPU group changes to take effect.${NC}"
echo ""
