#!/bin/bash
# =============================================================================
# start.sh — AI Vigilance System Start
# Activates hardware (Intel iGPU VAAPI + AMD RX 550 ROCm) and launches app.
# =============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ── Load ROCm + VAAPI environment ─────────────────────────────────────────────
ROCM_ENV="$HOME/.rocm_env"
if [ -f "$ROCM_ENV" ]; then
    source "$ROCM_ENV"
    info "ROCm/VAAPI environment loaded from $ROCM_ENV"
else
    # Inline defaults if setup hasn't been run yet
    export HSA_OVERRIDE_GFX_VERSION=8.0.3   # RX 550 = gfx803
    export ROCR_VISIBLE_DEVICES=0
    export HIP_VISIBLE_DEVICES=0
    export LIBVA_DRIVER_NAME=iHD
    export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
    export OPENCV_LOG_LEVEL=OFF
    export FFMPEG_LOG_LEVEL=quiet
    export PYTHONWARNINGS=ignore
fi

# ── CPU affinity — pin app to all 4 cores ─────────────────────────────────────
CPU_CORES=$(nproc)
info "CPU cores available: $CPU_CORES"

# ── Activate virtual environment ──────────────────────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
    ok "Virtual environment activated."
else
    warn "No .venv found — run ./setup_linux.sh first. Using system Python."
fi

# ── Hardware status check ─────────────────────────────────────────────────────
echo ""
info "Hardware check:"

# AMD dGPU
if [ -e /dev/kfd ]; then
    ok "  AMD dGPU  : /dev/kfd present (ROCm active)"
    # Verify PyTorch sees it
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'\033[0;32m[ OK ]\033[0m   ROCm device: {torch.cuda.get_device_name(0)}')
else:
    print('\033[1;33m[WARN]\033[0m   ROCm PyTorch not detecting GPU — face recognition on CPU')
" 2>/dev/null || warn "  PyTorch check failed — continuing anyway."
else
    warn "  AMD dGPU  : /dev/kfd not found — face recognition on CPU"
fi

# Intel iGPU VAAPI
if vainfo 2>/dev/null | grep -q "iHD\|Intel"; then
    ok "  Intel iGPU: VAAPI active (hardware video decode)"
else
    warn "  Intel iGPU: VAAPI not detected — video decode on CPU"
fi

# DRI render nodes
if ls /dev/dri/renderD* &>/dev/null 2>&1; then
    ok "  DRI nodes : $(ls /dev/dri/renderD* | tr '\n' ' ')"
fi

# ── Set process CPU limit (nice + taskset) ────────────────────────────────────
# Use all 4 cores, slightly lower priority so system stays responsive
TASKSET_CMD=""
if command -v taskset &>/dev/null; then
    TASKSET_CMD="taskset -c 0-$((CPU_CORES-1))"
fi
NICE_CMD="nice -n 5"

# ── Launch ─────────────────────────────────────────────────────────────────────
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║   AI Vigilance System Starting...            ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Dashboard  : ${YELLOW}http://${IP:-localhost}:8000${NC}"
echo -e "  HW Status  : ${YELLOW}http://${IP:-localhost}:8000/api/hw_status${NC}"
echo -e "  Logs       : ${YELLOW}tail -f app.log${NC}"
echo -e "  Stop       : ${YELLOW}Ctrl+C${NC}"
echo ""

exec $TASKSET_CMD $NICE_CMD python3 app.py