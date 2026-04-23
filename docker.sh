#!/usr/bin/env bash
# =============================================================================
# docker.sh — AI Vigilance System Docker deploy
# Ubuntu | Intel iGPU (VAAPI) + AMD RX 550 dGPU (ROCm) | 4 CPU | 4.5 GB RAM
# =============================================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   AI Vigilance — Docker Deploy Script        ║${NC}"
echo -e "${BOLD}${CYAN}║   Intel iGPU + AMD RX 550 + 4-core CPU       ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Docker ────────────────────────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    warn "Docker not found — installing..."
    sudo apt-get update -qq
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    ok "Docker installed."
else
    ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"
fi

# ── 2. Docker Compose ────────────────────────────────────────────────────────
info "Checking Docker Compose..."
if ! docker compose version &>/dev/null 2>&1; then
    warn "Docker Compose plugin not found — installing..."
    sudo apt-get install -y docker-compose-plugin
    ok "Docker Compose installed."
else
    ok "Docker Compose $(docker compose version --short)"
fi

# ── 3. AMD ROCm kernel driver (/dev/kfd) ─────────────────────────────────────
info "Checking AMD ROCm (/dev/kfd)..."
if [ -e /dev/kfd ]; then
    ok "AMD ROCm kernel driver found (/dev/kfd)"
    for grp in video render; do
        if ! groups "$USER" | grep -q "$grp"; then
            sudo usermod -aG "$grp" "$USER"
            warn "Added $USER to group '$grp' — re-login may be needed for GPU access."
        fi
    done
else
    warn "/dev/kfd not found — AMD ROCm unavailable. Face recognition will run on CPU."
    # Remove kfd line from compose so container starts without it
    sed -i '/\/dev\/kfd/d' docker-compose.yml 2>/dev/null || true
fi

# ── 4. Intel iGPU VAAPI ───────────────────────────────────────────────────────
info "Checking Intel iGPU (VAAPI)..."
if command -v vainfo &>/dev/null; then
    if vainfo 2>/dev/null | grep -q "iHD\|Intel"; then
        ok "Intel iGPU VAAPI detected"
    else
        warn "vainfo found but Intel iHD driver not active — VAAPI decode may not work."
    fi
else
    warn "vainfo not installed — install with: sudo apt install vainfo intel-media-va-driver-non-free"
fi

# ── 5. /dev/dri check ────────────────────────────────────────────────────────
info "Checking /dev/dri render nodes..."
if ls /dev/dri/renderD* &>/dev/null 2>&1; then
    ok "DRI render nodes: $(ls /dev/dri/renderD* | tr '\n' ' ')"
else
    warn "No /dev/dri render nodes found — removing GPU device mapping from compose."
    sed -i '/\/dev\/dri/d; /group_add:/,/- render/d' docker-compose.yml 2>/dev/null || true
fi

# ── 6. Data dirs ─────────────────────────────────────────────────────────────
info "Preparing data directories..."
mkdir -p snapshots recordings dataset
touch db.sqlite3 app.log
ok "Data directories ready."

# ── 7. Stop existing container ───────────────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^ai_vigilance$"; then
    warn "Stopping existing container..."
    docker compose down
    ok "Stopped."
fi

# ── 8. Build ─────────────────────────────────────────────────────────────────
info "Building Docker image (first build installs ROCm PyTorch — ~5 min)..."
docker compose build
ok "Image built."

# ── 9. Start ─────────────────────────────────────────────────────────────────
info "Starting container..."
docker compose up -d
ok "Container started."

# ── 10. Hardware status ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  AI Vigilance is running!                    ║${NC}"
echo -e "${BOLD}${GREEN}║  URL      : http://localhost:8000            ║${NC}"
echo -e "${BOLD}${GREEN}║  HW API   : http://localhost:8000/api/hw_status ║${NC}"
echo -e "${BOLD}${GREEN}║  RAM limit: 4.5 GB  |  CPU limit: 4 cores   ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

info "Container status:"
docker compose ps

echo ""
echo -e "${BOLD}Useful commands:${NC}"
echo -e "  ${YELLOW}docker stats ai_vigilance${NC}          — live CPU/RAM/GPU usage"
echo -e "  ${YELLOW}docker compose logs -f${NC}             — live logs"
echo -e "  ${YELLOW}docker compose down${NC}                — stop"
echo -e "  ${YELLOW}curl localhost:8000/api/hw_status${NC}  — hardware device status"
echo ""
