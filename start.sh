#!/bin/bash
# =============================================================================
# start.sh — AI Vigilance System Start (Linux)
# =============================================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# Load environment written by setup_linux.sh
ENV_FILE="$HOME/.ai_vigilance_env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    info "Environment loaded from $ENV_FILE"
fi

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
    ok "Virtual environment activated."
else
    warn "No .venv found — run ./setup_linux.sh first."
fi

# Quick hardware check
info "Hardware:"
if [ -e /dev/kfd ]; then
    ok "  AMD GPU: /dev/kfd present (ROCm)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    ok "  NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "  No GPU detected — CPU inference"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo -e "${BOLD}${GREEN}  AI Vigilance starting...${NC}"
echo -e "  Dashboard : ${YELLOW}http://${IP:-localhost}:8000${NC}"
echo -e "  Logs      : ${YELLOW}tail -f app.log${NC}"
echo ""

exec python3 app.py
