#!/bin/bash
# AI Vigilance - Start Script
# Works on Linux VM (headless). Activates venv and launches the app.

# ── Activate virtual environment ──────────────────────────────────────────────
if [ -d ".venv" ]; then
    # Linux
    source .venv/bin/activate 2>/dev/null || \
    # Windows Git Bash fallback
    source .venv/Scripts/activate 2>/dev/null || true
else
    echo "[WARN] No .venv found. Run setup_linux.sh first."
    echo "       Attempting to run with system Python..."
fi

# ── Headless Linux: suppress GUI and FFMPEG noise ─────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export DISPLAY=${DISPLAY:-""}
    export OPENCV_VIDEOIO_PRIORITY_MSMF=0
    export OPENCV_LOG_LEVEL=OFF
    export FFMPEG_LOG_LEVEL=quiet
    export PYTHONWARNINGS=ignore
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   AI Vigilance System Starting...    ║"
echo "  ╚══════════════════════════════════════╝"
echo ""
echo "  Dashboard → http://${IP:-localhost}:8000"
echo "  Logs      → tail -f app.log"
echo ""

python3 app.py
