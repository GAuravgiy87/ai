#!/bin/bash
# AI Vigilance Startup Script

# Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found. Run bash setup_linux.sh first."
    exit 1
fi

echo "-------------------------------------------------------"
echo "🚀 AI VIGILANCE - Surveillance System"
echo "   Mode: 10 FPS Accuracy Mode"
echo "   Hardware: Auto-detected (dGPU > iGPU > CPU)"
echo "-------------------------------------------------------"

# Single worker — required for camera thread stability
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
