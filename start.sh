#!/bin/bash
# AI Vigilance Startup Script

# 1. Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found. Run bash setup_linux.sh first."
    exit 1
fi

# 2. Set environment variables for performance
# Force OpenVINO to use GPU if available
export OPENVINO_DEVICE="GPU"
# Prevent excessive fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "-------------------------------------------------------"
echo "🚀 AI VIGILANCE - Optimized Surveillance System"
echo "   Mode: High Performance (30 FPS Target)"
echo "   Hardware Acceleration: Enabled"
echo "-------------------------------------------------------"

# 3. Launch with Uvicorn (Single worker for camera thread stability)
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
