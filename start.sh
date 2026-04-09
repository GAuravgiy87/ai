#!/bin/bash
# Start script for AI Vigilance

# Determine if we're in a venv
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
fi

# Headless environment check for Ubuntu VM
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Disable GUI focus requirements for OpenCV
    export DISPLAY=${DISPLAY:-""}
    # Priority for FFmpeg backends
    export OPENCV_VIDEOIO_PRIORITY_MSMF=0
fi

# Set working directory to the script location
cd "$(dirname "$0")"

echo "Starting AI Vigilance Server..."
echo "Local Access: http://localhost:8000"

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 60
