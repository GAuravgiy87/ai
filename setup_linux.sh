#!/bin/bash
# Setup script for AI Vigilance on Linux VM (Headless / Ubuntu 22.04+)

set -e

echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    sqlite3 curl

echo "[2/6] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "[3/6] Upgrading pip and core tools..."
pip install --upgrade pip setuptools wheel

echo "[4/6] Installing Python dependencies..."
# If NVIDIA GPU is detected, install CUDA-ready torch. Otherwise, default to index.
if command -v nvidia-smi &> /dev/null; then
    echo "  -> NVIDIA GPU detected. Installing CUDA-optimized PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "  -> No GPU detected. Installing CPU version of PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

echo "[5/6] Creating workspace directories..."
mkdir -p snapshots dataset recordings vehicles

echo "[6/6] Done. To start the application, run:"
echo "  bash start.sh"
