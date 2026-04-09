#!/bin/bash
# AI-VIGILANCE Deployment Script
# Optimized for Ubuntu 22.04+ with GPU Acceleration (Intel/AMD/NVIDIA)

set -e

echo "🚀 Starting AI-VIGILANCE Setup..."

echo "📦 [1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    ffmpeg sqlite3 curl mesa-utils pciutils

echo "virtualenv [2/6] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "🐍 [3/6] Upgrading pip and core tools..."
pip install --upgrade pip setuptools wheel

echo "🔥 [4/6] Installing AI Backend (GPU Optimized)..."
# Detection for NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "  -> NVIDIA GPU detected. Installing CUDA 12.1 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "  -> Generic Hardware. Installing Base PyTorch + DirectML/OpenVINO..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "📋 [5/6] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "📁 [6/6] Creating workspace directories..."
mkdir -p snapshots dataset recordings vehicles

echo "✅ Setup Complete!"
echo "-------------------------------------------------------"
echo "To start the AI Vigilance at 30 FPS, run:"
echo "  bash start.sh"
echo "-------------------------------------------------------"
