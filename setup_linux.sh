#!/bin/bash
# AI Vigilance - Linux Setup Script (Headless VM)
# Run once on a fresh machine. Installs all system + Python dependencies.

set -e

echo "=============================================="
echo "  AI Vigilance - Linux Setup"
echo "=============================================="

echo ""
echo "[1/7] Updating package lists..."
sudo apt-get update -qq

echo ""
echo "[2/7] Installing system dependencies..."
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    libavcodec-dev libavformat-dev libswscale-dev \
    sqlite3 \
    curl wget git

echo ""
echo "[3/7] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "[4/7] Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "[5/7] Installing CPU-only PyTorch (skips ~2GB CUDA download)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

echo ""
echo "[6/7] Installing Python application dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "[7/7] Creating required workspace directories..."
mkdir -p snapshots dataset recordings database static/uploads

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  To start the application, run:"
echo "    chmod +x start.sh && ./start.sh"
echo ""
echo "  Or manually:"
echo "    source .venv/bin/activate"
echo "    python3 app.py"
echo ""
echo "  Access at: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
