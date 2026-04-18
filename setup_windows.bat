@echo off
REM ── AI Vigilance — Windows Setup ─────────────────────────────────────────────
REM GPU: onnxruntime-directml (DirectX 12 — AMD / Intel / NVIDIA on Windows)
REM Run once before starting the app.

echo.
echo  AI Vigilance — Windows Setup
echo  GPU: DirectML (DirectX 12 — AMD / Intel / NVIDIA)
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Create venv if not exists
if not exist ".venv" (
    echo [1/6] Creating virtual environment...
    python -m venv .venv
)

echo [2/6] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/6] Installing PyTorch CPU build (no torchvision — not needed)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

echo [4/6] Removing conflicting onnxruntime + opencv packages...
pip uninstall onnxruntime onnxruntime-gpu opencv-python -y >nul 2>&1

echo [5/6] Installing onnxruntime-directml + core dependencies...
pip install onnxruntime-directml --quiet
pip install -r requirements.txt --quiet

echo [6/6] Installing ultralytics (no-deps — only needed for ONNX export)...
pip install ultralytics --no-deps --quiet

echo.
echo  Verifying DirectML...
python -c "import onnxruntime as ort; p=ort.get_available_providers(); print('Providers:', p); print('DirectML:', 'DmlExecutionProvider' in p)"

echo.
echo  GPU Info...
python -c "from utils.hw_manager import hw; s=hw.get_status(); print('GPU:', s['gpu_name']); print('Available:', s['gpu_available'])"

echo.
echo  Setup complete! Start with: python app.py
echo.
pause
