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
    echo [1/5] Creating virtual environment...
    python -m venv .venv
)

echo [2/5] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/5] Installing PyTorch CPU build...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

echo [4/5] Removing conflicting onnxruntime packages...
pip uninstall onnxruntime onnxruntime-gpu -y >nul 2>&1

echo [5/5] Installing onnxruntime-directml + all dependencies...
pip install onnxruntime-directml --quiet
REM Skip torch/torchvision — already installed above
pip install fastapi uvicorn[standard] jinja2 python-multipart httpx aiofiles --quiet
pip install python-jose[cryptography] passlib[bcrypt] --quiet
pip install opencv-python-headless numpy scipy pytz --quiet
pip install ultralytics facenet-pytorch psutil --quiet

echo.
echo  Verifying DirectML...
python -c "import onnxruntime as ort; p=ort.get_available_providers(); print('Providers:', p); print('DirectML:', 'DmlExecutionProvider' in p)"

echo.
echo  GPU Info...
python -c "from utils.hw_manager import hw; s=hw.get_status(); print('GPU:', s['gpu_name']); print('Available:', s['gpu_available']); print('Provider:', s['ort_providers'])"

echo.
echo  Setup complete!
echo  Start: python app.py
echo.
pause
