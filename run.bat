@echo off
setlocal
title AI Vigilance System
echo [1/3] Checking dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo [2/3] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: FFmpeg not found in PATH. Video recording may fail.
    echo Please install FFmpeg from https://ffmpeg.org/download.html
)

echo [3/3] Starting AI Vigilance...
python app.py
if %errorlevel% neq 0 (
    echo AI Vigilance crashed.
    pause
)
pause
