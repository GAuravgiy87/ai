import os
import logging
import builtins as _builtins

def setup_logging(log_file="app.log"):
    # Silence noisy libs before any import
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["AV_LOG_FORCE_LEVEL"] = "0" # Silences decode errors
    os.environ["PYTHONWARNINGS"] = "ignore"

    print("[OK] AI Vigilance System Starting...")

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_h = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.INFO)

    # Also log INFO+ to terminal so user can see what's happening
    stderr_h = logging.StreamHandler()
    stderr_h.setFormatter(fmt)
    stderr_h.setLevel(logging.INFO)

    logging.root.handlers.clear()
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_h)
    logging.root.addHandler(stderr_h)

    # Silence noisy libs
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    # Completely stop uvicorn access logs (noise)
    logging.getLogger("uvicorn.access").disabled = True

    return logging.getLogger("app")
