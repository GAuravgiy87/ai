import os
import logging

def setup_logging(log_file="logs/app.log"):
    # Silence noisy env vars
    os.environ["OPENCV_LOG_LEVEL"]        = "OFF"
    os.environ["FFMPEG_LOG_LEVEL"]        = "quiet"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"]  = "-8"
    os.environ["AV_LOG_FORCE_LEVEL"]      = "0"
    os.environ["PYTHONWARNINGS"]          = "ignore"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|analyze_duration;100000|probesize;100000|loglevel;quiet"
    )

    print("[OK] AI Vigilance System Starting...")

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # ── File handler: everything INFO+ goes to app.log ────────────────────
    file_h = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.INFO)

    # ── Terminal handler: WARNING+ only, so the live table isn't polluted ─
    # INFO messages (normal startup, camera restored, etc.) go to file only.
    # WARNING/ERROR/CRITICAL still appear in the terminal above the table.
    term_h = logging.StreamHandler()
    term_h.setFormatter(fmt)
    term_h.setLevel(logging.WARNING)

    logging.root.handlers.clear()
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_h)
    logging.root.addHandler(term_h)

    # Silence noisy third-party libs
    for noisy in ["ultralytics", "httpx", "httpcore",
                  "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").disabled = True

    return logging.getLogger("app")
