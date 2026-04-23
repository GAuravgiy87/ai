"""
core/logging_config.py — Minimal, clean logging system.

Rules:
  app.log  : CRITICAL events only — startup, shutdown, crash, camera add/remove, critical errors
  Terminal : Real-time ERROR+ with exact cause, colored, always visible
  DB table : WARNING+ only — no verbose noise

System snapshots (CPU/GPU/RAM/temp) are captured at startup and shutdown/crash.
"""
import logging
import os
import platform
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from typing import Optional

# ── ANSI colours for terminal ─────────────────────────────────────────────
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# Disable colours on Windows if not supported
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass  # colours may not work on older Windows — that's fine


# ── System snapshot ───────────────────────────────────────────────────────

def capture_system_snapshot(reason: str = "") -> dict:
    """
    Collect CPU, RAM, GPU load, temperature, and uptime.
    Returns a dict — safe to call at any time, never raises.
    """
    snap = {
        "reason":    reason,
        "platform":  platform.platform(),
        "python":    sys.version.split()[0],
        "cpu_pct":   None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "ram_pct":   None,
        "gpu_name":  None,
        "gpu_load_pct": None,
        "gpu_vram_used_mb": None,
        "cpu_temp_c": None,
        "uptime_s":  None,
    }

    # CPU + RAM via psutil
    try:
        import psutil
        snap["cpu_pct"]      = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()
        snap["ram_used_mb"]  = round(vm.used  / 1024 / 1024, 1)
        snap["ram_total_mb"] = round(vm.total / 1024 / 1024, 1)
        snap["ram_pct"]      = vm.percent
        snap["uptime_s"]     = round(time.time() - psutil.boot_time(), 0)

        # CPU temperature (Linux sensors / Windows WMI)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                    if key in temps and temps[key]:
                        snap["cpu_temp_c"] = temps[key][0].current
                        break
        except Exception:
            pass
    except ImportError:
        pass

    # GPU via hw_manager (already initialised)
    try:
        from utils.hw_manager import hw
        snap["gpu_name"]     = hw.gpu_name
        snap["gpu_load_pct"] = round(hw.gpu_load * 100, 1)
    except Exception:
        pass

    # GPU VRAM (onnxruntime / torch)
    try:
        import onnxruntime as ort
        # DirectML doesn't expose VRAM easily — skip
    except Exception:
        pass

    return snap


def format_snapshot(snap: dict) -> str:
    """Format snapshot dict into a compact single-line string for the log file."""
    parts = [f"reason={snap.get('reason') or 'N/A'}"]
    if snap.get("cpu_pct")      is not None: parts.append(f"cpu={snap['cpu_pct']}%")
    if snap.get("ram_pct")      is not None: parts.append(f"ram={snap['ram_pct']}% ({snap['ram_used_mb']}/{snap['ram_total_mb']} MB)")
    if snap.get("gpu_name"):                 parts.append(f"gpu={snap['gpu_name']}")
    if snap.get("gpu_load_pct") is not None: parts.append(f"gpu_load={snap['gpu_load_pct']}%")
    if snap.get("cpu_temp_c")   is not None: parts.append(f"cpu_temp={snap['cpu_temp_c']}°C")
    if snap.get("uptime_s")     is not None: parts.append(f"uptime={int(snap['uptime_s'])}s")
    parts.append(f"platform={snap.get('platform','?')}")
    return " | ".join(parts)


# ── File handler — CRITICAL only, rotating 5 MB × 3 ─────────────────────

class _CriticalFileHandler(RotatingFileHandler):
    """Writes only CRITICAL-level records to app.log."""

    # Events we always want in the file even if level < CRITICAL
    _ALWAYS_LOG = {
        "system.startup", "system.shutdown", "system.crash",
        "system.signal", "camera.add", "camera.remove",
    }

    def emit(self, record: logging.LogRecord):
        # Always log CRITICAL
        if record.levelno >= logging.CRITICAL:
            super().emit(record)
            return
        # Always log startup/shutdown/crash/camera events at WARNING+
        src = getattr(record, "source", "") or record.name
        if src in self._ALWAYS_LOG and record.levelno >= logging.WARNING:
            super().emit(record)
            return
        # Log ERROR from any source
        if record.levelno >= logging.ERROR:
            super().emit(record)


# ── Terminal handler — ERROR+ with colour ────────────────────────────────

class _TerminalHandler(logging.StreamHandler):
    """Prints ERROR+ to terminal with colour and exact cause."""

    _LEVEL_COLOUR = {
        logging.ERROR:    _RED,
        logging.CRITICAL: _BOLD + _RED,
        logging.WARNING:  _YELLOW,
    }

    def emit(self, record: logging.LogRecord):
        if record.levelno < logging.ERROR:
            return
        colour = self._LEVEL_COLOUR.get(record.levelno, _RED)
        ts     = time.strftime("%H:%M:%S")
        msg    = record.getMessage()

        # Print the main message
        print(f"{colour}[{ts}] [{record.levelname}] {msg}{_RESET}", file=sys.stderr, flush=True)

        # Print traceback if present
        if record.exc_info:
            tb = "".join(traceback.format_exception(*record.exc_info)).strip()
            print(f"{_RED}{tb}{_RESET}", file=sys.stderr, flush=True)


# ── DB handler — WARNING+ only, no verbose noise ─────────────────────────

class MinimalDBLogHandler(logging.Handler):
    """
    Writes WARNING+ records to system_logs SQLite table.
    Skips noisy sources entirely.
    """
    _SKIP = {"uvicorn.access", "uvicorn", "httpx", "httpcore", "urllib3",
             "PIL", "ultralytics", "core.pipeline", "utils.detector",
             "utils.recognizer", "utils.tracker", "cameras.camera_manager"}

    def __init__(self, db_ref_getter):
        super().__init__(level=logging.WARNING)
        self._get_db = db_ref_getter
        import queue as _q
        self._queue  = _q.Queue(maxsize=500)
        self._worker = threading.Thread(target=self._drain, daemon=True, name="db-log-drain")
        self._worker.start()

    def emit(self, record: logging.LogRecord):
        if record.levelno < logging.WARNING:
            return
        if record.name in self._SKIP:
            return
        try:
            self._queue.put_nowait(record)
        except Exception:
            pass

    def _drain(self):
        import queue as _q
        while True:
            try:
                record = self._queue.get(timeout=2)
                db = self._get_db()
                if db is None:
                    continue
                extra = None
                if record.exc_info:
                    extra = "".join(traceback.format_exception(*record.exc_info)).strip()
                db.log_event(
                    level   = record.levelname,
                    message = record.getMessage(),
                    source  = getattr(record, "source", None) or record.name,
                    extra   = extra,
                )
            except _q.Empty:
                continue
            except Exception:
                pass


# ── Setup function — called once from app.py ─────────────────────────────

def setup_logging(log_file: str = "app.log") -> None:
    """
    Configure the root logger:
      - app.log  : CRITICAL + startup/shutdown/camera events + all ERRORs (rotating 5 MB × 3)
      - Terminal : ERROR+ with colour
    DB handler is added separately after db_manager is created.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    file_h = _CriticalFileHandler(log_file, maxBytes=5*1024*1024,
                                   backupCount=3, encoding="utf-8")
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.DEBUG)   # handler filters internally

    term_h = _TerminalHandler(sys.stderr)
    term_h.setFormatter(fmt)
    term_h.setLevel(logging.ERROR)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)     # let handlers decide what to keep
    root.addHandler(file_h)
    root.addHandler(term_h)

    # Silence completely useless loggers
    for name in ("uvicorn.access", "httpx", "httpcore", "urllib3",
                 "PIL", "ultralytics", "multipart"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # uvicorn error/lifespan — only CRITICAL to file
    for name in ("uvicorn", "uvicorn.error", "uvicorn.lifespan"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Pipeline / ML modules — only ERROR+ (no verbose info spam)
    for name in ("core.pipeline", "utils.detector", "utils.recognizer",
                 "utils.tracker", "cameras.camera_manager", "utils.hw_manager",
                 "core.startup", "core.state"):
        logging.getLogger(name).setLevel(logging.ERROR)
