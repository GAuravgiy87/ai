"""
core/diagnostics.py — Live resource table + crash diagnostics.

Terminal display:
  - Single fixed table that updates IN PLACE every 2 seconds (no scrolling)
  - Uses ANSI cursor-up to overwrite the same lines each refresh
  - On crash: clears the table, prints full red crash report, saves to file

Crash log (crash_forensics.log):
  - Timestamp, exception type + message, source thread
  - Full traceback
  - System resources at crash moment (CPU, RAM, GPU, disk, threads)
"""

import os
import sys
import time
import logging
import platform
import threading
import traceback
from datetime import datetime

logger = logging.getLogger("diagnostics")

CRASH_LOG  = "crash_forensics.log"
_DIVIDER   = "=" * 72

# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _enable_ansi():
    """Enable ANSI escape codes on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_ansi()

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"

def _up(n):   return f"\033[{n}A"   # cursor up n lines
def _clr():   return "\033[2K"      # clear current line
def _col(pct, warn=70, crit=90):
    """Color a percentage value: green / yellow / red."""
    try:
        v = float(str(pct).rstrip('%'))
        if v >= crit:  return RED
        if v >= warn:  return YELLOW
        return GREEN
    except Exception:
        return WHITE

# ── Resource collection ───────────────────────────────────────────────────────

def _get_resources() -> dict:
    r = {}
    try:
        import psutil
        # interval=None calculates average since last call (2 seconds), matching Task Manager
        r['cpu']         = psutil.cpu_percent(interval=None)
        mem              = psutil.virtual_memory()
        r['ram_used']    = round(mem.used  / 1073741824, 2)   # GB
        r['ram_total']   = round(mem.total / 1073741824, 2)
        r['ram_pct']     = mem.percent
        disk             = psutil.disk_usage('.')
        r['disk_used']   = round(disk.used  / 1073741824, 2)
        r['disk_total']  = round(disk.total / 1073741824, 2)
        r['disk_pct']    = disk.percent
        proc             = psutil.Process(os.getpid())
        r['proc_cpu']    = round(proc.cpu_percent(interval=0.1), 1)
        r['proc_ram']    = round(proc.memory_info().rss / 1048576, 1)  # MB
        r['threads']     = proc.num_threads()
    except Exception as e:
        r['err'] = str(e)

    try:
        from core.resource_guard import get_level, get_det_fps
        rg_level = get_level()
        rg_fps   = get_det_fps()
        r['rg_level'] = rg_level
        r['rg_fps']   = rg_fps if rg_fps > 0 else 0
    except Exception:
        r['rg_level'] = 'ok'
        r['rg_fps']   = 6.0

    try:
        from ml_inference.hw_manager import hw
        s   = hw.get_status()
        gpu = s.get('gpu')
        if gpu:
            r['gpu_name'] = gpu.get('name', 'GPU')[:24]
            r['gpu_load'] = gpu.get('load', 'N/A')
            r['gpu_mem']  = gpu.get('memory_used', 'N/A')
        else:
            r['gpu_name'] = 'N/A'
            r['gpu_load'] = 'N/A'
            r['gpu_mem']  = 'N/A'
    except Exception:
        r['gpu_name'] = 'N/A'
        r['gpu_load'] = 'N/A'
        r['gpu_mem']  = 'N/A'

    return r

# ── Live table ────────────────────────────────────────────────────────────────

# Number of lines the table occupies — used to move cursor back up
_TABLE_LINES = 10
_table_printed = False   # have we printed the table at least once?
_table_lock    = threading.Lock()

def _render_table(r: dict, uptime_s: float) -> str:
    """Build the fixed-height table string."""
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    h    = int(uptime_s // 3600)
    m    = int((uptime_s % 3600) // 60)
    s    = int(uptime_s % 60)
    up   = f"{h:02d}:{m:02d}:{s:02d}"

    cpu_c  = _col(r.get('cpu', 0))
    ram_c  = _col(r.get('ram_pct', 0))
    disk_c = _col(r.get('disk_pct', 0))
    gpu_c  = _col(r.get('gpu_load', 0))

    # Resource guard level color
    rg_level = r.get('rg_level', 'ok')
    rg_fps   = r.get('rg_fps', 6.0)
    if rg_level == 'crit':
        rg_color = RED;    rg_str = f"PAUSED  (CPU critical)"
    elif rg_level == 'high':
        rg_color = RED;    rg_str = f"THROTTLED  {rg_fps:.0f}fps  CLAHE off"
    elif rg_level == 'warn':
        rg_color = YELLOW; rg_str = f"THROTTLED  {rg_fps:.0f}fps"
    else:
        rg_color = GREEN;  rg_str = f"OK  {rg_fps:.0f}fps"

    W = 72   # table width

    def row(label, value, color=WHITE):
        label_w = 22
        val_str = str(value)
        pad     = W - label_w - len(val_str) - 4
        return (f"  {DIM}{label:<{label_w}}{RESET}"
                f"{color}{val_str}{RESET}"
                f"{' ' * max(0, pad)}  ")

    lines = [
        f"{BOLD}{CYAN}{'─'*W}{RESET}",
        f"  {BOLD}{WHITE}AI Vigilance  │  {now}  │  Uptime {up}{RESET}",
        f"{BOLD}{CYAN}{'─'*W}{RESET}",
        row("CPU  (system)",
            f"{r.get('cpu','?')}%",          cpu_c),
        row("CPU  (process)",
            f"{r.get('proc_cpu','?')}%",     cpu_c),
        row("RAM  (system)",
            f"{r.get('ram_used','?')} / {r.get('ram_total','?')} GB  "
            f"({r.get('ram_pct','?')}%)",    ram_c),
        row("RAM  (process)",
            f"{r.get('proc_ram','?')} MB  "
            f"│  {r.get('threads','?')} threads", WHITE),
        row("GPU",
            f"{r.get('gpu_name','N/A')}  "
            f"load={r.get('gpu_load','?')}  "
            f"mem={r.get('gpu_mem','?')} MB", gpu_c),
        row("Disk",
            f"{r.get('disk_used','?')} / {r.get('disk_total','?')} GB  "
            f"({r.get('disk_pct','?')}%)",   disk_c),
        row("Detection",    rg_str,           rg_color),
        f"{BOLD}{CYAN}{'─'*W}{RESET}",
    ]
    return "\n".join(lines)


def _print_table(r: dict, uptime_s: float):
    """Overwrite the table in place using cursor-up."""
    global _table_printed
    table = _render_table(r, uptime_s)
    n     = _TABLE_LINES + 1   # +1 for the bottom divider line

    with _table_lock:
        if _table_printed:
            # Move cursor up to the first line of the table and overwrite
            sys.stdout.write(_up(n))
        sys.stdout.write(table + "\n")
        sys.stdout.flush()
        _table_printed = True


def _clear_table():
    """Erase the table from the terminal before printing a crash report."""
    global _table_printed
    with _table_lock:
        if _table_printed:
            n = _TABLE_LINES + 1
            sys.stdout.write(_up(n))
            for _ in range(n):
                sys.stdout.write(_clr() + "\n")
            sys.stdout.write(_up(n))
            sys.stdout.flush()
        _table_printed = False

# ── Crash report ──────────────────────────────────────────────────────────────

def _format_resources_for_log(r: dict) -> str:
    lines = [
        f"  CPU  (system)  : {r.get('cpu','N/A')}%",
        f"  CPU  (process) : {r.get('proc_cpu','N/A')}%",
        f"  RAM  (system)  : {r.get('ram_used','N/A')} / {r.get('ram_total','N/A')} GB  ({r.get('ram_pct','N/A')}%)",
        f"  RAM  (process) : {r.get('proc_ram','N/A')} MB  |  {r.get('threads','N/A')} threads",
        f"  GPU            : {r.get('gpu_name','N/A')}  load={r.get('gpu_load','N/A')}  mem={r.get('gpu_mem','N/A')} MB",
        f"  Disk           : {r.get('disk_used','N/A')} / {r.get('disk_total','N/A')} GB  ({r.get('disk_pct','N/A')}%)",
    ]
    if 'err' in r:
        lines.append(f"  [psutil error] : {r['err']}")
    return "\n".join(lines)


def _write_crash_report(exc_type, value, tb, source: str = "main"):
    """Build crash report, print to terminal in red, save to file."""
    now       = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    resources = _get_resources()
    tb_str    = "".join(traceback.format_exception(exc_type, value, tb))

    report = (
        f"\n{_DIVIDER}\n"
        f"  !!!  CRASH DETECTED  !!!\n"
        f"{_DIVIDER}\n"
        f"  Time       : {timestamp}\n"
        f"  Source     : {source}\n"
        f"  Exception  : {exc_type.__name__ if exc_type else 'Unknown'}\n"
        f"  Message    : {value}\n"
        f"  Platform   : {platform.system()} {platform.release()}"
        f"  |  Python {sys.version.split()[0]}\n"
        f"{_DIVIDER}\n"
        f"  SYSTEM RESOURCES AT CRASH TIME\n"
        f"{_format_resources_for_log(resources)}\n"
        f"{_DIVIDER}\n"
        f"  FULL TRACEBACK\n"
        f"{tb_str}"
        f"{_DIVIDER}\n"
    )

    # Clear the live table first so crash report isn't mixed with it
    _clear_table()

    # Print in red to terminal
    try:
        print(f"{RED}{report}{RESET}", file=sys.stderr, flush=True)
    except Exception:
        print(report, file=sys.stderr, flush=True)

    # Save to crash log
    try:
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(report)
    except Exception:
        pass

    try:
        logger.critical(report)
    except Exception:
        pass

# ── Thread exception hook ─────────────────────────────────────────────────────

def _thread_excepthook(args):
    if args.exc_type is SystemExit:
        return
    _write_crash_report(
        args.exc_type, args.exc_value, args.exc_traceback,
        source=f"thread:{getattr(args.thread, 'name', 'unknown')}"
    )

# ── Monitor loop ──────────────────────────────────────────────────────────────

_monitor_stop = threading.Event()
_start_time   = time.time()

def _monitor_loop(interval: int):
    # Print a blank table area first so cursor-up works on first refresh
    print("\n" * (_TABLE_LINES + 1), end="", flush=True)
    while not _monitor_stop.wait(interval):
        try:
            r = _get_resources()
            _print_table(r, time.time() - _start_time)
        except Exception:
            pass

def start_resource_monitor(interval: int = 2):
    """Start the live table monitor. interval=2 → refresh every 2 seconds."""
    t = threading.Thread(
        target=_monitor_loop,
        args=(interval,),
        name="resource-monitor",
        daemon=True,
    )
    t.start()
    return t

# ── Auto-restart ──────────────────────────────────────────────────────────────

def _auto_restart(delay: int = 5):
    print(
        f"\n{YELLOW}[AUTO-RESTART] Restarting in {delay}s ..."
        f"  ({sys.executable} {' '.join(sys.argv)}){RESET}\n",
        file=sys.stderr, flush=True
    )
    time.sleep(delay)
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"{RED}[AUTO-RESTART] Failed: {e}{RESET}", file=sys.stderr, flush=True)

# ── Install ───────────────────────────────────────────────────────────────────

_installed = False

def install(auto_restart: bool = True, monitor_interval: int = 2):
    """
    Call once at startup.
    monitor_interval=0 → skip monitor (used by camera server to avoid duplicate).
    """
    global _installed
    if _installed:
        logger.debug("[Diagnostics] Already installed, skipping duplicate installation.")
        return
    _installed = True

    def _main_excepthook(exc_type, value, tb):
        if exc_type is KeyboardInterrupt:
            _clear_table()
            print(f"\n{YELLOW}[Shutdown] Stopped by user.{RESET}\n", flush=True)
            sys.exit(0)
            
        import multiprocessing
        is_main_proc = multiprocessing.current_process().name == 'MainProcess'
        is_main_thread = threading.current_thread() == threading.main_thread()
        
        source = "main-thread" if is_main_thread else f"thread:{threading.current_thread().name}"
        _write_crash_report(exc_type, value, tb, source=source)
        
        if is_main_thread:
            if auto_restart and is_main_proc:
                _auto_restart(delay=5)
            else:
                sys.exit(1)
        # For background threads (like multiprocessing connection listener), return without exit/restart

    sys.excepthook = _main_excepthook

    try:
        threading.excepthook = _thread_excepthook
    except AttributeError:
        pass

    if monitor_interval > 0:
        start_resource_monitor(interval=monitor_interval)

    logger.info(
        f"[Diagnostics] Ready — crash log: {CRASH_LOG} | "
        f"table refresh: {monitor_interval}s | auto-restart: {auto_restart}"
    )
