"""
core/resource_guard.py — Dynamic resource throttle.

Monitors CPU usage every second. When CPU stays above thresholds for
a sustained period it automatically:
  - Reduces detection FPS
  - Skips display-frame CLAHE
  - Reduces JPEG quality
  - At critical level: pauses detection entirely

Logs only on STATE CHANGES — never spams the terminal.
"""

import time
import threading
import logging
import psutil

logger = logging.getLogger("resource_guard")

# ── Thresholds ────────────────────────────────────────────────────────────────
_CPU_WARN_PCT  = 75.0
_CPU_HIGH_PCT  = 85.0
_CPU_CRIT_PCT  = 92.0
_WARN_SECS     = 4
_HIGH_SECS     = 5
_CRIT_SECS     = 5
_PAUSE_SECS    = 8
_COOLDOWN_SECS = 15

# ── State ─────────────────────────────────────────────────────────────────────
_lock             = threading.Lock()
_det_fps_override = None    # None = default 6fps
_skip_clahe       = False
_jpeg_quality     = 75
_detection_paused = False
_pause_until      = 0.0
_level            = "ok"

# Guard against double-start
_started = False
_started_lock = threading.Lock()


def get_det_fps() -> float:
    with _lock:
        if _detection_paused and time.time() < _pause_until:
            return 0.0
        return _det_fps_override if _det_fps_override is not None else 6.0


def is_paused() -> bool:
    with _lock:
        return _detection_paused and time.time() < _pause_until


def should_skip_clahe() -> bool:
    with _lock:
        return _skip_clahe


def get_jpeg_quality() -> int:
    with _lock:
        return _jpeg_quality


def get_level() -> str:
    with _lock:
        return _level


# ── Monitor ───────────────────────────────────────────────────────────────────

def _monitor():
    global _det_fps_override, _skip_clahe, _jpeg_quality
    global _detection_paused, _pause_until, _level

    warn_since     = None
    high_since     = None
    crit_since     = None
    cooldown_until = 0.0
    _last_level    = "ok"   # track previous level to log only on change

    # Prime psutil so first call returns a real value
    psutil.cpu_percent(interval=None)

    while True:
        time.sleep(1.0)
        try:
            cpu = psutil.cpu_percent(interval=None)
            now = time.time()

            # ── Track sustained high CPU ──────────────────────────────────
            if cpu >= _CPU_CRIT_PCT:
                crit_since = crit_since or now
                high_since = high_since or now
                warn_since = warn_since or now
            elif cpu >= _CPU_HIGH_PCT:
                crit_since = None
                high_since = high_since or now
                warn_since = warn_since or now
            elif cpu >= _CPU_WARN_PCT:
                crit_since = None
                high_since = None
                warn_since = warn_since or now
            else:
                crit_since = None
                high_since = None
                warn_since = None

            # ── Apply throttle — log ONLY on level change ─────────────────
            with _lock:
                new_level = _level   # assume no change

                if (crit_since is not None and
                        (now - crit_since) >= _CRIT_SECS and
                        not _detection_paused):
                    _detection_paused = True
                    _pause_until      = now + _PAUSE_SECS
                    cooldown_until    = now + _PAUSE_SECS + _COOLDOWN_SECS
                    _det_fps_override = 2.0
                    _skip_clahe       = True
                    _jpeg_quality     = 55
                    new_level         = "crit"
                    if new_level != _last_level:  # BUG-10 fix: only log on state change
                        logger.warning(
                            f"[ResourceGuard] CPU {cpu:.0f}% critical — "
                            f"detection paused for {_PAUSE_SECS}s"
                        )

                elif _detection_paused and now >= _pause_until:
                    _detection_paused = False
                    new_level         = "high"   # still throttled post-pause
                    logger.info("[ResourceGuard] Detection resumed at 2fps")

                elif (high_since is not None and
                        (now - high_since) >= _HIGH_SECS and
                        not _detection_paused):
                    _det_fps_override = 3.0
                    _skip_clahe       = True
                    _jpeg_quality     = 60
                    new_level         = "high"
                    if new_level != _last_level:
                        logger.warning(
                            f"[ResourceGuard] CPU {cpu:.0f}% high — "
                            f"throttled to 3fps, CLAHE off"
                        )

                elif (warn_since is not None and
                        (now - warn_since) >= _WARN_SECS and
                        not _detection_paused):
                    _det_fps_override = 4.0
                    _skip_clahe       = False
                    _jpeg_quality     = 65
                    new_level         = "warn"
                    if new_level != _last_level:
                        logger.warning(
                            f"[ResourceGuard] CPU {cpu:.0f}% elevated — "
                            f"throttled to 4fps"
                        )

                elif (warn_since is None and
                        not _detection_paused and
                        now > cooldown_until):
                    # Restore defaults — but only log once on transition
                    if _level != "ok":
                        _det_fps_override = None
                        _skip_clahe       = False
                        _jpeg_quality     = 75
                        new_level         = "ok"
                        logger.info("[ResourceGuard] CPU normal — full FPS restored")
                    else:
                        # Already ok — ensure state is clean, no log
                        _det_fps_override = None
                        _skip_clahe       = False
                        _jpeg_quality     = 75
                        new_level         = "ok"

                _level      = new_level
                _last_level = new_level

        except Exception as e:
            logger.debug(f"[ResourceGuard] error: {e}")


def start():
    """Start the resource guard. Safe to call multiple times — only starts once."""
    global _started
    with _started_lock:
        if _started:
            return
        _started = True

    t = threading.Thread(target=_monitor, name="resource-guard", daemon=True)
    t.start()
    logger.info(
        f"[ResourceGuard] Started — "
        f"warn>{_CPU_WARN_PCT}% high>{_CPU_HIGH_PCT}% crit>{_CPU_CRIT_PCT}%"
    )
