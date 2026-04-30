"""
camera_server/client.py — HTTP client for the Camera Server (port 9001).

Used by the main app routes to proxy all camera operations.
"""

import logging
import requests
from typing import Optional, Any, Dict, List

logger  = logging.getLogger("camera_client")
BASE    = "http://127.0.0.1:9001"
TIMEOUT = 5   # seconds per request


def _get(path: str, params: dict = None) -> Any:
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[CameraClient] GET {path} — {e}")
        return None


def _post(path: str, json: dict = None) -> Any:
    try:
        r = requests.post(f"{BASE}{path}", json=json, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[CameraClient] POST {path} — {e}")
        return None


def _delete(path: str) -> Any:
    try:
        r = requests.delete(f"{BASE}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[CameraClient] DELETE {path} — {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def is_alive() -> bool:
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_cameras() -> List[Dict]:
    return _get("/cameras") or []


def add_camera(camera_id: str, source: str, camera_type: str = "rtsp") -> Dict:
    result = _post("/cameras", {
        "camera_id":   camera_id,
        "source":      source,
        "camera_type": camera_type,
    })
    if result is None:
        return {"status": "error", "message": "Camera server unreachable."}
    return result


def remove_camera(camera_id: str) -> Dict:
    result = _delete(f"/cameras/{camera_id}")
    return result or {"status": "error", "message": "Camera server unreachable."}


def get_results(camera_id: str) -> Optional[Dict]:
    return _get(f"/results/{camera_id}")


def get_occupancy(camera_id: str = None) -> Dict:
    params = {"camera_id": camera_id} if camera_id else None
    return _get("/occupancy", params=params) or {}


def get_daily_stats() -> Dict:
    return _get("/daily_stats") or {}


def get_camera_settings(camera_id: str) -> Dict:
    return _get(f"/settings/{camera_id}") or {}


def set_camera_settings(camera_id: str, enabled: bool) -> Dict:
    result = _post(f"/settings/{camera_id}", {"enabled": enabled})
    return result or {"status": "error", "message": "Camera server unreachable."}


def video_feed_url(camera_id: str) -> str:
    return f"{BASE}/video_feed/{camera_id}"


def capture_url(camera_id: str) -> str:
    return f"{BASE}/capture/{camera_id}"
