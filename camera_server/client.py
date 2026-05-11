"""
camera_server/client.py — Async HTTP client for the Camera Server (port 9001).

Optimized to prevent blocking the main FastAPI event loop.
"""

import logging
import httpx
import asyncio
from typing import Optional, Any, Dict, List

logger  = logging.getLogger("camera_client")
BASE    = "http://127.0.0.1:9001"
TIMEOUT = 5.0   # seconds per request

async def _get_async(path: str, params: dict = None) -> Any:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{BASE}{path}", params=params)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"[CameraClient] GET {path} — {e}")
        return None

async def _post_async(path: str, json: dict = None) -> Any:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(f"{BASE}{path}", json=json)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"[CameraClient] POST {path} — {e}")
        return None

async def _delete_async(path: str) -> Any:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.delete(f"{BASE}{path}")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"[CameraClient] DELETE {path} — {e}")
        return None

# ── Public API (Now Async) ───────────────────────────────────────────────────

async def is_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{BASE}/health")
            return r.status_code == 200
    except Exception:
        return False

async def list_cameras() -> List[Dict]:
    return await _get_async("/cameras") or []

async def add_camera(camera_id: str, source: str, camera_type: str = "rtsp") -> Dict:
    result = await _post_async("/cameras", {
        "camera_id":   camera_id,
        "source":      source,
        "camera_type": camera_type,
    })
    if result is None:
        return {"status": "error", "message": "Camera server unreachable."}
    return result

async def remove_camera(camera_id: str) -> Dict:
    result = await _delete_async(f"/cameras/{camera_id}")
    return result or {"status": "error", "message": "Camera server unreachable."}

async def get_results(camera_id: str) -> Optional[Dict]:
    return await _get_async(f"/results/{camera_id}")

async def get_occupancy(camera_id: str = None) -> Dict:
    params = {"camera_id": camera_id} if camera_id else None
    return await _get_async("/occupancy", params=params) or {}

async def get_daily_stats() -> Dict:
    return await _get_async("/daily_stats") or {}

async def get_camera_settings(camera_id: str) -> Dict:
    return await _get_async(f"/settings/{camera_id}") or {}

async def set_camera_settings(camera_id: str, enabled: bool) -> Dict:
    result = await _post_async(f"/settings/{camera_id}", {"enabled": enabled})
    return result or {"status": "error", "message": "Camera server unreachable."}

def video_feed_url(camera_id: str) -> str:
    return f"{BASE}/video_feed/{camera_id}"

def capture_url(camera_id: str) -> str:
    return f"{BASE}/capture/{camera_id}"
