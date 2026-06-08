import asyncio
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import streaming_service.state as state
from core.state import camera_results, results_lock, sanitize_rtsp_url
from core.pipeline import process_camera

logger = logging.getLogger("camera_server")
router = APIRouter()

class AddCameraRequest(BaseModel):
    camera_id:   str
    source:      str
    camera_type: Optional[str] = "rtsp"

@router.get("")
@router.get("/")
def list_cameras():
    db_cams = {c[0]: c[1] for c in state.db_manager.get_cameras()}
    return [
        {"id": cam_id, "source": db_cams.get(cam_id, "unknown")}
        for cam_id in state.camera_manager.get_active_cameras()
    ]

@router.post("")
@router.post("/")
async def add_camera(req: AddCameraRequest):
    cam_id      = req.camera_id.strip()
    source      = req.source.strip()
    camera_type = (req.camera_type or "rtsp").strip()

    if camera_type == "webcam" and str(source).isdigit():
        parsed = int(source)
    elif camera_type == "rtsp":
        parsed = sanitize_rtsp_url(source)
    elif camera_type == "droidcam":
        parsed = f"http://{source}:4747/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "ipwebcam":
        parsed = f"http://{source}:8080/video" if ":" not in source else f"http://{source}/video"
    else:
        parsed = source

    loop = asyncio.get_event_loop()
    status, final_source = await loop.run_in_executor(
        None, state.camera_manager.add_camera, cam_id, parsed
    )

    if status == 0:
        state.db_manager.add_camera_to_db(cam_id, final_source)
        logger.info(f"[CameraServer] Added: {cam_id}")

        threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
        await asyncio.sleep(1)

        return {"status": "success", "camera_id": cam_id, "source": final_source}
    elif status == 1:
        raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' already exists.")
    else:
        raise HTTPException(status_code=502, detail=f"Cannot connect to camera at '{source}'.")

@router.delete("/{camera_id}")
def remove_camera(camera_id: str):
    state.camera_manager.remove_camera(camera_id)
    state.db_manager.remove_camera_from_db(camera_id)
    with results_lock:
        camera_results.pop(camera_id, None)
    logger.info(f"[CameraServer] Removed: {camera_id}")
    return {"status": "success"}
