import time
import asyncio
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from core.state import camera_results, results_lock

router = APIRouter()

async def _gen_frames(camera_id: str):
    INTERVAL = 1.0 / 4
    next_t   = time.time()
    last_fb  = None
    while True:
        wait = next_t - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        next_t += INTERVAL
        if next_t < time.time() - (3 * INTERVAL):
            next_t = time.time() + INTERVAL

        with results_lock:
            fb = camera_results.get(camera_id, {}).get("encoded_frame")
        fb = fb if fb is not None else last_fb
        if fb is None:
            continue
        last_fb = fb
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(fb)).encode() + b"\r\n\r\n"
            + fb + b"\r\n"
        )

@router.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(
        _gen_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@router.get("/capture/{camera_id}")
def capture_frame(camera_id: str):
    with results_lock:
        fb = camera_results.get(camera_id, {}).get("encoded_frame")
    if fb is None:
        raise HTTPException(status_code=404, detail="No frame available.")
    return Response(content=fb, media_type="image/jpeg")
