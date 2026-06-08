import cv2
import numpy as np
import os
import base64
import logging
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from core.auth import require_auth
from core import pipeline
from core.pipeline import stream_bytes_to_local, get_recognizer
from core.state import templates, DATASET_DIR
from streaming_service import client as camera_client

logger = logging.getLogger(__name__)

router = APIRouter()

_db_manager = None
_recognizer = None


def init_routes(db, rec):
    global _db_manager, _recognizer
    _db_manager = db
    _recognizer = rec

@router.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "people.html", {})

@router.get("/api/persons")
async def api_persons():
    return _db_manager.get_persons_with_last_seen()

@router.post("/register")
@router.post("/api/register_person")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    encoding = None

    rec = get_recognizer()
    if rec is not None:
        # Local recognizer available (local dev mode)
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "error", "message": "Invalid image"}
        encoding = rec.get_encoding(image)
    else:
        # Docker mode: delegate encoding to camera_server which owns the models
        result = await camera_client.get_face_encoding(content, file.filename or "image.jpg")
        if result.get("status") != "success":
            msg = result.get("message", "No face detected")
            return {"status": "error", "message": msg}
        encoding = np.frombuffer(
            base64.b64decode(result["encoding"]), dtype=np.float32
        )

    if encoding is not None:
        l_path = f"{DATASET_DIR}/{name}/{file.filename}"
        def _on_c(ok):
            if ok:
                _db_manager.register_person(name, l_path, encoding.tobytes())
                # Reload faces in camera_server so it picks up the new person immediately
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(camera_client.reload_faces())
                    else:
                        loop.run_until_complete(camera_client.reload_faces())
                except Exception:
                    pass
                if rec:
                    rec.load_known_faces(_db_manager)
            else:
                logger.error(f"[People] Failed to save image for '{name}' at {l_path}")
        if stream_bytes_to_local(content, l_path, callback=_on_c):
            return {"status": "success"}
    return {"status": "error", "message": "No face detected"}

@router.delete("/api/delete_person/{person_id}")
async def delete_person(person_id: int):
    persons = _db_manager.get_registered_persons()
    person = next((p for p in persons if str(p[0]) == str(person_id)), None)
    if person:
        if person[2] and os.path.exists(person[2]):
            try:
                os.remove(person[2])
            except Exception as e:
                logger.warning(f"[People] Could not delete image file {person[2]}: {e}")
        rec = get_recognizer()
        _db_manager.delete_person_from_db(person_id)
        if rec:
            rec.load_known_faces(_db_manager)
        # Reload faces in camera_server
        try:
            await camera_client.reload_faces()
        except Exception:
            pass
        return {"status": "success"}
    return {"status": "error", "message": "Not found"}

@router.put("/api/edit_person/{person_id}")
async def edit_person(person_id: int, name: str = Form(...), file: UploadFile = File(None)):
    n_path = None
    n_enc = None
    rec = get_recognizer()

    if file and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            if rec is not None:
                n_enc = rec.get_encoding(image)
            else:
                # Docker mode: delegate to camera_server
                result = await camera_client.get_face_encoding(content, file.filename)
                if result.get("status") == "success":
                    n_enc = np.frombuffer(
                        base64.b64decode(result["encoding"]), dtype=np.float32
                    )
            if n_enc is not None:
                n_path = f"{DATASET_DIR}/{name}/{file.filename}"
                stream_bytes_to_local(content, n_path)

    _db_manager.rename_person(person_id, name, n_path, n_enc)

    # Reload faces everywhere
    if rec:
        rec.load_known_faces(_db_manager)
    try:
        await camera_client.reload_faces()
    except Exception:
        pass

    return {"status": "success"}
