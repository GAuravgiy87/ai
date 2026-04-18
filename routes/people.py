"""routes/people.py — Register, edit, delete persons."""
import os
import shutil

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.auth import require_auth
from core.state import DATASET_DIR, stream_bytes_to_local

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

_db_manager  = None
_recognizer  = None   # lambda: recognizer


def init(db_manager, get_recognizer):
    global _db_manager, _recognizer
    _db_manager = db_manager
    _recognizer = get_recognizer


@router.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    if not require_auth(request):
        return templates.TemplateResponse(request, "login.html", {})
    return templates.TemplateResponse(request, "people.html", {})


@router.post("/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    recognizer = _recognizer() if _recognizer else None
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not ready")

    img_bytes = await file.read()
    nparr     = np.frombuffer(img_bytes, np.uint8)
    image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    encoding = recognizer.get_encoding(image)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected")

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    local_path = os.path.join(person_dir, file.filename or f"{name}.jpg")

    def on_save(success):
        if success:
            _db_manager.register_person(name, local_path, encoding.tobytes())
            recognizer.load_known_faces(_db_manager)

    stream_bytes_to_local(img_bytes, local_path, callback=on_save)
    return {"status": "success", "name": name}


@router.delete("/api/persons/{person_id}")
async def delete_person(person_id: str):
    persons = _db_manager.get_registered_persons()
    person  = next((p for p in persons if str(p[0]) == str(person_id)), None)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        image_path = person[2]
        if image_path:
            d = os.path.dirname(image_path)
            if d and os.path.exists(d):
                shutil.rmtree(d)
    except Exception:
        pass
    try:
        _db_manager.delete_person_from_db(person_id)
    except Exception:
        pass
    recognizer = _recognizer() if _recognizer else None
    if recognizer:
        recognizer.load_known_faces(_db_manager)
    return {"status": "success"}


@router.post("/api/persons/{person_id}/edit")
async def edit_person(person_id: str,
                      new_name: str = Form(...),
                      file: UploadFile = File(None)):
    recognizer = _recognizer() if _recognizer else None
    persons    = _db_manager.get_registered_persons()
    person     = next((p for p in persons if str(p[0]) == str(person_id)), None)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    new_image_path = None
    new_encoding   = None

    if file and file.filename:
        img_bytes = await file.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None and recognizer:
            new_encoding = recognizer.get_encoding(image)
            if new_encoding is not None:
                person_dir     = os.path.join(DATASET_DIR, new_name)
                os.makedirs(person_dir, exist_ok=True)
                new_image_path = os.path.join(person_dir, file.filename)
                with open(new_image_path, "wb") as f:
                    f.write(img_bytes)

    _db_manager.rename_person(person_id, new_name, new_image_path, new_encoding)
    if recognizer:
        recognizer.load_known_faces(_db_manager)
    return {"status": "success"}
