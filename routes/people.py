import cv2
import numpy as np
import os
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from core.auth import require_auth
from core.state import templates, DATASET_DIR
from core.pipeline import stream_bytes_to_local

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
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: return {"status": "error", "message": "Invalid image"}
    encoding = _recognizer.get_encoding(image)
    if encoding is not None:
        l_path = f"{DATASET_DIR}/{name}/{file.filename}"
        def _on_c(ok):
            if ok: 
                _db_manager.register_person(name, l_path, encoding.tobytes())
                _recognizer.load_known_faces(_db_manager)
        if stream_bytes_to_local(content, l_path, callback=_on_c):
            return {"status": "success"}
    return {"status": "error", "message": "No face detected"}

@router.delete("/api/delete_person/{person_id}")
async def delete_person(person_id: int):
    persons = _db_manager.get_registered_persons()
    person = next((p for p in persons if str(p[0]) == str(person_id)), None)
    if person:
        if person[2]:
            try:
                import shutil
                d = os.path.dirname(person[2])
                if d and os.path.exists(d): shutil.rmtree(d)
            except: pass
        _db_manager.delete_person_from_db(person_id)
        _recognizer.load_known_faces(_db_manager)
        return {"status": "success"}
    return {"status": "error", "message": "Not found"}

@router.put("/api/edit_person/{person_id}")
async def edit_person(person_id: int, name: str = Form(...), file: UploadFile = File(None)):
    n_path = None; n_enc = None
    if file and file.filename:
        content = await file.read(); nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            n_enc = _recognizer.get_encoding(image)
            if n_enc is not None:
                n_path = f"{DATASET_DIR}/{name}/{file.filename}"
                stream_bytes_to_local(content, n_path)
    _db_manager.rename_person(person_id, name, n_path, n_enc)
    _recognizer.load_known_faces(_db_manager)
    return {"status": "success"}
