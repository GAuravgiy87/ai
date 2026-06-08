from fastapi import APIRouter, HTTPException, Request
import streaming_service.state as state

router = APIRouter()

@router.post("/reload_faces")
def reload_faces():
    """Reload known face encodings from the database into the recognizer."""
    if state.recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not available.")
    try:
        state.recognizer.load_known_faces(state.db_manager)
        return {"status": "success", "loaded": len(state.recognizer.known_face_names)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get_encoding")
async def get_face_encoding(request: Request):
    """
    Extract a face encoding from an uploaded image.
    Accepts multipart/form-data with a 'file' field.
    Returns the encoding as a base64-encoded float32 array.
    """
    import base64
    import numpy as np
    import cv2
    
    if state.recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not available.")
    form = await request.form()
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided.")
    content = await file.read()
    
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image.")
    encoding = state.recognizer.get_encoding(image)
    if encoding is None:
        raise HTTPException(status_code=422, detail="No face detected in image.")
    return {
        "status": "success",
        "encoding": base64.b64encode(encoding.astype(np.float32).tobytes()).decode(),
    }
