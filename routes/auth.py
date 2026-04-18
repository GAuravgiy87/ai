"""routes/auth.py — Login / logout routes."""
import secrets
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from core.state import templates
from core.auth import (
    authenticated_sessions, require_auth, verify_credentials, security,
)
from fastapi import Depends
from fastapi.security import HTTPBasicCredentials

router     = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@router.post("/api/login")
async def api_login(request: Request,
                    username: str = Form(...),
                    password: str = Form(...)):
    from fastapi.security import HTTPBasicCredentials
    from fastapi.responses import JSONResponse
    creds = HTTPBasicCredentials(username=username, password=password)
    user  = verify_credentials(creds)
    if user:
        token = secrets.token_hex(32)
        authenticated_sessions.add(token)
        resp = JSONResponse({"status": "ok"})
        resp.set_cookie("session", token, httponly=True, samesite="lax")
        return resp
    return JSONResponse({"status": "error", "detail": "Invalid credentials"}, status_code=401)


@router.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        authenticated_sessions.discard(token)
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("session")
    return resp
