"""routes/auth.py — Login / logout routes."""
import secrets
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.auth import (
    authenticated_sessions, require_auth, verify_credentials, security,
)
from fastapi import Depends
from fastapi.security import HTTPBasicCredentials

router     = APIRouter()
templates  = Jinja2Templates(directory="templates")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@router.post("/api/login")
async def api_login(request: Request,
                    username: str = Form(...),
                    password: str = Form(...)):
    from fastapi.security import HTTPBasicCredentials
    creds = HTTPBasicCredentials(username=username, password=password)
    user  = verify_credentials(creds)
    if user:
        token = secrets.token_hex(32)
        authenticated_sessions.add(token)
        resp = RedirectResponse(url="/", status_code=302)
        resp.set_cookie("session", token, httponly=True, samesite="lax")
        return resp
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        authenticated_sessions.discard(token)
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("session")
    return resp
