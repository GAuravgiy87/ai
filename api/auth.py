from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from core.auth import ADMIN_USERNAME, ADMIN_PASSWORD, create_session_token
from core.state import templates

router = APIRouter()

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})

@router.post("/api/login")
async def api_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session_token = create_session_token(username)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session", value=session_token, httponly=True)
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/logout")
async def logout(request: Request):
    """Logout and clear session."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session")
    return response
