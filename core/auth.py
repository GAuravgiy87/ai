"""core/auth.py — Credentials, session management."""
import secrets
from fastapi import Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "deiadmin@789"

security = HTTPBasic(auto_error=False)
authenticated_sessions: set = set()


def verify_credentials(credentials: HTTPBasicCredentials) -> str | None:
    if credentials:
        ok_u = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        ok_p = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if ok_u and ok_p:
            return credentials.username
    return None


def require_auth(request: Request) -> bool:
    token = request.cookies.get("session")
    return bool(token and token in authenticated_sessions)
