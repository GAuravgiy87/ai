import secrets
from fastapi import Request, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jose import jwt
import datetime
import os

security = HTTPBasic(auto_error=False)

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "deiadmin@789"
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "ai_vigilance_super_secret_key_default")
ALGORITHM = "HS256"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials."""
    if credentials:
        is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if is_correct_username and is_correct_password:
            return credentials.username
    return None

def create_session_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def require_auth(request: Request):
    """Check if user is authenticated via session cookie."""
    session_token = request.cookies.get("session")
    if not session_token:
        return False
    try:
        payload = jwt.decode(session_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") == ADMIN_USERNAME:
            return True
    except Exception:
        pass
    return False
