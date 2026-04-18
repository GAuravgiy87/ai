import secrets
from fastapi import Request, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic(auto_error=False)

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "deiadmin@789"

# Session storage
authenticated_sessions: set = set()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials."""
    if credentials:
        is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if is_correct_username and is_correct_password:
            return credentials.username
    return None

def require_auth(request: Request):
    """Check if user is authenticated via session cookie."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        return True
    return False
