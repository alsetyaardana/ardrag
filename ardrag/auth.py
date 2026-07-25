import secrets

from fastapi import HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ardrag.config import ADMIN_PASSWORD, ADMIN_USER, SESSION_COOKIE_NAME, SESSION_MAX_AGE_SECONDS, SESSION_SECRET

_serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="ardrag-session")


def verify_credentials(username: str, password: str) -> bool:
    user_ok = secrets.compare_digest(username, ADMIN_USER)
    pass_ok = secrets.compare_digest(password, ADMIN_PASSWORD)
    return user_ok and pass_ok


def create_session_token(username: str) -> str:
    return _serializer.dumps({"user": username})


def get_current_user(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("user")


def require_auth(request: Request) -> str:
    """Dependency for JSON API endpoints — 401 if no valid session cookie."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
