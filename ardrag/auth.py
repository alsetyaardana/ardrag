from fastapi import HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ardrag import db
from ardrag.config import SESSION_COOKIE_NAME, SESSION_MAX_AGE_SECONDS, SESSION_SECRET

_serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="ardrag-session")

# Permission matrix: each key is a permission string, value is the set of roles that have it.
# superadmin has all permissions implicitly (checked first in require_permission).
WRITE_ROUTES = {"upload", "delete", "reembed", "settings", "classify", "reindex", "chat", "manage_users"}


def verify_credentials(username: str, password: str) -> dict | None:
    """Verify against DB. Returns user dict {user_id, username, role} or None."""
    user = db.app_user_verify(username, password)
    if not user:
        return None
    return {"user_id": user.id, "username": user.username, "role": user.role}


def create_session_token(user_info: dict) -> str:
    return _serializer.dumps(user_info)


def get_current_user(request: Request) -> dict | None:
    """Returns {user_id, username, role} or None."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    # Backward compat: old tokens only had {"user": username}
    if "user_id" not in data and "user" in data:
        user = db.app_user_get_by_username(data["user"])
        if user:
            return {"user_id": user.id, "username": user.username, "role": user.role}
        return None
    return data if data.get("user_id") else None


def require_auth(request: Request) -> dict:
    """Dependency — 401 if no valid session. Returns {user_id, username, role}."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_role(*roles: str):
    """Dependency factory — 403 if current user's role is not in `roles`."""
    def dependency(user: dict = None) -> dict:
        # user is injected via Depends(require_auth) in the route
        return user
    # We return a dependency that checks role; the actual check happens in the wrapper below
    def checker(request: Request) -> dict:
        user = require_auth(request)
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker


def require_permission(permission: str):
    """Dependency factory — 403 if current user lacks the given permission.
    superadmin has all permissions. user role is denied WRITE_ROUTES."""
    def checker(request: Request) -> dict:
        user = require_auth(request)
        if user["role"] == "superadmin":
            return user
        if permission in WRITE_ROUTES:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker
