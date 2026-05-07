from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.security import decode_token
from app.core.dependencies import rate_limiter
security = HTTPBearer()

def get_token_payload(creds: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(creds.credentials)
    if not payload:
        raise HTTPException(401, "Invalid token")
    return payload

def get_current_user(payload = Depends(get_token_payload)):
    return payload["sub"]

def get_current_user_payload(payload = Depends(get_token_payload)):
    return payload

def require_admin(user = Depends(get_current_user_payload)):
    if user["role"] != "admin":
        raise HTTPException(403, "Not authorized")
    return user
    
def rate_limiter_dep(user_id: str = Depends(get_current_user)):
    allowed, retry_after = rate_limiter.allow(user_id)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": str(retry_after)}
        )