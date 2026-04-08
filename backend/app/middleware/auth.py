import jwt
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.supabase_client import supabase
from app.models.schemas import UserInfo

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> UserInfo:
    token = credentials.credentials
    try:
        # We verify the token with Supabase
        response = supabase.auth.get_user(token)
        if hasattr(response, "user") and response.user:
            return UserInfo(id=response.user.id, email=response.user.email)
        elif hasattr(response, "data") and response.data.user:
             return UserInfo(id=response.data.user.id, email=response.data.user.email)
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
