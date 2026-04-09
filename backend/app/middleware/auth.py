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
        
        user_id = None
        user_email = None
        
        # Robustly handle different response structures (GotrueResponse, UserResponse, dict)
        if hasattr(response, "user") and response.user:
            user_id = response.user.id
            user_email = response.user.email
        elif hasattr(response, "data") and hasattr(response.data, "user") and response.data.user:
            user_id = response.data.user.id
            user_email = response.data.user.email
        elif isinstance(response, dict) and "user" in response:
            user_id = response["user"]["id"]
            user_email = response["user"]["email"]
        else:
            print(f"Auth error: Unexpected response structure: {response}")
            raise HTTPException(status_code=401, detail="Invalid token structure")

        # Auto-create profile for legacy users
        try:
            # Check if profile exists
            prof = supabase.table('profiles').select('id').eq('id', user_id).execute()
            if not prof.data:
                supabase.table('profiles').insert({
                    'id': user_id, 
                    'email': user_email, 
                    'full_name': user_email.split('@')[0] if user_email else "User"
                }).execute()
        except Exception as e:
            print(f"Auto-profile warning: {e}")

        return UserInfo(id=user_id, email=user_email)

    except Exception as e:
        print(f"Critical Auth Error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
