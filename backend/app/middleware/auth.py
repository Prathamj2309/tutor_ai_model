import asyncio
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool
from app.core.supabase_client import supabase
from app.models.schemas import UserInfo

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> UserInfo:
    token = credentials.credentials
    
    # Retry logic for network/SSL glitches
    max_retries = 3
    retry_delay = 1
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # supabase.auth.get_user is synchronous, run in threadpool
            response = await run_in_threadpool(supabase.auth.get_user, token)
            
            user_id = None
            user_email = None
            
            # Robustly handle different response structures
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
                print(f"Auth error: Unexpected response structure on attempt {attempt+1}: {response}")
                continue # Try again or fail later

            # Auto-create profile for legacy users
            try:
                # Use threadpool for DB sync calls too
                def check_and_create_profile():
                    prof = supabase.table('profiles').select('id').eq('id', user_id).execute()
                    if not prof.data:
                        supabase.table('profiles').insert({
                            'id': user_id, 
                            'email': user_email, 
                            'full_name': user_email.split('@')[0] if user_email else "User"
                        }).execute()
                
                await run_in_threadpool(check_and_create_profile)
            except Exception as e:
                print(f"Auto-profile warning: {e}")

            return UserInfo(id=user_id, email=user_email)

        except Exception as e:
            last_error = str(e)
            print(f"Auth Attempt {attempt + 1} failed: {last_error}")
            if "handshake operation timed out" in last_error or "timeout" in last_error.lower():
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            else:
                break # Non-timeout errors don't benefit much from retries

    print(f"Critical Auth Error after {max_retries} attempts: {last_error}")
    raise HTTPException(status_code=401, detail=f"Authentication failed: {last_error}")
