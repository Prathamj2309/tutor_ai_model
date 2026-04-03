from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from app.models.schemas import UserInfo, ProfileUpdate
from app.middleware.auth import get_current_user
from app.core.supabase_client import supabase
from app.services.weakness_service import get_weakness_report
from datetime import datetime

router = APIRouter(tags=["profile"])

@router.get("/profile")
async def get_profile(user: UserInfo = Depends(get_current_user)):
    res = supabase.table('profiles').select('*').eq('id', user.id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    return res.data[0]

@router.patch("/profile")
async def update_profile(updates: ProfileUpdate, user: UserInfo = Depends(get_current_user)):
    update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
    update_data['updated_at'] = datetime.utcnow().isoformat()
    
    res = supabase.table('profiles').update(update_data).eq('id', user.id).execute()
    return res.data[0] if res.data else {}

@router.get("/weakness-report")
async def weakness_report(subject: Optional[str] = None, user: UserInfo = Depends(get_current_user)):
    report = get_weakness_report(user.id, subject)
    return report
