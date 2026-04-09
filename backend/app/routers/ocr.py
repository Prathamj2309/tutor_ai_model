from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.middleware.auth import get_current_user
from app.models.schemas import UserInfo
from app.services.ocr_service import extract_text_from_image
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/ocr", tags=["ocr"])

@router.post("/extract")
async def extract_text(image: UploadFile = File(...), user: UserInfo = Depends(get_current_user)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await image.read()
        # Run OCR in threadpool to avoid blocking event loop
        text = await run_in_threadpool(extract_text_from_image, content)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
