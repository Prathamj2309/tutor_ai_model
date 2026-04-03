from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import OCRResponse
from app.services.ocr_service import extract_text_from_image

router = APIRouter(prefix="/ai", tags=["ocr"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@router.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file.
    Returns the extracted text content (math + text).
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    extracted_text = extract_text_from_image(image_bytes)

    return OCRResponse(extracted_text=extracted_text)
