from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional
from app.models.schemas import UserInfo, ChatResponse
from app.middleware.auth import get_current_user
from app.core.supabase_client import supabase
from app.services.history_service import get_conversation_history, save_user_message, save_assistant_message
from app.services.llm_service import generate_answer
from app.services.ocr_service import extract_text_from_image
import uuid
from datetime import datetime

router = APIRouter(tags=["chat"])

@router.get("/conversations")
async def get_conversations(user: UserInfo = Depends(get_current_user)):
    res = supabase.table('conversations').select('*').eq('user_id', user.id).order('updated_at', desc=True).execute()
    return res.data

@router.post("/conversations")
async def create_conversation(
    subject: str = Form("general"),
    title: Optional[str] = Form(None),
    user: UserInfo = Depends(get_current_user)
):
    res = supabase.table('conversations').insert({
        'user_id': user.id,
        'subject': subject,
        'title': title or f"New {subject} chat"
    }).execute()
    return res.data[0] if res.data else {}

@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, user: UserInfo = Depends(get_current_user)):
    conv = supabase.table('conversations').select('id').eq('id', conversation_id).eq('user_id', user.id).execute()
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    res = supabase.table('messages').select('*').eq('conversation_id', conversation_id).order('created_at', desc=False).execute()
    return res.data

@router.post("/chat", response_model=ChatResponse)
async def chat(
    conversationId: str = Form(...),
    content: Optional[str] = Form(""),
    image: Optional[UploadFile] = File(None),
    user: UserInfo = Depends(get_current_user)
):
    if not content.strip() and not image:
        raise HTTPException(status_code=400, detail="Content or image required")

    conv = supabase.table('conversations').select('id, subject').eq('id', conversationId).eq('user_id', user.id).execute()
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    subject = conv.data[0].get("subject", "general")
    image_url = None
    image_ocr_text = None

    if image:
        # Save to storage (we bypass for now if storage is not setup, but here's the code format)
        file_ext = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"{user.id}/{conversationId}/{uuid.uuid4()}.{file_ext}"
        image_bytes = await image.read()
        try:
            supabase.storage.from_('question-images').upload(filename, image_bytes, {'content-type': image.content_type})
            image_url = supabase.storage.from_('question-images').get_public_url(filename)
        except Exception as e:
            print(f"Image upload failed: {e}")
            
        try:
            image_ocr_text = extract_text_from_image(image_bytes)
        except Exception:
            pass

    user_msg_data = save_user_message(
        conversation_id=conversationId,
        user_id=user.id,
        content=content.strip(),
        image_url=image_url,
        image_ocr_text=image_ocr_text
    )

    history = get_conversation_history(conversationId, limit=5)
    
    ai_question = content.strip()
    if image_ocr_text:
        ai_question = f"{ai_question}\n[From image]: {image_ocr_text}".strip()

    ai_result = generate_answer(question=ai_question, history=history, subject=subject)
    
    ai_msg_data = save_assistant_message(
        conversation_id=conversationId,
        user_id=user.id,
        content=ai_result["answer"],
        topic_tags=ai_result.get("topic_tags", [])
    )

    return ChatResponse(
        userMessage=user_msg_data,
        aiMessage=ai_msg_data
    )
