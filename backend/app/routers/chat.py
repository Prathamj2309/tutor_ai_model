from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional, List
from starlette.concurrency import run_in_threadpool
from app.models.schemas import UserInfo, ChatResponse
from app.middleware.auth import get_current_user
from app.core.supabase_client import supabase
from app.services.history_service import get_conversation_history, save_user_message, save_assistant_message
from app.services.llm_service import generate_answer
from app.services.ocr_service import extract_text_from_image

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
    # Verify owner
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
        raise HTTPException(status_code=400, detail="Content required")

    # Verify conversation
    conv = supabase.table('conversations').select('subject').eq('id', conversationId).eq('user_id', user.id).execute()
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    subject = conv.data[0].get("subject", "general")
    history = get_conversation_history(conversationId, limit=5)
    
    # Process AI in threadpool to avoid blocking
    ai_result = await run_in_threadpool(generate_answer, question=content, history=history, subject=subject)
    
    user_msg = save_user_message(conversation_id=conversationId, user_id=user.id, content=content)
    ai_msg = save_assistant_message(conversation_id=conversationId, user_id=user.id, 
                                    content=ai_result["answer"], topic_tags=ai_result.get("topic_tags", []))
    
    return ChatResponse(userMessage=user_msg, aiMessage=ai_msg)

from fastapi.responses import StreamingResponse
import json
from app.services.llm_service import generate_answer_stream, format_question_latex, generate_answer

@router.post("/chat/stream")
async def chat_stream(
    conversationId: str = Form(...),
    content: Optional[str] = Form(""),
    image: Optional[UploadFile] = File(None),
    user: UserInfo = Depends(get_current_user)
):
    if not content.strip() and not image:
        raise HTTPException(status_code=400, detail="Content required")

    # Verify conversation
    conv = supabase.table('conversations').select('subject').eq('id', conversationId).eq('user_id', user.id).execute()
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    subject = conv.data[0].get("subject", "general")
    history = get_conversation_history(conversationId, limit=5)
    
    formatted_content = await run_in_threadpool(format_question_latex, question=content)
    
    save_user_message(conversation_id=conversationId, user_id=user.id, content=formatted_content)
    
    async def event_generator():
        try:
            # Let the user know the local model is working
            yield f"data: {json.dumps({'content': '*(Generating answer natively... please wait)*\\n\\n'})}\n\n"
            
            # Fetch raw answer from local huggingface model (in threadpool to avoid blocking event loop)
            ai_result = await run_in_threadpool(generate_answer, question=formatted_content, history=history, subject=subject)
            raw_answer = ai_result.get("answer", "")
            topic_tags = ai_result.get("topic_tags", [])

            # Clear the loading message by formatting it as part of the structure (or just let Gemini format stream append)
            # We'll just stream the formatted version below it
            yield f"data: {json.dumps({'content': '---\\n\\n'})}\n\n"
            
            full_response = "*(Generating answer natively... please wait)*\n\n---\n\n"
            
            for chunk in generate_answer_stream(formatted_content, history, subject, raw_answer):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
                
            save_assistant_message(
                conversation_id=conversationId, 
                user_id=user.id, 
                content=full_response, 
                topic_tags=topic_tags
            )
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print("Stream error:", str(e))
            yield f"data: {json.dumps({'content': f'\\n\\nError: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

