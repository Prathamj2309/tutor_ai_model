from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import generate_answer

router = APIRouter(prefix="/ai", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accepts a question and conversation history.
    Returns an AI-generated answer and inferred topic tags.
    """
    result = generate_answer(
        question=request.question,
        history=request.history,
        subject=request.subject,
    )
    return ChatResponse(
        answer=result["answer"],
        topic_tags=result.get("topic_tags", []),
    )
