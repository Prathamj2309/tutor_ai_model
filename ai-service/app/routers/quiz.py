from fastapi import APIRouter
from app.models.schemas import QuizGenerateRequest, QuizGenerateResponse, QuizQuestion
from app.services.llm_service import generate_quiz

router = APIRouter(prefix="/ai", tags=["quiz"])


@router.post("/quiz/generate", response_model=QuizGenerateResponse)
async def generate_quiz_endpoint(request: QuizGenerateRequest):
    """
    Accepts a list of weak topics and a subject.
    Returns 5 MCQ questions targeting those topics.
    """
    result = generate_quiz(
        weak_topics=request.weak_topics,
        subject=request.subject,
    )
    questions = [QuizQuestion(**q) for q in result.get("questions", [])]
    return QuizGenerateResponse(questions=questions)
