from typing import Optional
from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role: str   # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []
    subject: str = "general"


class ChatResponse(BaseModel):
    answer: str
    topic_tags: list[str] = []


class OCRResponse(BaseModel):
    extracted_text: str


class QuizGenerateRequest(BaseModel):
    weak_topics: list[str]
    subject: str = "physics"


class QuizQuestion(BaseModel):
    id: int
    question: str
    options: list[str]
    correct_answer: str   # 'A' | 'B' | 'C' | 'D'
    explanation: str


class QuizGenerateResponse(BaseModel):
    questions: list[QuizQuestion]
