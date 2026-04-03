from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, ocr, quiz
from app.core.config import settings

app = FastAPI(
    title="TutorAI - AI Inference Service",
    description="Local GGUF-based inference service for IIT-JEE tutoring.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000"],  # Express backend only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router)
app.include_router(ocr.router)
app.include_router(quiz.router)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=False)
