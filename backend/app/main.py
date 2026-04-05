from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, quiz, profile
from app.core.config import settings

app = FastAPI(
    title="TutorAI API",
    description="Unified backend handling Auth, User Data, and AI inference.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(profile.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "unified-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
