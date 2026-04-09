from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, quiz, profile, ocr
from app.core.config import settings
from contextlib import asynccontextmanager
from app.services.llm_service import load_models
import traceback
from fastapi import Request
from starlette.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading MoE models into VRAM...")
    try:
        load_models()
    except Exception as e:
        print(f"FAILED TO LOAD MODELS: {e}")
        traceback.print_exc()
    yield

app = FastAPI(
    title="TutorAI API",
    lifespan=lifespan
)

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"GLOBAL ERROR CAUGHT: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

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
app.include_router(ocr.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=False)
