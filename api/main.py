from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.config import settings
from api.routes.transcription import router as transcription_router
from api.exceptions import register_exception_handlers
from api.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("==========================================================")
    logger.info(f" Starting System Initialization Suite for: {settings.APP_NAME}")
    logger.info(f" API Routing Space Context URL: {settings.API_V1_STR}")
    logger.info("==========================================================")
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-ready versioned REST API engine for regional Peshawari dialect Automatic Speech Recognition.",
    version="1.3",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

register_exception_handlers(app)

@app.get("/health", tags=["System Status"])
def health_check():
    return {"status": "healthy", "service": "Pashto ASR Core"}

app.include_router(transcription_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)