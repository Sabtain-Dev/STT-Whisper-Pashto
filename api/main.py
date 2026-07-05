# api/main.py

# Binds global components together, mounting versioned routes onto an explicit path space (/api/v1) while attaching exception handling layers cleanly.

from fastapi import FastAPI
from api.config import settings
from api.routes.transcription import router as transcription_router
from api.exceptions import register_exception_handlers
from api.logger import logger

# Initialize production configuration metadata
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-ready versioned REST API engine for regional Peshawari dialect Automatic Speech Recognition.",
    version="1.3",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc"
)

# Register robust architectural validation error handlers
register_exception_handlers(app)

# Mount the decoupled versioned routing structures securely
app.include_router(transcription_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    logger.info(f"==========================================================")
    logger.info(f" Starting System Initialization Suite for: {settings.APP_NAME}")
    logger.info(f" API Routing Space Context URL: {settings.API_V1_STR}")
    logger.info(f"==========================================================")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)