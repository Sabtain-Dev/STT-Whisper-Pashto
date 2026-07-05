# api/routes.py

# This file defines the FastAPI routing layer, exposing endpoints for health checks, model metadata retrieval, and transcription processing. It leverages the TranscriptionService class to handle the core business logic, ensuring a clean separation of concerns between request handling and service execution.
# Now that our core business logic lives cleanly inside services.py, we can update api/routes.py to keep the endpoints light, clean, and easily maintainable
# api/routes.py (The Traffic Controller): This acts as a mailroom. It defines the paths/URLs (/health, /model-info, /transcribe). It accepts the incoming request parameters, makes basic initial checks, and passes them off to the service layer. It does not know how the transcription happens; it only passes data along.

import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from api.schemas import HealthResponse, ModelInfoResponse, TranscriptionResponse
from api.services import TranscriptionService

router = APIRouter()

# ── Service Instantiation Variables ───────────────────────────────────────────
MERGED_MODEL_PATH = "Sabtain-Dev/STT-Whisper-Pashto"
SUPPORTED_FORMATS = {"wav", "mp3", "mp4", "m4a", "flac", "ogg", "opus", "webm", "aac", "wma"}
TEMP_API_DIR = "./workspace_data/api_temp"

os.makedirs(TEMP_API_DIR, exist_ok=True)

# Instantiate the centralized business service worker single source instance
transcription_service = TranscriptionService(
    model_path=MERGED_MODEL_PATH,
    temp_dir=TEMP_API_DIR,
    supported_formats=SUPPORTED_FORMATS
)

# ── 1. Health Check Endpoint ──────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def get_health():
    """
    Returns the server system status verification layer code.
    """
    return {"status": "healthy"}

# ── 2. Model Information Endpoint ─────────────────────────────────────────────
@router.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
async def get_model_info():
    """
    Exposes architecture tracking metadata metrics detailing performance evaluations.
    """
    return {
        "model": "STT-Whisper-Pashto",
        "version": "1.3",
        "language": "Pashto",
        "framework": "Transformers",
        "wer": 43.83
    }

# ── 3. Core Transcription Processing Endpoint ──────────────────────────────────
@router.post("/transcribe", response_model=TranscriptionResponse, tags=["Inference"])
async def post_transcribe(
    file: UploadFile = File(...),
    reference_text: Optional[str] = Form(None)
):
    """
    Primary endpoint ingest payload loop. Validates incoming multipart streams, 
    stages tracking segments inside a physical data layer space, and calls the 
    underlying Whisper models.
    """
    # Simply hand over parameters to the clean service abstraction layer
    text_output, calculated_wer = transcription_service.process_transcription(
        file=file, 
        reference_text=reference_text
    )
    
    return {
        "filename": file.filename,
        "transcription": text_output,
        "wer_score": calculated_wer
    }