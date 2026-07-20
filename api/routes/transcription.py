# api/routes/transcription.py

# Builds self-documenting parameters and integrates version endpoints under clean API specifications.

from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Optional
from api.schemas.response import HealthResponse, ModelInfoResponse, TranscriptionResponse
from api.dependencies import get_shared_transcriber_engine
from api.services.transcription_service import TranscriptionService
from api.config import settings

# Instantiate router with dedicated semantic tags for self-documenting UI generation
router = APIRouter(prefix="/transcription", tags=["Pashto Transcription Operations Engine"])

@router.get("/health", response_model=HealthResponse, summary="Retrieve API Operational Status")
async def get_health():
    """Returns a simple health check diagnostic response model confirmation."""
    return {"status": "healthy"}

@router.get("/model-info", response_model=ModelInfoResponse, summary="Retrieve Embedded Model Track Metrics")
async def get_model_info():
    """Exposes structured model metadata detailing current version and baseline Pashto dialect WER scoring."""
    return {
        "model": "STT-Whisper-Pashto",
        "version": "2.1",
        "language": "Pashto",
        "framework": "Transformers",
        "wer": 43.83
    }

@router.post("/transcribe", response_model=TranscriptionResponse, summary="Transcribe Audio File to Pashto Script")
async def post_transcribe(
    file: UploadFile = File(..., description="Target dialect audio stream track asset container payload."),
    reference_text: Optional[str] = Form(None, description="Optional baseline ground truth reference text script for computing WER metrics."),
    engine = Depends(get_shared_transcriber_engine)
):
    """
    Ingests multi-part audio streams, validates format criteria dynamically, 
    stages tracking components safely, and returns transcribed Pashto text 
    along with computation timestamps.
    """
    service = TranscriptionService(engine=engine)
    
    # Unpack all 4 metrics returned by the transcription service
    text_output, wer_score, processing_time, inference_time = service.generate_speech_to_text(file, reference_text)
    
    return {
        "filename": file.filename,
        "transcription": text_output,
        "wer_score": wer_score,
        "processing_time_sec": processing_time,
        "inference_time_sec": inference_time,
        "model_version": getattr(settings, "MERGED_MODEL_PATH", "Pashto-Whisper-v2.1-LoRA")
    }