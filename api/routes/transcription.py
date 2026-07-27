# api/routes/transcription.py

# Builds self-documenting parameters and integrates version endpoints under clean API specifications.

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.config import settings
from api.dependencies import get_shared_transcriber_engine
from api.schemas.response import (
    HealthResponse,
    ModelInfoResponse,
    TranscriptionResponse,
)
from api.services.transcription_service import TranscriptionService
from api.utils.validation import validate_audio_file

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
        "version": "2.7",
        "language": "Pashto",
        "framework": "Transformers",
        "wer": 43.83
    }

@router.post("/transcribe", response_model=TranscriptionResponse, summary="Transcribe Audio File to Pashto Script")
async def post_transcribe(
    file: UploadFile = File(..., description="Target dialect audio stream track asset container payload."),
    reference_text: str | None = Form(None, description="Optional baseline ground truth reference text script for computing WER metrics."),
    engine = Depends(get_shared_transcriber_engine)
):
    """
    Validates file formats & size bounds, processes multi-part audio streams safely, 
    stages tracking components securely, and returns transcribed Pashto text 
    along with computation timestamps.
    """
    # Secure file input and format validation layer
    await validate_audio_file(file)

    service = TranscriptionService(engine=engine)
    
    # Unpack all 4 metrics returned by the transcription service
    text_output, wer_score, processing_time, inference_time = service.generate_speech_to_text(file, reference_text)
    
    return {
        "filename": file.filename,
        "transcription": text_output,
        "wer_score": wer_score,
        "processing_time_sec": processing_time,
        "inference_time_sec": inference_time,
        "model_version": getattr(settings, "MERGED_MODEL_PATH", "Pashto-Whisper-v2.3.0-LoRA")
    }