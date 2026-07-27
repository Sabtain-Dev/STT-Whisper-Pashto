# api/schemas/response.py

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model: str
    version: str
    language: str
    framework: str
    wer: float

class TranscriptionResponse(BaseModel):
    filename: str
    transcription: str
    wer_score: float | None = None
    processing_time_sec: float
    inference_time_sec: float
    model_version: str