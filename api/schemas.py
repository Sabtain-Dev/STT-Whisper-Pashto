# api/schemas.py

# This handles standard Pydantic request data structure types and baseline model validation schemas.
# api/schemas.py (The Data Inspector): This file uses Pydantic to enforce exact shapes for data coming in or out. It protects the engine by making sure that what the server returns matches a structured JSON outline (HealthResponse, ModelInfoResponse, TranscriptionResponse).

from pydantic import BaseModel
from typing import Optional

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
    wer_score: Optional[float] = None