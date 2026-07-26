# api/schemas/request.py

# These files establish immutable contracts for operational payload data shapes, including input hyperparameters and detailed performance tracking metrics.

from pydantic import BaseModel, Field

# from typing import Optional

class TranscriptionGenerationConfig(BaseModel):
    beam_size: int = Field(default=1, ge=1, le=5, description="Decoding search beam depth optimization footprint. Keep at 1 for low RAM configurations.")
    task: str = Field(default="transcribe", description="Task mapping execution path: transcribe or translate.")