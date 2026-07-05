# api/dependencies.py

# Provides singletons and instance managers. This registers your model runner once at startup and dynamically shares it across threads to avoid multi-gigabyte memory spikes.

import functools
from api.config import settings
from api.logger import logger

try:
    from utils.inference import PashtoTranscriber
except ImportError:
    # Diagnostic Mocking Stub if execution pipeline components are evaluated independently
    class PashtoTranscriber:
        def __init__(self, model_id_or_path: str, hf_token: str = ""):
            self.model_id = model_id_or_path
        def transcribe(self, audio_path: str) -> str:
            return "متبادل متن (ML Fallback Test Output)"

@functools.lru_cache(maxsize=1)
def get_shared_transcriber_engine() -> PashtoTranscriber:
    """ Loads and caches the pipeline singleton object inside memory spaces to ensure clean resource reuse. """
    logger.info(f"Loading weights into Memory Space for Shared Pipeline Singleton targeting: {settings.MERGED_MODEL_PATH}")
    return PashtoTranscriber(
        model_id_or_path=settings.MERGED_MODEL_PATH,
        hf_token=settings.HF_TOKEN
    )