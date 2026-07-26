# api/config.py

# This utilizes Pydantic Settings to automatically capture environmental parameters (.env) with secure fallbacks, establishing a clean source of truth for runtime configurations.

import os
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App General Settings
    APP_NAME: str = "Pashto Whisper STT API"
    ENV_MODE: str = "development"  # development | production
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Model Settings
    MERGED_MODEL_PATH: str = "Sabtain-Dev/STT-Whisper-Pashto"
    HF_TOKEN: str = ""
    
    # Hardware Configuration
    OMP_NUM_THREADS: str = "2"
    
    # Audio Storage Settings
    TEMP_API_DIR: str = "./workspace_data/api_temp"
    SUPPORTED_FORMATS: Set[str] = {"wav", "mp3", "mp4", "m4a", "flac", "ogg", "opus", "webm", "aac", "wma"}
    MAX_UPLOAD_SIZE_MB: int = 50

    # Modern Pydantic v2 settings configuration mapping
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate configurations globally
settings = Settings()

# Apply system optimization flags based on settings configurations immediately at runtime
os.environ["OMP_NUM_THREADS"] = settings.OMP_NUM_THREADS
os.makedirs(settings.TEMP_API_DIR, exist_ok=True)