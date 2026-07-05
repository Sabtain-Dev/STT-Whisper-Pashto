# api/services.py

# This file encapsulates the core service layer logic for handling transcription requests, including file validation, temporary storage management, and interfacing with the underlying ML inference engine. It also provides optional Word Error Rate (WER) calculations when reference text is provided.
# api/services.py (The Business Brain): This is where the core logic lives. The service layer handles file-handling constraints, runs security validations on file extensions, manages temporary file allocations, calculates Word Error Rates (WER), and safely executes system resource cleanups.

import os
import shutil
import logging
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status

# Try pulling from your unified shared ML utilities folder
try:
    from utils.inference import PashtoTranscriber
except ImportError:
    # Safe fallback interface stub for early standalone scaffolding tests
    class PashtoTranscriber:
        def __init__(self, model_id_or_path: str): pass
        def transcribe(self, audio_path: str) -> str:
            return "متبادل متن (ML Inference Fallback Loop)"

try:
    from jiwer import wer as jiwer_wer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

logger = logging.getLogger("uvicorn.error")

class TranscriptionService:
    def __init__(self, model_path: str, temp_dir: str, supported_formats: set):
        self.temp_dir = temp_dir
        self.supported_formats = supported_formats
        
        # Initialize your backend model pipeline instance once upon service spin-up
        logger.info(f"Initializing Transcription Engine using model path: {model_path}")
        self.engine = PashtoTranscriber(model_id_or_path=model_path)

    def validate_file(self, filename: str) -> str:
        """
        Validates file formats prior to initiating expensive parsing calculations.
        """
        if not filename or "." not in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file layout string. Missing expected extension type dot separator."
            )
        
        ext = filename.split(".")[-1].lower()
        if ext not in self.supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format extension: '.{ext}'. Supported choices: {list(self.supported_formats)}"
            )
        return ext

    def process_transcription(self, file: UploadFile, reference_text: Optional[str] = None) -> Tuple[str, Optional[float]]:
        """
        Main execution service layer processing staging, transcription generation, 
        and text comparison scoring metrics.
        """
        # 1. Enforce strict type validation checks
        self.validate_file(file.filename)
        
        # 2. Build temporary system location tracking paths
        temp_file_path = os.path.join(self.temp_dir, f"api_stream_{os.getpid()}_{file.filename}")
        
        try:
            # 3. Stream binary multi-part upload contents to temporary storage locations
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 4. Delegate to single source of truth inference logic block
            inferred_text = self.engine.transcribe(temp_file_path)
            
            # 5. Handle optional baseline error calculations
            wer_score = None
            if reference_text and reference_text.strip():
                wer_score = self.calculate_wer(reference_text.strip(), inferred_text)
                
            return inferred_text, wer_score

        except Exception as e:
            logger.error(f"Internal processing block exception caught inside service layer: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Linguistic pipeline execution exception hit: {str(e)}"
            )
        finally:
            # 6. Strict environment protection loop ensuring temporary files are always purged
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as err:
                    logger.warning(f"Could not purge temporary asset track trace: {err}")

    def calculate_wer(self, reference: str, hypothesis: str) -> Optional[float]:
        """
        Computes accurate Word Error Rate mappings safely against target references.
        """
        if not JIWER_AVAILABLE:
            logger.warning("WER execution requested but 'jiwer' dependency package remains unindexed.")
            return None
        try:
            score = jiwer_wer(reference.strip(), hypothesis.strip())
            return round(score * 100, 2)
        except Exception as e:
            logger.warning(f"Failed to generate comparative metrics output loop: {e}")
            return None