# api/services/transcription_service.py

# Runs validation logic, writes binary streams dynamically using disk staging configurations, and coordinates text evaluations.
import os
import time
import shutil
from typing import Optional, Tuple
from fastapi import UploadFile
from api.config import settings
from api.logger import logger
from api.exceptions import CustomAPIException

try:
    from jiwer import wer as jiwer_wer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

class TranscriptionService:
    def __init__(self, engine):
        self.engine = engine

    def run_file_system_checks(self, file: UploadFile):
        if not file.filename or "." not in file.filename:
            raise CustomAPIException("Malformed file label identification sequence.", status_code=400)
            
        ext = file.filename.split(".")[-1].lower()
        if ext not in settings.SUPPORTED_FORMATS:
            raise CustomAPIException(f"Unsupported file format extension: '.{ext}'. Use valid options.", status_code=400)

    def generate_speech_to_text(self, file: UploadFile, reference_text: Optional[str] = None) -> Tuple[str, Optional[float], float]:
        self.run_file_system_checks(file)
        
        temp_file_path = os.path.join(settings.TEMP_API_DIR, f"prod_task_{os.getpid()}_{file.filename}")
        start_time = time.perf_counter()
        
        try:
            logger.info(f"Disk staging inbound track: {file.filename}")
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Perform ML Inference call using our dependency injected engine instance
            logger.info("Executing acoustic sequence inference loops...")
            inferred_text = self.engine.transcribe(temp_file_path)
            
            # Compute WER statistics if a reference script mapping is supplied
            wer_score = None
            if reference_text and reference_text.strip():
                if JIWER_AVAILABLE:
                    wer_score = round(jiwer_wer(reference_text.strip(), inferred_text.strip()) * 100, 2)
                    logger.info(f"Evaluated WER Metric score output calculated: {wer_score}%")
                else:
                    logger.warning("WER metrics skipped: 'jiwer' package is unindexed in runtime context.")
            
            end_time = time.perf_counter()
            duration = round(end_time - start_time, 3)
            logger.info(f"Successfully processed audio in {duration} seconds.")
            
            return inferred_text, wer_score, duration

        except CustomAPIException:
            raise
        except Exception as e:
            logger.error(f"Execution error hit inside operational service routines: {str(e)}", exc_info=True)
            raise CustomAPIException("Linguistic engine dropped during transcription operations.", status_code=500)
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as err:
                    logger.warning(f"Disk tracking garbage purge error loop trace: {err}")