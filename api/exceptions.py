# api/exceptions.py

# Defines tailored domain runtime errors to decouple operational business layer exceptions from raw HTTP transport layers.

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("PashtoWhisperAPI.Exceptions")

class CustomAPIException(Exception):
    """Base exception class for domain level validation or runtime system drops."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(CustomAPIException)
    async def custom_exception_handler(request: Request, exc: CustomAPIException):
        logger.warning(f"Domain validation exception triggered on path '{request.url.path}': {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "detail": exc.message}
        )

    @app.exception_handler(Exception)
    async def global_panic_handler(request: Request, exc: Exception):
        logger.critical(f"Unhandled critical system exception occurred on path '{request.url.path}': {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": "An unexpected server error occurred in the transcription pipeline."}
        )