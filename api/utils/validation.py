# api/utils/validation.py

from fastapi import HTTPException, UploadFile, status

# Configuration Limits
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Expanded and verified supported audio formats
ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".flac", 
    ".ogg", ".opus", ".webm", ".aac", ".wma"
}

ALLOWED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", 
    "audio/mpeg", "audio/mp3", 
    "audio/mp4", "audio/m4a", "audio/x-m4a", 
    "audio/flac", "audio/ogg", "application/ogg",
    "audio/opus", "audio/webm", "video/webm",
    "audio/aac", "audio/x-ms-wma"
}

async def validate_audio_file(file: UploadFile) -> None:
    """
    Validates file extension, MIME type, and size restrictions 
    prior to processing audio streams through the ASR engine.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File payload missing a valid filename."
        )

    # 1. Check Extension
    ext = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{ext}'. Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Check MIME Type (if provided)
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported MIME type '{file.content_type}'."
        )

    # 3. Read & Check File Size + Non-Empty Payload
    file_bytes = await file.read()
    await file.seek(0)  # Reset stream pointer for downstream processing

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file payload is empty (0 bytes)."
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed threshold of {MAX_FILE_SIZE_MB}MB."
        )