import logging
import sys

from api.config import settings


def setup_logger():
    logger = logging.getLogger("pashto_asr")
    
    # Avoid duplicate handlers if re-initialized
    if logger.hasHandlers():
        return logger

    # Toggle verbosity based on operational environment profiles
    if settings.ENV_MODE == "production":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    # Standard clean timestamp format for terminal tracking
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] ── %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()