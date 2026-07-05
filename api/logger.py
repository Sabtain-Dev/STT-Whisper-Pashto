# api/logger.py

# Replaces raw terminal print statements with unified, structured runtime logs capturing performance timeframes and runtime events.

import logging
import sys

# Configure standard root formatting guidelines
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("PashtoWhisperAPI")