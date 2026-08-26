"""
Centralised logging setup for the Airline Passenger Satisfaction project.

Importing this module configures the root logger to write to a fresh,
timestamped log file inside ``logs/`` (created automatically).

Usage::

    import logging
    from logger import LOG_FILE_PATH  # noqa: F401  (configures logging)

    logger = logging.getLogger(__name__)
    logger.info("Something happened")
"""

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] line %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
