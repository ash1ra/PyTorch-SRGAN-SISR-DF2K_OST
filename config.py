import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def create_logger(
    log_level: str,
    caller_file_name: str,
    log_file_name: str | None = None,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y.%m.%d %H:%M:%S"
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    if not log_file_name:
        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file_name = f"logs/srgan_{Path(caller_file_name).stem}_{current_date}.log"

    log_file_path = Path(log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
