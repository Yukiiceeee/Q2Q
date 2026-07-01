import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "q2q",
    level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"q2q_{timestamp}.log", encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Also set child loggers
    for child_name in ["src.memory", "src.retrieval", "src.embedding", "src.storage"]:
        child = logging.getLogger(child_name)
        child.setLevel(logger.level)
        if not child.handlers:
            child.parent = logger

    return logger
