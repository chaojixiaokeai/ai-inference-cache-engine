"""Structured logging system with rotation and multiple log levels."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str = "~/.cache/ai_engine/logs",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB per file
    backup_count: int = 5,
):
    """Configure the logging system with file rotation."""
    log_path = Path(os.path.expanduser(log_dir))
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Runtime log
    runtime_handler = RotatingFileHandler(
        log_path / "runtime.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    runtime_handler.setLevel(logging.DEBUG)
    runtime_handler.setFormatter(fmt)
    root.addHandler(runtime_handler)

    # Error log
    error_handler = RotatingFileHandler(
        log_path / "error.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)

    # Console handler (minimal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )
    root.addHandler(console_handler)

    logging.getLogger("llama_cpp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
