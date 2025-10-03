"""
Logging configuration for Remi Voice Reminder Assistant.
Sets up console and file handlers with appropriate formatting.
"""

import logging
import logging.handlers
from pathlib import Path
import colorlog
from config.settings import LOGS_DIR, DEBUG_MODE


def setup_logging(name: str = "remi", level: int = None) -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.

    Args:
        name: Logger name (typically module name)
        level: Logging level (defaults based on DEBUG_MODE)

    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.DEBUG if DEBUG_MODE else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    log_file = LOGS_DIR / "remi.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Silence noisy third-party libraries
    logging.getLogger("vosk").setLevel(logging.WARNING)
    logging.getLogger("pvporcupine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If root logger not configured, configure it first.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # Configure root logger if not already done
    if not logging.getLogger("remi").handlers:
        setup_logging("remi")

    return logging.getLogger(name)
