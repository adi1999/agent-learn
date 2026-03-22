"""Structured logging for agentlearn."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for agentlearn."""
    logger = logging.getLogger(f"agentlearn.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        level = os.environ.get("AGENTLEARN_LOG_LEVEL", "INFO").upper()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
