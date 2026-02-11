import sys
from pathlib import Path

from loguru import logger


_LOGGER_INITIALIZED = False


def get_logger(
    name: str = "app",
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
):
    global _LOGGER_INITIALIZED

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not _LOGGER_INITIALIZED:
        logger.remove()

        if console:
            logger.add(
                sys.stdout,
                level=level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                    "<level>{message}</level>"
                ),
            )

        _LOGGER_INITIALIZED = True

    if file:
        logger.add(
            log_dir / f"{name}.log",
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{module}:{function}:{line} - "
                "{message}"
            ),
        )

    return logger