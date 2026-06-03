import logging
import sys
from typing import Any
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handlers: list[Any] = [logging.StreamHandler(sys.stdout)]

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    handlers.append(logging.FileHandler(log_dir / "app.log", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
    )

    # Silence noisy libs
    for noisy in ("uvicorn.access", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
