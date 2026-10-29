"""Structured JSON logs to stdout, a rotating file on the data volume, and an in-memory ring for /logs."""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from collections import deque
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


class RingHandler(logging.Handler):
    def __init__(self, size: int = 5000):
        super().__init__()
        self.lines: deque[str] = deque(maxlen=size)

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


RING = RingHandler()


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    fmt = JsonFormatter()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    handlers: list[logging.Handler] = [logging.StreamHandler(), RING]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.TimedRotatingFileHandler(
            log_dir / "polybot.log", when="midnight", backupCount=14, utc=True))
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("aiohttp.access").setLevel("WARNING")
    logging.getLogger("websockets").setLevel("WARNING")
