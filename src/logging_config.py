"""Structured JSON logging for the API (observability).

Emits one JSON line per log record so logs are machine-parseable by tools
like Datadog, Loki, or CloudWatch — the kind of observability expected of a
production AI service.
"""

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        # Attach any structured extras passed via logger.info(..., extra={...}).
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str = "resume_analyser") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **fields) -> None:
    """Log a message with arbitrary structured fields."""
    logger.info(message, extra={"extra_fields": fields})
