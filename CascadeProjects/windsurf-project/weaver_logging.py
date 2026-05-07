#!/usr/bin/env python3
"""
weaver_logging.py — Structured JSON Logging
=============================================
Configures per-lobe loggers with JSON output, rotation, and a shared
format across all Weaver modules.

Usage:
    from weaver_logging import get_logger
    log = get_logger("phone_bridge")
    log.info("Call started", caller="+1555...", quantum_pathway="Resonance")
"""

import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone

PROJ = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJ, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = os.environ.get("WEAVER_LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.environ.get("WEAVER_LOG_MAX_MB", "10")) * 1024 * 1024
LOG_BACKUP_COUNT = int(os.environ.get("WEAVER_LOG_BACKUPS", "5"))

_BOOT_TIME = time.monotonic()
_configured_loggers: dict[str, logging.Logger] = {}


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "lobe": getattr(record, "lobe", record.name),
            "msg": record.getMessage(),
            "uptime_s": round(time.monotonic() - _BOOT_TIME, 1),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        for key in ("caller", "quantum_pathway", "tool", "event", "detail",
                     "port", "latency_ms", "status_code"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


def get_logger(lobe: str) -> logging.Logger:
    """Return a named logger with JSON file + console handlers."""
    if lobe in _configured_loggers:
        return _configured_loggers[lobe]

    logger = logging.getLogger(f"weaver.{lobe}")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        fmt = JSONFormatter()

        fh = logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, f"{lobe}.jsonl"),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        combined = logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, "weaver.jsonl"),
            maxBytes=LOG_MAX_BYTES * 3,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        combined.setFormatter(fmt)
        logger.addHandler(combined)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            "\033[36m[%(name)s]\033[0m %(levelname)s %(message)s"
        ))
        ch.setLevel(logging.WARNING)
        logger.addHandler(ch)

    logger = logging.LoggerAdapter(logger, {"lobe": lobe})
    _configured_loggers[lobe] = logger
    return logger


def log_event(lobe: str, event: str, **kwargs):
    """Quick structured log entry without getting a logger first."""
    logger = get_logger(lobe)
    logger.info(event, extra=kwargs)
