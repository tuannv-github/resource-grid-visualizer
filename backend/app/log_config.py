"""Backend logging via the stdlib :mod:`logging` package.

File output uses automatic rotation:

* :class:`logging.handlers.RotatingFileHandler` when ``rotate_when`` is unset (rotate by size).
* :class:`logging.handlers.TimedRotatingFileHandler` when ``rotate_when`` is set (rotate by time).
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

APP_LOGGER_NAME = "resgrid"

_file_handler_added = False
_console_handler_added = False

_valid_timed_when = frozenset(
    ["S", "M", "H", "D", "midnight", *[f"W{i}" for i in range(7)]]
)


class _SkipStatusPollOnConsoleFilter(logging.Filter):
    """Drop status-poll app log records on stderr (file handler on root still receives them)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "status_poll", False)


class _SkipUvicornStatusAccessLineFilter(logging.Filter):
    """Drop uvicorn access-log lines that mention GET /api/stream/status on stderr only."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except (ValueError, TypeError):
            msg = str(record.msg)
        return "/api/stream/status" not in msg


def _attach_stderr_only_filters(handler: logging.StreamHandler) -> None:
    if not any(isinstance(f, _SkipStatusPollOnConsoleFilter) for f in handler.filters):
        handler.addFilter(_SkipStatusPollOnConsoleFilter())


def _suppress_status_poll_on_stderr_stream_handlers() -> None:
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            _attach_stderr_only_filters(h)


def attach_uvicorn_access_stderr_filters() -> None:
    for h in logging.getLogger("uvicorn.access").handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            if not any(isinstance(f, _SkipUvicornStatusAccessLineFilter) for f in h.filters):
                h.addFilter(_SkipUvicornStatusAccessLineFilter())


def get_logger(name: str = APP_LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def _level(name: str) -> int:
    return getattr(logging, name, logging.INFO) if hasattr(logging, name) else logging.INFO


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _normalize_timed_when(raw: str) -> str | None:
    w = raw.strip()
    if not w:
        return None
    u = w.upper()
    if u in ("S", "M", "H", "D"):
        return u
    if len(u) == 2 and u[0] == "W" and u[1].isdigit():
        return u if u in _valid_timed_when else None
    if w.lower() == "midnight":
        return "midnight"
    return None


def ensure_console_logging(level_name: str) -> None:
    """Attach a single stderr :class:`logging.StreamHandler` to the root logger."""
    global _console_handler_added
    if _console_handler_added:
        return
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            _console_handler_added = True
            return
    level = _level(level_name.upper())
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(_formatter())
    root.addHandler(sh)
    if root.level == logging.NOTSET:
        root.setLevel(level)
    else:
        root.setLevel(min(root.level, level))
    _console_handler_added = True


def configure_file_logging(
    log_file: str,
    level_name: str = "INFO",
    *,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    rotate_when: str | None = None,
    rotate_interval: int = 1,
    rotate_utc: bool = False,
) -> bool:
    """Attach a size- or time-based rotating file handler on the root logger (once).

    Returns whether a file sink is active (so callers can hide noisy status lines from stderr only).
    """
    global _file_handler_added
    raw = log_file.strip()
    if not raw or raw.lower() in ("none", "off", "false", "0", "-"):
        return False
    if _file_handler_added:
        return True

    path = Path(raw).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    target = str(path)
    path_s = str(path)
    root = logging.getLogger()

    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            _file_handler_added = True
            return True

    level = _level(level_name.upper())
    timed_key = _normalize_timed_when(rotate_when) if rotate_when else None
    if rotate_when and rotate_when.strip() and timed_key is None:
        get_logger().warning(
            "Invalid LOG_ROTATE_WHEN=%r; falling back to size-based rotation",
            rotate_when,
        )

    if timed_key is not None:
        fh: logging.Handler = logging.handlers.TimedRotatingFileHandler(
            path_s,
            when=timed_key,
            interval=max(1, rotate_interval),
            backupCount=backup_count,
            encoding="utf-8",
            utc=rotate_utc,
        )
        kind = f"time-based when={timed_key!r} interval={rotate_interval} utc={rotate_utc}"
    else:
        fh = logging.handlers.RotatingFileHandler(
            path_s,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        kind = f"size-based max_bytes={max_bytes}"

    fh.setLevel(level)
    fh.setFormatter(_formatter())
    root.addHandler(fh)
    if root.level == logging.NOTSET:
        root.setLevel(level)
    else:
        root.setLevel(min(root.level, level))
    _file_handler_added = True
    get_logger().info("Writing logs to %s (%s, backups=%s)", target, kind, backup_count)
    return True


def setup_backend_logging(
    log_file: str,
    level_name: str,
    *,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    rotate_when: str | None = None,
    rotate_interval: int = 1,
    rotate_utc: bool = False,
) -> None:
    """Stderr always; rotating file when *log_file* is enabled.

    When file logging is on, ``GET /api/stream/status`` app logs and uvicorn access lines for that
    path are filtered off stderr only (they still go to the file).
    """
    ensure_console_logging(level_name)
    file_on = configure_file_logging(
        log_file,
        level_name,
        max_bytes=max_bytes,
        backup_count=backup_count,
        rotate_when=rotate_when,
        rotate_interval=rotate_interval,
        rotate_utc=rotate_utc,
    )
    if file_on:
        _suppress_status_poll_on_stderr_stream_handlers()
        attach_uvicorn_access_stderr_filters()
