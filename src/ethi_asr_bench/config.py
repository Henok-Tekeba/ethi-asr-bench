"""Configuration loaded from the environment.

No secret is ever logged or persisted. ``HF_TOKEN`` is read once and held in memory
only for the lifetime of the process.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ethi_asr_bench.errors import ConfigurationError, ErrorCode

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 0.5


def default_data_dir() -> Path:
    """Return a platform-appropriate directory for local state."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "ethi-asr-bench"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "ethi-asr-bench"
    return Path.home() / ".local" / "share" / "ethi-asr-bench"


def default_db_path() -> Path:
    """Return the default SQLite database path."""
    return default_data_dir() / "records.db"


def _float_from_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            ErrorCode.INVALID_CONFIGURATION,
            f"{name} must be a number, got {raw!r}.",
        ) from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            ErrorCode.INVALID_CONFIGURATION,
            f"{name} must be between {minimum} and {maximum}, got {value}.",
        )
    return value


def _int_from_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            ErrorCode.INVALID_CONFIGURATION,
            f"{name} must be an integer, got {raw!r}.",
        ) from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            ErrorCode.INVALID_CONFIGURATION,
            f"{name} must be between {minimum} and {maximum}, got {value}.",
        )
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings."""

    db_path: Path
    hf_token: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    log_level: str = "WARNING"

    @classmethod
    def from_env(cls, *, db_path: Path | None = None) -> Settings:
        """Build settings from environment variables and an optional CLI override."""
        resolved_db = db_path if db_path is not None else _db_path_from_env()
        token = (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
        ).strip()
        return cls(
            db_path=resolved_db.expanduser(),
            hf_token=token or None,
            timeout_seconds=_float_from_env(
                "ETHI_ASR_HF_TIMEOUT",
                DEFAULT_TIMEOUT_SECONDS,
                minimum=1.0,
                maximum=300.0,
            ),
            max_retries=_int_from_env(
                "ETHI_ASR_HF_MAX_RETRIES",
                DEFAULT_MAX_RETRIES,
                minimum=0,
                maximum=10,
            ),
            backoff_seconds=_float_from_env(
                "ETHI_ASR_HF_BACKOFF",
                DEFAULT_BACKOFF_SECONDS,
                minimum=0.0,
                maximum=30.0,
            ),
            log_level=_log_level_from_env(),
        )


def _db_path_from_env() -> Path:
    raw = os.environ.get("ETHI_ASR_DB_PATH")
    if raw is None or raw.strip() == "":
        return default_db_path()
    return Path(raw)


def _log_level_from_env() -> str:
    level = os.environ.get("ETHI_ASR_LOG_LEVEL", "WARNING").strip().upper()
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
        raise ConfigurationError(
            ErrorCode.INVALID_CONFIGURATION,
            f"ETHI_ASR_LOG_LEVEL must be one of {', '.join(sorted(allowed))}, got {level!r}.",
        )
    return level
