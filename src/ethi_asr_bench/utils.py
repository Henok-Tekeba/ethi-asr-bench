"""Small, dependency-free helpers shared across the application."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, quote_plus

UTC = UTC
_JSON_TERMINAL_CONTROL_RE = re.compile(r"[\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


def utc_now() -> datetime:
    """Return the current UTC time, truncated to whole seconds."""
    return datetime.now(tz=UTC).replace(microsecond=0)


def format_utc(value: datetime) -> str:
    """Format an aware datetime as canonical UTC ISO 8601 (``...Z``)."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    """Parse an ISO 8601 timestamp and return an aware UTC datetime."""
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def truncate(value: str, limit: int) -> tuple[str, bool]:
    """Return ``value`` truncated to ``limit`` characters and a flag."""
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def redact(value: str, secret: str | None) -> str:
    """Remove a known secret and common URL-encoded forms from a message."""
    if not secret:
        return value
    variants = {secret, quote(secret, safe=""), quote_plus(secret)}
    for variant in sorted(variants, key=len, reverse=True):
        value = value.replace(variant, "[REDACTED]")
    return value


def safe_json_dumps(payload: Any, *, indent: int | None = None) -> str:
    """Serialize JSON while escaping terminal control and bidi override characters.

    ``json.dumps`` escapes C0 controls but leaves C1 controls and bidi formatting
    characters raw when ``ensure_ascii=False``. The replacement is applied to the
    serialized text, where these characters can only occur inside JSON strings.
    """
    serialized = json.dumps(payload, ensure_ascii=False, indent=indent)
    serialized = serialized.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return _JSON_TERMINAL_CONTROL_RE.sub(lambda match: f"\\u{ord(match.group()):04x}", serialized)
