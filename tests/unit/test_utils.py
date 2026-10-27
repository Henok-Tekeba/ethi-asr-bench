"""Utility hardening tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ethi_asr_bench.utils import format_utc, parse_utc, redact


def test_redact_removes_exact_and_url_encoded_secret_forms() -> None:
    secret = "test/secret+token"
    message = "raw=test/secret+token encoded=test%2Fsecret%2Btoken query=test%2Fsecret%2Btoken"
    redacted = redact(message, secret)
    assert secret not in redacted
    assert "test%2Fsecret%2Btoken" not in redacted
    assert redacted.count("[REDACTED]") == 3


def test_redact_without_secret_is_a_no_op() -> None:
    assert redact("message", None) == "message"


def test_format_and_parse_utc() -> None:
    value = datetime(2026, 10, 27, 12, 30, 45, tzinfo=UTC)
    assert format_utc(value) == "2026-10-27T12:30:45Z"
    assert parse_utc("2026-10-27T12:30:45Z") == value


def test_format_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        format_utc(datetime(2026, 10, 27, 12, 30, 45))


def test_safe_json_dumps_escapes_c1_and_bidi_controls_but_preserves_amharic() -> None:
    from ethi_asr_bench.utils import safe_json_dumps

    serialized = safe_json_dumps({"c1": "x\x9by", "bidi": "x\u202ey", "amharic": "የአማርኛ ድምፅ"})
    assert "\\u009b" in serialized
    assert "\\u202e" in serialized
    assert "\x9b" not in serialized
    assert "\u202e" not in serialized
    assert "የአማርኛ ድምፅ" in serialized


def test_safe_json_dumps_handles_lone_surrogates_without_crashing() -> None:
    from ethi_asr_bench.utils import safe_json_dumps

    serialized = safe_json_dumps({"value": "bad\ud800text"})
    serialized.encode("utf-8")
    assert "\\ud800" in serialized
