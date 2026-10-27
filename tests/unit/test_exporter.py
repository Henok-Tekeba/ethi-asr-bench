"""Export envelope and safe file-writing tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from tests.conftest import FIXED_TIME, make_record

from ethi_asr_bench.errors import ErrorCode, InputError
from ethi_asr_bench.services.exporter import (
    EXPORT_FORMAT_VERSION,
    build_export_payload,
    serialize_export,
    write_export,
)
from ethi_asr_bench.storage.repository_store import StoredInspection


def stored(*, raw: dict[str, object] | None = None) -> StoredInspection:
    return StoredInspection(
        inspection_id=1,
        record=make_record(),
        raw_source=raw,
        provider="fake",
        provider_version="test",
    )


def test_export_envelope_contains_versioned_records() -> None:
    payload = build_export_payload([stored()], exported_at=FIXED_TIME)
    assert payload["export_format_version"] == EXPORT_FORMAT_VERSION
    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"].startswith("ethi-asr-bench/")
    assert payload["exported_at"] == "2026-10-27T12:00:00Z"
    assert payload["count"] == 1
    assert payload["records"][0]["identity"]["repo_id"] == "b1n1yam/shook-medium-amharic-2k"
    assert "raw_source" not in payload["records"][0]


def test_export_can_include_raw_source() -> None:
    payload = build_export_payload(
        [stored(raw={"provider": "fake", "model_info": {"id": "x"}})],
        exported_at=FIXED_TIME,
        include_raw=True,
    )
    assert payload["records"][0]["raw_source"]["provider"] == "fake"


def test_export_serialization_is_deterministic() -> None:
    records = [stored()]
    first = serialize_export(build_export_payload(records, exported_at=FIXED_TIME))
    second = serialize_export(build_export_payload(records, exported_at=FIXED_TIME))
    assert first == second
    assert first.endswith("\n")
    json.loads(first)


def test_export_preserves_unicode() -> None:
    serialized = serialize_export({"record": {"description": "የአማርኛ ድምፅ"}})
    assert "የአማርኛ ድምፅ" in serialized


def test_write_export_creates_parent_and_writes_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "metadata.json"
    write_export(destination, '{"ok": true}\n')
    assert destination.read_text(encoding="utf-8") == '{"ok": true}\n'
    assert list(destination.parent.glob(".*.tmp")) == []


def test_write_export_refuses_directory(tmp_path: Path) -> None:
    with pytest.raises(InputError) as captured:
        write_export(tmp_path, "{}\n")
    assert captured.value.code is ErrorCode.INVALID_EXPORT_PATH


def test_write_export_overwrites_existing_file_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "metadata.json"
    destination.write_text("old", encoding="utf-8")
    write_export(destination, "new\n")
    assert destination.read_text(encoding="utf-8") == "new\n"


def test_build_export_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValueError):
        build_export_payload(
            [stored()],
            exported_at=datetime(2026, 10, 27, 12, 0, 0),
        )


def test_export_does_not_mutate_stored_record() -> None:
    record = stored(raw={"provider": "fake"})
    before = record.record.model_dump(mode="json")
    payload = build_export_payload([record], exported_at=FIXED_TIME, include_raw=True)
    payload["records"][0]["identity"]["repo_id"] = "mutated/model"
    assert record.record.model_dump(mode="json") == before


def test_export_creates_private_parent_and_file_permissions(tmp_path: Path) -> None:
    if __import__("sys").platform == "win32":
        pytest.skip("POSIX permission bits are not meaningful on Windows")
    destination = tmp_path / "private" / "metadata.json"
    write_export(destination, "{}\n")
    assert destination.parent.stat().st_mode & 0o777 == 0o700
    assert destination.stat().st_mode & 0o777 == 0o600


def test_export_escapes_terminal_control_characters() -> None:
    stored_inspection = stored()
    payload = build_export_payload([stored_inspection], exported_at=FIXED_TIME)
    payload["records"][0]["custom_note"] = "unsafe \x9b31m"
    serialized = serialize_export(payload)
    assert "\x9b" not in serialized
    assert "\\u009b31m" in serialized
