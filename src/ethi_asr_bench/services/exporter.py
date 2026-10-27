"""Deterministic JSON export and safe atomic file writing."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ethi_asr_bench import __version__
from ethi_asr_bench.errors import ErrorCode, InputError, StorageError
from ethi_asr_bench.models.repository import SCHEMA_VERSION
from ethi_asr_bench.storage.repository_store import StoredInspection
from ethi_asr_bench.utils import format_utc, safe_json_dumps

EXPORT_FORMAT_VERSION = "1.0.0"


def build_export_payload(
    records: Sequence[StoredInspection],
    *,
    exported_at: datetime,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Build a versioned export envelope with deterministic record order."""
    exported_records: list[dict[str, Any]] = []
    for stored in records:
        payload = stored.record.model_dump(mode="json", exclude_none=True)
        if include_raw:
            payload["raw_source"] = stored.raw_source
        exported_records.append(payload)
    return {
        "export_format_version": EXPORT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_by": f"ethi-asr-bench/{__version__}",
        "exported_at": format_utc(exported_at),
        "count": len(exported_records),
        "records": exported_records,
    }


def serialize_export(payload: dict[str, Any]) -> str:
    """Serialize an export deterministically with a trailing newline."""
    return safe_json_dumps(payload, indent=2) + "\n"


def write_export(path: Path, content: str) -> None:
    """Write an export atomically, refusing to replace a directory.

    The destination is chosen by the user. No path is derived from remote metadata
    or repository IDs.
    """
    destination = path.expanduser()
    if destination.exists() and destination.is_dir():
        raise InputError(
            ErrorCode.INVALID_EXPORT_PATH,
            f"Export destination is a directory: {destination}",
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    except InputError:
        raise
    except OSError as exc:
        raise StorageError(
            ErrorCode.EXPORT_FAILED,
            f"Could not write export to {destination}: {exc}",
        ) from exc
