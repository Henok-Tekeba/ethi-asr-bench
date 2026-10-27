"""Repository-facing persistence interface and its SQLite implementation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError as PydanticValidationError

from ethi_asr_bench.errors import ErrorCode, StorageError
from ethi_asr_bench.models.repository import (
    InspectionRecord,
    InspectionStatus,
    RepositorySummary,
)
from ethi_asr_bench.storage.database import Database
from ethi_asr_bench.utils import parse_utc


@dataclass(frozen=True, slots=True)
class StoredInspection:
    """A persisted inspection row and its optional raw provider payload."""

    inspection_id: int
    record: InspectionRecord
    raw_source: dict[str, Any] | None
    provider: str
    provider_version: str | None


class RepositoryStore(Protocol):
    """Persistence boundary used by services and the CLI."""

    def save_inspection(
        self,
        record: InspectionRecord,
        *,
        raw_source: dict[str, Any] | None = None,
    ) -> None: ...

    def get_latest(
        self, repo_id: str, *, revision: str | None = None
    ) -> StoredInspection | None: ...

    def list_repositories(self, *, limit: int = 50) -> list[RepositorySummary]: ...

    def export_records(
        self,
        *,
        all_history: bool = False,
        repo_id: str | None = None,
        include_raw: bool = False,
    ) -> list[StoredInspection]: ...

    def delete_repository(self, repo_id: str) -> int: ...

    def close(self) -> None: ...


class SQLiteRepositoryStore:
    """Store normalized inspections in SQLite with revision-aware history."""

    def __init__(self, database: Database | Path) -> None:
        self._database = database if isinstance(database, Database) else Database(Path(database))
        self._database.initialize()

    def save_inspection(
        self,
        record: InspectionRecord,
        *,
        raw_source: dict[str, Any] | None = None,
    ) -> None:
        identity = record.identity
        retrieved_at = identity.retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        record_json = _dump_record(record)
        raw_json = _dump_json(raw_source) if raw_source is not None else None
        first_error = record.errors[0] if record.errors else None

        try:
            with self._database.connection as connection:
                connection.execute(
                    """
                    INSERT INTO repositories (
                        repo_id, repo_type, canonical_url, owner, name,
                        first_seen_at, last_inspected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(repo_id) DO UPDATE SET
                        repo_type = excluded.repo_type,
                        canonical_url = excluded.canonical_url,
                        owner = excluded.owner,
                        name = excluded.name,
                        last_inspected_at = excluded.last_inspected_at
                    """,
                    (
                        identity.repo_id,
                        identity.repo_type.value,
                        identity.canonical_url,
                        identity.owner,
                        identity.name,
                        retrieved_at,
                        retrieved_at,
                    ),
                )

                existing_id: int | None = None
                if identity.resolved_commit is not None and record.status in {
                    InspectionStatus.SUCCESS,
                    InspectionStatus.PARTIAL,
                }:
                    row = connection.execute(
                        """
                        SELECT id FROM inspections
                        WHERE repo_id = ?
                          AND requested_revision IS ?
                          AND resolved_commit = ?
                          AND status IN ('success', 'partial')
                        """,
                        (
                            identity.repo_id,
                            identity.requested_revision,
                            identity.resolved_commit,
                        ),
                    ).fetchone()
                    if row is not None:
                        existing_id = int(row["id"])

                values = (
                    identity.repo_id,
                    identity.requested_revision,
                    identity.resolved_commit,
                    retrieved_at,
                    record.status.value,
                    record.schema_version,
                    record_json,
                    raw_json,
                    record.provenance.provider,
                    record.provenance.provider_version,
                    first_error.code if first_error is not None else None,
                    first_error.message if first_error is not None else None,
                )
                if existing_id is not None:
                    connection.execute(
                        """
                        UPDATE inspections SET
                            requested_revision = ?,
                            resolved_commit = ?,
                            retrieved_at = ?,
                            status = ?,
                            schema_version = ?,
                            record_json = ?,
                            raw_json = ?,
                            provider = ?,
                            provider_version = ?,
                            error_code = ?,
                            error_message = ?
                        WHERE id = ?
                        """,
                        (*values[1:], existing_id),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO inspections (
                            repo_id, requested_revision, resolved_commit, retrieved_at,
                            status, schema_version, record_json, raw_json, provider,
                            provider_version, error_code, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
        except (sqlite3.Error, UnicodeError) as exc:
            raise StorageError(
                ErrorCode.STORAGE_ERROR,
                f"Could not save the inspection record: {exc}",
            ) from exc

    def get_latest(self, repo_id: str, *, revision: str | None = None) -> StoredInspection | None:
        sql = "SELECT * FROM inspections WHERE repo_id = ?"
        parameters: list[Any] = [repo_id]
        if revision is not None:
            sql += (
                " AND (requested_revision = ? OR resolved_commit = ?"
                " OR substr(resolved_commit, 1, length(?)) = ?)"
            )
            parameters.extend([revision, revision, revision, revision])
        sql += " ORDER BY retrieved_at DESC, id DESC LIMIT 1"
        try:
            row = self._database.connection.execute(sql, parameters).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(
                ErrorCode.STORAGE_ERROR, f"Could not read the inspection record: {exc}"
            ) from exc
        return _row_to_stored(row, include_raw=True) if row is not None else None

    def list_repositories(self, *, limit: int = 50) -> list[RepositorySummary]:
        if limit < 1:
            raise ValueError("limit must be positive")
        sql = """
            WITH ranked AS (
                SELECT
                    i.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY i.repo_id
                        ORDER BY i.retrieved_at DESC, i.id DESC
                    ) AS row_number,
                    COUNT(*) OVER (PARTITION BY i.repo_id) AS inspection_count
                FROM inspections AS i
            )
            SELECT
                r.repo_id AS repository_id,
                r.canonical_url,
                r.owner,
                r.name,
                ranked.status,
                ranked.requested_revision,
                ranked.resolved_commit,
                ranked.retrieved_at,
                ranked.record_json,
                ranked.inspection_count
            FROM repositories AS r
            JOIN ranked ON ranked.repo_id = r.repo_id AND ranked.row_number = 1
            ORDER BY ranked.retrieved_at DESC, r.repo_id COLLATE NOCASE
            LIMIT ?
        """
        try:
            rows = self._database.connection.execute(sql, (limit,)).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(
                ErrorCode.STORAGE_ERROR, f"Could not list repositories: {exc}"
            ) from exc

        summaries: list[RepositorySummary] = []
        for row in rows:
            record = _parse_record_json(str(row["record_json"]), repo_id=str(row["repository_id"]))
            summaries.append(
                RepositorySummary(
                    repo_id=str(row["repository_id"]),
                    canonical_url=str(row["canonical_url"]),
                    owner=_optional_str(row["owner"]),
                    name=str(row["name"]),
                    status=InspectionStatus(str(row["status"])),
                    requested_revision=_optional_str(row["requested_revision"]),
                    resolved_commit=_optional_str(row["resolved_commit"]),
                    retrieved_at=parse_utc(str(row["retrieved_at"])),
                    inspection_count=int(row["inspection_count"]),
                    warning_count=len(record.warnings),
                )
            )
        return summaries

    def export_records(
        self,
        *,
        all_history: bool = False,
        repo_id: str | None = None,
        include_raw: bool = False,
    ) -> list[StoredInspection]:
        if all_history:
            sql = "SELECT * FROM inspections"
            conditions: list[str] = []
            parameters: list[Any] = []
            if repo_id is not None:
                conditions.append("repo_id = ?")
                parameters.append(repo_id)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY repo_id COLLATE NOCASE, retrieved_at, id"
        else:
            sql = """
                WITH ranked AS (
                    SELECT
                        i.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY i.repo_id
                            ORDER BY i.retrieved_at DESC, i.id DESC
                        ) AS row_number
                    FROM inspections AS i
                )
                SELECT * FROM ranked WHERE row_number = 1
            """
            parameters = []
            if repo_id is not None:
                sql += " AND repo_id = ?"
                parameters.append(repo_id)
            sql += " ORDER BY repo_id COLLATE NOCASE, retrieved_at, id"
        try:
            rows = self._database.connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(
                ErrorCode.STORAGE_ERROR, f"Could not export inspection records: {exc}"
            ) from exc
        return [_row_to_stored(row, include_raw=include_raw) for row in rows]

    def delete_repository(self, repo_id: str) -> int:
        try:
            with self._database.connection as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM inspections WHERE repo_id = ?",
                    (repo_id,),
                ).fetchone()
                inspection_count = int(row["count"]) if row is not None else 0
                connection.execute("DELETE FROM repositories WHERE repo_id = ?", (repo_id,))
        except sqlite3.Error as exc:
            raise StorageError(
                ErrorCode.STORAGE_ERROR, f"Could not delete the repository: {exc}"
            ) from exc
        return inspection_count

    def close(self) -> None:
        self._database.close()


def _dump_record(record: InspectionRecord) -> str:
    try:
        payload = record.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise StorageError(
            ErrorCode.STORAGE_ERROR,
            f"The normalized record is not JSON serializable: {exc}",
        ) from exc


def _dump_json(value: dict[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise StorageError(
            ErrorCode.STORAGE_ERROR,
            f"Raw provider metadata is not JSON serializable: {exc}",
        ) from exc


def _parse_record_json(value: str, *, repo_id: str) -> InspectionRecord:
    try:
        return InspectionRecord.model_validate_json(value)
    except PydanticValidationError as exc:
        raise StorageError(
            ErrorCode.STORAGE_ERROR,
            f"Stored record for {repo_id!r} does not match schema version 1.0.0: "
            f"{exc.error_count()} validation error(s).",
        ) from exc


def _row_to_stored(row: sqlite3.Row, *, include_raw: bool) -> StoredInspection:
    repo_id = str(row["repo_id"])
    record = _parse_record_json(str(row["record_json"]), repo_id=repo_id)
    raw_source: dict[str, Any] | None = None
    if include_raw and row["raw_json"] is not None:
        try:
            parsed = json.loads(str(row["raw_json"]))
        except json.JSONDecodeError as exc:
            raise StorageError(
                ErrorCode.STORAGE_ERROR,
                f"Stored raw metadata for {repo_id!r} is not valid JSON: {exc}",
            ) from exc
        if not isinstance(parsed, dict):
            raise StorageError(
                ErrorCode.STORAGE_ERROR,
                f"Stored raw metadata for {repo_id!r} is not a JSON object.",
            )
        raw_source = parsed
    return StoredInspection(
        inspection_id=int(row["id"]),
        record=record,
        raw_source=raw_source,
        provider=str(row["provider"]),
        provider_version=_optional_str(row["provider_version"]),
    )


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
