"""SQLite persistence, history, export, and deletion tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.conftest import COMMIT, OTHER_COMMIT, make_record, make_snapshot

from ethi_asr_bench.errors import ErrorCode, RecordNotFoundError, StorageError
from ethi_asr_bench.models.repository import (
    InspectionError,
    InspectionRecord,
    InspectionStatus,
    RepositoryRef,
    WarningSeverity,
)
from ethi_asr_bench.providers.base import ProviderTextFileError
from ethi_asr_bench.services.metadata_extractor import build_failed_record
from ethi_asr_bench.storage.database import Database
from ethi_asr_bench.storage.repository_store import SQLiteRepositoryStore, StoredInspection

BASE_TIME = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)


def record_for(
    *,
    repo_id: str = "owner/model",
    commit: str | None = COMMIT,
    revision: str | None = None,
    retrieved_at: datetime = BASE_TIME,
    partial: bool = False,
) -> InspectionRecord:
    snapshot = make_snapshot(
        repo_id=repo_id,
        resolved_commit=commit,
        requested_revision=revision,
        notes=[] if commit else ["no commit"],
    )
    if partial:
        snapshot = make_snapshot(
            repo_id=repo_id,
            resolved_commit=commit,
            requested_revision=revision,
            text_files={},
            text_file_errors=[
                ProviderTextFileError(
                    filename="README.md",
                    code=ErrorCode.TIMEOUT.value,
                    message="timed out",
                    retryable=True,
                )
            ],
        )
    return make_record(snapshot, retrieved_at=retrieved_at)


def failure_for(
    *,
    repo_id: str = "owner/model",
    revision: str | None = None,
    retrieved_at: datetime = BASE_TIME,
    message: str = "not found",
) -> InspectionRecord:
    ref = RepositoryRef(
        repo_id=repo_id,
        owner=repo_id.partition("/")[0] if "/" in repo_id else None,
        name=repo_id.rsplit("/", maxsplit=1)[-1],
        canonical_url=f"https://huggingface.co/{repo_id}",
        revision=revision,
    )
    return build_failed_record(
        ref,
        InspectionError(code=ErrorCode.REPOSITORY_NOT_FOUND.value, message=message),
        retrieved_at=retrieved_at,
        provider="fake",
    )


def test_save_and_reload_round_trip(store: SQLiteRepositoryStore) -> None:
    record = record_for()
    raw = {"provider": "fake", "model_info": {"id": "owner/model"}, "text_files": {}}
    store.save_inspection(record, raw_source=raw)

    stored = store.get_latest("owner/model")
    assert isinstance(stored, StoredInspection)
    assert stored.record == record
    assert stored.raw_source == raw
    assert stored.provider == "fake"
    assert stored.provider_version == "test"


def test_reinspection_of_same_revision_and_commit_is_idempotent(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(retrieved_at=BASE_TIME))
    later = BASE_TIME + timedelta(hours=1)
    store.save_inspection(record_for(retrieved_at=later))

    summaries = store.list_repositories()
    assert len(summaries) == 1
    assert summaries[0].inspection_count == 1
    assert summaries[0].retrieved_at == later


def test_new_commit_preserves_previous_revision_history(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(commit=COMMIT, retrieved_at=BASE_TIME))
    store.save_inspection(
        record_for(commit=OTHER_COMMIT, retrieved_at=BASE_TIME + timedelta(hours=1))
    )

    summaries = store.list_repositories()
    assert summaries[0].inspection_count == 2
    assert summaries[0].resolved_commit == OTHER_COMMIT

    history = store.export_records(all_history=True)
    assert [stored.record.identity.resolved_commit for stored in history] == [
        COMMIT,
        OTHER_COMMIT,
    ]

    old = store.get_latest("owner/model", revision=COMMIT)
    assert old is not None
    assert old.record.identity.resolved_commit == COMMIT


def test_commit_prefix_lookup(store: SQLiteRepositoryStore) -> None:
    store.save_inspection(record_for(commit=COMMIT))
    stored = store.get_latest("owner/model", revision=COMMIT[:10])
    assert stored is not None
    assert stored.record.identity.resolved_commit == COMMIT


def test_same_commit_under_different_requested_revision_is_preserved(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(revision="main", retrieved_at=BASE_TIME))
    store.save_inspection(record_for(revision="dev", retrieved_at=BASE_TIME + timedelta(hours=1)))
    summaries = store.list_repositories()
    assert summaries[0].inspection_count == 2
    assert summaries[0].requested_revision == "dev"


def test_failed_attempts_are_recorded_with_reasons(store: SQLiteRepositoryStore) -> None:
    store.save_inspection(failure_for(retrieved_at=BASE_TIME, message="first"))
    store.save_inspection(
        failure_for(retrieved_at=BASE_TIME + timedelta(minutes=1), message="second")
    )

    latest = store.get_latest("owner/model")
    assert latest is not None
    assert latest.record.status is InspectionStatus.FAILED
    assert latest.record.errors[0].message == "second"
    assert store.list_repositories()[0].inspection_count == 2


def test_partial_status_is_stored_honestly(store: SQLiteRepositoryStore) -> None:
    store.save_inspection(record_for(partial=True))
    stored = store.get_latest("owner/model")
    assert stored is not None
    assert stored.record.status is InspectionStatus.PARTIAL
    assert any(warning.severity is WarningSeverity.WARNING for warning in stored.record.warnings)


def test_latest_record_returns_none_for_unknown_repository(
    store: SQLiteRepositoryStore,
) -> None:
    assert store.get_latest("missing/model") is None


def test_list_repositories_returns_latest_status_and_counts(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(repo_id="owner/a", retrieved_at=BASE_TIME))
    store.save_inspection(
        record_for(repo_id="owner/b", retrieved_at=BASE_TIME + timedelta(minutes=1))
    )
    store.save_inspection(
        record_for(
            repo_id="owner/b",
            commit=OTHER_COMMIT,
            retrieved_at=BASE_TIME + timedelta(minutes=2),
        )
    )

    summaries = store.list_repositories()
    by_id = {summary.repo_id: summary for summary in summaries}
    assert set(by_id) == {"owner/a", "owner/b"}
    assert by_id["owner/a"].inspection_count == 1
    assert by_id["owner/b"].inspection_count == 2
    assert by_id["owner/b"].resolved_commit == OTHER_COMMIT
    assert by_id["owner/a"].retrieved_at == BASE_TIME
    assert by_id["owner/a"].canonical_url == "https://huggingface.co/owner/a"


def test_list_repositories_limit(store: SQLiteRepositoryStore) -> None:
    for index in range(3):
        store.save_inspection(
            record_for(
                repo_id=f"owner/{index}",
                retrieved_at=BASE_TIME + timedelta(minutes=index),
            )
        )
    assert len(store.list_repositories(limit=2)) == 2
    with pytest.raises(ValueError):
        store.list_repositories(limit=0)


def test_export_latest_and_all_history(store: SQLiteRepositoryStore) -> None:
    store.save_inspection(record_for(commit=COMMIT, retrieved_at=BASE_TIME))
    store.save_inspection(
        record_for(commit=OTHER_COMMIT, retrieved_at=BASE_TIME + timedelta(hours=1))
    )
    store.save_inspection(record_for(repo_id="other/model", retrieved_at=BASE_TIME))

    latest = store.export_records()
    assert {stored.record.identity.repo_id for stored in latest} == {
        "owner/model",
        "other/model",
    }
    assert len(latest) == 2

    history = store.export_records(all_history=True)
    assert len(history) == 3

    filtered = store.export_records(all_history=True, repo_id="other/model")
    assert len(filtered) == 1
    assert filtered[0].raw_source is None

    with_raw = store.export_records(all_history=True, repo_id="owner/model", include_raw=True)
    assert len(with_raw) == 2


def test_delete_repository_removes_history_and_cascades(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(commit=COMMIT))
    store.save_inspection(record_for(commit=OTHER_COMMIT))
    store.save_inspection(record_for(repo_id="other/model"))

    removed = store.delete_repository("OWNER/MODEL")
    assert removed == 2
    assert store.get_latest("owner/model") is None
    assert store.get_latest("other/model") is not None
    assert store.delete_repository("owner/model") == 0


def test_store_case_insensitive_repository_lookup(store: SQLiteRepositoryStore) -> None:
    store.save_inspection(record_for(repo_id="Owner/Model"))
    stored = store.get_latest("owner/model")
    assert stored is not None
    assert stored.record.identity.repo_id == "Owner/Model"


def test_raw_source_must_be_json_serializable(store: SQLiteRepositoryStore) -> None:
    with pytest.raises(StorageError):
        store.save_inspection(record_for(), raw_source={"bad": object()})


def test_corrupt_normalized_record_raises_storage_error(tmp_path: Path) -> None:
    database = Database(tmp_path / "records.db")
    store = SQLiteRepositoryStore(database)
    try:
        store.save_inspection(record_for())
        with database.connection:
            database.connection.execute("UPDATE inspections SET record_json = '{}'")
        with pytest.raises(StorageError) as captured:
            store.get_latest("owner/model")
        assert captured.value.code is ErrorCode.STORAGE_ERROR
    finally:
        store.close()


def test_future_database_schema_is_refused(tmp_path: Path) -> None:
    database = Database(tmp_path / "records.db")
    database.initialize()
    with database.connection:
        database.connection.execute("PRAGMA user_version = 99")
    with pytest.raises(StorageError) as captured:
        SQLiteRepositoryStore(database)
    assert captured.value.code is ErrorCode.STORAGE_INITIALIZATION_FAILED
    database.close()


def test_database_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state" / "records.db"
    store = SQLiteRepositoryStore(path)
    try:
        store.save_inspection(record_for())
        assert path.exists()
    finally:
        store.close()


def test_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    first = SQLiteRepositoryStore(path)
    first.save_inspection(record_for())
    first.close()

    second = SQLiteRepositoryStore(path)
    try:
        assert second.get_latest("owner/model") is not None
    finally:
        second.close()


def test_repository_table_is_updated_on_reinspection(store: SQLiteRepositoryStore) -> None:
    store.save_inspection(record_for(revision="main", retrieved_at=BASE_TIME))
    store.save_inspection(
        record_for(
            revision="main", commit=OTHER_COMMIT, retrieved_at=BASE_TIME + timedelta(hours=1)
        )
    )
    summary = store.list_repositories()[0]
    assert summary.resolved_commit == OTHER_COMMIT
    assert summary.retrieved_at == BASE_TIME + timedelta(hours=1)


def test_stored_json_preserves_untrusted_instructions_without_execution() -> None:
    record = record_for()
    payload = json.loads(record.model_dump_json())
    serialized = json.dumps(payload)
    assert "pip install transformers" in serialized
    assert payload["custom_code"]["code_executed"] is False


def test_record_not_found_error_is_separate_from_storage_failure() -> None:
    error = RecordNotFoundError("missing")
    assert error.code is ErrorCode.RECORD_NOT_FOUND
    assert error.exit_code == 1


def test_revision_lookup_does_not_treat_underscore_as_a_wildcard(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(commit=COMMIT))
    assert store.get_latest("owner/model", revision=COMMIT[:5]) is not None
    assert store.get_latest("owner/model", revision="60b7_") is None
    assert store.get_latest("owner/model", revision="60b7%") is None


def test_case_insensitive_repository_reinspection_updates_one_row(
    store: SQLiteRepositoryStore,
) -> None:
    store.save_inspection(record_for(repo_id="Owner/Model"))
    store.save_inspection(record_for(repo_id="owner/model"))
    summaries = store.list_repositories()
    assert len(summaries) == 1
    assert summaries[0].inspection_count == 1
