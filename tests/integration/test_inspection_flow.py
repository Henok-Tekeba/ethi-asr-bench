"""End-to-end service, persistence, batch, and export integration tests (mocked network)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.conftest import COMMIT, FIXED_TIME, FakeProvider, make_snapshot

from ethi_asr_bench.errors import ErrorCode, ProviderError, StorageError
from ethi_asr_bench.models.repository import InspectionStatus
from ethi_asr_bench.services.batch import BatchItemStatus, run_batch
from ethi_asr_bench.services.exporter import build_export_payload, serialize_export
from ethi_asr_bench.services.inspection import InspectionService
from ethi_asr_bench.services.link_parser import parse_batch_text, parse_repository_target
from ethi_asr_bench.storage.database import Database
from ethi_asr_bench.storage.repository_store import SQLiteRepositoryStore
from ethi_asr_bench.utils import utc_now

EXAMPLE_ID = "b1n1yam/shook-medium-amharic-2k"


@pytest.fixture
def repository_store(tmp_path: Path) -> SQLiteRepositoryStore:
    store = SQLiteRepositoryStore(Database(tmp_path / "records.db"))
    try:
        yield store
    finally:
        store.close()


def test_inspect_persist_reload_and_export(
    repository_store: SQLiteRepositoryStore,
    sample_snapshot: object,
) -> None:
    provider = FakeProvider(snapshots={EXAMPLE_ID: sample_snapshot})
    service = InspectionService(provider, repository_store, clock=lambda: FIXED_TIME)
    outcome = service.inspect(parse_repository_target(EXAMPLE_ID))

    assert outcome.record.status is InspectionStatus.SUCCESS
    assert outcome.record.identity.resolved_commit == COMMIT

    stored = repository_store.get_latest(EXAMPLE_ID)
    assert stored is not None
    assert stored.record == outcome.record
    assert stored.raw_source is not None
    assert stored.raw_source["model_info"]["id"] == EXAMPLE_ID

    payload = build_export_payload(
        repository_store.export_records(include_raw=True),
        exported_at=FIXED_TIME,
    )
    serialized = serialize_export(payload)
    assert EXAMPLE_ID in serialized
    assert payload["records"][0]["identity"]["retrieved_at"] == "2026-10-27T12:00:00Z"


def test_batch_continues_after_provider_failures(
    repository_store: SQLiteRepositoryStore,
    sample_snapshot: object,
) -> None:
    provider = FakeProvider(
        snapshots={EXAMPLE_ID: sample_snapshot},
        errors={
            "owner/missing": ProviderError(ErrorCode.REPOSITORY_NOT_FOUND, "Repository not found.")
        },
    )
    service = InspectionService(provider, repository_store, clock=lambda: FIXED_TIME)
    parsed = parse_batch_text(
        f"{EXAMPLE_ID}\nhttps://huggingface.co/{EXAMPLE_ID}\nowner/missing\nnot valid\n# comment\n"
    )
    report = run_batch(service, parsed)

    assert report.summary.total_inputs == 4
    assert report.summary.succeeded == 1
    assert report.summary.failed == 1
    assert report.summary.invalid == 1
    assert report.summary.duplicates == 1
    assert report.summary.skipped == 1
    assert report.summary.exit_code == 2
    assert [item.status for item in report.items] == [
        BatchItemStatus.SUCCESS,
        BatchItemStatus.DUPLICATE,
        BatchItemStatus.FAILED,
        BatchItemStatus.INVALID,
    ]
    assert len(provider.calls) == 2
    assert repository_store.get_latest("owner/missing") is not None


def test_repeated_inspection_is_idempotent(
    repository_store: SQLiteRepositoryStore,
    sample_snapshot: object,
) -> None:
    provider = FakeProvider(snapshots={EXAMPLE_ID: sample_snapshot})
    service = InspectionService(provider, repository_store, clock=lambda: FIXED_TIME)
    ref = parse_repository_target(EXAMPLE_ID)
    service.inspect(ref)
    service.inspect(ref)

    summaries = repository_store.list_repositories()
    assert summaries[0].inspection_count == 1
    assert len(provider.calls) == 2


def test_provider_failure_is_persisted_without_raw_metadata(
    repository_store: SQLiteRepositoryStore,
) -> None:
    provider = FakeProvider(
        errors={EXAMPLE_ID: ProviderError(ErrorCode.RATE_LIMITED, "Rate limited.", retryable=True)}
    )
    service = InspectionService(provider, repository_store, clock=lambda: FIXED_TIME)
    outcome = service.inspect(parse_repository_target(EXAMPLE_ID))

    assert outcome.record.status is InspectionStatus.FAILED
    assert outcome.record.errors[0].code == "rate_limited"
    stored = repository_store.get_latest(EXAMPLE_ID)
    assert stored is not None
    assert stored.raw_source is None


def test_unexpected_extraction_failure_is_stored_with_raw_payload(
    repository_store: SQLiteRepositoryStore,
    sample_snapshot: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic extraction bug")

    monkeypatch.setattr("ethi_asr_bench.services.inspection.extract_metadata", explode)
    provider = FakeProvider(snapshots={EXAMPLE_ID: sample_snapshot})
    service = InspectionService(provider, repository_store, clock=lambda: FIXED_TIME)
    outcome = service.inspect(parse_repository_target(EXAMPLE_ID))

    assert outcome.record.status is InspectionStatus.FAILED
    assert outcome.record.errors[0].code == "extraction_failed"
    stored = repository_store.get_latest(EXAMPLE_ID)
    assert stored is not None
    assert stored.raw_source is not None
    assert "model_info" in stored.raw_source


def test_storage_failure_propagates(
    sample_snapshot: object,
) -> None:
    class BrokenStore:
        def save_inspection(self, *_args: object, **_kwargs: object) -> None:
            raise StorageError(ErrorCode.STORAGE_ERROR, "disk full")

        def close(self) -> None:
            pass

    provider = FakeProvider(snapshots={EXAMPLE_ID: sample_snapshot})
    service = InspectionService(provider, BrokenStore())  # type: ignore[arg-type]
    with pytest.raises(StorageError):
        service.inspect(parse_repository_target(EXAMPLE_ID))


def test_export_is_reproducible_for_fixed_time(
    repository_store: SQLiteRepositoryStore,
    sample_snapshot: object,
) -> None:
    service = InspectionService(
        FakeProvider(snapshots={EXAMPLE_ID: sample_snapshot}),
        repository_store,
        clock=lambda: FIXED_TIME,
    )
    service.inspect(parse_repository_target(EXAMPLE_ID))
    first = serialize_export(
        build_export_payload(
            repository_store.export_records(), exported_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    )
    second = serialize_export(
        build_export_payload(
            repository_store.export_records(), exported_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    )
    assert first == second


def test_batch_all_invalid_does_not_require_a_store() -> None:
    from ethi_asr_bench.services.batch import report_without_inspection

    parsed = parse_batch_text("not valid\nhttps://evil.example/a/b\n")
    report = report_without_inspection(parsed)
    assert report.summary.invalid == 2
    assert report.summary.exit_code == 2
    assert all(item.record is None for item in report.items)


def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == datetime.now(tz=UTC).utcoffset()


def test_snapshot_notes_affect_status(
    repository_store: SQLiteRepositoryStore,
    sample_snapshot: object,
) -> None:
    partial = make_snapshot(notes=["The provider did not return a file listing."])
    service = InspectionService(
        FakeProvider(snapshots={EXAMPLE_ID: partial}),
        repository_store,
        clock=lambda: FIXED_TIME,
    )
    outcome = service.inspect(parse_repository_target(EXAMPLE_ID))
    assert outcome.record.status is InspectionStatus.PARTIAL


def test_provider_cannot_store_metadata_under_a_different_repository(
    repository_store: SQLiteRepositoryStore,
) -> None:
    class MismatchedProvider:
        name = "fake"

        def fetch_repository(self, _ref: object) -> object:
            return make_snapshot(repo_id="someone/else")

    service = InspectionService(MismatchedProvider(), repository_store, clock=lambda: FIXED_TIME)  # type: ignore[arg-type]
    outcome = service.inspect(parse_repository_target(EXAMPLE_ID))

    assert outcome.record.status is InspectionStatus.FAILED
    assert outcome.record.errors[0].code == "invalid_provider_response"
    assert repository_store.get_latest("someone/else") is None
    assert repository_store.get_latest(EXAMPLE_ID) is not None
