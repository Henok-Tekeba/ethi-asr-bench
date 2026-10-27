"""Inspection orchestration: provider fetch, normalization, persistence."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ethi_asr_bench.errors import ErrorCode, ProviderError
from ethi_asr_bench.models.repository import (
    InspectionError,
    InspectionRecord,
    InspectionStatus,
    RepositoryRef,
)
from ethi_asr_bench.providers.base import ProviderSnapshot, RepositoryProvider
from ethi_asr_bench.services.metadata_extractor import (
    build_failed_record,
    extract_metadata,
)
from ethi_asr_bench.storage.repository_store import RepositoryStore
from ethi_asr_bench.utils import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InspectionOutcome:
    """The normalized record for one attempted inspection."""

    ref: RepositoryRef
    record: InspectionRecord

    @property
    def succeeded(self) -> bool:
        return self.record.status is InspectionStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.record.status is InspectionStatus.FAILED


class InspectionService:
    """Coordinates one inspection without knowing CLI or SQL details."""

    def __init__(
        self,
        provider: RepositoryProvider,
        store: RepositoryStore,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._provider = provider
        self._store = store
        self._clock = clock

    def inspect(self, ref: RepositoryRef) -> InspectionOutcome:
        """Inspect one repository and persist either the result or the failure.

        Provider failures are converted into stored failed records. Persistence
        failures propagate: continuing a batch without a working database would be
        misleading.
        """
        try:
            snapshot = self._provider.fetch_repository(ref)
            if snapshot.repo_id.casefold() != ref.repo_id.casefold():
                raise ProviderError(
                    ErrorCode.INVALID_PROVIDER_RESPONSE,
                    "The provider returned metadata for a different repository.",
                )
        except ProviderError as exc:
            logger.warning(
                "Provider inspection failed for %s (code=%s).", ref.repo_id, exc.code.value
            )
            record = build_failed_record(
                ref,
                InspectionError(
                    code=exc.code.value,
                    message=exc.message,
                    retryable=exc.retryable,
                    http_status=exc.http_status,
                ),
                retrieved_at=self._clock(),
                provider=self._provider.name,
            )
            self._store.save_inspection(record)
            return InspectionOutcome(ref=ref, record=record)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error(
                "Unexpected provider failure for %s (type=%s).",
                ref.repo_id,
                type(exc).__name__,
            )
            record = build_failed_record(
                ref,
                InspectionError(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=(
                        "The provider raised an unexpected error "
                        f"({type(exc).__name__}). No metadata was accepted."
                    ),
                ),
                retrieved_at=self._clock(),
                provider=self._provider.name,
            )
            self._store.save_inspection(record)
            return InspectionOutcome(ref=ref, record=record)

        raw_source = build_raw_source(snapshot)
        try:
            record = extract_metadata(snapshot, retrieved_at=self._clock())
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.debug(
                "Unexpected metadata extraction failure for %s.",
                ref.repo_id,
                exc_info=True,
            )
            record = build_failed_record(
                ref,
                InspectionError(
                    code=ErrorCode.EXTRACTION_FAILED.value,
                    message=(
                        "Metadata was retrieved but could not be normalized "
                        f"({type(exc).__name__})."
                    ),
                ),
                retrieved_at=self._clock(),
                provider=snapshot.provider_name,
                resolved_commit=snapshot.resolved_commit,
                provider_version=snapshot.provider_version,
            )
            self._store.save_inspection(record, raw_source=raw_source)
            return InspectionOutcome(ref=ref, record=record)

        self._store.save_inspection(record, raw_source=raw_source)
        logger.info(
            "Inspected %s at %s (status=%s).",
            record.identity.repo_id,
            record.identity.resolved_commit or "unresolved revision",
            record.status.value,
        )
        return InspectionOutcome(ref=ref, record=record)


def build_raw_source(snapshot: ProviderSnapshot) -> dict[str, Any]:
    """Assemble a bounded raw payload for debugging and future reprocessing."""
    return {
        "provider": snapshot.provider_name,
        "provider_version": snapshot.provider_version,
        "requested_revision": snapshot.requested_revision,
        "resolved_commit": snapshot.resolved_commit,
        "model_info": snapshot.raw_model_info,
        "text_files": dict(sorted(snapshot.text_files.items())),
        "text_file_errors": [asdict(error) for error in snapshot.text_file_errors],
        "notes": list(snapshot.notes),
    }
