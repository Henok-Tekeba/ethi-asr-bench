"""Batch orchestration with per-input accounting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ethi_asr_bench.models.repository import (
    InspectionError,
    InspectionRecord,
    InspectionStatus,
)
from ethi_asr_bench.services.inspection import InspectionService
from ethi_asr_bench.services.link_parser import ParsedInput, ParseResult


class BatchItemStatus(StrEnum):
    """Terminal state of one batch line."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class BatchItem:
    """Input-to-result correspondence for one submitted line or argument."""

    raw_input: str
    line_number: int | None
    repo_id: str | None
    status: BatchItemStatus
    record: InspectionRecord | None = None
    error: InspectionError | None = None
    duplicate_of: str | None = None
    duplicate_of_line: int | None = None


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Counts and documented exit-code policy for a batch run."""

    total_inputs: int
    unique_repositories: int
    succeeded: int
    partial: int
    failed: int
    invalid: int
    duplicates: int
    skipped: int

    @property
    def attempted(self) -> int:
        return self.succeeded + self.partial + self.failed

    @property
    def exit_code(self) -> int:
        """2 for unusable input, 1 for any non-successful inspection, else 0."""
        if self.invalid:
            return 2
        if self.failed or self.partial:
            return 1
        return 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_inputs": self.total_inputs,
            "unique_repositories": self.unique_repositories,
            "succeeded": self.succeeded,
            "partial": self.partial,
            "failed": self.failed,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "skipped": self.skipped,
            "attempted": self.attempted,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Ordered item results plus aggregate counts."""

    items: list[BatchItem]
    summary: BatchSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "items": [
                {
                    "input": item.raw_input,
                    "line_number": item.line_number,
                    "repo_id": item.repo_id,
                    "status": item.status.value,
                    "error": item.error.model_dump(exclude_none=True) if item.error else None,
                    "duplicate_of": item.duplicate_of,
                    "duplicate_of_line": item.duplicate_of_line,
                }
                for item in self.items
            ],
            "summary": self.summary.to_dict(),
        }


def run_batch(service: InspectionService, parsed: ParseResult) -> BatchReport:
    """Inspect every valid, unique input, continuing after individual failures."""
    items: list[BatchItem] = []
    for entry in parsed.items:
        if entry.ref is not None and entry.error is None and entry.duplicate_of is None:
            items.append(_inspect_entry(service, entry))
        else:
            items.append(_static_entry(entry))
    return BatchReport(items=items, summary=_summarize(items, skipped=parsed.ignored_lines))


def report_without_inspection(parsed: ParseResult) -> BatchReport:
    """Build a report for input that was entirely rejected before any provider call."""
    if any(
        entry.ref is not None and entry.error is None and entry.duplicate_of is None
        for entry in parsed.items
    ):
        raise ValueError("a valid input requires an inspection service")
    items = [_static_entry(entry) for entry in parsed.items]
    return BatchReport(items=items, summary=_summarize(items, skipped=parsed.ignored_lines))


def _summarize(items: list[BatchItem], *, skipped: int) -> BatchSummary:
    succeeded = sum(1 for item in items if item.status is BatchItemStatus.SUCCESS)
    partial = sum(1 for item in items if item.status is BatchItemStatus.PARTIAL)
    failed = sum(1 for item in items if item.status is BatchItemStatus.FAILED)
    invalid = sum(1 for item in items if item.status is BatchItemStatus.INVALID)
    duplicates = sum(1 for item in items if item.status is BatchItemStatus.DUPLICATE)
    unique = sum(
        1
        for item in items
        if item.status not in {BatchItemStatus.INVALID, BatchItemStatus.DUPLICATE}
    )
    return BatchSummary(
        total_inputs=len(items),
        unique_repositories=unique,
        succeeded=succeeded,
        partial=partial,
        failed=failed,
        invalid=invalid,
        duplicates=duplicates,
        skipped=skipped,
    )


def _static_entry(entry: ParsedInput) -> BatchItem:
    if entry.error is not None:
        return BatchItem(
            raw_input=entry.raw_input,
            line_number=entry.line_number,
            repo_id=None,
            status=BatchItemStatus.INVALID,
            error=entry.error,
        )
    if entry.duplicate_of is not None:
        return BatchItem(
            raw_input=entry.raw_input,
            line_number=entry.line_number,
            repo_id=entry.ref.repo_id if entry.ref is not None else None,
            status=BatchItemStatus.DUPLICATE,
            duplicate_of=entry.duplicate_of,
            duplicate_of_line=entry.duplicate_of_line,
        )
    raise ValueError("parsed input requires an inspection")


def _inspect_entry(service: InspectionService, entry: ParsedInput) -> BatchItem:
    if entry.ref is None:
        raise ValueError("parsed input has no repository reference")
    outcome = service.inspect(entry.ref)
    record = outcome.record
    status = {
        InspectionStatus.SUCCESS: BatchItemStatus.SUCCESS,
        InspectionStatus.PARTIAL: BatchItemStatus.PARTIAL,
        InspectionStatus.FAILED: BatchItemStatus.FAILED,
    }[record.status]
    return BatchItem(
        raw_input=entry.raw_input,
        line_number=entry.line_number,
        repo_id=record.identity.repo_id,
        status=status,
        record=record,
        error=record.errors[0] if record.errors else None,
    )
