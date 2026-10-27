"""Command-line interface for EthiASR Bench."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ethi_asr_bench import __version__
from ethi_asr_bench.config import Settings
from ethi_asr_bench.errors import (
    ErrorCode,
    EthiAsrBenchError,
    InputError,
    RecordNotFoundError,
    exit_code_for,
)
from ethi_asr_bench.models.compatibility import (
    CompatibilityAssessment,
    CompatibilityClass,
    EvidenceKind,
)
from ethi_asr_bench.models.repository import (
    InspectionRecord,
    ObservedValue,
    RepositorySummary,
    ValueState,
)
from ethi_asr_bench.providers.huggingface import HuggingFaceProvider
from ethi_asr_bench.services.batch import (
    BatchItem,
    BatchItemStatus,
    BatchReport,
    report_without_inspection,
    run_batch,
)
from ethi_asr_bench.services.classifier import classify_record
from ethi_asr_bench.services.exporter import (
    build_export_payload,
    serialize_export,
    write_export,
)
from ethi_asr_bench.services.inspection import InspectionService
from ethi_asr_bench.services.link_parser import (
    parse_batch_text,
    parse_inputs,
    parse_repository_target,
    read_batch_file,
)
from ethi_asr_bench.storage.database import Database
from ethi_asr_bench.storage.repository_store import SQLiteRepositoryStore
from ethi_asr_bench.utils import format_utc, safe_json_dumps, utc_now

app = typer.Typer(
    name="ethi-asr",
    help=(
        "Discover and inspect Amharic ASR model metadata, then classify likely inference "
        "compatibility. No weights are downloaded and no model code is executed."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

stdout_console = Console()
stderr_console = Console(stderr=True)
logger = logging.getLogger(__name__)
_TERMINAL_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
)


def _plain_safe(value: str) -> str:
    """Remove terminal control characters from untrusted text."""
    return _TERMINAL_CONTROL_RE.sub("", value)


def _safe(value: str) -> str:
    """Remove control characters and escape Rich markup in untrusted text."""
    return escape(_plain_safe(value))


@dataclass(frozen=True, slots=True)
class CliState:
    """Validated global options shared by commands."""

    settings: Settings


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    db: Annotated[
        Path | None,
        typer.Option(
            "--db",
            help="SQLite database path (default: $ETHI_ASR_DB_PATH or user data dir).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging (never logs tokens)."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """Global options."""
    if version:
        typer.echo(f"ethi-asr-bench {__version__}")
        raise typer.Exit()
    try:
        settings = Settings.from_env(db_path=db)
    except EthiAsrBenchError as exc:
        _abort(exc)
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level, logging.WARNING)
    _configure_logging(level)
    ctx.obj = CliState(settings=settings)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _configure_logging(level: int) -> None:
    """Enable application debug logging without enabling HTTP client debug logs.

    ``httpx``/``httpcore``/``huggingface_hub`` debug logging can expose sensitive
    request metadata (and Hugging Face's optional curl tracing can expose the
    Authorization header). Keep those libraries at WARNING even when the user asks
    for application debug output.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("ethi_asr_bench").setLevel(level)
    for noisy_logger in ("httpcore", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


@app.command()
def inspect(
    ctx: typer.Context,
    targets: Annotated[
        list[str],
        typer.Argument(metavar="REPOSITORY...", help="Hugging Face model IDs or URLs."),
    ],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Branch, tag, or commit for bare IDs."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Inspect one or more Hugging Face model repositories."""
    state = _state(ctx)
    try:
        if not targets:
            raise InputError(ErrorCode.EMPTY_INPUT, "At least one repository is required.")
        parsed = parse_inputs(targets, revision=revision)
        if not any(entry.is_valid for entry in parsed.items):
            report = report_without_inspection(parsed)
        else:
            with open_service(state.settings) as service:
                report = run_batch(service, parsed)
    except EthiAsrBenchError as exc:
        _abort(exc, json_output=json_output)

    if json_output:
        _print_json(_batch_payload(report, command="inspect"))
    else:
        _render_batch(report, show_single_detail=True)
    raise typer.Exit(code=report.summary.exit_code)


@app.command()
def batch(
    ctx: typer.Context,
    input_file: Annotated[
        Path, typer.Argument(metavar="FILE", help="Text file with one ID or URL per line.")
    ],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Revision applied to every bare repository ID."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Inspect every repository in a text file, continuing after individual failures."""
    state = _state(ctx)
    try:
        content = read_batch_file(input_file)
        parsed = parse_batch_text(content, revision=revision)
        if not any(entry.is_valid for entry in parsed.items):
            report = report_without_inspection(parsed)
        else:
            with open_service(state.settings) as service:
                report = run_batch(service, parsed)
    except EthiAsrBenchError as exc:
        _abort(exc, json_output=json_output)

    if json_output:
        _print_json(_batch_payload(report, command="batch"))
    else:
        _render_batch(report, show_single_detail=False)
    raise typer.Exit(code=report.summary.exit_code)


@app.command("list")
def list_repositories(
    ctx: typer.Context,
    limit: Annotated[
        int, typer.Option("--limit", min=1, max=1000, help="Maximum repositories to show.")
    ] = 50,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """List locally stored repositories and their latest inspection."""
    state = _state(ctx)
    try:
        with open_store(state.settings.db_path) as store:
            summaries = store.list_repositories(limit=limit)
    except EthiAsrBenchError as exc:
        _abort(exc, json_output=json_output)

    if json_output:
        _print_json(
            {
                "repositories": [
                    summary.model_dump(mode="json", exclude_none=True) for summary in summaries
                ],
                "count": len(summaries),
            }
        )
        return
    _render_repository_list(summaries)


@app.command()
def show(
    ctx: typer.Context,
    repository: Annotated[str, typer.Argument(help="Stored repository ID or Hugging Face URL.")],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Filter by requested revision or commit prefix."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the normalized record as JSON.")
    ] = False,
    include_raw: Annotated[
        bool,
        typer.Option("--include-raw", help="Include stored raw provider metadata (with --json)."),
    ] = False,
) -> None:
    """Show the latest stored inspection for one repository."""
    state = _state(ctx)
    if include_raw and not json_output:
        _abort(
            InputError(
                ErrorCode.INVALID_INPUT,
                "--include-raw requires --json so the output remains unambiguous.",
            )
        )
    try:
        ref = parse_repository_target(repository, revision=revision)
        with open_store(state.settings.db_path) as store:
            stored = store.get_latest(ref.repo_id, revision=ref.revision)
        if stored is None:
            raise RecordNotFoundError(
                f"No stored inspection found for {ref.repo_id!r}"
                + (f" at revision {ref.revision!r}." if ref.revision else ".")
            )
    except EthiAsrBenchError as exc:
        _abort(exc, json_output=json_output)

    if json_output:
        record_payload = stored.record.model_dump(mode="json", exclude_none=True)
        if include_raw:
            _print_json({"record": record_payload, "raw_source": stored.raw_source})
        else:
            _print_json(record_payload)
        return
    _render_record(stored.record, show_description=True)


@app.command()
def classify(
    ctx: typer.Context,
    repository: Annotated[str, typer.Argument(help="Stored repository ID or Hugging Face URL.")],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Filter by requested revision or commit prefix."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the compatibility assessment as JSON.")
    ] = False,
) -> None:
    """Classify a stored inspection without loading weights or running the model."""
    state = _state(ctx)
    try:
        ref = parse_repository_target(repository, revision=revision)
        with open_store(state.settings.db_path) as store:
            stored = store.get_latest(ref.repo_id, revision=ref.revision)
        if stored is None:
            raise RecordNotFoundError(
                f"No stored inspection found for {ref.repo_id!r}"
                + (f" at revision {ref.revision!r}." if ref.revision else ".")
            )
    except EthiAsrBenchError as exc:
        _abort(exc, json_output=json_output)

    assessment = classify_record(stored.record)
    if json_output:
        _print_json(assessment.model_dump(mode="json", exclude_none=True))
        return
    _render_assessment(assessment)


@app.command()
def export(
    ctx: typer.Context,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Destination path, or '-' for standard output.",
        ),
    ] = "-",
    export_format: Annotated[
        str,
        typer.Option("--format", help="Export format. Stage 1 supports 'json'."),
    ] = "json",
    include_raw: Annotated[
        bool, typer.Option("--include-raw", help="Include raw provider metadata.")
    ] = False,
    all_history: Annotated[
        bool,
        typer.Option(
            "--all-history", help="Export every inspection instead of the latest per repository."
        ),
    ] = False,
    repository: Annotated[
        str | None,
        typer.Option("--repo", help="Limit export to one repository ID or URL."),
    ] = None,
) -> None:
    """Export stored normalized records as deterministic JSON."""
    state = _state(ctx)
    try:
        if export_format != "json":
            raise InputError(
                ErrorCode.INVALID_INPUT,
                f"Unsupported export format {export_format!r}; only 'json' is available.",
            )
        repo_id: str | None = None
        if repository is not None:
            repo_id = _repository_id_without_revision(repository, purpose="export --repo")
        with open_store(state.settings.db_path) as store:
            records = store.export_records(
                all_history=all_history,
                repo_id=repo_id,
                include_raw=include_raw,
            )
        payload = build_export_payload(records, exported_at=utc_now(), include_raw=include_raw)
        content = serialize_export(payload)
        if output == "-":
            typer.echo(content, nl=False)
            stderr_console.print(
                f"Exported {len(records)} record(s) to standard output.", style="dim"
            )
        else:
            write_export(Path(output), content)
            stderr_console.print(
                f"Exported {len(records)} record(s) to {_safe(output)}.", style="dim"
            )
    except EthiAsrBenchError as exc:
        _abort(exc)


@app.command()
def delete(
    ctx: typer.Context,
    repository: Annotated[
        str, typer.Argument(help="Repository ID or URL whose local history should be deleted.")
    ],
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Delete without interactive confirmation.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Delete all local inspection history for one repository."""
    state = _state(ctx)
    try:
        repo_id = _repository_id_without_revision(repository, purpose="delete")
        if not yes:
            confirmed = typer.confirm(
                f"Delete all local inspection history for {repo_id}?",
                abort=True,
            )
            if not confirmed:  # pragma: no cover - typer.confirm aborts first
                raise typer.Exit(code=1)
        with open_store(state.settings.db_path) as store:
            removed = store.delete_repository(repo_id)
        if removed == 0:
            raise RecordNotFoundError(f"No stored inspection found for {repo_id!r}.")
    except EthiAsrBenchError as exc:
        _abort(exc, json_output=json_output)

    if json_output:
        _print_json({"deleted": True, "repo_id": repo_id, "inspections_removed": removed})
    else:
        stdout_console.print(
            f"Deleted {removed} inspection record(s) for {_safe(repo_id)}.",
            style="green",
        )


def create_provider(settings: Settings) -> HuggingFaceProvider:
    """Construct the configured provider. Patched by tests."""
    return HuggingFaceProvider(
        token=settings.hf_token,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        backoff_seconds=settings.backoff_seconds,
    )


@contextmanager
def open_store(db_path: Path) -> Iterator[SQLiteRepositoryStore]:
    store = SQLiteRepositoryStore(Database(db_path))
    try:
        yield store
    finally:
        store.close()


@contextmanager
def open_service(settings: Settings) -> Iterator[InspectionService]:
    with open_store(settings.db_path) as store:
        yield InspectionService(create_provider(settings), store)


def _state(ctx: typer.Context) -> CliState:
    if not isinstance(ctx.obj, CliState):
        raise RuntimeError("CLI context was not initialized")
    return ctx.obj


def _repository_id_without_revision(raw: str, *, purpose: str) -> str:
    """Reject revision-scoped references for commands that act on all history."""
    ref = parse_repository_target(raw)
    if ref.revision is not None:
        raise InputError(
            ErrorCode.INVALID_INPUT,
            f"{purpose} operates on every stored revision for a repository; "
            "submit the repository ID or URL without a revision.",
        )
    return ref.repo_id


def _abort(error: EthiAsrBenchError, *, json_output: bool = False) -> NoReturn:
    if json_output:
        _print_json({"status": "error", "error": error.to_dict()})
    stderr_console.print(f"[bold red]Error[/bold red] ({error.code.value}): {_safe(error.message)}")
    raise typer.Exit(code=exit_code_for(error))


def _print_json(payload: Any) -> None:
    typer.echo(safe_json_dumps(payload, indent=2))


def _batch_payload(report: BatchReport, *, command: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in report.items:
        results.append(
            {
                "input": item.raw_input,
                "line_number": item.line_number,
                "repo_id": item.repo_id,
                "status": item.status.value,
                "error": item.error.model_dump(exclude_none=True) if item.error else None,
                "duplicate_of": item.duplicate_of,
                "duplicate_of_line": item.duplicate_of_line,
                "record": (
                    item.record.model_dump(mode="json", exclude_none=True)
                    if item.record is not None
                    else None
                ),
            }
        )
    return {
        "command": command,
        "results": results,
        "summary": report.summary.to_dict(),
    }


def _render_batch(report: BatchReport, *, show_single_detail: bool) -> None:
    if show_single_detail and len(report.items) == 1:
        item = report.items[0]
        if item.record is not None:
            _render_record(item.record, show_description=True)
        elif item.status is BatchItemStatus.INVALID and item.error is not None:
            stderr_console.print(
                f"[bold red]Invalid input[/bold red] ({item.error.code}): "
                f"{_safe(item.error.message)}"
            )
        elif item.status is BatchItemStatus.FAILED and item.error is not None:
            stderr_console.print(
                f"[bold red]Inspection failed[/bold red] ({item.error.code}): "
                f"{_safe(item.error.message)}"
            )
        _render_batch_summary(report)
        return

    for item in report.items:
        _render_batch_line(item)
    _render_batch_summary(report)


def _render_batch_line(item: BatchItem) -> None:
    location = f"line {item.line_number}: " if item.line_number is not None else ""
    style, marker = {
        BatchItemStatus.SUCCESS: ("green", "OK"),
        BatchItemStatus.PARTIAL: ("yellow", "PARTIAL"),
        BatchItemStatus.FAILED: ("red", "FAILED"),
        BatchItemStatus.INVALID: ("red", "INVALID"),
        BatchItemStatus.DUPLICATE: ("dim", "DUPLICATE"),
    }[item.status]
    detail = ""
    if item.status is BatchItemStatus.DUPLICATE and item.duplicate_of is not None:
        duplicate_line = (
            f"line {item.duplicate_of_line}"
            if item.duplicate_of_line is not None
            else "earlier input"
        )
        detail = f" (duplicate of {item.duplicate_of} at {duplicate_line})"
    elif item.error is not None:
        detail = f": {item.error.code}: {item.error.message}"
    target = item.repo_id or item.raw_input
    stdout_console.print(f"[{style}]{marker:<9}[/{style}] {_safe(location + target + detail)}")


def _render_batch_summary(report: BatchReport) -> None:
    summary = report.summary
    table = Table(title="Batch summary", show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Succeeded", str(summary.succeeded))
    table.add_row("Partial", str(summary.partial))
    table.add_row("Failed", str(summary.failed))
    table.add_row("Invalid inputs", str(summary.invalid))
    table.add_row("Duplicates", str(summary.duplicates))
    table.add_row("Skipped blank/comment lines", str(summary.skipped))
    table.add_row("Unique repositories attempted", str(summary.unique_repositories))
    stdout_console.print(table)


def _render_repository_list(summaries: list[RepositorySummary]) -> None:
    if not summaries:
        stdout_console.print("No stored inspections. Run 'ethi-asr inspect <repository>' first.")
        return
    table = Table(title="Stored repositories")
    table.add_column("Repository", style="bold")
    table.add_column("Status")
    table.add_column("Commit")
    table.add_column("Retrieved (UTC)")
    table.add_column("Inspections", justify="right")
    table.add_column("Warnings", justify="right")
    for summary in summaries:
        status_style = {
            "success": "green",
            "partial": "yellow",
            "failed": "red",
        }[summary.status.value]
        table.add_row(
            _safe(summary.repo_id),
            Text(summary.status.value, style=status_style),
            (summary.resolved_commit or "-")[:12],
            format_utc(summary.retrieved_at),
            str(summary.inspection_count),
            str(summary.warning_count),
        )
    stdout_console.print(table)


def _render_record(record: InspectionRecord, *, show_description: bool) -> None:
    identity = record.identity
    status_style = {
        "success": "green",
        "partial": "yellow",
        "failed": "red",
    }[record.status.value]
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Repository", Text(_plain_safe(identity.repo_id)))
    table.add_row("URL", Text(_plain_safe(identity.canonical_url)))
    table.add_row("Status", Text(record.status.value, style=status_style))
    table.add_row("Requested revision", Text(_plain_safe(identity.requested_revision or "-")))
    table.add_row("Resolved commit", Text(identity.resolved_commit or "unresolved"))
    table.add_row("Retrieved (UTC)", Text(format_utc(identity.retrieved_at)))
    table.add_row("Display name", Text(_plain_safe(_observed_text(record.model.display_name))))
    table.add_row("Pipeline/task", Text(_plain_safe(_observed_text(record.model.pipeline_tag))))
    table.add_row("Architectures", Text(_plain_safe(_observed_text(record.model.architectures))))
    table.add_row("Model type", Text(_plain_safe(_observed_text(record.model.model_type))))
    table.add_row("Library", Text(_plain_safe(_observed_text(record.model.library_name))))
    table.add_row("Languages", Text(_plain_safe(_observed_text(record.model.languages))))
    table.add_row("Licenses", Text(_plain_safe(_observed_text(record.model.licenses))))
    table.add_row(
        "Files",
        Text(
            f"{len(record.contents.files)} listed, "
            f"{record.contents.known_size_bytes} known bytes, "
            f"weights={'yes' if record.contents.has_weight_files else 'no'}, "
            f"custom code={'yes' if record.contents.has_custom_code else 'no'}"
        ),
    )
    table.add_row("Tags", Text(_plain_safe(", ".join(record.model.tags[:20]) or "none")))
    stdout_console.print(table)

    if show_description and record.model.description.state is ValueState.KNOWN:
        description = record.model.description.value or ""
        stdout_console.print(
            Panel(
                Text(_plain_safe(description)),
                title="Model-card summary (untrusted descriptive text)",
                border_style="dim",
            )
        )
    if record.warnings:
        stdout_console.print("[bold]Warnings[/bold]")
        for warning in record.warnings:
            style = "yellow" if warning.severity.value == "warning" else "dim"
            stdout_console.print(
                f"  [{style}]{_safe(warning.code)}[/{style}]: {_safe(warning.message)}"
            )
    if record.errors:
        stdout_console.print("[bold red]Errors[/bold red]")
        for error in record.errors:
            stdout_console.print(f"  [red]{_safe(error.code)}[/red]: {_safe(error.message)}")


def _render_assessment(assessment: CompatibilityAssessment) -> None:
    style = {
        CompatibilityClass.STANDARD_WHISPER_CANDIDATE: "green",
        CompatibilityClass.KNOWN_VARIATION: "cyan",
        CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED: "yellow",
        CompatibilityClass.INSUFFICIENT_INFORMATION: "dim",
    }[assessment.classification]
    table = Table(title="Compatibility assessment", show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Repository", Text(_plain_safe(assessment.repo_id)))
    table.add_row("Resolved commit", Text(assessment.resolved_commit or "unresolved"))
    table.add_row("Inspection status", Text(assessment.inspection_status.value))
    table.add_row("Inspection schema", Text(assessment.inspection_schema_version))
    table.add_row("Classifier version", Text(assessment.classifier_version))
    table.add_row(
        "Classification",
        Text(assessment.classification.value, style=style),
    )
    stdout_console.print(table)

    stdout_console.print("[bold]Reasons[/bold]")
    for reason in assessment.reasons:
        stdout_console.print(f"  - {_safe(reason)}")

    if assessment.evidence:
        evidence_table = Table(title="Evidence", box=None)
        evidence_table.add_column("Kind")
        evidence_table.add_column("Rule")
        evidence_table.add_column("Signal")
        evidence_table.add_column("Value")
        evidence_table.add_column("Source")
        for item in assessment.evidence:
            kind_style = "cyan" if item.kind is EvidenceKind.OBSERVED else "dim"
            evidence_table.add_row(
                Text(item.kind.value, style=kind_style),
                Text(_plain_safe(item.rule)),
                Text(_plain_safe(item.signal)),
                Text(_plain_safe(item.value)),
                Text(_plain_safe(item.source or "-")),
            )
        stdout_console.print(evidence_table)

    if assessment.warnings:
        stdout_console.print("[bold]Warnings[/bold]")
        for warning in assessment.warnings:
            stdout_console.print(
                f"  [yellow]{_safe(warning.code)}[/yellow]: {_safe(warning.message)}"
            )
    stdout_console.print(
        "[dim]Rule-based assessment only; no model weights were downloaded and no model "
        "code was executed.[/dim]"
    )


def _observed_text(value: ObservedValue[Any]) -> str:
    if value.state is ValueState.KNOWN and value.value is not None:
        raw = value.value
        if isinstance(raw, list):
            text = ", ".join(str(item) for item in raw)
        elif isinstance(raw, bool):
            text = "yes" if raw else "no"
        else:
            text = str(raw)
        if value.note:
            return f"{text} ({value.note})"
        return text
    if value.state is ValueState.EXTRACTION_FAILED:
        return "extraction failed" + (f" ({value.note})" if value.note else "")
    if value.state is ValueState.NOT_APPLICABLE:
        return "not applicable"
    return "unknown" + (f" ({value.note})" if value.note else "")


if __name__ == "__main__":  # pragma: no cover
    app()
