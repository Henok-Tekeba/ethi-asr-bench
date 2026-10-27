"""CLI behavior and exit-code tests using a deterministic fake provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import FakeProvider, make_snapshot
from typer.testing import CliRunner

from ethi_asr_bench import cli
from ethi_asr_bench.errors import ErrorCode, ProviderError

EXAMPLE_ID = "b1n1yam/shook-medium-amharic-2k"
EXAMPLE_URL = f"https://huggingface.co/{EXAMPLE_ID}"
pytestmark = pytest.mark.usefixtures("cli_provider")

runner = CliRunner()


@pytest.fixture
def cli_provider(monkeypatch: pytest.MonkeyPatch, sample_snapshot: object) -> FakeProvider:
    provider = FakeProvider(snapshots={EXAMPLE_ID: sample_snapshot})
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    return provider


def invoke(db: Path, *args: str) -> object:
    return runner.invoke(cli.app, ["--db", str(db), *args])


def test_version_does_not_touch_the_database(tmp_path: Path) -> None:
    result = invoke(tmp_path / "db.sqlite", "--version")
    assert result.exit_code == 0
    assert "ethi-asr-bench 0.1.0" in result.stdout
    assert not (tmp_path / "db.sqlite").exists()


def test_inspect_success_json_and_persistence(tmp_path: Path, cli_provider: FakeProvider) -> None:
    db = tmp_path / "records.db"
    result = invoke(db, "inspect", EXAMPLE_URL, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["command"] == "inspect"
    assert payload["summary"]["succeeded"] == 1
    assert payload["results"][0]["record"]["identity"]["repo_id"] == EXAMPLE_ID
    assert payload["results"][0]["record"]["status"] == "success"
    assert payload["results"][0]["record"]["custom_code"]["code_executed"] is False
    assert len(cli_provider.calls) == 1
    assert db.exists()


def test_inspect_multiple_inputs_deduplicates_without_reinspection(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    result = invoke(tmp_path / "db.sqlite", "inspect", EXAMPLE_ID, EXAMPLE_URL, "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["duplicates"] == 1
    assert len(cli_provider.calls) == 1


def test_inspect_partial_metadata_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ethi_asr_bench.providers.base import ProviderTextFileError

    partial = make_snapshot(
        text_files={},
        text_file_errors=[
            ProviderTextFileError(
                filename="README.md",
                code=ErrorCode.TIMEOUT.value,
                message="timed out",
            )
        ],
    )
    provider = FakeProvider(snapshots={EXAMPLE_ID: partial})
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)

    result = invoke(tmp_path / "db.sqlite", "inspect", EXAMPLE_ID, "--json")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["results"][0]["status"] == "partial"
    assert payload["summary"]["partial"] == 1


def test_inspect_provider_failure_is_stored_and_exits_one(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    cli_provider.errors["owner/missing"] = ProviderError(
        ErrorCode.REPOSITORY_NOT_FOUND, "Repository not found."
    )
    db = tmp_path / "records.db"
    result = invoke(db, "inspect", "owner/missing", "--json")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["results"][0]["status"] == "failed"
    assert payload["results"][0]["error"]["code"] == "repository_not_found"

    show = invoke(db, "show", "owner/missing", "--json")
    assert show.exit_code == 0
    assert json.loads(show.stdout)["status"] == "failed"


def test_invalid_input_exits_two_without_provider_or_database(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    db = tmp_path / "records.db"
    result = invoke(db, "inspect", "https://huggingface.co.attacker.example/a/b", "--json")
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["results"][0]["status"] == "invalid"
    assert payload["results"][0]["error"]["code"] == "unsupported_host"
    assert cli_provider.calls == []
    assert not db.exists()


def test_single_invalid_input_reports_an_invalid_result(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "inspect", "not a repository", "--json")
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["results"][0]["status"] == "invalid"
    assert payload["results"][0]["error"]["code"] == "invalid_input"


def test_conflicting_revision_is_rejected_before_provider(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    result = invoke(
        tmp_path / "records.db",
        "inspect",
        f"{EXAMPLE_URL}/tree/main",
        "--revision",
        "dev",
        "--json",
    )
    assert result.exit_code == 2
    assert cli_provider.calls == []


def test_batch_continues_after_failure_and_reports_summary(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    cli_provider.errors["owner/missing"] = ProviderError(
        ErrorCode.REPOSITORY_NOT_FOUND, "not found"
    )
    batch = tmp_path / "models.txt"
    batch.write_text(
        f"{EXAMPLE_ID}\n{EXAMPLE_URL}\nowner/missing\nnot a repository\n\n# comment\n",
        encoding="utf-8",
    )
    result = invoke(tmp_path / "records.db", "batch", str(batch), "--json")
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    summary = payload["summary"]
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["duplicates"] == 1
    assert summary["invalid"] == 1
    assert summary["skipped"] == 2
    assert len(cli_provider.calls) == 2


def test_batch_without_invalid_inputs_exits_one_on_failure(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    cli_provider.errors["owner/missing"] = ProviderError(
        ErrorCode.REPOSITORY_NOT_FOUND, "not found"
    )
    batch = tmp_path / "models.txt"
    batch.write_text(f"{EXAMPLE_ID}\nowner/missing\n", encoding="utf-8")
    result = invoke(tmp_path / "records.db", "batch", str(batch), "--json")
    assert result.exit_code == 1


def test_batch_missing_file_exits_two(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "batch", str(tmp_path / "nope.txt"), "--json")
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "invalid_batch_file"


def test_list_show_export_and_delete_flow(tmp_path: Path, cli_provider: FakeProvider) -> None:
    db = tmp_path / "records.db"
    assert invoke(db, "inspect", EXAMPLE_ID, "--json").exit_code == 0

    listed = invoke(db, "list", "--json")
    assert listed.exit_code == 0
    listed_payload = json.loads(listed.stdout)
    assert listed_payload["count"] == 1
    assert listed_payload["repositories"][0]["warning_count"] == 3
    assert listed_payload["repositories"][0]["retrieved_at"].endswith("Z")

    shown = invoke(db, "show", EXAMPLE_ID, "--json")
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["identity"]["repo_id"] == EXAMPLE_ID

    exported = tmp_path / "metadata.json"
    export_result = invoke(db, "export", "--output", str(exported), "--include-raw")
    assert export_result.exit_code == 0
    export_payload = json.loads(exported.read_text(encoding="utf-8"))
    assert export_payload["count"] == 1
    assert export_payload["records"][0]["raw_source"]["provider"] == "fake"

    stdout_export = invoke(db, "export")
    assert stdout_export.exit_code == 0
    assert json.loads(stdout_export.stdout)["count"] == 1

    deleted = invoke(db, "delete", EXAMPLE_ID, "--yes", "--json")
    assert deleted.exit_code == 0
    assert json.loads(deleted.stdout)["inspections_removed"] == 1
    assert json.loads(invoke(db, "list", "--json").stdout)["count"] == 0


def test_classify_uses_stored_record_without_network_or_reinspection(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    db = tmp_path / "records.db"
    assert invoke(db, "inspect", EXAMPLE_ID, "--json").exit_code == 0
    calls_after_inspection = len(cli_provider.calls)

    result = invoke(db, "classify", EXAMPLE_ID, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["classification"] == "standard_whisper_candidate"
    assert payload["classifier_version"] == "1.0.0"
    assert payload["inspection_schema_version"] == "1.0.0"
    assert payload["evidence"]
    assert any(item["kind"] == "observed" for item in payload["evidence"])
    assert any(item["kind"] == "assumption" for item in payload["evidence"])
    assert len(cli_provider.calls) == calls_after_inspection


def test_classify_human_output_and_revision_filter(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_URL, "--revision", "main", "--json")

    result = invoke(db, "classify", f"{EXAMPLE_URL}/tree/main")
    assert result.exit_code == 0, result.output
    assert "standard_whisper_candidate" in result.stdout
    assert "Rule-based assessment only" in result.stdout
    assert "Reasons" in result.stdout
    assert "Evidence" in result.stdout


def test_classify_missing_record_exits_one(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "classify", "owner/missing", "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "record_not_found"


def test_show_missing_record_exits_one(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "show", "owner/missing", "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "record_not_found"


def test_delete_missing_record_exits_one(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "delete", "owner/missing", "--yes", "--json")
    assert result.exit_code == 1


def test_show_include_raw_requires_json(tmp_path: Path, cli_provider: FakeProvider) -> None:
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_ID, "--json")
    result = invoke(db, "show", EXAMPLE_ID, "--include-raw")
    assert result.exit_code == 2


def test_export_rejects_unsupported_format(tmp_path: Path, cli_provider: FakeProvider) -> None:
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_ID, "--json")
    result = invoke(db, "export", "--format", "yaml")
    assert result.exit_code == 2


def test_list_empty_database(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "list")
    assert result.exit_code == 0
    assert "No stored inspections" in result.stdout


def test_human_output_does_not_render_model_card_markup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readme = "# Model\n\n[red]This is untrusted[/red] text."
    provider = FakeProvider(
        snapshots={
            EXAMPLE_ID: make_snapshot(text_files={"README.md": readme}),
        }
    )
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_ID)
    result = invoke(db, "show", EXAMPLE_ID)
    assert result.exit_code == 0
    assert "[red]This is untrusted[/red] text." in result.stdout


def test_human_output_strips_terminal_control_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        snapshots={
            EXAMPLE_ID: make_snapshot(
                raw_model_info={
                    "id": EXAMPLE_ID,
                    "tags": ["safe", "\x1b[31mevil"],
                    "pipeline_tag": "automatic-speech-recognition",
                }
            )
        }
    )
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_ID)
    result = invoke(db, "show", EXAMPLE_ID)
    assert result.exit_code == 0
    assert "\x1b[31m" not in result.stdout


def test_show_accepts_url_and_revision_filters(tmp_path: Path, cli_provider: FakeProvider) -> None:
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_URL, "--revision", "main", "--json")
    result = invoke(db, "show", f"{EXAMPLE_URL}/tree/main", "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["identity"]["requested_revision"] == "main"


def test_export_and_delete_reject_revision_pinned_references(
    tmp_path: Path, cli_provider: FakeProvider
) -> None:
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_URL, "--revision", "main", "--json")

    export_result = invoke(db, "export", "--repo", f"{EXAMPLE_URL}/tree/main")
    assert export_result.exit_code == 2
    assert "every stored revision" in export_result.stderr

    delete_result = invoke(db, "delete", f"{EXAMPLE_URL}/tree/main", "--yes", "--json")
    assert delete_result.exit_code == 2
    assert json.loads(delete_result.stdout)["error"]["code"] == "invalid_input"
    assert json.loads(invoke(db, "list", "--json").stdout)["count"] == 1


def test_verbose_logging_does_not_enable_http_client_debug(tmp_path: Path) -> None:
    import logging

    result = runner.invoke(cli.app, ["--db", str(tmp_path / "records.db"), "--verbose", "list"])
    assert result.exit_code == 0
    assert logging.getLogger("ethi_asr_bench").getEffectiveLevel() == logging.DEBUG
    for name in ("httpcore", "httpx", "huggingface_hub", "urllib3"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


def test_missing_arguments_are_usage_errors(tmp_path: Path) -> None:
    inspect_result = runner.invoke(cli.app, ["--db", str(tmp_path / "db"), "inspect"])
    assert inspect_result.exit_code == 2
    assert "Missing argument" in inspect_result.stderr

    show_result = runner.invoke(cli.app, ["--db", str(tmp_path / "db"), "show"])
    assert show_result.exit_code == 2
    assert "Missing argument" in show_result.stderr


def test_list_rejects_out_of_range_limit(tmp_path: Path) -> None:
    result = invoke(tmp_path / "records.db", "list", "--limit", "0")
    assert result.exit_code == 2
    assert "range" in result.stderr


def test_human_output_strips_c1_control_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        snapshots={
            EXAMPLE_ID: make_snapshot(
                raw_model_info={
                    "id": EXAMPLE_ID,
                    "tags": ["safe", "\x9b31mevil", "\u202eevil"],
                    "pipeline_tag": "automatic-speech-recognition",
                }
            )
        }
    )
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    db = tmp_path / "records.db"
    invoke(db, "inspect", EXAMPLE_ID)
    result = invoke(db, "show", EXAMPLE_ID)
    assert result.exit_code == 0
    assert "\x9b" not in result.stdout
    assert "\u202e" not in result.stdout


def test_database_path_that_is_a_directory_is_a_clean_storage_error(tmp_path: Path) -> None:
    result = invoke(tmp_path, "list", "--json")
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "storage_initialization_failed"
    assert "Traceback" not in result.stderr
