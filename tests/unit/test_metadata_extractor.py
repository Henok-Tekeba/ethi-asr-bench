"""Metadata normalization, epistemic states, and classification tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.conftest import FIXED_TIME, make_record, make_snapshot

from ethi_asr_bench.errors import ErrorCode
from ethi_asr_bench.models.repository import (
    InspectionError,
    InspectionRecord,
    InspectionStatus,
    ObservedValue,
    RepositoryRef,
    ValueState,
    WarningSeverity,
)
from ethi_asr_bench.providers.base import ProviderFile, ProviderTextFileError
from ethi_asr_bench.services.metadata_extractor import (
    build_failed_record,
    extract_metadata,
)


def test_structured_and_descriptive_metadata_are_extracted() -> None:
    record = make_record()
    model = record.model
    assert record.status is InspectionStatus.SUCCESS
    assert model.display_name.value == "Shook Medium Amharic"
    assert model.architectures.value == ["WhisperForConditionalGeneration"]
    assert model.model_type.value == "whisper"
    assert model.library_name.value == "transformers"
    assert model.pipeline_tag.value == "automatic-speech-recognition"
    assert model.languages.value == ["am"]
    assert model.licenses.value == ["apache-2.0"]
    assert model.task_tags == ["automatic-speech-recognition"]
    assert model.declared_metrics == ["wer"]
    assert "speech-recognition model for Amharic" in (model.description.value or "")
    assert model.description.source == "model_card.README.md"


def test_file_inventory_is_heuristic_and_weights_are_not_downloaded() -> None:
    record = make_record()
    contents = record.contents
    assert contents.has_readme
    assert contents.has_config_json
    assert contents.has_tokenizer_or_processor
    assert contents.has_weight_files
    assert contents.weight_files == ["model.safetensors"]
    assert "training_args.bin" not in contents.weight_files
    assert contents.known_size_bytes == 3_003_478_386
    assert record.custom_code.code_executed is False


def test_dependency_clues_are_descriptive_only() -> None:
    record = make_record()
    assert record.dependencies.declared_requirements == [
        "transformers==4.40.0",
        "torch>=2.0",
        "soundfile",
    ]
    assert record.dependencies.install_instructions == ["pip install transformers"]
    assert record.dependencies.requirements_files == ["requirements.txt"]
    assert any("auto_model" in clue for clue in record.dependencies.compatibility_clues)
    assert record.custom_code.trust_remote_code_mentioned.value is True
    assert record.custom_code.auto_map_declared.value is False


def test_prose_mention_of_whisper_does_not_create_architecture_claims() -> None:
    snapshot = make_snapshot(
        config=None,
        files=[ProviderFile("README.md", 100, False)],
        card_data={},
        text_files={
            "README.md": "# Model\n\nThis is a Whisper model for Amharic.",
        },
    )
    record = make_record(snapshot)
    assert record.model.architectures.state is ValueState.UNKNOWN
    assert record.model.model_type.state is ValueState.UNKNOWN
    assert record.status is InspectionStatus.SUCCESS


def test_absent_optional_metadata_is_unknown_not_failed() -> None:
    snapshot = make_snapshot(
        files=[],
        card_data={},
        config=None,
        transformers_info={},
        text_files={},
        raw_model_info={"id": "owner/model", "tags": []},
    )
    record = make_record(snapshot)
    assert record.model.architectures.state is ValueState.UNKNOWN
    assert record.model.languages.state is ValueState.UNKNOWN
    assert record.model.licenses.state is ValueState.UNKNOWN
    assert record.model.display_name.state is ValueState.UNKNOWN
    assert record.model.description.state is ValueState.UNKNOWN
    assert record.custom_code.auto_map_declared.state is ValueState.UNKNOWN
    assert record.custom_code.trust_remote_code_mentioned.state is ValueState.UNKNOWN
    assert record.status is InspectionStatus.SUCCESS
    assert record.warnings == []


def test_config_downloaded_file_takes_precedence_over_hub_config() -> None:
    downloaded = {"architectures": ["DownloadedArchitecture"], "model_type": "downloaded"}
    snapshot = make_snapshot(
        config={"architectures": ["HubArchitecture"], "model_type": "hub"},
        text_files={"config.json": json.dumps(downloaded)},
    )
    record = make_record(snapshot)
    assert record.model.architectures.value == ["DownloadedArchitecture"]
    assert record.model.architectures.source == "config.json.architectures"
    assert record.model.model_type.source == "config.json.model_type"


def test_malformed_config_json_falls_back_with_partial_status() -> None:
    snapshot = make_snapshot(text_files={"config.json": "{not json"})
    record = make_record(snapshot)
    assert record.model.architectures.value == ["WhisperForConditionalGeneration"]
    assert record.model.architectures.source == "hub.config.architectures"
    assert record.status is InspectionStatus.PARTIAL
    assert any(warning.code == "config_json_malformed" for warning in record.warnings)


def test_unreadable_config_without_fallback_is_extraction_failed() -> None:
    snapshot = make_snapshot(
        config=None,
        text_files={},
        text_file_errors=[
            ProviderTextFileError(
                filename="config.json",
                code=ErrorCode.FILE_UNAVAILABLE.value,
                message="config.json exceeded the limit.",
            )
        ],
    )
    record = make_record(snapshot)
    assert record.model.architectures.state is ValueState.EXTRACTION_FAILED
    assert record.model.model_type.state is ValueState.EXTRACTION_FAILED
    assert record.status is InspectionStatus.PARTIAL
    assert any(
        warning.code == "provider_text_file_error" and warning.severity is WarningSeverity.WARNING
        for warning in record.warnings
    )


def test_unreadable_readme_marks_description_failed_and_status_partial() -> None:
    snapshot = make_snapshot(
        text_files={},
        text_file_errors=[
            ProviderTextFileError(
                filename="README.md",
                code=ErrorCode.TIMEOUT.value,
                message="timed out",
            )
        ],
    )
    record = make_record(snapshot)
    assert record.model.description.state is ValueState.EXTRACTION_FAILED
    assert record.status is InspectionStatus.PARTIAL
    assert any(warning.code == "provider_text_file_error" for warning in record.warnings)


def test_malformed_architecture_type_is_reported_and_partial() -> None:
    snapshot = make_snapshot(
        config={"architectures": {"not": "a list"}, "model_type": 42},
        text_files={},
    )
    record = make_record(snapshot)
    assert record.model.architectures.state is ValueState.EXTRACTION_FAILED
    assert record.model.model_type.state is ValueState.EXTRACTION_FAILED
    assert record.status is InspectionStatus.PARTIAL
    assert sum(w.code == "malformed_metadata" for w in record.warnings) >= 2


def test_architecture_string_is_normalized_with_warning() -> None:
    snapshot = make_snapshot(
        config={"architectures": "WhisperForConditionalGeneration", "model_type": "whisper"},
        text_files={},
    )
    record = make_record(snapshot)
    assert record.model.architectures.value == ["WhisperForConditionalGeneration"]
    assert record.status is InspectionStatus.PARTIAL


def test_language_tags_use_allowlist_not_arbitrary_short_tags() -> None:
    snapshot = make_snapshot(
        card_data={},
        raw_model_info={
            "id": "owner/model",
            "tags": ["am", "en", "asr", "abc", "license:mit"],
            "pipeline_tag": "automatic-speech-recognition",
        },
    )
    record = make_record(snapshot)
    assert record.model.languages.value == ["am", "en"]
    assert record.model.licenses.value == ["mit"]


def test_license_from_tags_and_card_are_deduplicated() -> None:
    snapshot = make_snapshot(
        card_data={"license": ["MIT"]},
        raw_model_info={"id": "owner/model", "tags": ["license:mit"]},
    )
    record = make_record(snapshot)
    assert record.model.licenses.value == ["MIT"]


def test_amharic_card_language_is_kept_even_if_not_a_hub_tag() -> None:
    snapshot = make_snapshot(
        card_data={"language": "amh"},
        raw_model_info={"id": "owner/model", "tags": []},
    )
    record = make_record(snapshot)
    assert record.model.languages.value == ["amh"]


def test_malformed_language_field_is_reported() -> None:
    snapshot = make_snapshot(
        card_data={"language": {"unexpected": True}},
        raw_model_info={"id": "owner/model", "tags": []},
    )
    record = make_record(snapshot)
    assert record.model.languages.state is ValueState.EXTRACTION_FAILED
    assert record.status is InspectionStatus.PARTIAL


def test_custom_code_and_auto_map_are_detected_not_executed() -> None:
    snapshot = make_snapshot(
        files=[
            ProviderFile("README.md", 100, False),
            ProviderFile("custom_modeling.py", 1000, False),
            ProviderFile("config.json", 100, False),
        ],
        config={
            "architectures": ["CustomModel"],
            "model_type": "custom",
            "auto_map": {"AutoModel": "custom_modeling.CustomModel"},
        },
        text_files={"README.md": "Use trust_remote_code=True."},
    )
    record = make_record(snapshot)
    assert record.contents.has_custom_code is True
    assert record.custom_code.python_files == ["custom_modeling.py"]
    assert record.custom_code.auto_map_declared.value is True
    assert record.custom_code.trust_remote_code_mentioned.value is True
    assert record.custom_code.code_executed is False
    assert record.status is InspectionStatus.SUCCESS
    assert {warning.code for warning in record.warnings} >= {
        "auto_map_declared",
        "trust_remote_code_mentioned",
        "custom_code_present",
    }


def test_unsafe_provider_paths_are_skipped() -> None:
    snapshot = make_snapshot(
        files=[
            ProviderFile("../escape.txt", 10, False),
            ProviderFile("/absolute.txt", 10, False),
            ProviderFile("bad\x1bname.txt", 10, False),
            ProviderFile("bad\x9bname.txt", 10, False),
            ProviderFile("safe/file.txt", 10, False),
        ]
    )
    record = make_record(snapshot)
    assert [file.path for file in record.contents.files] == ["safe/file.txt"]
    assert record.status is InspectionStatus.PARTIAL
    assert any(warning.code == "unsafe_file_path" for warning in record.warnings)


def test_nested_readme_is_not_treated_as_the_root_model_card() -> None:
    snapshot = make_snapshot(
        files=[ProviderFile("docs/README.md", 100, False)],
        text_files={},
        config=None,
    )
    record = make_record(snapshot)
    assert record.contents.has_readme is False
    assert record.contents.readme_filename is None
    assert record.model.description.state is ValueState.UNKNOWN


def test_file_inventory_is_bounded() -> None:
    files = [ProviderFile(f"file-{index:04d}.txt", 1, False) for index in range(2005)]
    record = make_record(make_snapshot(files=files))
    assert len(record.contents.files) == 2000
    assert record.contents.files_truncated is True
    assert record.status is InspectionStatus.PARTIAL


def test_model_card_summary_removes_front_matter_and_badges() -> None:
    readme = """---
language: am
license: mit
---
# Title

[![badge](https://example.invalid/badge.svg)](https://example.invalid)
<!-- hidden -->
This is the first readable paragraph.
"""
    record = make_record(make_snapshot(text_files={"README.md": readme}))
    assert record.model.description.value == "This is the first readable paragraph."


def test_description_is_truncated_and_noted() -> None:
    record = make_record(make_snapshot(text_files={"README.md": "A" * 3000}))
    assert len(record.model.description.value or "") == 2000
    assert "truncated" in (record.model.description.note or "")


def test_requirements_with_options_are_not_treated_as_packages() -> None:
    snapshot = make_snapshot(
        text_files={"requirements.txt": "--extra-index-url https://evil.invalid\nfoo==1\n"}
    )
    record = make_record(snapshot)
    assert record.dependencies.declared_requirements == ["foo==1"]


def test_record_json_round_trip_and_utc_timestamp() -> None:
    record = make_record()
    payload = record.model_dump(mode="json", exclude_none=True)
    assert payload["identity"]["retrieved_at"] == "2026-10-27T12:00:00Z"
    assert payload["schema_version"] == "1.0.0"
    restored = InspectionRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_record_schema_rejects_failed_status_without_error() -> None:
    record = make_record()
    with pytest.raises(ValidationError):
        InspectionRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "status": "failed",
                "errors": [],
            }
        )


def test_failed_record_is_honest() -> None:
    ref = RepositoryRef(
        repo_id="owner/model",
        owner="owner",
        name="model",
        canonical_url="https://huggingface.co/owner/model",
        revision="main",
    )
    record = build_failed_record(
        ref,
        InspectionError(code="repository_not_found", message="not found"),
        retrieved_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
        provider="fake",
    )
    assert record.status is InspectionStatus.FAILED
    assert record.identity.resolved_commit is None
    assert record.identity.requested_revision == "main"
    assert record.errors[0].code == "repository_not_found"
    assert record.model.architectures.state is ValueState.UNKNOWN


def test_observed_value_states_are_exclusive() -> None:
    with pytest.raises(ValidationError):
        ObservedValue[str](state=ValueState.KNOWN, value=None)
    with pytest.raises(ValidationError):
        ObservedValue[str](state=ValueState.UNKNOWN, value="invented")
    with pytest.raises(ValidationError):
        ObservedValue[list[str]](state=ValueState.KNOWN, value=[])
    assert ObservedValue[bool].known(False, source="test").value is False


def test_record_uses_provider_canonical_url_for_future_registries() -> None:
    snapshot = make_snapshot(canonical_url="https://mirror.example/models/owner/model")
    record = make_record(snapshot)
    assert record.identity.canonical_url == "https://mirror.example/models/owner/model"
    assert record.identity.repo_id == "b1n1yam/shook-medium-amharic-2k"


def test_missing_commit_makes_the_record_partial() -> None:
    record = make_record(make_snapshot(resolved_commit=None))
    assert record.identity.resolved_commit is None
    assert record.status is InspectionStatus.PARTIAL
    assert any(warning.code == "commit_unresolved" for warning in record.warnings)


def test_notes_are_preserved_as_warnings() -> None:
    record = make_record(make_snapshot(notes=["The provider did not return a file listing."]))
    assert record.status is InspectionStatus.PARTIAL
    assert record.warnings[0].message == "The provider did not return a file listing."


def test_provenance_records_sources() -> None:
    record = make_record()
    assert record.provenance.provider == "fake"
    assert "README.md" in record.provenance.sources
    assert record.provenance.field_sources["files"].source == "hub.model_info.siblings"
    assert record.model.architectures.source == "config.json.architectures"


def test_extraction_is_deterministic_for_the_same_snapshot() -> None:
    snapshot = make_snapshot()
    first = extract_metadata(snapshot, retrieved_at=FIXED_TIME)
    second = extract_metadata(replace(snapshot), retrieved_at=FIXED_TIME)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_duplicate_valid_tags_do_not_look_like_malformed_metadata() -> None:
    snapshot = make_snapshot(
        raw_model_info={
            "id": "owner/model",
            "tags": ["am", "am", "whisper"],
        },
        card_data={},
        config=None,
        files=[],
        text_files={},
    )
    record = make_record(snapshot)
    assert record.status is InspectionStatus.SUCCESS
    assert not any(w.code == "malformed_metadata" for w in record.warnings)
    assert record.model.tags == ["am", "whisper"]


def test_license_name_is_preserved_but_marked_as_non_identifier() -> None:
    snapshot = make_snapshot(
        card_data={"license_name": "Apache License 2.0"},
        raw_model_info={"id": "owner/model", "tags": []},
    )
    record = make_record(snapshot)
    assert record.model.licenses.value == ["Apache License 2.0"]
    assert "license_name" in (record.model.licenses.source or "")
    assert "not a normalized identifier" in (record.model.licenses.note or "")


def test_listed_but_unreadable_files_make_status_partial_even_without_provider_error() -> None:
    snapshot = make_snapshot(
        config=None,
        text_files={},
        text_file_errors=[],
        files=[
            ProviderFile("README.md", 100, False),
            ProviderFile("config.json", 100, False),
        ],
    )
    record = make_record(snapshot)
    assert record.status is InspectionStatus.PARTIAL
    assert {warning.code for warning in record.warnings} >= {
        "readme_unavailable",
        "config_unavailable",
    }


def test_root_config_flags_do_not_claim_nested_configuration() -> None:
    snapshot = make_snapshot(
        files=[ProviderFile("nested/config.json", 100, False)],
        config=None,
        text_files={},
    )
    record = make_record(snapshot)
    assert record.contents.has_config_json is False
    assert record.contents.config_files == ["config.json"]


def test_unclosed_model_card_front_matter_does_not_become_description() -> None:
    readme = "---\nlanguage: am\nlicense: mit\n# Heading\n\nA real paragraph."
    record = make_record(make_snapshot(text_files={"README.md": readme}))
    assert record.model.description.state is ValueState.UNKNOWN


def test_c1_control_characters_are_removed_from_description() -> None:
    record = make_record(make_snapshot(text_files={"README.md": "Text \x9b31m before."}))
    assert "\x9b" not in (record.model.description.value or "")
