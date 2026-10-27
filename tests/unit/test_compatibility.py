"""Deterministic compatibility classification tests (no network, no model loading)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from tests.conftest import FIXED_TIME, make_record, make_snapshot

from ethi_asr_bench.models.compatibility import (
    CLASSIFIER_VERSION,
    CompatibilityAssessment,
    CompatibilityClass,
    CompatibilityEvidence,
    EvidenceKind,
)
from ethi_asr_bench.models.repository import (
    InspectionError,
    InspectionStatus,
    RepositoryRef,
)
from ethi_asr_bench.providers.base import ProviderFile
from ethi_asr_bench.services.classifier import classify_record
from ethi_asr_bench.services.metadata_extractor import build_failed_record


def model_info(
    *,
    pipeline: str | None = "automatic-speech-recognition",
    library: str | None = "transformers",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"id": "owner/model", "tags": tags or []}
    if pipeline is not None:
        value["pipeline_tag"] = pipeline
    if library is not None:
        value["library_name"] = library
    return value


def record_with_config(config: dict[str, Any], **overrides: Any):
    overrides.setdefault("files", [])
    overrides.setdefault("text_files", {})
    overrides.setdefault("raw_model_info", model_info())
    return make_record(make_snapshot(config=config, **overrides))


def test_standard_whisper_record_is_a_candidate_with_observed_evidence() -> None:
    record = make_record()
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE
    assert assessment.classifier_version == CLASSIFIER_VERSION
    assert assessment.repo_id == record.identity.repo_id
    assert assessment.resolved_commit == record.identity.resolved_commit
    assert any(item.kind is EvidenceKind.OBSERVED for item in assessment.evidence)
    assert any(item.kind is EvidenceKind.ASSUMPTION for item in assessment.evidence)
    assert any(item.signal == "model.model_type" for item in assessment.evidence)
    assert any(item.signal == "custom_code.auto_map_declared" for item in assessment.evidence)
    assert any("not proven" in reason for reason in assessment.reasons)


def test_whisper_architecture_without_model_type_is_still_a_candidate() -> None:
    record = record_with_config({"architectures": ["WhisperForConditionalGeneration"]})
    assessment = classify_record(record)
    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE


def test_whisper_model_type_without_architecture_is_still_a_candidate() -> None:
    record = record_with_config({"model_type": "whisper"})
    assessment = classify_record(record)
    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE


def test_known_non_whisper_asr_family_is_a_known_variation() -> None:
    record = record_with_config({"architectures": ["Wav2Vec2ForCTC"], "model_type": "wav2vec2"})
    assessment = classify_record(record)
    assert assessment.classification is CompatibilityClass.KNOWN_VARIATION
    assert any(item.signal == "model.architectures" for item in assessment.evidence)
    assert any("family-specific" in reason for reason in assessment.reasons)


def test_known_variation_can_be_resolved_from_model_type_alone() -> None:
    record = record_with_config({"model_type": "mms"})
    assessment = classify_record(record)
    assert assessment.classification is CompatibilityClass.KNOWN_VARIATION


def test_missing_structured_metadata_is_insufficient_not_incompatible() -> None:
    snapshot = make_snapshot(
        files=[],
        config=None,
        text_files={},
        card_data={},
        transformers_info={},
        raw_model_info={"id": "owner/model", "tags": []},
    )
    assessment = classify_record(make_record(snapshot))

    assert assessment.classification is CompatibilityClass.INSUFFICIENT_INFORMATION
    assert any(warning.code == "structured_metadata_missing" for warning in assessment.warnings)
    assert any(
        "not treated as evidence of incompatibility" in reason for reason in assessment.reasons
    )
    assert not any(item.kind is EvidenceKind.ASSUMPTION for item in assessment.evidence)


def test_malformed_structured_metadata_is_insufficient_and_flagged() -> None:
    record = record_with_config({"architectures": {"not": "a list"}, "model_type": 42})
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.INSUFFICIENT_INFORMATION
    assert any(warning.code == "structured_metadata_malformed" for warning in assessment.warnings)
    assert assessment.inspection_status is InspectionStatus.PARTIAL


def test_failed_inspection_is_insufficient_with_the_failure_recorded() -> None:
    ref = RepositoryRef(
        repo_id="owner/model",
        owner="owner",
        name="model",
        canonical_url="https://huggingface.co/owner/model",
    )
    record = build_failed_record(
        ref,
        InspectionError(code="repository_not_found", message="not found"),
        retrieved_at=FIXED_TIME,
        provider="fake",
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.INSUFFICIENT_INFORMATION
    assert any(item.signal == "inspection.errors" for item in assessment.evidence)
    assert any("repository_not_found" in item.value for item in assessment.evidence)
    assert any("failure is not evidence" in reason for reason in assessment.reasons)


def test_unsupported_architecture_requires_a_custom_implementation() -> None:
    record = record_with_config({"architectures": ["AcmeSpeechModel"], "model_type": "acme_speech"})
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED
    assert any(warning.code == "unsupported_architecture" for warning in assessment.warnings)
    assert not record.custom_code.code_executed


def test_explicit_non_asr_pipeline_for_unsupported_architecture_is_custom() -> None:
    record = record_with_config(
        {"architectures": ["BertForMaskedLM"], "model_type": "bert"},
        raw_model_info=model_info(pipeline="fill-mask"),
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED
    assert any(warning.code == "non_asr_task" for warning in assessment.warnings)


def test_auto_map_forces_custom_implementation_even_for_whisper_config() -> None:
    record = record_with_config(
        {
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
            "auto_map": {"AutoModel": "custom_modeling.CustomWhisper"},
        }
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED
    assert any(warning.code == "auto_map_declared" for warning in assessment.warnings)
    assert not record.custom_code.code_executed


def test_python_files_without_auto_map_do_not_force_custom_classification() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        files=[ProviderFile("training_script.py", 100, False)],
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE
    assert any(warning.code == "custom_code_present" for warning in assessment.warnings)


def test_trust_remote_code_prose_is_a_warning_not_a_custom_verdict() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        text_files={"README.md": "Use trust_remote_code=True when loading."},
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE
    assert any(warning.code == "trust_remote_code_mentioned" for warning in assessment.warnings)


def test_architecture_and_model_type_conflict_is_insufficient() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "wav2vec2"}
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.INSUFFICIENT_INFORMATION
    assert any(warning.code == "structured_signals_conflict" for warning in assessment.warnings)


def test_conflicting_architecture_list_is_insufficient() -> None:
    record = record_with_config(
        {
            "architectures": ["WhisperForConditionalGeneration", "Wav2Vec2ForCTC"],
            "model_type": "whisper",
        }
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.INSUFFICIENT_INFORMATION
    assert any(warning.code == "structured_signals_conflict" for warning in assessment.warnings)


def test_task_and_architecture_conflict_is_insufficient() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        raw_model_info=model_info(pipeline="text-generation"),
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.INSUFFICIENT_INFORMATION
    assert any(warning.code == "task_architecture_conflict" for warning in assessment.warnings)


def test_whisper_with_non_transformers_library_is_a_known_variation() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        raw_model_info=model_info(library="tensorflow"),
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.KNOWN_VARIATION
    assert any(warning.code == "non_transformers_library" for warning in assessment.warnings)


def test_missing_task_and_library_are_assumptions_not_conflicts() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        raw_model_info=model_info(pipeline=None, library=None),
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE
    assert {warning.code for warning in assessment.warnings} >= {
        "task_not_declared",
        "library_not_declared",
    }
    assert any(item.rule == "classification.task_assumption" for item in assessment.evidence)
    assert any(item.rule == "classification.library_assumption" for item in assessment.evidence)


def test_asr_task_tag_without_pipeline_is_recorded_as_evidence() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        raw_model_info=model_info(
            pipeline=None,
            tags=["automatic-speech-recognition"],
        ),
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE
    assert any(item.signal == "model.task_tags" for item in assessment.evidence)


def test_missing_weight_and_tokenizer_files_are_warnings_not_incompatibility() -> None:
    record = record_with_config(
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        files=[],
    )
    assessment = classify_record(record)

    assert assessment.classification is CompatibilityClass.STANDARD_WHISPER_CANDIDATE
    assert {warning.code for warning in assessment.warnings} >= {
        "weights_not_advertised",
        "tokenizer_not_advertised",
    }


def test_classification_is_deterministic_and_does_not_mutate_the_record() -> None:
    record = make_record()
    before = record.model_dump(mode="json")

    first = classify_record(record)
    second = classify_record(record)

    assert first == second
    assert record.model_dump(mode="json") == before


def test_assessment_json_round_trip() -> None:
    assessment = classify_record(make_record())
    restored = CompatibilityAssessment.model_validate_json(assessment.model_dump_json())
    assert restored == assessment


def test_assessment_requires_observed_evidence_and_reasons() -> None:
    with pytest.raises(ValidationError):
        CompatibilityAssessment(
            classification=CompatibilityClass.INSUFFICIENT_INFORMATION,
            repo_id="owner/model",
            inspection_schema_version="1.0.0",
            inspection_status=InspectionStatus.SUCCESS,
            reasons=["synthetic"],
            evidence=[
                CompatibilityEvidence(
                    rule="synthetic",
                    kind=EvidenceKind.ASSUMPTION,
                    signal="synthetic",
                    value="synthetic",
                    message="synthetic",
                )
            ],
        )
