"""Deterministic compatibility classification from inspected metadata.

The classifier consumes a validated :class:`InspectionRecord`. It does not call a
provider, resolve auto_map entries, fetch model files, install dependencies, load
weights, or execute repository code.

Rules are intentionally conservative:

* structured architecture, model_type, task, and library declarations come first;
* model-card prose is never used as architecture evidence;
* unknown metadata produces ``insufficient_information``, never a negative verdict;
* conflicting structured signals produce ``insufficient_information``;
* ``auto_map`` is the only custom-code signal that can force a custom verdict on
  its own.
"""

from __future__ import annotations

from enum import StrEnum

from ethi_asr_bench.models.compatibility import (
    CLASSIFIER_VERSION,
    CompatibilityAssessment,
    CompatibilityClass,
    CompatibilityEvidence,
    CompatibilityWarning,
    EvidenceKind,
)
from ethi_asr_bench.models.repository import (
    InspectionRecord,
    InspectionStatus,
    ObservedValue,
    ValueState,
)

ASR_TASK = "automatic-speech-recognition"
TRANSFORMERS_LIBRARY = "transformers"

WHISPER_ARCHITECTURES = frozenset(
    {
        "whisperforconditionalgeneration",
    }
)
WHISPER_MODEL_TYPES = frozenset({"whisper"})

KNOWN_VARIATION_ARCHITECTURES = frozenset(
    {
        "data2vecaudioforctc",
        "hubertforctc",
        "mctctforctc",
        "seamlessm4tforspeechtotext",
        "seamlessm4tv2forspeechtotext",
        "speechencoderdecodermodel",
        "speech2textforconditionalgeneration",
        "unispeechforctc",
        "unispeechsatforctc",
        "wav2vec2conformerforctc",
        "wav2vec2forctc",
        "wavlmforctc",
    }
)
KNOWN_VARIATION_MODEL_TYPES = frozenset(
    {
        "data2vec-audio",
        "hubert",
        "mctct",
        "mms",
        "seamless_m4t",
        "seamless_m4t_v2",
        "speech_to_text",
        "speech_to_text_2",
        "unispeech",
        "unispeech-sat",
        "wav2vec2",
        "wav2vec2-bert",
        "wav2vec2-conformer",
        "wavlm",
    }
)


class _Family(StrEnum):
    """Internal family resolution result."""

    WHISPER = "whisper"
    KNOWN_VARIATION = "known_variation"
    UNSUPPORTED = "unsupported"


def classify_record(record: InspectionRecord) -> CompatibilityAssessment:
    """Classify one inspection record without changing it.

    The result is deterministic for the same record and classifier version.
    """
    evidence: list[CompatibilityEvidence] = []
    warnings: list[CompatibilityWarning] = []

    def add(
        rule: str,
        kind: EvidenceKind,
        signal: str,
        value: str,
        message: str,
        *,
        source: str | None = None,
    ) -> None:
        evidence.append(
            CompatibilityEvidence(
                rule=rule,
                kind=kind,
                signal=signal,
                value=value,
                source=source,
                message=message,
            )
        )

    def warn(code: str, message: str) -> None:
        warning = CompatibilityWarning(code=code, message=message)
        if all(item.code != warning.code or item.message != warning.message for item in warnings):
            warnings.append(warning)

    def finish(
        classification: CompatibilityClass,
        reasons: list[str],
    ) -> CompatibilityAssessment:
        return CompatibilityAssessment(
            classifier_version=CLASSIFIER_VERSION,
            classification=classification,
            repo_id=record.identity.repo_id,
            resolved_commit=record.identity.resolved_commit,
            inspection_schema_version=record.schema_version,
            inspection_status=record.status,
            reasons=reasons,
            evidence=evidence,
            warnings=warnings,
        )

    add(
        "inspection.status",
        EvidenceKind.OBSERVED,
        "inspection.status",
        record.status.value,
        "Inspection status is recorded in the Stage 1 record.",
    )
    if record.status is InspectionStatus.PARTIAL:
        warn(
            "inspection_partial",
            "The inspection is partial; classification uses only the structured metadata "
            "that was available.",
        )

    architectures = _observed_list(record.model.architectures)
    model_type = _observed_string(record.model.model_type)
    pipeline_tag = _observed_string(record.model.pipeline_tag)
    library_name = _observed_string(record.model.library_name)

    add(
        "model.architectures",
        EvidenceKind.OBSERVED,
        "model.architectures",
        ", ".join(architectures) if architectures else record.model.architectures.state.value,
        (
            "Structured architecture classes were observed."
            if architectures
            else "No known architecture class was observed."
        ),
        source=record.model.architectures.source,
    )
    add(
        "model.model_type",
        EvidenceKind.OBSERVED,
        "model.model_type",
        model_type or record.model.model_type.state.value,
        (
            "A structured model_type was observed."
            if model_type
            else "No known model_type was observed."
        ),
        source=record.model.model_type.source,
    )
    add(
        "model.pipeline_tag",
        EvidenceKind.OBSERVED,
        "model.pipeline_tag",
        pipeline_tag or record.model.pipeline_tag.state.value,
        (
            "A task/pipeline declaration was observed."
            if pipeline_tag
            else "No task/pipeline declaration was observed."
        ),
        source=record.model.pipeline_tag.source,
    )
    add(
        "model.library_name",
        EvidenceKind.OBSERVED,
        "model.library_name",
        library_name or record.model.library_name.state.value,
        (
            "A library declaration was observed."
            if library_name
            else "No library declaration was observed."
        ),
        source=record.model.library_name.source,
    )
    asr_task_tags = [tag for tag in record.model.task_tags if tag.casefold() == ASR_TASK]
    if pipeline_tag is None and asr_task_tags:
        add(
            "model.task_tags",
            EvidenceKind.OBSERVED,
            "model.task_tags",
            ", ".join(record.model.task_tags),
            "An ASR task tag was observed without a pipeline_tag declaration.",
        )
    add(
        "custom_code.auto_map_declared",
        EvidenceKind.OBSERVED,
        "custom_code.auto_map_declared",
        _observed_bool_text(record.custom_code.auto_map_declared),
        "auto_map is a structured configuration signal for repository-provided code.",
        source=record.custom_code.auto_map_declared.source,
    )

    if record.model.architectures.state is ValueState.EXTRACTION_FAILED:
        warn(
            "structured_metadata_malformed",
            "Architecture metadata was present but could not be normalized.",
        )
    if record.model.model_type.state is ValueState.EXTRACTION_FAILED:
        warn(
            "structured_metadata_malformed",
            "Model-type metadata was present but could not be normalized.",
        )
    if record.model.pipeline_tag.state is ValueState.EXTRACTION_FAILED:
        warn(
            "structured_metadata_malformed",
            "Task/pipeline metadata was present but could not be normalized.",
        )
    if record.model.library_name.state is ValueState.EXTRACTION_FAILED:
        warn(
            "structured_metadata_malformed",
            "Library metadata was present but could not be normalized.",
        )
    if record.custom_code.auto_map_declared.state is ValueState.EXTRACTION_FAILED:
        warn(
            "structured_metadata_malformed",
            "The auto_map declaration could not be normalized.",
        )

    if record.status is InspectionStatus.FAILED:
        error_codes = ", ".join(error.code for error in record.errors) or "unknown"
        add(
            "inspection.errors",
            EvidenceKind.OBSERVED,
            "inspection.errors",
            error_codes,
            "The inspection failed before usable structured inference metadata was stored.",
        )
        return finish(
            CompatibilityClass.INSUFFICIENT_INFORMATION,
            [
                "The inspection failed, so inference compatibility cannot be assessed.",
                "A failure is not evidence that the model is incompatible.",
            ],
        )

    if _auto_map_declared(record):
        warn(
            "auto_map_declared",
            "The repository declares auto_map entries. Stage 2 does not resolve, "
            "download, or execute repository-provided code.",
        )
        add(
            "classification.custom_auto_map",
            EvidenceKind.ASSUMPTION,
            "classification",
            CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED.value,
            "Loading this repository may require a custom implementation and a separately "
            "designed isolation boundary. That path has not been implemented or tested.",
        )
        return finish(
            CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED,
            [
                "The repository declares auto_map, which can require repository-provided code.",
                "The standard Whisper loading path cannot be assumed.",
            ],
        )

    architecture_family, architecture_conflict = _architecture_family(architectures)
    type_family = _family_for_model_type(model_type) if model_type else None

    if architecture_conflict:
        warn(
            "structured_signals_conflict",
            "The architecture list resolves to more than one model family.",
        )
        add(
            "classification.architecture_conflict",
            EvidenceKind.OBSERVED,
            "model.architectures",
            ", ".join(architectures),
            "The architecture entries do not agree on a single known family.",
        )
        return finish(
            CompatibilityClass.INSUFFICIENT_INFORMATION,
            [
                "Architecture metadata conflicts within itself.",
                "No compatibility verdict is asserted from conflicting architecture signals.",
            ],
        )

    if (
        architecture_family is not None
        and type_family is not None
        and architecture_family is not type_family
    ):
        warn(
            "structured_signals_conflict",
            "Architecture and model_type declarations resolve to different model families.",
        )
        add(
            "classification.architecture_model_type_conflict",
            EvidenceKind.OBSERVED,
            "model.architectures + model.model_type",
            f"{', '.join(architectures)} / {model_type}",
            "Architecture and model_type do not agree.",
        )
        return finish(
            CompatibilityClass.INSUFFICIENT_INFORMATION,
            [
                "Architecture and model_type metadata conflict.",
                "No compatibility verdict is asserted from conflicting structured signals.",
            ],
        )

    family = architecture_family or type_family
    if family is None:
        if (
            record.model.architectures.state is ValueState.EXTRACTION_FAILED
            or record.model.model_type.state is ValueState.EXTRACTION_FAILED
        ):
            warn(
                "structured_metadata_malformed",
                "Structured architecture/model_type metadata was present but malformed.",
            )
        else:
            warn(
                "structured_metadata_missing",
                "Neither architecture nor model_type is available in structured metadata.",
            )
        return finish(
            CompatibilityClass.INSUFFICIENT_INFORMATION,
            [
                "No usable structured architecture or model_type is available.",
                "Missing metadata is not treated as evidence of incompatibility.",
            ],
        )

    if family is _Family.UNSUPPORTED:
        warn(
            "unsupported_architecture",
            "The structured architecture/model_type is outside the current ASR family allowlist.",
        )
        add(
            "classification.unsupported_architecture",
            EvidenceKind.ASSUMPTION,
            "classification",
            CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED.value,
            "Using this model for ASR would require a custom or additional implementation. "
            "No repository code was downloaded or executed.",
        )
        reasons = [
            "Structured architecture/model_type is not in the current ASR family allowlist.",
            "The standard Whisper path cannot be assumed for this model.",
        ]
        if pipeline_tag is not None and pipeline_tag.casefold() != ASR_TASK:
            warn(
                "non_asr_task",
                f"The declared pipeline is {pipeline_tag!r}, not {ASR_TASK!r}.",
            )
            reasons.append("The declared pipeline is not automatic speech recognition.")
        return finish(CompatibilityClass.CUSTOM_IMPLEMENTATION_REQUIRED, reasons)

    task_asr = _task_is_asr(record)
    if task_asr is None:
        warn(
            "task_not_declared",
            "No ASR pipeline or task tag was declared; ASR is inferred from the architecture.",
        )
        add(
            "classification.task_assumption",
            EvidenceKind.ASSUMPTION,
            "model.pipeline_tag",
            "automatic-speech-recognition",
            "The classifier assumes an ASR-style inference task because the structured "
            "architecture is an ASR family.",
        )
    elif task_asr is False:
        warn(
            "task_architecture_conflict",
            "The declared pipeline is not ASR, but the structured architecture is an ASR family.",
        )
        return finish(
            CompatibilityClass.INSUFFICIENT_INFORMATION,
            [
                "Task and architecture metadata conflict.",
                "No compatibility verdict is asserted from conflicting structured signals.",
            ],
        )

    library_variation = library_name is not None and library_name.casefold() != TRANSFORMERS_LIBRARY
    if library_name is None:
        warn(
            "library_not_declared",
            "No library declaration was observed; a standard Transformers path is assumed.",
        )
        add(
            "classification.library_assumption",
            EvidenceKind.ASSUMPTION,
            "model.library_name",
            TRANSFORMERS_LIBRARY,
            "The classifier assumes standard Transformers loading because no library was "
            "declared. This has not been verified.",
        )
    elif library_variation:
        warn(
            "non_transformers_library",
            f"The declared library is {library_name!r}, not {TRANSFORMERS_LIBRARY!r}.",
        )

    add(
        "contents.weight_files",
        EvidenceKind.OBSERVED,
        "contents.weight_files",
        f"{len(record.contents.weight_files)} weight-like file(s)",
        "Weight-like files are inventory entries only; none were downloaded.",
    )
    if not record.contents.has_weight_files:
        warn(
            "weights_not_advertised",
            "The file inventory does not advertise weight files; inference assets were not "
            "verified.",
        )
    if not record.contents.has_tokenizer_or_processor:
        warn(
            "tokenizer_not_advertised",
            "The file inventory does not advertise tokenizer or processor files; inference "
            "assets were not verified.",
        )
    if record.contents.files_truncated:
        warn(
            "file_inventory_truncated",
            "The file inventory was truncated; absent files are not proof of incompatibility.",
        )
    if record.custom_code.python_files:
        add(
            "custom_code.python_files",
            EvidenceKind.OBSERVED,
            "custom_code.python_files",
            f"{len(record.custom_code.python_files)} Python file(s)",
            "Python files are inventory entries only; they were not downloaded or executed.",
        )
        warn(
            "custom_code_present",
            "Python files are present, but no auto_map declaration was observed. Python "
            "files alone are not treated as requiring a custom inference path.",
        )
    if (
        record.custom_code.trust_remote_code_mentioned.state is ValueState.KNOWN
        and record.custom_code.trust_remote_code_mentioned.value
    ):
        add(
            "custom_code.trust_remote_code_mentioned",
            EvidenceKind.OBSERVED,
            "custom_code.trust_remote_code_mentioned",
            "true",
            "The model card mentions trust_remote_code; this is an untrusted hint, not proof "
            "that custom code is required.",
        )
        warn(
            "trust_remote_code_mentioned",
            "The model card mentions trust_remote_code. This does not by itself change the "
            "classification.",
        )

    if record.custom_code.auto_map_declared.state is not ValueState.KNOWN:
        warn(
            "auto_map_not_available",
            "No usable auto_map declaration was available; a standard loader is assumed.",
        )
        add(
            "classification.auto_map_assumption",
            EvidenceKind.ASSUMPTION,
            "custom_code.auto_map_declared",
            "false",
            "The classifier assumes no custom auto_map loader is required because the "
            "declaration is unavailable. This has not been verified.",
        )

    if family is _Family.WHISPER:
        if library_variation:
            add(
                "classification.library_variation",
                EvidenceKind.ASSUMPTION,
                "classification",
                CompatibilityClass.KNOWN_VARIATION.value,
                "A non-Transformers Whisper runtime may be supported by a known adapter, but "
                "no adapter has been implemented or tested.",
            )
            return finish(
                CompatibilityClass.KNOWN_VARIATION,
                [
                    "Structured metadata declares Whisper, but the declared library is not "
                    "Transformers.",
                    "The standard Transformers Whisper path is not established for this record.",
                ],
            )

        add(
            "classification.standard_whisper_runtime",
            EvidenceKind.ASSUMPTION,
            "classification",
            CompatibilityClass.STANDARD_WHISPER_CANDIDATE.value,
            "A standard Transformers Whisper ASR runtime is assumed to be able to load this "
            "configuration. No weights were downloaded and no inference was run.",
        )
        return finish(
            CompatibilityClass.STANDARD_WHISPER_CANDIDATE,
            [
                "Structured metadata declares a Whisper architecture or Whisper model_type.",
                "No auto_map custom loader declaration was observed.",
                "This is a candidate for a standard Whisper path; compatibility is not proven.",
            ],
        )

    add(
        "classification.known_variation_adapter",
        EvidenceKind.ASSUMPTION,
        "classification",
        CompatibilityClass.KNOWN_VARIATION.value,
        "A family-specific ASR adapter may support this known family. No adapter has been "
        "implemented or tested.",
    )
    return finish(
        CompatibilityClass.KNOWN_VARIATION,
        [
            "Structured metadata matches a known non-Whisper ASR family.",
            "A family-specific inference adapter would be required; none has been run.",
            "This is a known variation, not a standard Whisper candidate.",
        ],
    )


def _observed_string(value: ObservedValue[str]) -> str | None:
    if value.state is ValueState.KNOWN and isinstance(value.value, str) and value.value:
        return value.value
    return None


def _observed_list(value: ObservedValue[list[str]]) -> list[str]:
    if value.state is ValueState.KNOWN and isinstance(value.value, list):
        return [item for item in value.value if isinstance(item, str) and item]
    return []


def _observed_bool_text(value: ObservedValue[bool]) -> str:
    if value.state is ValueState.KNOWN and isinstance(value.value, bool):
        return "true" if value.value else "false"
    return value.state.value


def _auto_map_declared(record: InspectionRecord) -> bool:
    auto_map = record.custom_code.auto_map_declared
    return auto_map.state is ValueState.KNOWN and auto_map.value is True


def _task_is_asr(record: InspectionRecord) -> bool | None:
    pipeline = record.model.pipeline_tag
    if pipeline.state is ValueState.KNOWN and pipeline.value:
        return pipeline.value.casefold() == ASR_TASK
    if any(tag.casefold() == ASR_TASK for tag in record.model.task_tags):
        return True
    return None


def _architecture_family(values: list[str]) -> tuple[_Family | None, bool]:
    if not values:
        return None, False
    families = {_family_for_architecture(value) for value in values}
    if len(families) > 1:
        return None, True
    return families.pop(), False


def _family_for_architecture(value: str) -> _Family:
    normalized = value.casefold()
    if normalized in WHISPER_ARCHITECTURES:
        return _Family.WHISPER
    if normalized in KNOWN_VARIATION_ARCHITECTURES:
        return _Family.KNOWN_VARIATION
    return _Family.UNSUPPORTED


def _family_for_model_type(value: str) -> _Family:
    normalized = value.casefold()
    if normalized in WHISPER_MODEL_TYPES:
        return _Family.WHISPER
    if normalized in KNOWN_VARIATION_MODEL_TYPES:
        return _Family.KNOWN_VARIATION
    return _Family.UNSUPPORTED
