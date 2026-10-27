"""Normalize provider facts into the versioned inspection schema.

All strings originating in a model card or repository configuration are untrusted
descriptive metadata. They are stored, never interpreted as instructions, and never
used to fetch additional URLs or run code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from ethi_asr_bench.errors import ErrorCode
from ethi_asr_bench.models.repository import (
    SCHEMA_VERSION,
    CustomCodeIndicators,
    DependencyIndicators,
    FieldProvenance,
    InspectionError,
    InspectionRecord,
    InspectionStatus,
    InspectionWarning,
    ModelMetadata,
    ObservedValue,
    Provenance,
    RepositoryContents,
    RepositoryFile,
    RepositoryIdentity,
    RepositoryRef,
    ValueState,
    WarningSeverity,
)
from ethi_asr_bench.providers.base import ProviderSnapshot
from ethi_asr_bench.utils import truncate

MAX_DESCRIPTION_CHARS = 2_000
MAX_FILES = 2_000
MAX_REQUIREMENTS = 50
MAX_REQUIREMENT_CHARS = 200
MAX_INSTALL_INSTRUCTIONS = 5
MAX_INSTALL_INSTRUCTION_CHARS = 300
MAX_PYTHON_FILES = 100

CONFIG_FILE_NAMES = frozenset(
    {
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "generation_config.json",
    }
)
TOKENIZER_OR_PROCESSOR_NAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
        "normalizer.json",
        "sentencepiece.bpe.model",
        "spm.model",
        "tokenizer.model",
    }
)
WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".onnx",
    ".gguf",
    ".msgpack",
    ".h5",
    ".tflite",
    ".ckpt",
    ".pth",
    ".bin",
)
NON_WEIGHT_BIN_NAMES = frozenset(
    {
        "training_args.bin",
        "optimizer.bin",
        "scheduler.bin",
        "training_state.bin",
    }
)
NON_WEIGHT_PT_NAMES = frozenset({"optimizer.pt", "scheduler.pt", "training_args.pt"})

KNOWN_TASK_TAGS = frozenset(
    {
        "audio-classification",
        "audio-to-audio",
        "automatic-speech-recognition",
        "feature-extraction",
        "text-to-audio",
        "text-to-speech",
        "voice-activity-detection",
    }
)

# ISO 639-1 plus common three-letter and Ethiopian language codes. Matching is
# intentionally allowlist-based: arbitrary short tags are not language evidence.
LANGUAGE_CODES = frozenset(
    [
        "aa",
        "ab",
        "ae",
        "af",
        "ak",
        "am",
        "an",
        "ar",
        "as",
        "av",
        "ay",
        "az",
        "ba",
        "be",
        "bg",
        "bi",
        "bm",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "ce",
        "ch",
        "co",
        "cr",
        "cs",
        "cu",
        "cv",
        "cy",
        "da",
        "de",
        "dv",
        "dz",
        "ee",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "ff",
        "fi",
        "fj",
        "fo",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gn",
        "gu",
        "gv",
        "ha",
        "he",
        "hi",
        "ho",
        "hr",
        "ht",
        "hu",
        "hy",
        "hz",
        "ia",
        "id",
        "ie",
        "ig",
        "ii",
        "ik",
        "io",
        "is",
        "it",
        "iu",
        "ja",
        "jv",
        "ka",
        "kg",
        "ki",
        "kj",
        "kk",
        "kl",
        "km",
        "kn",
        "ko",
        "kr",
        "ks",
        "ku",
        "kv",
        "kw",
        "ky",
        "la",
        "lb",
        "lg",
        "li",
        "ln",
        "lo",
        "lt",
        "lu",
        "lv",
        "mg",
        "mh",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "na",
        "nb",
        "nd",
        "ne",
        "ng",
        "nl",
        "nn",
        "no",
        "nr",
        "nv",
        "ny",
        "oc",
        "oj",
        "om",
        "or",
        "os",
        "pa",
        "pi",
        "pl",
        "ps",
        "pt",
        "qu",
        "rm",
        "rn",
        "ro",
        "ru",
        "rw",
        "sa",
        "sc",
        "sd",
        "se",
        "sg",
        "si",
        "sk",
        "sl",
        "sm",
        "sn",
        "so",
        "sq",
        "sr",
        "ss",
        "st",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "ti",
        "tk",
        "tl",
        "tn",
        "to",
        "tr",
        "ts",
        "tt",
        "tw",
        "ty",
        "ug",
        "uk",
        "ur",
        "uz",
        "ve",
        "vi",
        "vo",
        "wa",
        "wo",
        "xh",
        "yi",
        "yo",
        "za",
        "zh",
        "zu",
        "amh",
        "byn",
        "gez",
        "sid",
        "tir",
        "wal",
    ]
)

_INSTALL_RE = re.compile(r"(?i)\b(?:pip3?|python\s+-m\s+pip|conda)\s+install\b")
_TRUST_REMOTE_CODE_RE = re.compile(r"\btrust_remote_code\b", re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def extract_metadata(
    snapshot: ProviderSnapshot,
    *,
    retrieved_at: datetime,
) -> InspectionRecord:
    """Build a normalized record from one provider snapshot."""
    warnings: list[InspectionWarning] = []

    def warn(
        code: str,
        message: str,
        severity: WarningSeverity = WarningSeverity.WARNING,
    ) -> None:
        warning = InspectionWarning(code=code, message=message, severity=severity)
        if all(
            not (existing.code == warning.code and existing.message == warning.message)
            for existing in warnings
        ):
            warnings.append(warning)

    repo_id = snapshot.repo_id
    owner = repo_id.partition("/")[0] if "/" in repo_id else None
    name = repo_id.rsplit("/", maxsplit=1)[-1]
    identity = RepositoryIdentity(
        repo_id=repo_id,
        owner=owner,
        name=name,
        canonical_url=snapshot.canonical_url or f"https://huggingface.co/{repo_id}",
        requested_revision=snapshot.requested_revision,
        resolved_commit=snapshot.resolved_commit,
        retrieved_at=retrieved_at,
    )

    for note in snapshot.notes:
        warn("provider_note", note, WarningSeverity.WARNING)
    if snapshot.resolved_commit is None:
        warn(
            "commit_unresolved",
            "The provider did not resolve a commit; revision-specific history may be incomplete.",
            WarningSeverity.WARNING,
        )

    card_data = snapshot.card_data
    config, config_source = _resolve_config(snapshot, warn)
    readme = snapshot.text_files.get("README.md")
    readme_present = any(file.path == "README.md" for file in snapshot.files)
    if (
        readme_present
        and readme is None
        and not any(error.filename == "README.md" for error in snapshot.text_file_errors)
    ):
        warn(
            "readme_unavailable",
            "README.md was listed but could not be retrieved; the description is partial.",
            WarningSeverity.WARNING,
        )
    config_present = any(file.path == "config.json" for file in snapshot.files)
    if (
        config_present
        and config is None
        and not any(error.filename == "config.json" for error in snapshot.text_file_errors)
    ):
        warn(
            "config_unavailable",
            "config.json was listed but could not be read; architecture metadata is partial.",
            WarningSeverity.WARNING,
        )

    contents = _build_contents(snapshot, warn)
    model = _build_model(
        snapshot=snapshot,
        card_data=card_data,
        config=config,
        config_source=config_source,
        readme=readme,
        readme_present=readme_present,
        warn=warn,
    )
    dependencies = _build_dependencies(snapshot, contents, readme, warn)
    custom_code = _build_custom_code(snapshot, contents, config, readme, warn)
    _add_text_file_warnings(snapshot, config, warn)
    _add_security_notes(contents, custom_code, warn)

    provenance = Provenance(
        provider=snapshot.provider_name,
        provider_version=snapshot.provider_version,
        sources=[
            "hub.model_info",
            "hub.file_listing",
            *(["README.md"] if readme is not None else []),
            *(sorted([config_source]) if config_source else []),
            *(["requirements.txt"] if "requirements.txt" in snapshot.text_files else []),
        ],
        field_sources={
            "tags": FieldProvenance(source="hub.model_info.tags"),
            "task_tags": FieldProvenance(
                source="hub.model_info.pipeline_tag + hub.model_info.tags"
            ),
            "files": FieldProvenance(source="hub.model_info.siblings"),
            "display_name": FieldProvenance(source="model_card.cardData"),
            "description": FieldProvenance(
                source="model_card.README.md", note="untrusted descriptive text"
            ),
            "languages": FieldProvenance(
                source="model_card.cardData.language + hub.model_info.tags"
            ),
            "licenses": FieldProvenance(source="model_card.cardData.license + hub.model_info.tags"),
            "architectures": FieldProvenance(source=config_source or "unavailable"),
            "model_type": FieldProvenance(source=config_source or "unavailable"),
            "install_instructions": FieldProvenance(
                source="model_card.README.md", note="untrusted descriptive text"
            ),
            "declared_requirements": FieldProvenance(
                source="repository.requirements.txt", note="untrusted descriptive text"
            ),
        },
    )

    has_warning = any(warning.severity is WarningSeverity.WARNING for warning in warnings)
    status = InspectionStatus.PARTIAL if has_warning else InspectionStatus.SUCCESS
    return InspectionRecord(
        schema_version=SCHEMA_VERSION,
        identity=identity,
        status=status,
        model=model,
        contents=contents,
        dependencies=dependencies,
        custom_code=custom_code,
        warnings=warnings,
        errors=[],
        provenance=provenance,
    )


def build_failed_record(
    ref: RepositoryRef,
    error: InspectionError,
    *,
    retrieved_at: datetime,
    provider: str,
    resolved_commit: str | None = None,
    provider_version: str | None = None,
) -> InspectionRecord:
    """Represent an unsuccessful attempt without pretending metadata was obtained."""
    return InspectionRecord(
        schema_version=SCHEMA_VERSION,
        identity=RepositoryIdentity(
            repo_id=ref.repo_id,
            owner=ref.owner,
            name=ref.name,
            canonical_url=ref.canonical_url,
            requested_revision=ref.revision,
            resolved_commit=resolved_commit,
            retrieved_at=retrieved_at,
        ),
        status=InspectionStatus.FAILED,
        errors=[error],
        provenance=Provenance(provider=provider, provider_version=provider_version),
    )


def _resolve_config(
    snapshot: ProviderSnapshot,
    warn: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    downloaded = snapshot.text_files.get("config.json")
    if downloaded is not None:
        try:
            parsed = json.loads(downloaded)
        except json.JSONDecodeError as exc:
            warn(
                "config_json_malformed",
                f"config.json is not valid JSON ({exc.msg}); falling back to Hub metadata.",
            )
        else:
            if isinstance(parsed, dict):
                return parsed, "config.json"
            warn(
                "config_json_malformed",
                "config.json does not contain a JSON object; falling back to Hub metadata.",
            )
    if snapshot.config is not None:
        return snapshot.config, "hub.config"
    return None, None


def _build_model(
    *,
    snapshot: ProviderSnapshot,
    card_data: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    config_source: str | None,
    readme: str | None,
    readme_present: bool,
    warn: Any,
) -> ModelMetadata:
    raw_info = snapshot.raw_model_info
    tags = _string_list(raw_info.get("tags"), field="tags", warn=warn) or []

    display_name = _read_display_name(card_data, warn)
    description = _read_description(
        readme=readme,
        readme_present=readme_present,
        warn=warn,
    )
    config_present = any(file.path == "config.json" for file in snapshot.files)
    architectures, model_type = _read_config_fields(
        config, config_source, config_present=config_present, warn=warn
    )
    library_name = _read_string_field(
        raw_info.get("library_name"),
        field="library_name",
        source="hub.model_info.library_name",
        fallback=_optional_string(card_data.get("library_name")),
        fallback_source="model_card.cardData.library_name",
        warn=warn,
    )
    pipeline_tag = _read_string_field(
        raw_info.get("pipeline_tag"),
        field="pipeline_tag",
        source="hub.model_info.pipeline_tag",
        fallback=_optional_string(card_data.get("pipeline_tag")),
        fallback_source="model_card.cardData.pipeline_tag",
        warn=warn,
    )
    languages = _read_languages(card_data, tags, warn)
    licenses = _read_licenses(card_data, tags, warn)
    metrics = _string_list(card_data.get("metrics"), field="metrics", warn=warn) or []

    task_tags: list[str] = []
    if pipeline_tag.state is ValueState.KNOWN and pipeline_tag.value is not None:
        task_tags.append(pipeline_tag.value)
    task_tags.extend(tag for tag in tags if tag in KNOWN_TASK_TAGS)
    task_tags = _unique(task_tags)

    return ModelMetadata(
        display_name=display_name,
        description=description,
        architectures=architectures,
        model_type=model_type,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
        languages=languages,
        licenses=licenses,
        task_tags=task_tags,
        tags=tags,
        declared_metrics=metrics,
    )


def _read_display_name(card_data: Mapping[str, Any], warn: Any) -> ObservedValue[str]:
    for key in ("model_name", "name"):
        value = card_data.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return ObservedValue.known(value.strip(), source=f"model_card.cardData.{key}")
        warn(
            "malformed_metadata",
            f"Model card field {key!r} is not a non-empty string; display name is unknown.",
        )
        return ObservedValue.extraction_failed(
            note=f"model card field {key!r} had an unexpected type"
        )
    return ObservedValue.unknown()


def _read_description(
    *,
    readme: str | None,
    readme_present: bool,
    warn: Any,
) -> ObservedValue[str]:
    if readme is None:
        if readme_present:
            return ObservedValue.extraction_failed(note="README.md could not be retrieved")
        return ObservedValue.unknown()
    summary = _summarize_model_card(readme)
    if not summary:
        return ObservedValue.unknown(note="README.md has no plain-text summary paragraph")
    truncated, was_truncated = truncate(summary, MAX_DESCRIPTION_CHARS)
    return ObservedValue.known(
        truncated,
        source="model_card.README.md",
        note=(f"truncated to {MAX_DESCRIPTION_CHARS} characters" if was_truncated else None),
    )


def _read_config_fields(
    config: Mapping[str, Any] | None,
    config_source: str | None,
    *,
    config_present: bool,
    warn: Any,
) -> tuple[ObservedValue[list[str]], ObservedValue[str]]:
    if config is None:
        if config_present:
            note = "config.json was present but could not be read"
        else:
            note = "no structured configuration was provided"
        return (
            ObservedValue.extraction_failed(note=note)
            if config_present
            else ObservedValue.unknown(note=note),
            ObservedValue.extraction_failed(note=note)
            if config_present
            else ObservedValue.unknown(note=note),
        )
    source = config_source or "config"

    raw_architectures = config.get("architectures")
    if raw_architectures is None:
        architectures: ObservedValue[list[str]] = ObservedValue.unknown(
            note="config does not declare architectures"
        )
    elif isinstance(raw_architectures, str) and raw_architectures.strip():
        architectures = ObservedValue.known(
            [raw_architectures.strip()], source=f"{source}.architectures"
        )
        warn(
            "malformed_metadata",
            "config.architectures was a string instead of a list; it was normalized to a list.",
        )
    elif isinstance(raw_architectures, list) and all(
        isinstance(item, str) and item.strip() for item in raw_architectures
    ):
        values = _unique([item.strip() for item in raw_architectures])
        architectures = (
            ObservedValue.known(values, source=f"{source}.architectures")
            if values
            else ObservedValue.unknown(note="config declares an empty architecture list")
        )
    else:
        architectures = ObservedValue.extraction_failed(
            note="config.architectures had an unexpected type"
        )
        warn(
            "malformed_metadata",
            "config.architectures had an unexpected type; architectures are unknown.",
        )

    raw_model_type = config.get("model_type")
    if raw_model_type is None:
        model_type: ObservedValue[str] = ObservedValue.unknown(
            note="config does not declare model_type"
        )
    elif isinstance(raw_model_type, str) and raw_model_type.strip():
        model_type = ObservedValue.known(raw_model_type.strip(), source=f"{source}.model_type")
    else:
        model_type = ObservedValue.extraction_failed(
            note="config.model_type had an unexpected type"
        )
        warn(
            "malformed_metadata",
            "config.model_type had an unexpected type; model type is unknown.",
        )
    return architectures, model_type


def _read_string_field(
    value: Any,
    *,
    field: str,
    source: str,
    fallback: str | None,
    fallback_source: str,
    warn: Any,
) -> ObservedValue[str]:
    if isinstance(value, str) and value.strip():
        return ObservedValue.known(value.strip(), source=source)
    if value is not None:
        warn(
            "malformed_metadata",
            f"Hub field {field!r} was not a non-empty string; trying the model card.",
        )
    if fallback:
        return ObservedValue.known(fallback, source=fallback_source)
    if value is not None:
        return ObservedValue.extraction_failed(note=f"Hub field {field!r} was malformed")
    return ObservedValue.unknown()


def _read_languages(
    card_data: Mapping[str, Any],
    tags: list[str],
    warn: Any,
) -> ObservedValue[list[str]]:
    language_values, malformed = _extract_languages(card_data.get("language"))
    tag_values = [
        tag.split(":", 1)[1].lower() for tag in tags if tag.lower().startswith("language:")
    ]
    tag_values.extend(tag.lower() for tag in tags if tag.lower() in LANGUAGE_CODES)
    tag_values = [value for value in tag_values if value in LANGUAGE_CODES]
    values = _unique([*language_values, *tag_values])
    if malformed:
        warn("malformed_metadata", "model card language field had an unexpected type.")
    if malformed and not values:
        return ObservedValue.extraction_failed(note="language field was malformed")
    if values:
        return ObservedValue.known(
            values,
            source="model_card.cardData.language + hub.model_info.tags",
        )
    return ObservedValue.unknown()


def _read_licenses(
    card_data: Mapping[str, Any],
    tags: list[str],
    warn: Any,
) -> ObservedValue[list[str]]:
    raw = card_data.get("license")
    values: list[str] = []
    source_parts: list[str] = []
    note: str | None = None
    malformed = False
    if isinstance(raw, str) and raw.strip():
        values.append(raw.strip())
        source_parts.append("model_card.cardData.license")
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
            else:
                malformed = True
        if values:
            source_parts.append("model_card.cardData.license")
    elif raw is not None:
        malformed = True

    license_name = card_data.get("license_name")
    if not values and isinstance(license_name, str) and license_name.strip():
        values.append(license_name.strip())
        source_parts.append("model_card.cardData.license_name")
        note = "human-readable license name, not a normalized identifier"

    tag_values = [
        tag.split(":", 1)[1].strip()
        for tag in tags
        if tag.lower().startswith("license:") and tag.split(":", 1)[1].strip()
    ]
    values.extend(tag_values)
    if tag_values:
        source_parts.append("hub.model_info.tags")
    unique_values = _unique(values)
    if malformed:
        warn("malformed_metadata", "model card license field had an unexpected type.")
    if malformed and not unique_values:
        return ObservedValue.extraction_failed(note="license field was malformed")
    if unique_values:
        return ObservedValue.known(
            unique_values,
            source=" + ".join(source_parts) or "model_card.cardData.license",
            note=note,
        )
    return ObservedValue.unknown()


def _extract_languages(value: Any) -> tuple[list[str], bool]:
    if value is None:
        return [], False
    if isinstance(value, str):
        candidate = value.strip().lower()
        return ([candidate] if candidate else []), False
    if isinstance(value, list):
        values: list[str] = []
        malformed = False
        for item in value:
            if isinstance(item, str) and item.strip():
                values.append(item.strip().lower())
            else:
                malformed = True
        return values, malformed
    return [], True


def _build_contents(snapshot: ProviderSnapshot, warn: Any) -> RepositoryContents:
    validated: list[RepositoryFile] = []
    skipped_paths = 0
    for provider_file in snapshot.files:
        try:
            validated.append(
                RepositoryFile(
                    path=provider_file.path,
                    size_bytes=provider_file.size_bytes,
                    is_lfs=provider_file.is_lfs,
                )
            )
        except PydanticValidationError:
            skipped_paths += 1

    if skipped_paths:
        warn(
            "unsafe_file_path",
            f"Skipped {skipped_paths} provider file entr"
            f"{'y' if skipped_paths == 1 else 'ies'} with unsafe or invalid paths.",
        )

    validated.sort(key=lambda file: file.path)
    truncated = len(validated) > MAX_FILES
    if truncated:
        validated = validated[:MAX_FILES]
        warn(
            "file_inventory_truncated",
            f"File inventory was truncated to the first {MAX_FILES} paths.",
            WarningSeverity.WARNING,
        )

    known_size = sum(file.size_bytes for file in validated if file.size_bytes is not None)
    unknown_size_count = sum(1 for file in validated if file.size_bytes is None)
    paths = [file.path for file in validated]
    basenames = {PurePosixPath(path).name for path in paths}

    readme_filename = "README.md" if "README.md" in paths else None
    config_files = sorted(name for name in basenames if name in CONFIG_FILE_NAMES)
    tokenizer_files = sorted(name for name in basenames if name in TOKENIZER_OR_PROCESSOR_NAMES)
    weight_files = sorted(path for path in paths if _is_weight_file(path))
    python_files = sorted(path for path in paths if path.casefold().endswith(".py"))

    return RepositoryContents(
        files=validated,
        files_truncated=truncated,
        known_size_bytes=known_size,
        unknown_size_file_count=unknown_size_count,
        has_readme=readme_filename is not None,
        readme_filename=readme_filename,
        has_config_json="config.json" in paths,
        config_files=config_files,
        tokenizer_or_processor_files=tokenizer_files,
        has_tokenizer_or_processor=bool(tokenizer_files),
        weight_files=weight_files,
        has_weight_files=bool(weight_files),
        has_custom_code=bool(python_files),
        generation_config_present="generation_config.json" in paths,
    )


def _build_dependencies(
    snapshot: ProviderSnapshot,
    contents: RepositoryContents,
    readme: str | None,
    warn: Any,
) -> DependencyIndicators:
    basenames = {PurePosixPath(file.path).name for file in contents.files}
    requirements_files = sorted(
        name for name in basenames if name.startswith("requirements") and name.endswith(".txt")
    )
    requirements_text = snapshot.text_files.get("requirements.txt")
    declared_requirements: list[str] = []
    if requirements_text is not None:
        declared_requirements = _parse_requirements(requirements_text)
        if len(declared_requirements) > MAX_REQUIREMENTS:
            declared_requirements = declared_requirements[:MAX_REQUIREMENTS]
            warn(
                "requirements_truncated",
                f"Declared requirements were truncated to {MAX_REQUIREMENTS} entries.",
                WarningSeverity.INFO,
            )
    install_instructions = _extract_install_instructions(readme)
    if install_instructions:
        warn(
            "install_instructions_present",
            "The model card contains installation instructions; they are stored as "
            "untrusted descriptive text and are never executed.",
            WarningSeverity.INFO,
        )

    clues: list[str] = []
    raw_info = snapshot.raw_model_info
    library = _optional_string(raw_info.get("library_name"))
    pipeline = _optional_string(raw_info.get("pipeline_tag"))
    if library:
        clues.append(f"Declared library: {library}")
    if pipeline:
        clues.append(f"Declared pipeline: {pipeline}")
    auto_model = _optional_string(snapshot.transformers_info.get("auto_model"))
    if auto_model:
        clues.append(f"Transformers auto_model: {auto_model}")
    if contents.weight_files:
        extensions = sorted({PurePosixPath(path).suffix for path in contents.weight_files})
        clues.append(f"Weight-like files present ({', '.join(extensions)}); not downloaded.")
    if contents.has_custom_code:
        clues.append("Repository contains Python files; custom code indicators detected.")

    return DependencyIndicators(
        requirements_files=requirements_files,
        declared_requirements=declared_requirements,
        install_instructions=install_instructions,
        compatibility_clues=clues,
    )


def _build_custom_code(
    snapshot: ProviderSnapshot,
    contents: RepositoryContents,
    config: Mapping[str, Any] | None,
    readme: str | None,
    warn: Any,
) -> CustomCodeIndicators:
    python_files = [
        path for path in (file.path for file in contents.files) if path.casefold().endswith(".py")
    ][:MAX_PYTHON_FILES]

    if config is None:
        auto_map: ObservedValue[bool] = ObservedValue.unknown(note="configuration unavailable")
    else:
        raw_auto_map = config.get("auto_map")
        auto_map = ObservedValue.known(
            bool(raw_auto_map),
            source="config.auto_map",
        )

    if readme is None:
        trust_mention: ObservedValue[bool] = ObservedValue.unknown(note="README.md unavailable")
    else:
        mentioned = bool(_TRUST_REMOTE_CODE_RE.search(readme))
        trust_mention = ObservedValue.known(mentioned, source="model_card.README.md")

    if auto_map.state is ValueState.KNOWN and auto_map.value:
        warn(
            "auto_map_declared",
            "The repository declares auto_map entries that may reference custom code. "
            "Stage 1 does not resolve, download, or execute that code.",
            WarningSeverity.INFO,
        )
    if trust_mention.state is ValueState.KNOWN and trust_mention.value:
        warn(
            "trust_remote_code_mentioned",
            "The model card mentions trust_remote_code. This is an untrusted hint, "
            "not a verified requirement.",
            WarningSeverity.INFO,
        )

    return CustomCodeIndicators(
        python_files=python_files,
        auto_map_declared=auto_map,
        trust_remote_code_mentioned=trust_mention,
        code_executed=False,
    )


def _add_text_file_warnings(
    snapshot: ProviderSnapshot,
    config: Mapping[str, Any] | None,
    warn: Any,
) -> None:
    for failure in snapshot.text_file_errors:
        severity = WarningSeverity.INFO
        if failure.filename == "README.md" or (
            failure.filename == "config.json" and config is None
        ):
            severity = WarningSeverity.WARNING
        warn(
            "provider_text_file_error",
            f"Could not retrieve {failure.filename}: {failure.message}",
            severity,
        )


def _add_security_notes(
    contents: RepositoryContents,
    custom_code: CustomCodeIndicators,
    warn: Any,
) -> None:
    if custom_code.python_files:
        warn(
            "custom_code_present",
            f"{len(custom_code.python_files)} Python file(s) are present. They were not "
            "downloaded or executed.",
            WarningSeverity.INFO,
        )
    if contents.has_weight_files:
        warn(
            "weights_not_downloaded",
            "Model weight-like files were detected from the file listing and were not downloaded.",
            WarningSeverity.INFO,
        )


def _is_weight_file(path: str) -> bool:
    name = PurePosixPath(path).name
    lowered = name.casefold()
    if lowered in NON_WEIGHT_BIN_NAMES or lowered in NON_WEIGHT_PT_NAMES:
        return False
    return lowered.endswith(WEIGHT_EXTENSIONS)


def _parse_requirements(text: str) -> list[str]:
    values: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        value, _ = truncate(stripped, MAX_REQUIREMENT_CHARS)
        if value not in values:
            values.append(value)
    return values


def _extract_install_instructions(readme: str | None) -> list[str]:
    if not readme:
        return []
    instructions: list[str] = []
    for line in readme.splitlines():
        stripped = line.strip().lstrip("$>").strip()
        if not stripped or not _INSTALL_RE.search(stripped):
            continue
        value, _ = truncate(_CONTROL_RE.sub("", stripped), MAX_INSTALL_INSTRUCTION_CHARS)
        if value not in instructions:
            instructions.append(value)
        if len(instructions) >= MAX_INSTALL_INSTRUCTIONS:
            break
    return instructions


def _summarize_model_card(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    normalized = _HTML_COMMENT_RE.sub("", normalized)
    lines = normalized.split("\n")
    if lines and lines[0].strip() == "---":
        closing_index = next(
            (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
            None,
        )
        if closing_index is None:
            return ""
        lines = lines[closing_index + 1 :]

    paragraphs: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if current:
                paragraphs.append(current)
                current = []
            continue
        if in_fence:
            continue
        if not stripped:
            if current:
                paragraphs.append(current)
                current = []
            continue
        if stripped.startswith(("#", "|", "<", "![", "[![", "---")):
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(current)

    for paragraph in paragraphs:
        text_value = " ".join(paragraph).strip()
        if text_value:
            return _CONTROL_RE.sub("", text_value)
    return ""


def _string_list(value: Any, *, field: str, warn: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    if isinstance(value, list):
        string_values = [item for item in value if isinstance(item, str)]
        invalid_count = len(value) - len(string_values)
        values = _unique([item.strip() for item in string_values if item.strip()])
        if invalid_count:
            warn(
                "malformed_metadata",
                f"Hub field {field!r} contained {invalid_count} non-string "
                f"entr{'y' if invalid_count == 1 else 'ies'}; they were ignored.",
            )
        return values or None
    warn(
        "malformed_metadata",
        f"Hub field {field!r} had an unexpected type; it was ignored.",
    )
    return None


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def extraction_failure(ref: RepositoryRef, message: str) -> InspectionError:
    """Build the structured error stored when normalization fails unexpectedly."""
    return InspectionError(code=ErrorCode.EXTRACTION_FAILED.value, message=message)
