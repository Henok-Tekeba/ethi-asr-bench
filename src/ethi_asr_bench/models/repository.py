"""Versioned, persistence-facing domain models.

The normalized record is deliberately conservative: every classification-relevant
field is wrapped in :class:`ObservedValue` so that "unknown", "extraction failed",
and "not applicable" are distinguishable from a known value. Nothing in this module
executes or interprets code from a model repository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Generic, Literal, Self, TypeVar
from urllib.parse import urlsplit

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ethi_asr_bench.utils import format_utc

SCHEMA_VERSION: SchemaVersion = "1.0.0"
SchemaVersion = Literal["1.0.0"]

# Owner and name segments: 1-96 characters, no path traversal characters.
REPOSITORY_SEGMENT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
REPOSITORY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
REVISION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$"

T = TypeVar("T")

NonEmptyStr = Annotated[str, Field(min_length=1)]


class RepositoryType(StrEnum):
    """Source repository category. Only models are accepted in Stage 1."""

    MODEL = "model"


class ValueState(StrEnum):
    """Why an observed value does or does not carry data."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    EXTRACTION_FAILED = "extraction_failed"
    NOT_APPLICABLE = "not_applicable"


class InspectionStatus(StrEnum):
    """Outcome of an inspection attempt."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class WarningSeverity(StrEnum):
    """Whether a warning signals incomplete metadata or is informational."""

    INFO = "info"
    WARNING = "warning"


class ObservedValue(BaseModel, Generic[T]):
    """A value together with its epistemic state and source.

    ``state=known`` requires a non-empty value. Other states must not carry a value,
    which prevents missing metadata from being mistaken for a real value.
    """

    model_config = ConfigDict(frozen=True)

    state: ValueState
    value: T | None = None
    source: NonEmptyStr | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _state_matches_value(self) -> Self:
        empty = (
            self.value is None
            or (isinstance(self.value, str) and self.value == "")
            or (isinstance(self.value, (list, tuple, set, dict)) and len(self.value) == 0)
        )
        if self.state is ValueState.KNOWN and empty:
            raise ValueError("a known value must be non-empty")
        if self.state is ValueState.KNOWN and not self.source:
            raise ValueError("a known value must record its source")
        if self.state is not ValueState.KNOWN and self.value is not None:
            raise ValueError("only known values may carry a value")
        return self

    @classmethod
    def known(
        cls,
        value: T,
        *,
        source: str,
        note: str | None = None,
    ) -> Self:
        return cls(state=ValueState.KNOWN, value=value, source=source, note=note)

    @classmethod
    def unknown(cls, *, note: str | None = None) -> Self:
        return cls(state=ValueState.UNKNOWN, note=note)

    @classmethod
    def extraction_failed(cls, *, note: str | None = None) -> Self:
        return cls(state=ValueState.EXTRACTION_FAILED, note=note)

    @classmethod
    def not_applicable(cls, *, note: str | None = None) -> Self:
        return cls(state=ValueState.NOT_APPLICABLE, note=note)


class RepositoryRef(BaseModel):
    """A validated, canonical reference to a repository revision.

    Instances are produced by :mod:`ethi_asr_bench.services.link_parser`; provider
    implementations must treat this object as already validated.
    """

    model_config = ConfigDict(frozen=True)

    repo_id: str = Field(pattern=REPOSITORY_ID_PATTERN)
    owner: str | None = Field(default=None, pattern=REPOSITORY_SEGMENT_PATTERN)
    name: str = Field(pattern=REPOSITORY_SEGMENT_PATTERN)
    repo_type: RepositoryType = RepositoryType.MODEL
    canonical_url: str
    revision: str | None = Field(default=None, pattern=REVISION_PATTERN)


class RepositoryIdentity(BaseModel):
    """Canonical identity and retrieval coordinates for one inspection."""

    model_config = ConfigDict(frozen=True)

    repo_id: str = Field(pattern=REPOSITORY_ID_PATTERN)
    owner: str | None = Field(default=None, pattern=REPOSITORY_SEGMENT_PATTERN)
    name: str = Field(pattern=REPOSITORY_SEGMENT_PATTERN)
    repo_type: RepositoryType = RepositoryType.MODEL
    canonical_url: str
    requested_revision: str | None = Field(default=None, pattern=REVISION_PATTERN)
    resolved_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,64}$")
    retrieved_at: AwareDatetime

    @field_validator("canonical_url")
    @classmethod
    def _canonical_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError("canonical_url must be a valid URL") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "canonical_url must be a credential-free https URL without query or fragment"
            )
        return value

    @field_validator("retrieved_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_serializer("retrieved_at")
    def _serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)

    @model_validator(mode="after")
    def _consistent_identity(self) -> Self:
        if self.owner is not None:
            owner, separator, name = self.repo_id.partition("/")
            if separator != "/" or owner != self.owner or name != self.name:
                raise ValueError("owner and name do not match repo_id")
        elif "/" in self.repo_id:
            raise ValueError("owner is required for namespaced repository IDs")
        elif self.name != self.repo_id:
            raise ValueError("name does not match repo_id")
        return self


class RepositoryFile(BaseModel):
    """One file advertised by the provider. Paths are never used for local access."""

    model_config = ConfigDict(frozen=True)

    path: NonEmptyStr
    size_bytes: int | None = Field(default=None, ge=0)
    is_lfs: bool | None = None

    @field_validator("path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        if (
            value.startswith("/")
            or "\\" in value
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)
        ):
            raise ValueError("file paths must be relative POSIX paths without control characters")
        parts = value.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("file paths must not contain empty, '.' or '..' segments")
        return value


class ModelMetadata(BaseModel):
    """Descriptive and structured model metadata declared by the provider."""

    display_name: ObservedValue[str] = Field(default_factory=lambda: ObservedValue[str].unknown())
    description: ObservedValue[str] = Field(default_factory=lambda: ObservedValue[str].unknown())
    architectures: ObservedValue[list[str]] = Field(
        default_factory=lambda: ObservedValue[list[str]].unknown()
    )
    model_type: ObservedValue[str] = Field(default_factory=lambda: ObservedValue[str].unknown())
    library_name: ObservedValue[str] = Field(default_factory=lambda: ObservedValue[str].unknown())
    pipeline_tag: ObservedValue[str] = Field(default_factory=lambda: ObservedValue[str].unknown())
    languages: ObservedValue[list[str]] = Field(
        default_factory=lambda: ObservedValue[list[str]].unknown()
    )
    licenses: ObservedValue[list[str]] = Field(
        default_factory=lambda: ObservedValue[list[str]].unknown()
    )
    task_tags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    declared_metrics: list[str] = Field(default_factory=list)


class RepositoryContents(BaseModel):
    """File inventory plus derived presence flags. No file is downloaded here."""

    files: list[RepositoryFile] = Field(default_factory=list)
    files_truncated: bool = False
    known_size_bytes: int = Field(default=0, ge=0)
    unknown_size_file_count: int = Field(default=0, ge=0)
    has_readme: bool = False
    readme_filename: str | None = None
    has_config_json: bool = False
    config_files: list[str] = Field(default_factory=list)
    tokenizer_or_processor_files: list[str] = Field(default_factory=list)
    has_tokenizer_or_processor: bool = False
    weight_files: list[str] = Field(default_factory=list)
    has_weight_files: bool = False
    has_custom_code: bool = False
    generation_config_present: bool = False


class CustomCodeIndicators(BaseModel):
    """Signals that a repository may rely on repository-provided Python code.

    These are indicators only. Stage 1 never downloads or executes the files named
    here, and ``code_executed`` is fixed to ``False``.
    """

    python_files: list[str] = Field(default_factory=list)
    auto_map_declared: ObservedValue[bool] = Field(
        default_factory=lambda: ObservedValue[bool].unknown()
    )
    trust_remote_code_mentioned: ObservedValue[bool] = Field(
        default_factory=lambda: ObservedValue[bool].unknown()
    )
    code_executed: Literal[False] = False


class DependencyIndicators(BaseModel):
    """Untrusted dependency and loading clues. Nothing here is installed or run."""

    requirements_files: list[str] = Field(default_factory=list)
    declared_requirements: list[str] = Field(default_factory=list)
    install_instructions: list[str] = Field(default_factory=list)
    compatibility_clues: list[str] = Field(default_factory=list)


class FieldProvenance(BaseModel):
    """Where a composite or derived field came from."""

    model_config = ConfigDict(frozen=True)

    source: NonEmptyStr
    note: str | None = None


class Provenance(BaseModel):
    """Provider and source-file provenance for the whole record."""

    provider: NonEmptyStr
    provider_version: str | None = None
    sources: list[str] = Field(default_factory=list)
    field_sources: dict[str, FieldProvenance] = Field(default_factory=dict)


class InspectionWarning(BaseModel):
    """A non-fatal observation. ``warning`` severity implies partial metadata."""

    model_config = ConfigDict(frozen=True)

    code: NonEmptyStr
    message: NonEmptyStr
    severity: WarningSeverity = WarningSeverity.WARNING


class InspectionError(BaseModel):
    """A structured failure stored alongside the inspection record."""

    model_config = ConfigDict(frozen=True)

    code: NonEmptyStr
    message: NonEmptyStr
    retryable: bool = False
    http_status: int | None = None


class InspectionRecord(BaseModel):
    """Normalized, versioned output of inspecting one repository revision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: SchemaVersion = SCHEMA_VERSION
    identity: RepositoryIdentity
    status: InspectionStatus
    model: ModelMetadata = Field(default_factory=ModelMetadata)
    contents: RepositoryContents = Field(default_factory=RepositoryContents)
    dependencies: DependencyIndicators = Field(default_factory=DependencyIndicators)
    custom_code: CustomCodeIndicators = Field(default_factory=CustomCodeIndicators)
    warnings: list[InspectionWarning] = Field(default_factory=list)
    errors: list[InspectionError] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def _status_is_honest(self) -> Self:
        if self.status is InspectionStatus.FAILED and not self.errors:
            raise ValueError("a failed inspection must include at least one error")
        if self.status is InspectionStatus.SUCCESS and self.errors:
            raise ValueError("a successful inspection must not include errors")
        return self


class RepositorySummary(BaseModel):
    """Compact view used by ``ethi-asr list``."""

    repo_id: str
    canonical_url: str
    owner: str | None
    name: str
    status: InspectionStatus
    requested_revision: str | None
    resolved_commit: str | None
    retrieved_at: AwareDatetime
    inspection_count: int = Field(ge=1)
    warning_count: int = Field(ge=0)

    @field_serializer("retrieved_at")
    def _serialize_timestamp(self, value: datetime) -> str:
        return format_utc(value)
