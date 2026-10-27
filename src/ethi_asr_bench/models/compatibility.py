"""Versioned, rule-based compatibility assessment contract.

A compatibility assessment is derived from one validated :class:`InspectionRecord`.
It is deliberately separate from the inspection schema: classifying a record does
not mutate historical metadata, require a database migration, download weights, or
execute repository code.

Evidence distinguishes what was observed in metadata from what the classifier
assumes. An assessment is a candidate classification, not proof that inference will
succeed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ethi_asr_bench.models.repository import InspectionStatus, SchemaVersion

ClassifierVersion = Literal["1.0.0"]
CLASSIFIER_VERSION: ClassifierVersion = "1.0.0"


class CompatibilityClass(StrEnum):
    """Rule-based inference compatibility categories."""

    STANDARD_WHISPER_CANDIDATE = "standard_whisper_candidate"
    KNOWN_VARIATION = "known_variation"
    CUSTOM_IMPLEMENTATION_REQUIRED = "custom_implementation_required"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class EvidenceKind(StrEnum):
    """Whether a statement was read from metadata or inferred by a rule."""

    OBSERVED = "observed"
    ASSUMPTION = "assumption"


class CompatibilityEvidence(BaseModel):
    """One machine-readable observation or assumption behind a classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str = Field(min_length=1)
    kind: EvidenceKind
    signal: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: str | None = None
    message: str = Field(min_length=1)


class CompatibilityWarning(BaseModel):
    """A limitation or conflict that does not itself prove incompatibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class CompatibilityAssessment(BaseModel):
    """Derived classification plus the evidence needed to review it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classifier_version: ClassifierVersion = CLASSIFIER_VERSION
    classification: CompatibilityClass
    repo_id: str = Field(min_length=1)
    resolved_commit: str | None = None
    inspection_schema_version: SchemaVersion
    inspection_status: InspectionStatus
    reasons: list[str] = Field(min_length=1)
    evidence: list[CompatibilityEvidence] = Field(min_length=1)
    warnings: list[CompatibilityWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _every_verdict_has_observed_evidence(self) -> Self:
        if not any(item.kind is EvidenceKind.OBSERVED for item in self.evidence):
            raise ValueError("every compatibility assessment must include observed evidence")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("compatibility reasons must be non-empty")
        return self
