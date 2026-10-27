"""Provider interface and transport-neutral snapshot types.

A provider is responsible only for retrieving remote facts. It must not normalize
them into the application schema, execute repository code, or download weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ethi_asr_bench.models.repository import RepositoryRef


@dataclass(frozen=True, slots=True)
class ProviderFile:
    """A file advertised by a provider."""

    path: str
    size_bytes: int | None = None
    is_lfs: bool | None = None


@dataclass(frozen=True, slots=True)
class ProviderTextFileError:
    """A non-fatal failure to retrieve one small, allowlisted text file."""

    filename: str
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class ProviderSnapshot:
    """Untrusted facts retrieved from a model registry for one commit."""

    repo_id: str
    canonical_url: str
    owner: str | None
    name: str
    requested_revision: str | None
    resolved_commit: str | None
    raw_model_info: dict[str, Any]
    files: list[ProviderFile] = field(default_factory=list)
    card_data: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] | None = None
    transformers_info: dict[str, Any] = field(default_factory=dict)
    text_files: dict[str, str] = field(default_factory=dict)
    text_file_errors: list[ProviderTextFileError] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provider_name: str = "huggingface"
    provider_version: str | None = None


class RepositoryProvider(Protocol):
    """Replaceable source of model repository metadata."""

    name: str

    def fetch_repository(self, ref: RepositoryRef) -> ProviderSnapshot:
        """Retrieve metadata for a validated repository reference.

        Implementations must raise :class:`ethi_asr_bench.errors.ProviderError`
        for expected remote failures.
        """
        ...
