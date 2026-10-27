"""Hugging Face Hub provider.

The provider uses ``huggingface_hub`` for model metadata and downloads only a
small, fixed allowlist of text files (model card, ``config.json``, and
``requirements.txt``). It never downloads weights and never follows URLs found in
model metadata.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import httpx
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import __version__ as hf_hub_version
from huggingface_hub import errors as hf_errors
from huggingface_hub.hf_api import ModelInfo

from ethi_asr_bench import __version__ as app_version
from ethi_asr_bench.errors import ErrorCode, ProviderError
from ethi_asr_bench.models.repository import RepositoryRef
from ethi_asr_bench.providers.base import (
    ProviderFile,
    ProviderSnapshot,
    ProviderTextFileError,
)
from ethi_asr_bench.utils import format_utc, redact, truncate

logger = logging.getLogger(__name__)

TEXT_FILE_ALLOWLIST: tuple[str, ...] = ("README.md", "config.json", "requirements.txt")
DEFAULT_MAX_TEXT_FILE_BYTES = 1_000_000
MAX_RETRY_DELAY_SECONDS = 30.0
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


class HuggingFaceProvider:
    """Retrieve model metadata from the Hugging Face Hub."""

    name = "huggingface"
    provider_version = hf_hub_version

    def __init__(
        self,
        *,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        max_text_file_bytes: int = DEFAULT_MAX_TEXT_FILE_BYTES,
        api: HfApi | None = None,
        downloader: Callable[..., str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        if max_text_file_bytes < 1:
            raise ValueError("max_text_file_bytes must be positive")
        self._token = token
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._max_text_file_bytes = max_text_file_bytes
        self._api = api or HfApi(
            token=token,
            library_name="ethi-asr-bench",
            library_version=app_version,
        )
        self._download = downloader or cast(Callable[..., str], hf_hub_download)
        self._sleep = sleep

    def fetch_repository(self, ref: RepositoryRef) -> ProviderSnapshot:
        """Fetch metadata for ``ref`` without executing or downloading model code."""
        info = self._with_retries(
            lambda: self._api.model_info(
                ref.repo_id,
                revision=ref.revision,
                timeout=self._timeout,
                files_metadata=True,
            ),
            operation_name="model_info",
        )
        if not isinstance(info, ModelInfo):
            raise ProviderError(
                ErrorCode.INVALID_PROVIDER_RESPONSE,
                "The Hub client returned an unexpected response type.",
            )
        repo_id = self._validate_model_info(info, ref)
        resolved_commit = self._resolved_commit(info, ref)
        files = self._provider_files(info)
        text_files, text_file_errors = self._fetch_text_files(
            repo_id=repo_id,
            revision=resolved_commit or ref.revision,
            files=files,
        )

        notes: list[str] = []
        if info.siblings is None:
            notes.append("The provider did not return a file listing.")
        if resolved_commit is None:
            notes.append("The provider did not resolve a commit SHA.")

        owner = info.author if isinstance(info.author, str) and info.author else ref.owner
        name = repo_id.rsplit("/", maxsplit=1)[-1]
        return ProviderSnapshot(
            repo_id=repo_id,
            canonical_url=f"https://huggingface.co/{repo_id}",
            owner=owner,
            name=name,
            requested_revision=ref.revision,
            resolved_commit=resolved_commit,
            raw_model_info=self._model_info_to_dict(info),
            files=files,
            card_data=self._card_data(info),
            config=info.config if isinstance(info.config, dict) else None,
            transformers_info=self._transformers_info(info),
            text_files=text_files,
            text_file_errors=text_file_errors,
            notes=notes,
            provider_name=self.name,
            provider_version=self.provider_version,
        )

    def _with_retries(self, operation: Callable[[], Any], *, operation_name: str) -> Any:
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except hf_errors.OfflineModeIsEnabled as exc:
                raise ProviderError(
                    ErrorCode.NETWORK_ERROR,
                    self._safe_message(exc, self._token),
                    retryable=False,
                ) from exc
            except (hf_errors.HfHubHTTPError, httpx.HTTPError) as exc:
                error = self._map_http_error(exc, token=self._token)
                if not error.retryable or attempt >= attempts:
                    raise error from exc
                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "Transient Hub failure during %s (attempt %d/%d, code=%s); retrying in %.2fs.",
                    operation_name,
                    attempt,
                    attempts,
                    error.code.value,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("retry loop exhausted without a result")

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        exponential = self._backoff * (2 ** (attempt - 1))
        return float(min(MAX_RETRY_DELAY_SECONDS, max(exponential, retry_after)))

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return 0.0
        raw = headers.get("Retry-After")
        if raw is None:
            return 0.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                return 0.0
            if parsed.tzinfo is None:
                return 0.0
            seconds = (parsed - datetime.now(tz=parsed.tzinfo)).total_seconds()
            return max(0.0, seconds)

    def _map_http_error(self, exc: Exception, *, token: str | None) -> ProviderError:
        message = self._safe_message(exc, token)
        if isinstance(exc, hf_errors.GatedRepoError):
            return ProviderError(ErrorCode.REPOSITORY_GATED, message)
        if isinstance(exc, hf_errors.RevisionNotFoundError):
            return ProviderError(ErrorCode.REVISION_NOT_FOUND, message)
        if isinstance(exc, hf_errors.RepositoryNotFoundError):
            return ProviderError(ErrorCode.REPOSITORY_NOT_FOUND, message)
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(ErrorCode.TIMEOUT, message, retryable=True)
        if isinstance(exc, hf_errors.HfHubHTTPError):
            response = exc.response
            status = response.status_code if response is not None else None
            if status == 401:
                return ProviderError(ErrorCode.AUTHENTICATION_FAILED, message, http_status=status)
            if status == 403:
                return ProviderError(
                    ErrorCode.PERMISSION_DENIED,
                    message,
                    http_status=status,
                )
            if status == 404:
                return ProviderError(
                    ErrorCode.REPOSITORY_NOT_FOUND,
                    message,
                    http_status=status,
                )
            if status == 408:
                return ProviderError(
                    ErrorCode.TIMEOUT,
                    message,
                    retryable=True,
                    http_status=status,
                )
            if status == 429:
                return ProviderError(
                    ErrorCode.RATE_LIMITED,
                    message,
                    retryable=True,
                    http_status=status,
                )
            if status is not None and 500 <= status <= 599:
                return ProviderError(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    message,
                    retryable=True,
                    http_status=status,
                )
            return ProviderError(
                ErrorCode.HTTP_ERROR,
                message,
                http_status=status,
            )
        if isinstance(exc, httpx.HTTPError):
            return ProviderError(ErrorCode.NETWORK_ERROR, message, retryable=True)
        return ProviderError(ErrorCode.NETWORK_ERROR, message, retryable=True)

    @staticmethod
    def _safe_message(exc: Exception, token: str | None) -> str:
        message, _ = truncate(redact(str(exc) or exc.__class__.__name__, token), 500)
        return message

    def _validate_model_info(self, info: ModelInfo, ref: RepositoryRef) -> str:
        if not isinstance(info.id, str) or not info.id:
            raise ProviderError(
                ErrorCode.INVALID_PROVIDER_RESPONSE,
                "The Hub response did not include a repository ID.",
            )
        if info.id.casefold() != ref.repo_id.casefold():
            raise ProviderError(
                ErrorCode.INVALID_PROVIDER_RESPONSE,
                "The Hub returned metadata for a different repository.",
            )
        return info.id

    @staticmethod
    def _resolved_commit(info: ModelInfo, ref: RepositoryRef) -> str | None:
        sha = info.sha
        if sha is None:
            return None
        if not isinstance(sha, str) or not _COMMIT_RE.match(sha):
            raise ProviderError(
                ErrorCode.INVALID_PROVIDER_RESPONSE,
                "The Hub response included an invalid commit identifier.",
            )
        return sha

    @staticmethod
    def _provider_files(info: ModelInfo) -> list[ProviderFile]:
        files: list[ProviderFile] = []
        for sibling in info.siblings or []:
            if not isinstance(sibling.rfilename, str):
                continue
            files.append(
                ProviderFile(
                    path=sibling.rfilename,
                    size_bytes=sibling.size if isinstance(sibling.size, int) else None,
                    is_lfs=sibling.lfs is not None,
                )
            )
        return files

    def _fetch_text_files(
        self,
        *,
        repo_id: str,
        revision: str | None,
        files: list[ProviderFile],
    ) -> tuple[dict[str, str], list[ProviderTextFileError]]:
        by_path = {file.path: file for file in files}
        text_files: dict[str, str] = {}
        errors: list[ProviderTextFileError] = []
        for filename in TEXT_FILE_ALLOWLIST:
            advertised = by_path.get(filename)
            if advertised is None:
                continue
            if advertised.is_lfs:
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=ErrorCode.FILE_UNAVAILABLE.value,
                        message=(
                            f"{filename} is LFS-backed and was not downloaded; "
                            "Stage 1 avoids LFS text files to prevent large artifact retrieval."
                        ),
                    )
                )
                continue
            if advertised.size_bytes is None:
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=ErrorCode.FILE_UNAVAILABLE.value,
                        message=f"{filename} has no advertised size; it was not downloaded.",
                    )
                )
                continue
            if advertised.size_bytes > self._max_text_file_bytes:
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=ErrorCode.FILE_UNAVAILABLE.value,
                        message=(
                            f"{filename} ({advertised.size_bytes} bytes) exceeds the "
                            f"{self._max_text_file_bytes}-byte text-file limit."
                        ),
                    )
                )
                continue
            try:
                content = self._with_retries(
                    partial(
                        self._download_text,
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                    ),
                    operation_name=f"download:{filename}",
                )
            except ProviderError as exc:
                code = exc.code
                if code is ErrorCode.REPOSITORY_NOT_FOUND:
                    code = ErrorCode.FILE_UNAVAILABLE
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=code.value,
                        message=exc.message,
                        retryable=exc.retryable,
                    )
                )
                continue
            except OSError as exc:
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=ErrorCode.FILE_UNAVAILABLE.value,
                        message=self._safe_message(exc, self._token),
                    )
                )
                continue
            except hf_errors.EntryNotFoundError as exc:
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=ErrorCode.FILE_UNAVAILABLE.value,
                        message=self._safe_message(exc, self._token),
                    )
                )
                continue
            except Exception as exc:  # optional file boundary
                logger.warning(
                    "Unexpected error retrieving optional text file %s (type=%s); continuing.",
                    filename,
                    type(exc).__name__,
                )
                errors.append(
                    ProviderTextFileError(
                        filename=filename,
                        code=ErrorCode.FILE_UNAVAILABLE.value,
                        message=(
                            "An unexpected provider error prevented retrieval "
                            f"({type(exc).__name__})."
                        ),
                    )
                )
                continue
            text_files[filename] = content
        return text_files, errors

    def _download_text(self, *, repo_id: str, filename: str, revision: str | None) -> str:
        local_path = self._download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            token=self._token,
            etag_timeout=self._timeout,
        )
        path = Path(local_path)
        with path.open("rb") as handle:
            payload = handle.read(self._max_text_file_bytes + 1)
        if len(payload) > self._max_text_file_bytes:
            raise ProviderError(
                ErrorCode.FILE_UNAVAILABLE,
                f"{filename} exceeds the {self._max_text_file_bytes}-byte text-file limit.",
            )
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _card_data_payload(info: ModelInfo) -> dict[str, Any] | None:
        if info.card_data is None:
            return None
        try:
            data = info.card_data.to_dict()  # type: ignore[no-untyped-call]
        except Exception as exc:  # optional provider object
            logger.warning(
                "Could not serialize model-card metadata (type=%s); continuing without it.",
                type(exc).__name__,
            )
            return {}
        return cast(dict[str, Any], _jsonable(data)) if isinstance(data, dict) else {}

    @classmethod
    def _card_data(cls, info: ModelInfo) -> dict[str, Any]:
        return cls._card_data_payload(info) or {}

    @staticmethod
    def _transformers_info(info: ModelInfo) -> dict[str, Any]:
        value = info.transformers_info
        if value is None:
            return {}
        try:
            if dataclasses.is_dataclass(value):
                return cast(dict[str, Any], _jsonable(asdict(value)))
        except Exception as exc:  # optional provider object
            logger.warning(
                "Could not serialize Transformers metadata (type=%s); continuing without it.",
                type(exc).__name__,
            )
        return {}

    @staticmethod
    def _safetensors_payload(info: ModelInfo) -> dict[str, Any] | None:
        value = info.safetensors
        if value is None or not dataclasses.is_dataclass(value):
            return None
        try:
            return cast(dict[str, Any], _jsonable(asdict(value)))
        except Exception as exc:  # optional provider object
            logger.warning(
                "Could not serialize safetensors metadata (type=%s); continuing without it.",
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _model_info_to_dict(info: ModelInfo) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": info.id,
            "modelId": info.id,
            "author": info.author,
            "sha": info.sha,
            "private": info.private,
            "gated": info.gated,
            "disabled": info.disabled,
            "pipeline_tag": info.pipeline_tag,
            "library_name": info.library_name,
            "tags": info.tags,
            "downloads": info.downloads,
            "likes": info.likes,
            "createdAt": info.created_at,
            "lastModified": info.last_modified,
            "cardData": HuggingFaceProvider._card_data_payload(info),
            "config": info.config,
            "transformersInfo": HuggingFaceProvider._transformers_info(info) or None,
            "safetensors": HuggingFaceProvider._safetensors_payload(info),
            "siblings": [
                {
                    "rfilename": sibling.rfilename,
                    "size": sibling.size,
                    "lfs": asdict(sibling.lfs) if sibling.lfs is not None else None,
                }
                for sibling in (info.siblings or [])
            ],
            "spaces": info.spaces,
            "usedStorage": info.used_storage,
        }
        return cast(dict[str, Any], _jsonable(payload))


def _jsonable(value: Any) -> Any:
    """Convert provider objects to deterministic JSON-safe values."""
    if isinstance(value, datetime):
        return format_utc(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
