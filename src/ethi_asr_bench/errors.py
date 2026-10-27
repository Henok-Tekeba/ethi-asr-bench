"""Structured error types with stable machine-readable codes."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable error identifiers. Values are part of the public CLI contract."""

    # Input and validation
    INVALID_INPUT = "invalid_input"
    INVALID_REPOSITORY_ID = "invalid_repository_id"
    INVALID_REVISION = "invalid_revision"
    CONFLICTING_REVISION = "conflicting_revision"
    UNSUPPORTED_URL = "unsupported_url"
    UNSUPPORTED_HOST = "unsupported_host"
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    UNSUPPORTED_REPOSITORY_TYPE = "unsupported_repository_type"
    UNSUPPORTED_URL_PATH = "unsupported_url_path"
    CREDENTIALS_IN_URL = "credentials_in_url"
    UNSUPPORTED_PORT = "unsupported_port"
    INVALID_BATCH_FILE = "invalid_batch_file"
    INVALID_EXPORT_PATH = "invalid_export_path"
    EMPTY_INPUT = "empty_input"

    # Configuration
    INVALID_CONFIGURATION = "invalid_configuration"

    # Provider access
    REPOSITORY_NOT_FOUND = "repository_not_found"
    REPOSITORY_GATED = "repository_gated"
    PERMISSION_DENIED = "permission_denied"
    AUTHENTICATION_FAILED = "authentication_failed"
    REVISION_NOT_FOUND = "revision_not_found"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    HTTP_ERROR = "http_error"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    FILE_UNAVAILABLE = "file_unavailable"

    # Extraction
    EXTRACTION_FAILED = "extraction_failed"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"

    # Persistence and export
    STORAGE_ERROR = "storage_error"
    STORAGE_INITIALIZATION_FAILED = "storage_initialization_failed"
    RECORD_NOT_FOUND = "record_not_found"
    EXPORT_FAILED = "export_failed"

    # Catch-all
    INTERNAL_ERROR = "internal_error"


class EthiAsrBenchError(Exception):
    """Base class for expected application failures."""

    exit_code = 3

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.http_status = http_status

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation without credentials or tracebacks."""
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "http_status": self.http_status,
        }


class InputError(EthiAsrBenchError):
    """The caller supplied something that cannot be accepted as a repository target."""

    exit_code = 2


class ConfigurationError(EthiAsrBenchError):
    """The local configuration or environment is invalid."""

    exit_code = 2


class ProviderError(EthiAsrBenchError):
    """A remote metadata provider failed or returned an unusable response."""

    exit_code = 1


class ExtractionError(EthiAsrBenchError):
    """Metadata was retrieved but could not be normalized."""

    exit_code = 1


class StorageError(EthiAsrBenchError):
    """Local persistence failed."""

    exit_code = 3


class RecordNotFoundError(StorageError):
    """A requested local record does not exist."""

    exit_code = 1

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.RECORD_NOT_FOUND, message)


def exit_code_for(error: EthiAsrBenchError) -> int:
    """Map an application error to its documented process exit code."""
    return error.exit_code
