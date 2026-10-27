"""Safe acceptance and canonicalization of Hugging Face model references.

Only ``https`` URLs on an exact allowlist of Hugging Face hosts are accepted.
Domain matching is performed on the parsed hostname, never by substring search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from ethi_asr_bench.errors import ErrorCode, InputError
from ethi_asr_bench.models.repository import (
    REPOSITORY_SEGMENT_PATTERN,
    InspectionError,
    RepositoryRef,
    RepositoryType,
)

ALLOWED_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})
CANONICAL_HF_HOST = "huggingface.co"
MAX_INPUT_LENGTH = 2048
MAX_BATCH_BYTES = 1_000_000
MAX_BATCH_LINES = 10_000
MAX_REVISION_LENGTH = 200

_SEGMENT_RE = re.compile(REPOSITORY_SEGMENT_PATTERN)
_SCHEME_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_RESERVED_SINGLE_SEGMENTS = frozenset(
    {
        "api",
        "audio",
        "blog",
        "datasets",
        "docs",
        "inference",
        "join",
        "login",
        "models",
        "new",
        "organizations",
        "papers",
        "search",
        "settings",
        "spaces",
        "tasks",
    }
)
_FILE_PATH_SEGMENTS = frozenset({"blob", "raw", "resolve"})
_DATASET_OR_SPACE_SEGMENTS = frozenset({"datasets", "spaces"})


@dataclass(frozen=True, slots=True)
class ParsedInput:
    """One line or CLI argument and its parse outcome.

    ``ref`` is set only for syntactically valid, non-duplicate inputs. ``error``
    describes a rejected input without aborting the rest of a batch.
    """

    raw_input: str
    line_number: int | None
    ref: RepositoryRef | None = None
    error: InspectionError | None = None
    duplicate_of: str | None = None
    duplicate_of_line: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.ref is not None and self.error is None and self.duplicate_of is None


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Parsed inputs plus a count of blank/comment lines ignored."""

    items: list[ParsedInput] = field(default_factory=list)
    ignored_lines: int = 0


def parse_repository_target(raw: str, *, revision: str | None = None) -> RepositoryRef:
    """Parse one repository ID or Hugging Face URL into a canonical reference.

    ``revision`` optionally pins a revision for a bare repository ID. When a URL
    already carries a revision, both values must agree.
    """
    candidate = raw.strip()
    if not candidate:
        raise InputError(ErrorCode.EMPTY_INPUT, "The repository input is empty.")
    if len(candidate) > MAX_INPUT_LENGTH:
        raise InputError(
            ErrorCode.INVALID_INPUT,
            f"The repository input exceeds {MAX_INPUT_LENGTH} characters.",
        )
    if any(character.isspace() for character in candidate):
        raise InputError(
            ErrorCode.INVALID_INPUT,
            "Repository inputs must not contain whitespace.",
        )

    if _SCHEME_PREFIX_RE.match(candidate):
        return _parse_url(candidate, revision=revision)
    if candidate.lower().startswith(("http:", "https:")):
        return _parse_url(candidate, revision=revision)
    return _parse_repository_id(candidate, revision=revision)


def parse_inputs(raw_inputs: list[str], *, revision: str | None = None) -> ParseResult:
    """Parse CLI arguments, preserving order and deduplicating valid targets."""
    items: list[ParsedInput] = []
    seen: dict[tuple[str, str], ParsedInput] = {}
    for raw in raw_inputs:
        item = _parse_and_dedupe(raw, line_number=None, revision=revision, seen=seen)
        if item is not None:
            items.append(item)
    return ParseResult(items=items)


def parse_batch_text(content: str, *, revision: str | None = None) -> ParseResult:
    """Parse batch file content, ignoring blank lines and ``#`` comments."""
    content = content.removeprefix("\ufeff")
    lines = content.splitlines()
    if len(lines) > MAX_BATCH_LINES:
        raise InputError(
            ErrorCode.INVALID_BATCH_FILE,
            f"Batch input exceeds the {MAX_BATCH_LINES}-line limit.",
        )

    items: list[ParsedInput] = []
    seen: dict[tuple[str, str], ParsedInput] = {}
    ignored = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            ignored += 1
            continue
        item = _parse_and_dedupe(
            stripped,
            line_number=line_number,
            revision=revision,
            seen=seen,
        )
        if item is not None:
            items.append(item)
    return ParseResult(items=items, ignored_lines=ignored)


def read_batch_file(path: Path) -> str:
    """Read a user-supplied batch file with size and encoding checks."""
    try:
        if not path.exists():
            raise InputError(ErrorCode.INVALID_BATCH_FILE, f"Batch file does not exist: {path}")
        if not path.is_file():
            raise InputError(ErrorCode.INVALID_BATCH_FILE, f"Batch path is not a file: {path}")
        if path.stat().st_size > MAX_BATCH_BYTES:
            raise InputError(
                ErrorCode.INVALID_BATCH_FILE,
                f"Batch file exceeds the {MAX_BATCH_BYTES}-byte limit: {path}",
            )
        return path.read_text(encoding="utf-8-sig")
    except InputError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(
            ErrorCode.INVALID_BATCH_FILE,
            f"Could not read batch file {path}: {exc}",
        ) from exc


def _parse_and_dedupe(
    raw: str,
    *,
    line_number: int | None,
    revision: str | None,
    seen: dict[tuple[str, str], ParsedInput],
) -> ParsedInput | None:
    try:
        ref = parse_repository_target(raw, revision=revision)
    except InputError as exc:
        return ParsedInput(
            raw_input=raw,
            line_number=line_number,
            error=InspectionError(code=exc.code.value, message=exc.message),
        )

    key = (ref.repo_id.casefold(), ref.revision or "")
    previous = seen.get(key)
    if previous is not None:
        return ParsedInput(
            raw_input=raw,
            line_number=line_number,
            ref=ref,
            duplicate_of=previous.raw_input,
            duplicate_of_line=previous.line_number,
        )
    item = ParsedInput(raw_input=raw, line_number=line_number, ref=ref)
    seen[key] = item
    return item


def _parse_repository_id(candidate: str, *, revision: str | None) -> RepositoryRef:
    if "?" in candidate or "#" in candidate:
        raise InputError(
            ErrorCode.INVALID_REPOSITORY_ID,
            "Query strings and fragments are only accepted in full Hugging Face URLs.",
        )
    if "@" in candidate:
        raise InputError(
            ErrorCode.INVALID_REPOSITORY_ID,
            "The 'owner/model@revision' form is not supported; use --revision.",
        )
    if ":" in candidate:
        raise InputError(
            ErrorCode.INVALID_REPOSITORY_ID,
            "A repository ID must not contain ':'. URLs must start with https://.",
        )
    if candidate.startswith("/") or candidate.endswith("/") or "//" in candidate:
        raise InputError(
            ErrorCode.INVALID_REPOSITORY_ID,
            "A repository ID must be 'owner/model' or a single model name, without slashes.",
        )

    segments = candidate.split("/")
    if len(segments) > 2:
        raise InputError(
            ErrorCode.INVALID_REPOSITORY_ID,
            "A repository ID must contain at most one slash ('owner/model').",
        )
    _validate_segment(segments[0], role="owner or model name")
    if len(segments) == 2:
        _validate_segment(segments[1], role="model name")

    resolved_revision = _validate_revision(revision) if revision is not None else None
    return _build_ref(segments, revision=resolved_revision)


def _parse_url(candidate: str, *, revision: str | None) -> RepositoryRef:
    try:
        parsed = urlsplit(candidate)
    except ValueError as exc:
        raise InputError(ErrorCode.UNSUPPORTED_URL, f"Malformed URL: {exc}") from exc

    if parsed.scheme.lower() != "https":
        raise InputError(
            ErrorCode.UNSUPPORTED_SCHEME,
            "Only https:// URLs are accepted.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise InputError(
            ErrorCode.CREDENTIALS_IN_URL,
            "URLs containing embedded credentials are rejected.",
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise InputError(ErrorCode.UNSUPPORTED_PORT, "The URL contains an invalid port.") from exc
    if port is not None and port != 443:
        raise InputError(
            ErrorCode.UNSUPPORTED_PORT,
            f"Unexpected port {port}; use the default HTTPS port.",
        )

    hostname = parsed.hostname
    if hostname is None:
        raise InputError(ErrorCode.UNSUPPORTED_URL, "The URL has no hostname.")
    normalized_host = hostname.lower()
    if normalized_host not in ALLOWED_HF_HOSTS:
        raise InputError(
            ErrorCode.UNSUPPORTED_HOST,
            f"Unsupported host {hostname!r}; only huggingface.co is accepted.",
        )
    if "%" in parsed.path:
        raise InputError(
            ErrorCode.UNSUPPORTED_URL_PATH,
            "Percent-encoded URL paths are not accepted.",
        )
    if parsed.fragment:
        raise InputError(
            ErrorCode.UNSUPPORTED_URL,
            "URL fragments are not accepted; use a /tree/<revision> path or ?revision=.",
        )

    url_revision = _revision_from_query(parsed.query)
    if revision is not None and url_revision is not None:
        requested = _validate_revision(revision)
        if requested != url_revision:
            raise InputError(
                ErrorCode.CONFLICTING_REVISION,
                f"URL revision {url_revision!r} conflicts with --revision {requested!r}.",
            )
    effective_revision = url_revision if url_revision is not None else revision
    resolved_revision = (
        _validate_revision(effective_revision) if effective_revision is not None else None
    )

    path = parsed.path
    if not path.startswith("/"):
        raise InputError(ErrorCode.UNSUPPORTED_URL, "The URL has no repository path.")
    path_without_leading_slash = path[1:].rstrip("/")
    segments = path_without_leading_slash.split("/") if path_without_leading_slash else []
    if any(segment == "" for segment in segments):
        raise InputError(
            ErrorCode.UNSUPPORTED_URL_PATH,
            "The URL contains an empty path segment.",
        )
    if not segments:
        raise InputError(
            ErrorCode.UNSUPPORTED_URL_PATH,
            "The URL does not identify a model repository.",
        )

    if segments[0].lower() in _DATASET_OR_SPACE_SEGMENTS:
        raise InputError(
            ErrorCode.UNSUPPORTED_REPOSITORY_TYPE,
            "Only model repositories are supported, not datasets or spaces.",
        )

    if len(segments) == 1:
        _validate_segment(segments[0], role="model name")
        if segments[0].lower() in _RESERVED_SINGLE_SEGMENTS:
            raise InputError(
                ErrorCode.UNSUPPORTED_URL_PATH,
                "The URL is a Hugging Face site path, not a model repository.",
            )
        _reject_duplicate_revision(resolved_revision, source="URL", explicit=revision)
        return _build_ref(segments, revision=resolved_revision)

    if len(segments) < 2:
        raise InputError(ErrorCode.UNSUPPORTED_URL_PATH, "Unsupported Hugging Face URL path.")

    _validate_segment(segments[0], role="owner")
    _validate_segment(segments[1], role="model name")

    if len(segments) == 2:
        _reject_duplicate_revision(resolved_revision, source="URL", explicit=revision)
        return _build_ref(segments, revision=resolved_revision)

    marker = segments[2].lower()
    if marker == "tree":
        if len(segments) < 4:
            raise InputError(
                ErrorCode.INVALID_REVISION,
                "The /tree/ URL path is missing a revision.",
            )
        path_revision = _validate_revision("/".join(segments[3:]))
        if resolved_revision is not None and path_revision != resolved_revision:
            raise InputError(
                ErrorCode.CONFLICTING_REVISION,
                f"URL path revision {path_revision!r} conflicts with "
                f"revision {resolved_revision!r}.",
            )
        return _build_ref(segments[:2], revision=path_revision)
    if marker in _FILE_PATH_SEGMENTS:
        raise InputError(
            ErrorCode.UNSUPPORTED_URL_PATH,
            "File URLs are not accepted; submit the model repository, not a file.",
        )
    raise InputError(
        ErrorCode.UNSUPPORTED_URL_PATH,
        "Unsupported Hugging Face URL path; expected /owner/model or /owner/model/tree/<revision>.",
    )


def _revision_from_query(query: str) -> str | None:
    if not query:
        return None
    pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=False)
    keys = [key for key, _ in pairs]
    if any(key != "revision" for key in keys):
        unexpected = sorted({key for key in keys if key != "revision"})
        raise InputError(
            ErrorCode.UNSUPPORTED_URL,
            f"Unsupported URL query parameter(s): {', '.join(unexpected)}.",
        )
    if keys.count("revision") > 1:
        raise InputError(
            ErrorCode.UNSUPPORTED_URL,
            "The revision query parameter must appear at most once.",
        )
    revision = next((value for key, value in pairs if key == "revision"), "")
    if revision == "":
        raise InputError(ErrorCode.INVALID_REVISION, "The revision query value is empty.")
    return _validate_revision(revision)


def _reject_duplicate_revision(
    resolved_revision: str | None,
    *,
    source: str,
    explicit: str | None,
) -> None:
    if resolved_revision is not None and explicit is not None:
        requested = _validate_revision(explicit)
        if requested != resolved_revision:
            raise InputError(
                ErrorCode.CONFLICTING_REVISION,
                f"{source} revision {resolved_revision!r} conflicts with --revision {requested!r}.",
            )


def _validate_segment(value: str, *, role: str) -> None:
    if value in {".", ".."} or not _SEGMENT_RE.match(value):
        raise InputError(
            ErrorCode.INVALID_REPOSITORY_ID,
            f"Invalid {role} {value!r}; use letters, digits, '.', '_' or '-'.",
        )


def _validate_revision(value: str | None) -> str:
    if value is None:
        raise InputError(ErrorCode.INVALID_REVISION, "A revision value is required.")
    candidate = value.strip()
    if not candidate or len(candidate) > MAX_REVISION_LENGTH:
        raise InputError(
            ErrorCode.INVALID_REVISION,
            f"Revisions must be 1-{MAX_REVISION_LENGTH} characters long.",
        )
    if any(character.isspace() for character in candidate):
        raise InputError(ErrorCode.INVALID_REVISION, "Revisions must not contain whitespace.")
    if (
        not _REVISION_RE.match(candidate)
        or ".." in candidate
        or candidate.startswith("/")
        or candidate.endswith("/")
        or "//" in candidate
    ):
        raise InputError(
            ErrorCode.INVALID_REVISION,
            "Invalid revision; use a branch, tag, or commit name without traversal segments.",
        )
    return candidate


def _build_ref(segments: list[str], *, revision: str | None) -> RepositoryRef:
    repo_id = "/".join(segments)
    owner = segments[0] if len(segments) == 2 else None
    name = segments[1] if len(segments) == 2 else segments[0]
    return RepositoryRef(
        repo_id=repo_id,
        owner=owner,
        name=name,
        repo_type=RepositoryType.MODEL,
        canonical_url=f"https://{CANONICAL_HF_HOST}/{repo_id}",
        revision=revision,
    )
