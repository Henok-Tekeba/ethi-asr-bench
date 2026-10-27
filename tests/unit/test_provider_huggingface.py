"""Hugging Face provider tests. All network access is mocked."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub import ModelInfo
from huggingface_hub import errors as hf_errors
from tests.conftest import COMMIT

from ethi_asr_bench.errors import ErrorCode, ProviderError
from ethi_asr_bench.providers.huggingface import HuggingFaceProvider
from ethi_asr_bench.services.link_parser import parse_repository_target

REF = parse_repository_target("owner/model")
ENDPOINT = "https://huggingface.co/api/models/owner/model"


def response(status: int, *, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        request=httpx.Request("GET", ENDPOINT),
        headers=headers or {},
    )


def make_model_info(**overrides: Any) -> ModelInfo:
    payload: dict[str, Any] = {
        "id": "owner/model",
        "author": "owner",
        "sha": COMMIT,
        "private": False,
        "gated": False,
        "pipeline_tag": "automatic-speech-recognition",
        "library_name": "transformers",
        "tags": ["transformers", "whisper", "am", "license:mit"],
        "siblings": [
            {"rfilename": "README.md", "size": 100},
            {"rfilename": "config.json", "size": 80},
            {"rfilename": "requirements.txt", "size": 20},
            {"rfilename": "model.safetensors", "size": 9_000_000_000},
            {"rfilename": "training_args.bin", "size": 100},
        ],
        "config": {
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
        },
        "cardData": {"language": ["am"], "license": "mit", "metrics": ["wer"]},
        "transformersInfo": {
            "auto_model": "AutoModelForSpeechSeq2Seq",
            "custom_class": None,
            "pipeline_tag": "automatic-speech-recognition",
            "processor": "AutoProcessor",
        },
        "safetensors": {"parameters": {"F32": 100}, "total": 100},
        "lastModified": "2026-10-01T00:00:00Z",
        "createdAt": "2026-09-01T00:00:00Z",
    }
    payload.update(overrides)
    return ModelInfo(**payload)


class StubApi:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def model_info(self, repo_id: str, **kwargs: Any) -> ModelInfo:
        self.calls.append({"repo_id": repo_id, **kwargs})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[no-any-return]


class StubDownloader:
    def __init__(
        self,
        contents: dict[str, str] | None = None,
        *,
        error: Exception | None = None,
        directory: Path | None = None,
    ) -> None:
        self.contents = contents or {}
        self.error = error
        self.directory = directory
        self.calls: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        filename = str(kwargs["filename"])
        self.calls.append(filename)
        if self.error is not None:
            raise self.error
        target = (self.directory or Path("/tmp")) / filename.replace("/", "_")
        target.write_text(self.contents.get(filename, ""), encoding="utf-8")
        return str(target)


def provider(
    api: StubApi,
    *,
    downloader: Callable[..., str] | None = None,
    sleep: Callable[[float], None] | None = None,
    token: str | None = None,
    max_retries: int = 2,
    max_text_file_bytes: int = 1_000_000,
) -> HuggingFaceProvider:
    return HuggingFaceProvider(
        api=api,  # type: ignore[arg-type]
        downloader=downloader or StubDownloader(),
        sleep=sleep or (lambda _seconds: None),
        token=token,
        timeout_seconds=12.5,
        max_retries=max_retries,
        backoff_seconds=0.25,
        max_text_file_bytes=max_text_file_bytes,
    )


def test_successful_snapshot_contains_metadata_and_only_allowlisted_text(
    tmp_path: Path,
) -> None:
    downloader = StubDownloader(
        {
            "README.md": "# Model\n\nAmharic ASR.",
            "config.json": json.dumps(
                {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"}
            ),
            "requirements.txt": "transformers==4.40.0\n",
        },
        directory=tmp_path,
    )
    api = StubApi([make_model_info()])
    snapshot = provider(api, downloader=downloader).fetch_repository(REF)

    assert snapshot.repo_id == "owner/model"
    assert snapshot.canonical_url == "https://huggingface.co/owner/model"
    assert snapshot.resolved_commit == COMMIT
    assert snapshot.requested_revision is None
    assert snapshot.card_data["license"] == "mit"
    assert snapshot.config is not None
    assert snapshot.transformers_info["auto_model"] == "AutoModelForSpeechSeq2Seq"
    assert set(snapshot.text_files) == {"README.md", "config.json", "requirements.txt"}
    assert snapshot.text_file_errors == []
    assert downloader.calls == ["README.md", "config.json", "requirements.txt"]
    assert "model.safetensors" not in downloader.calls
    assert api.calls[0]["timeout"] == 12.5
    assert api.calls[0]["files_metadata"] is True
    assert api.calls[0]["revision"] is None
    json.dumps(snapshot.raw_model_info)
    assert snapshot.raw_model_info["lastModified"] == "2026-10-01T00:00:00Z"


def test_requested_revision_is_passed_and_used_for_file_downloads(tmp_path: Path) -> None:
    api = StubApi([make_model_info()])
    downloader = StubDownloader(
        {"README.md": "card", "config.json": "{}", "requirements.txt": "x"},
        directory=tmp_path,
    )
    ref = parse_repository_target("owner/model", revision="dev")
    snapshot = provider(api, downloader=downloader).fetch_repository(ref)
    assert api.calls[0]["revision"] == "dev"
    assert snapshot.requested_revision == "dev"
    assert snapshot.resolved_commit == COMMIT


def test_text_files_above_the_size_limit_are_not_downloaded() -> None:
    info = make_model_info(
        siblings=[
            {"rfilename": "README.md", "size": 2_000_000},
            {"rfilename": "config.json", "size": 80},
        ]
    )
    downloader = StubDownloader({"config.json": "{}"})
    snapshot = provider(api=StubApi([info]), downloader=downloader).fetch_repository(REF)
    assert "README.md" not in snapshot.text_files
    assert any(error.filename == "README.md" for error in snapshot.text_file_errors)
    assert downloader.calls == ["config.json"]


def test_text_file_with_unknown_size_is_not_downloaded() -> None:
    info = make_model_info(siblings=[{"rfilename": "README.md", "size": None}])
    downloader = StubDownloader({"README.md": "card"})
    snapshot = provider(api=StubApi([info]), downloader=downloader).fetch_repository(REF)
    assert "README.md" not in snapshot.text_files
    assert snapshot.text_file_errors[0].filename == "README.md"


def test_text_file_download_failure_is_recorded_not_fatal() -> None:
    info = make_model_info(siblings=[{"rfilename": "README.md", "size": 100}])
    downloader = StubDownloader(error=OSError("cache unavailable"))
    snapshot = provider(api=StubApi([info]), downloader=downloader).fetch_repository(REF)
    assert snapshot.repo_id == "owner/model"
    assert snapshot.text_files == {}
    assert snapshot.text_file_errors[0].filename == "README.md"
    assert snapshot.text_file_errors[0].code == "file_unavailable"


def test_missing_siblings_produces_a_note() -> None:
    info = make_model_info(siblings=None)
    snapshot = provider(api=StubApi([info])).fetch_repository(REF)
    assert snapshot.files == []
    assert any("file listing" in note for note in snapshot.notes)


def test_repository_not_found_is_not_retried() -> None:
    api = StubApi([hf_errors.RepositoryNotFoundError("missing", response=response(404))])
    sleeps: list[float] = []
    with pytest.raises(ProviderError) as captured:
        provider(api, sleep=sleeps.append).fetch_repository(REF)
    assert captured.value.code is ErrorCode.REPOSITORY_NOT_FOUND
    assert captured.value.retryable is False
    assert len(api.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (
            hf_errors.GatedRepoError("gated", response=response(403)),
            ErrorCode.REPOSITORY_GATED,
            False,
        ),
        (
            hf_errors.RevisionNotFoundError("no revision", response=response(404)),
            ErrorCode.REVISION_NOT_FOUND,
            False,
        ),
        (
            hf_errors.HfHubHTTPError("unauthorized", response=response(401)),
            ErrorCode.AUTHENTICATION_FAILED,
            False,
        ),
        (
            hf_errors.HfHubHTTPError("forbidden", response=response(403)),
            ErrorCode.PERMISSION_DENIED,
            False,
        ),
        (
            hf_errors.HfHubHTTPError("rate limited", response=response(429)),
            ErrorCode.RATE_LIMITED,
            True,
        ),
        (
            hf_errors.HfHubHTTPError("unavailable", response=response(503)),
            ErrorCode.SERVICE_UNAVAILABLE,
            True,
        ),
        (
            httpx.ConnectTimeout("slow"),
            ErrorCode.TIMEOUT,
            True,
        ),
        (
            httpx.ConnectError("offline"),
            ErrorCode.NETWORK_ERROR,
            True,
        ),
        (
            hf_errors.OfflineModeIsEnabled("offline"),
            ErrorCode.NETWORK_ERROR,
            False,
        ),
    ],
)
def test_error_mapping(error: Exception, code: ErrorCode, retryable: bool) -> None:
    api = StubApi([error])
    with pytest.raises(ProviderError) as captured:
        provider(api, sleep=lambda _s: None).fetch_repository(REF)
    assert captured.value.code is code
    assert captured.value.retryable is retryable


def test_transient_failure_is_retried_with_bounded_backoff() -> None:
    api = StubApi(
        [
            hf_errors.HfHubHTTPError("unavailable", response=response(503)),
            hf_errors.HfHubHTTPError("unavailable", response=response(503)),
            make_model_info(),
        ]
    )
    sleeps: list[float] = []
    snapshot = provider(api, sleep=sleeps.append, max_retries=2).fetch_repository(REF)
    assert snapshot.repo_id == "owner/model"
    assert len(api.calls) == 3
    assert sleeps == [0.25, 0.5]


def test_retry_respects_retry_after_header() -> None:
    api = StubApi(
        [
            hf_errors.HfHubHTTPError(
                "rate limited", response=response(429, headers={"Retry-After": "7"})
            ),
            make_model_info(),
        ]
    )
    sleeps: list[float] = []
    provider(api, sleep=sleeps.append, max_retries=1).fetch_repository(REF)
    assert sleeps == [7.0]


def test_retries_are_bounded() -> None:
    api = StubApi([hf_errors.HfHubHTTPError("unavailable", response=response(503))])
    with pytest.raises(ProviderError) as captured:
        provider(api, sleep=lambda _s: None, max_retries=2).fetch_repository(REF)
    assert captured.value.code is ErrorCode.SERVICE_UNAVAILABLE
    assert len(api.calls) == 3


def test_token_is_redacted_from_error_messages() -> None:
    token = "test-token-not-real-1"
    api = StubApi([hf_errors.HfHubHTTPError(f"invalid token {token}", response=response(403))])
    with pytest.raises(ProviderError) as captured:
        provider(api, token=token, sleep=lambda _s: None).fetch_repository(REF)
    assert token not in captured.value.message
    assert "[REDACTED]" in captured.value.message


def test_mismatched_repository_id_is_rejected() -> None:
    api = StubApi([make_model_info(id="other/repository")])
    with pytest.raises(ProviderError) as captured:
        provider(api).fetch_repository(REF)
    assert captured.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_invalid_commit_identifier_is_rejected() -> None:
    api = StubApi([make_model_info(sha="not-a-sha")])
    with pytest.raises(ProviderError) as captured:
        provider(api).fetch_repository(REF)
    assert captured.value.code is ErrorCode.INVALID_PROVIDER_RESPONSE


def test_missing_commit_is_preserved_as_none() -> None:
    api = StubApi([make_model_info(sha=None)])
    snapshot = provider(api).fetch_repository(REF)
    assert snapshot.resolved_commit is None
    assert any("commit" in note for note in snapshot.notes)


def test_representative_fixture_is_valid_hub_metadata() -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "model_info.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    info = ModelInfo(**data)
    assert info.id == "b1n1yam/shook-medium-amharic-2k"
    assert info.sha == COMMIT
    assert info.siblings is not None
    assert info.card_data is not None
    assert info.card_data.language == ["am"]


def test_offline_mode_is_not_retried() -> None:
    api = StubApi([hf_errors.OfflineModeIsEnabled("offline")])
    with pytest.raises(ProviderError) as captured:
        provider(api, max_retries=3).fetch_repository(REF)
    assert captured.value.code is ErrorCode.NETWORK_ERROR
    assert len(api.calls) == 1


def test_token_is_not_included_in_raw_snapshot_metadata() -> None:
    token = "test-token-not-real-2"
    snapshot = provider(StubApi([make_model_info()]), token=token).fetch_repository(REF)
    assert token not in json.dumps(snapshot.raw_model_info)
    assert token not in json.dumps(snapshot.card_data)


def test_missing_text_entry_is_recorded_not_fatal() -> None:
    info = make_model_info(siblings=[{"rfilename": "README.md", "size": 100}])
    downloader = StubDownloader(error=hf_errors.EntryNotFoundError("missing entry"))
    snapshot = provider(api=StubApi([info]), downloader=downloader).fetch_repository(REF)
    assert snapshot.repo_id == "owner/model"
    assert snapshot.text_files == {}
    assert snapshot.text_file_errors[0].filename == "README.md"
    assert snapshot.text_file_errors[0].code == "file_unavailable"


def test_unexpected_optional_file_exception_does_not_fail_the_inspection() -> None:
    info = make_model_info(siblings=[{"rfilename": "README.md", "size": 100}])
    downloader = StubDownloader(error=ValueError("unexpected library bug"))
    snapshot = provider(api=StubApi([info]), downloader=downloader).fetch_repository(REF)
    assert snapshot.repo_id == "owner/model"
    assert snapshot.text_files == {}
    assert snapshot.text_file_errors[0].filename == "README.md"
    assert snapshot.text_file_errors[0].code == "file_unavailable"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"backoff_seconds": -1},
        {"max_text_file_bytes": 0},
        {"max_retries": -1},
    ],
)
def test_invalid_provider_bounds_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HuggingFaceProvider(api=StubApi([make_model_info()]), **kwargs)  # type: ignore[arg-type]


def test_unserializable_card_metadata_does_not_fail_the_snapshot() -> None:
    class BadCardData:
        def to_dict(self) -> dict[str, object]:
            raise TypeError("unexpected card object")

    info = make_model_info()
    info.card_data = BadCardData()  # type: ignore[assignment]
    snapshot = provider(StubApi([info])).fetch_repository(REF)
    assert snapshot.repo_id == "owner/model"
    assert snapshot.card_data == {}
    assert snapshot.raw_model_info["cardData"] == {}


def test_lfs_backed_text_files_are_not_downloaded() -> None:
    info = make_model_info(
        siblings=[
            {
                "rfilename": "README.md",
                "size": 500,
                "lfs": {"size": 500, "sha256": "a" * 64, "pointerSize": 134},
            }
        ]
    )
    downloader = StubDownloader({"README.md": "should-not-download"})
    snapshot = provider(api=StubApi([info]), downloader=downloader).fetch_repository(REF)
    assert snapshot.text_files == {}
    assert downloader.calls == []
    assert snapshot.text_file_errors[0].filename == "README.md"
    assert snapshot.text_file_errors[0].code == "file_unavailable"
