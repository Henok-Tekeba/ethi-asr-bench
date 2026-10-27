"""Shared test fixtures. No fixture contacts the network."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from ethi_asr_bench.errors import ErrorCode, ProviderError
from ethi_asr_bench.models.repository import InspectionRecord, RepositoryRef
from ethi_asr_bench.providers.base import ProviderFile, ProviderSnapshot
from ethi_asr_bench.services.metadata_extractor import extract_metadata
from ethi_asr_bench.storage.database import Database
from ethi_asr_bench.storage.repository_store import SQLiteRepositoryStore

COMMIT = "60b7ae174266ed42362ee02e636aea144011378d"
OTHER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
FIXED_TIME = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)

SAMPLE_README = """---
language: am
license: apache-2.0
---

# Shook Medium Amharic

A fine-tuned speech-recognition model for Amharic. The model card may mention
Whisper, but architecture claims in this file are not treated as structured facts.

## Usage

```bash
pip install transformers
```

Set trust_remote_code=True only after reviewing any custom code.
"""

SAMPLE_CONFIG = {
    "architectures": ["WhisperForConditionalGeneration"],
    "model_type": "whisper",
    "tokenizer_config": {"bos_token": "<|startoftranscript|>"},
}

SAMPLE_REQUIREMENTS = """# declared by the repository; never installed automatically
transformers==4.40.0
torch>=2.0
soundfile
"""


def make_snapshot(
    repo_id: str = "b1n1yam/shook-medium-amharic-2k",
    **overrides: Any,
) -> ProviderSnapshot:
    parts = repo_id.split("/")
    owner = parts[0] if len(parts) == 2 else None
    name = parts[-1]
    defaults: dict[str, Any] = {
        "repo_id": repo_id,
        "canonical_url": f"https://huggingface.co/{repo_id}",
        "owner": owner,
        "name": name,
        "requested_revision": None,
        "resolved_commit": COMMIT,
        "raw_model_info": {
            "id": repo_id,
            "modelId": repo_id,
            "author": owner,
            "sha": COMMIT,
            "pipeline_tag": "automatic-speech-recognition",
            "library_name": "transformers",
            "tags": [
                "transformers",
                "safetensors",
                "whisper",
                "am",
                "license:apache-2.0",
                "automatic-speech-recognition",
            ],
            "siblings": [],
        },
        "files": [
            ProviderFile(".gitattributes", 1519, False),
            ProviderFile("README.md", 3019, False),
            ProviderFile("config.json", 1328, False),
            ProviderFile("generation_config.json", 200, False),
            ProviderFile("model.safetensors", 3_000_000_000, True),
            ProviderFile("training_args.bin", 5000, False),
            ProviderFile("tokenizer.json", 3_466_000, False),
            ProviderFile("tokenizer_config.json", 1200, False),
            ProviderFile("requirements.txt", 120, False),
        ],
        "card_data": {
            "language": ["am"],
            "license": "apache-2.0",
            "metrics": ["wer"],
            "model_name": "Shook Medium Amharic",
        },
        "config": SAMPLE_CONFIG,
        "transformers_info": {"auto_model": "AutoModelForSpeechSeq2Seq"},
        "text_files": {
            "README.md": SAMPLE_README,
            "config.json": json.dumps(SAMPLE_CONFIG),
            "requirements.txt": SAMPLE_REQUIREMENTS,
        },
        "text_file_errors": [],
        "notes": [],
        "provider_name": "fake",
        "provider_version": "test",
    }
    defaults.update(overrides)
    return ProviderSnapshot(**defaults)


def make_record(
    snapshot: ProviderSnapshot | None = None,
    *,
    retrieved_at: datetime = FIXED_TIME,
) -> InspectionRecord:
    return extract_metadata(snapshot or make_snapshot(), retrieved_at=retrieved_at)


class FakeProvider:
    """Deterministic provider used by service, storage, and CLI tests."""

    name = "fake"

    def __init__(
        self,
        snapshots: dict[str, ProviderSnapshot] | None = None,
        errors: dict[str, ProviderError] | None = None,
    ) -> None:
        self.snapshots = {key.casefold(): value for key, value in (snapshots or {}).items()}
        self.errors = {key.casefold(): value for key, value in (errors or {}).items()}
        self.calls: list[RepositoryRef] = []

    def fetch_repository(self, ref: RepositoryRef) -> ProviderSnapshot:
        self.calls.append(ref)
        key = ref.repo_id.casefold()
        if key in self.errors:
            raise self.errors[key]
        snapshot = self.snapshots.get(key)
        if snapshot is None:
            raise ProviderError(
                ErrorCode.REPOSITORY_NOT_FOUND,
                f"Repository {ref.repo_id!r} was not found.",
            )
        return replace(
            snapshot,
            repo_id=ref.repo_id,
            canonical_url=f"https://huggingface.co/{ref.repo_id}",
            owner=ref.owner,
            name=ref.name,
            requested_revision=ref.revision,
            raw_model_info={**snapshot.raw_model_info, "id": ref.repo_id},
        )


@pytest.fixture
def snapshot_factory() -> Callable[..., ProviderSnapshot]:
    return make_snapshot


@pytest.fixture
def sample_snapshot() -> ProviderSnapshot:
    return make_snapshot()


@pytest.fixture
def sample_record(sample_snapshot: ProviderSnapshot) -> InspectionRecord:
    return make_record(sample_snapshot)


@pytest.fixture
def fake_provider(sample_snapshot: ProviderSnapshot) -> FakeProvider:
    return FakeProvider(snapshots={"b1n1yam/shook-medium-amharic-2k": sample_snapshot})


@pytest.fixture
def store(tmp_path: Any) -> SQLiteRepositoryStore:
    database = Database(tmp_path / "records.db")
    repository_store = SQLiteRepositoryStore(database)
    try:
        yield repository_store
    finally:
        repository_store.close()
