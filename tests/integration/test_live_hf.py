"""Optional live Hugging Face smoke test.

Run explicitly with:

    ETHI_ASR_LIVE=1 uv run pytest tests/integration/test_live_hf.py -m live -q

It is skipped in the default test suite and must never be required for CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ethi_asr_bench.models.repository import InspectionStatus, ValueState
from ethi_asr_bench.providers.huggingface import HuggingFaceProvider
from ethi_asr_bench.services.inspection import InspectionService
from ethi_asr_bench.services.link_parser import parse_repository_target
from ethi_asr_bench.storage.database import Database
from ethi_asr_bench.storage.repository_store import SQLiteRepositoryStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("ETHI_ASR_LIVE") != "1",
        reason="set ETHI_ASR_LIVE=1 to run live Hub tests",
    ),
]

LIVE_MODEL = "b1n1yam/shook-medium-amharic-2k"


def test_live_inspection_of_the_reference_model(tmp_path: Path) -> None:
    store = SQLiteRepositoryStore(Database(tmp_path / "live.db"))
    try:
        service = InspectionService(
            HuggingFaceProvider(timeout_seconds=30.0, max_retries=1),
            store,
        )
        outcome = service.inspect(parse_repository_target(LIVE_MODEL))
        record = outcome.record

        assert record.identity.repo_id == LIVE_MODEL
        assert record.status in {InspectionStatus.SUCCESS, InspectionStatus.PARTIAL}
        assert record.identity.resolved_commit is not None
        assert record.contents.has_weight_files is True
        assert record.custom_code.code_executed is False
        if record.model.architectures.state is ValueState.KNOWN:
            assert isinstance(record.model.architectures.value, list)
        if record.model.pipeline_tag.state is ValueState.KNOWN:
            assert record.model.pipeline_tag.value

        stored = store.get_latest(LIVE_MODEL)
        assert stored is not None
        assert stored.record.identity.resolved_commit == record.identity.resolved_commit
    finally:
        store.close()
