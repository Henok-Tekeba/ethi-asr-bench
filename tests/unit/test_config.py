"""Configuration precedence, validation, and secret-handling tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ethi_asr_bench.config import Settings, default_data_dir, default_db_path
from ethi_asr_bench.errors import ConfigurationError, ErrorCode


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "ETHI_ASR_DB_PATH",
        "ETHI_ASR_HF_TIMEOUT",
        "ETHI_ASR_HF_MAX_RETRIES",
        "ETHI_ASR_HF_BACKOFF",
        "ETHI_ASR_LOG_LEVEL",
        "XDG_DATA_HOME",
        "LOCALAPPDATA",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_paths_use_home_and_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_data_dir() == tmp_path / ".local" / "share" / "ethi-asr-bench"
    assert default_db_path() == tmp_path / ".local" / "share" / "ethi-asr-bench" / "records.db"

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_data_dir() == tmp_path / "xdg" / "ethi-asr-bench"


def test_windows_default_uses_local_app_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert default_data_dir() == tmp_path / "local" / "ethi-asr-bench"


def test_environment_path_and_cli_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETHI_ASR_DB_PATH", str(tmp_path / "env.db"))
    assert Settings.from_env().db_path == tmp_path / "env.db"
    assert Settings.from_env(db_path=tmp_path / "cli.db").db_path == tmp_path / "cli.db"


def test_token_is_read_from_environment_and_hidden_from_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    settings = Settings.from_env()
    assert settings.hf_token == "hf_secret"
    assert "hf_secret" not in repr(settings)


def test_alternative_token_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_alt")
    assert Settings.from_env().hf_token == "hf_alt"


def test_hf_token_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "first")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "second")
    assert Settings.from_env().hf_token == "first"


def test_numeric_settings_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETHI_ASR_HF_TIMEOUT", "12.5")
    monkeypatch.setenv("ETHI_ASR_HF_MAX_RETRIES", "1")
    monkeypatch.setenv("ETHI_ASR_HF_BACKOFF", "2.5")
    monkeypatch.setenv("ETHI_ASR_LOG_LEVEL", "debug")
    settings = Settings.from_env()
    assert settings.timeout_seconds == 12.5
    assert settings.max_retries == 1
    assert settings.backoff_seconds == 2.5
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETHI_ASR_LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigurationError) as captured:
        Settings.from_env()
    assert captured.value.code is ErrorCode.INVALID_CONFIGURATION


def test_token_whitespace_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert Settings.from_env().hf_token is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ETHI_ASR_HF_TIMEOUT", "not-a-number"),
        ("ETHI_ASR_HF_TIMEOUT", "0"),
        ("ETHI_ASR_HF_TIMEOUT", "301"),
        ("ETHI_ASR_HF_MAX_RETRIES", "eleven"),
        ("ETHI_ASR_HF_MAX_RETRIES", "-1"),
        ("ETHI_ASR_HF_MAX_RETRIES", "11"),
        ("ETHI_ASR_HF_BACKOFF", "31"),
        ("ETHI_ASR_HF_BACKOFF", "oops"),
    ],
)
def test_invalid_numeric_settings_are_rejected(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError) as captured:
        Settings.from_env()
    assert captured.value.code is ErrorCode.INVALID_CONFIGURATION


def test_empty_numeric_settings_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ETHI_ASR_HF_TIMEOUT", "   ")
    settings = Settings.from_env()
    assert settings.timeout_seconds == 30.0


def test_cli_path_is_expanded(tmp_path: Path) -> None:
    settings = Settings.from_env(db_path=Path("~/custom.db"))
    assert settings.db_path == Path.home() / "custom.db"
