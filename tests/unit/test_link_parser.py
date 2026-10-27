"""Link acceptance, canonicalization, and batch parsing tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ethi_asr_bench.errors import ErrorCode, InputError
from ethi_asr_bench.services.link_parser import (
    MAX_BATCH_BYTES,
    parse_batch_text,
    parse_inputs,
    parse_repository_target,
    read_batch_file,
)

EXAMPLE_ID = "b1n1yam/shook-medium-amharic-2k"
EXAMPLE_URL = f"https://huggingface.co/{EXAMPLE_ID}"


@pytest.mark.parametrize(
    ("raw", "expected_id", "expected_revision"),
    [
        (EXAMPLE_ID, EXAMPLE_ID, None),
        (EXAMPLE_URL, EXAMPLE_ID, None),
        (f"  {EXAMPLE_URL}  ", EXAMPLE_ID, None),
        (f"https://www.huggingface.co/{EXAMPLE_ID}", EXAMPLE_ID, None),
        (f"HTTPS://HUGGINGFACE.CO/{EXAMPLE_ID}", EXAMPLE_ID, None),
        (f"{EXAMPLE_URL}/", EXAMPLE_ID, None),
        (f"https://huggingface.co:443/{EXAMPLE_ID}", EXAMPLE_ID, None),
        ("gpt2", "gpt2", None),
        ("owner/model.name_1", "owner/model.name_1", None),
        (f"{EXAMPLE_URL}/tree/main", EXAMPLE_ID, "main"),
        (f"{EXAMPLE_URL}/tree/feature/amharic-v2", EXAMPLE_ID, "feature/amharic-v2"),
        (f"{EXAMPLE_URL}?revision=main", EXAMPLE_ID, "main"),
        (f"{EXAMPLE_URL}/tree/main?revision=main", EXAMPLE_ID, "main"),
    ],
)
def test_valid_inputs_are_canonicalized(
    raw: str, expected_id: str, expected_revision: str | None
) -> None:
    ref = parse_repository_target(raw)
    assert ref.repo_id == expected_id
    assert ref.canonical_url == f"https://huggingface.co/{expected_id}"
    assert ref.revision == expected_revision
    if "/" in expected_id:
        assert ref.owner == expected_id.split("/")[0]
        assert ref.name == expected_id.split("/")[1]
    else:
        assert ref.owner is None
        assert ref.name == expected_id
    assert ref.repo_type == "model"


def test_cli_revision_can_pin_a_bare_id() -> None:
    ref = parse_repository_target(EXAMPLE_ID, revision="refs/tags/v1.0")
    assert ref.revision == "refs/tags/v1.0"


def test_cli_revision_matching_url_revision_is_accepted() -> None:
    ref = parse_repository_target(f"{EXAMPLE_URL}/tree/main", revision="main")
    assert ref.revision == "main"


def test_conflicting_cli_and_url_revisions_are_rejected() -> None:
    with pytest.raises(InputError) as captured:
        parse_repository_target(f"{EXAMPLE_URL}/tree/main", revision="dev")
    assert captured.value.code is ErrorCode.CONFLICTING_REVISION


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "owner/model/extra",
        "owner//model",
        "/owner/model",
        "owner/model/",
        "owner/model?revision=main",
        "owner/model#fragment",
        "owner/model@main",
        "owner:model",
        "owner model/name",
        "owner/model name",
        "owner/-model",
        "owner/.model",
        "owner/..",
    ],
)
def test_malformed_repository_ids_are_rejected(raw: str) -> None:
    with pytest.raises(InputError):
        parse_repository_target(raw)


@pytest.mark.parametrize(
    "raw",
    [
        f"http://huggingface.co/{EXAMPLE_ID}",
        f"ftp://huggingface.co/{EXAMPLE_ID}",
        f"https://huggingface.co.attacker.example/{EXAMPLE_ID}",
        f"https://attacker.example/huggingface.co/{EXAMPLE_ID}",
        f"https://api.huggingface.co/{EXAMPLE_ID}",
        f"https://huggingface.co.evil/{EXAMPLE_ID}",
        f"https://huggingface.co@attacker.example/{EXAMPLE_ID}",
        f"https://user:pass@huggingface.co/{EXAMPLE_ID}",
        f"https://user@huggingface.co/{EXAMPLE_ID}",
        f"https://huggingface.co:8443/{EXAMPLE_ID}",
        f"https://huggingface.co/{EXAMPLE_ID}#readme",
        f"https://huggingface.co/{EXAMPLE_ID}?library=transformers",
        f"https://huggingface.co/{EXAMPLE_ID}?revision=main&revision=dev",
        f"https://huggingface.co/{EXAMPLE_ID}?revision=",
        "https://huggingface.co/%62%31%6e%31%79%61%6d/shook-medium-amharic-2k",
        f"https://huggingface.co//{EXAMPLE_ID}",
        "https://huggingface.co",
        "https://huggingface.co/",
        "https://huggingface.co/models",
        f"https://huggingface.co/models/{EXAMPLE_ID}",
        f"https://huggingface.co/datasets/{EXAMPLE_ID}",
        f"https://huggingface.co/spaces/{EXAMPLE_ID}",
        f"{EXAMPLE_URL}/blob/main/config.json",
        f"{EXAMPLE_URL}/resolve/main/config.json",
        f"{EXAMPLE_URL}/tree",
    ],
)
def test_unsafe_or_unsupported_urls_are_rejected(raw: str) -> None:
    with pytest.raises(InputError):
        parse_repository_target(raw)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (f"http://huggingface.co/{EXAMPLE_ID}", ErrorCode.UNSUPPORTED_SCHEME),
        (
            f"https://huggingface.co.attacker.example/{EXAMPLE_ID}",
            ErrorCode.UNSUPPORTED_HOST,
        ),
        (
            f"https://user:pass@huggingface.co/{EXAMPLE_ID}",
            ErrorCode.CREDENTIALS_IN_URL,
        ),
        (
            f"https://huggingface.co:8443/{EXAMPLE_ID}",
            ErrorCode.UNSUPPORTED_PORT,
        ),
        (
            f"https://huggingface.co/datasets/{EXAMPLE_ID}",
            ErrorCode.UNSUPPORTED_REPOSITORY_TYPE,
        ),
        (
            f"{EXAMPLE_URL}/blob/main/config.json",
            ErrorCode.UNSUPPORTED_URL_PATH,
        ),
        (
            f"{EXAMPLE_URL}/tree",
            ErrorCode.INVALID_REVISION,
        ),
        (
            f"{EXAMPLE_URL}?revision=",
            ErrorCode.INVALID_REVISION,
        ),
    ],
)
def test_error_codes_are_stable(raw: str, code: ErrorCode) -> None:
    with pytest.raises(InputError) as captured:
        parse_repository_target(raw)
    assert captured.value.code is code


@pytest.mark.parametrize(
    "revision",
    ["", " ", "main branch", "../main", "/main", "main/", "main//next", "a" * 201],
)
def test_invalid_revisions_are_rejected(revision: str) -> None:
    with pytest.raises(InputError):
        parse_repository_target(EXAMPLE_ID, revision=revision)


@pytest.mark.parametrize(
    "raw",
    [
        f"https://huggingface.co/{EXAMPLE_ID}/tree/{'x' * 201}",
        f"https://huggingface.co/{EXAMPLE_ID}/tree/../main",
        f"https://huggingface.co/{EXAMPLE_ID}/tree/main//next",
    ],
)
def test_invalid_url_revisions_are_rejected(raw: str) -> None:
    with pytest.raises(InputError):
        parse_repository_target(raw)


def test_query_string_is_not_treated_as_a_repository_name() -> None:
    with pytest.raises(InputError) as captured:
        parse_repository_target("huggingface.co/owner/model?foo=bar")
    assert captured.value.code is ErrorCode.INVALID_REPOSITORY_ID


def test_batch_parsing_preserves_lines_and_deduplicates() -> None:
    content = f"""
# comment
{EXAMPLE_ID}

{EXAMPLE_URL}
https://huggingface.co/{EXAMPLE_ID}/tree/main
not a repository
{EXAMPLE_URL}/tree/main
"""
    parsed = parse_batch_text(content)
    assert parsed.ignored_lines == 3
    assert len(parsed.items) == 5
    assert parsed.items[0].line_number == 3
    assert parsed.items[0].is_valid
    assert parsed.items[1].duplicate_of == EXAMPLE_ID
    assert parsed.items[2].ref is not None
    assert parsed.items[2].ref.revision == "main"
    assert parsed.items[3].error is not None
    assert not parsed.items[3].is_valid
    assert parsed.items[4].duplicate_of is not None


def test_batch_deduplication_is_case_insensitive_for_repository_id() -> None:
    parsed = parse_batch_text(f"{EXAMPLE_ID}\n{EXAMPLE_ID.upper()}\n")
    assert len(parsed.items) == 2
    assert parsed.items[1].duplicate_of == EXAMPLE_ID


def test_batch_accepts_utf8_bom() -> None:
    parsed = parse_batch_text(f"\ufeff{EXAMPLE_ID}\n")
    assert parsed.items[0].is_valid


def test_batch_revision_conflict_marks_the_line_invalid() -> None:
    parsed = parse_batch_text(f"{EXAMPLE_URL}/tree/main\n", revision="dev")
    assert len(parsed.items) == 1
    assert parsed.items[0].error is not None
    assert parsed.items[0].error.code == "conflicting_revision"


def test_parse_inputs_deduplicates_direct_arguments() -> None:
    parsed = parse_inputs([EXAMPLE_ID, EXAMPLE_URL, "other/model"])
    assert len(parsed.items) == 3
    assert parsed.items[1].duplicate_of == EXAMPLE_ID
    assert parsed.items[2].is_valid
    assert parsed.ignored_lines == 0


def test_read_batch_file_missing(tmp_path: Path) -> None:
    with pytest.raises(InputError) as captured:
        read_batch_file(tmp_path / "missing.txt")
    assert captured.value.code is ErrorCode.INVALID_BATCH_FILE


def test_read_batch_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        read_batch_file(tmp_path)


def test_read_batch_file_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.txt"
    path.write_bytes(b"x" * (MAX_BATCH_BYTES + 1))
    with pytest.raises(InputError):
        read_batch_file(path)


def test_read_batch_file_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "batch.txt"
    path.write_bytes(b"\xef\xbb\xbf" + EXAMPLE_ID.encode())
    assert read_batch_file(path).startswith(EXAMPLE_ID)


def test_read_batch_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "batch.txt"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(InputError) as captured:
        read_batch_file(path)
    assert captured.value.code is ErrorCode.INVALID_BATCH_FILE


def test_provider_neutral_canonical_urls_do_not_expand_accepted_input_hosts() -> None:
    with pytest.raises(InputError) as captured:
        parse_repository_target("https://mirror.example/models/owner/model")
    assert captured.value.code is ErrorCode.UNSUPPORTED_HOST
