# Contributing to EthiASR Bench

Thank you for considering a contribution. This project is intended to become shared
research infrastructure for Amharic and, eventually, other Ethiopian-language speech
technology. Small, well-tested, honest improvements are more valuable than broad
unfinished features.

## Scope

Stage 1 covers link acceptance, metadata inspection, normalization, SQLite storage,
JSON export, and the CLI. It does **not** run models, download weights, install model
dependencies, or calculate WER/CER. Please do not add inference or evaluation code
without a design discussion and a security review.

Good first contributions include:

- parser and host-validation edge cases;
- fixtures representing diverse metadata shapes;
- tests for malformed or incomplete repository metadata;
- documentation corrections;
- language-tag or license normalization improvements;
- CLI usability fixes that preserve machine-readable output and exit codes.

## Development setup

Python 3.11+ is required.

### Recommended: uv

```bash
uv sync --all-groups
uv run ethi-asr --version
```

### pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install "pytest>=8.3" "pytest-cov>=5" "ruff>=0.9" "mypy>=1.14"
```

## Commands

```bash
# Full offline test suite
uv run pytest

# Coverage
uv run pytest --cov=ethi_asr_bench --cov-report=term-missing

# Lint and formatting
uv run ruff check .
uv run ruff format --check .

# Static types for the application package
uv run mypy src

# Optional live Hub test (not required for most changes)
ETHI_ASR_LIVE=1 uv run pytest tests/integration/test_live_hf.py -m live -q
```

Run `uv run ruff format .` before submitting. The CI also runs `uv build`.

## Code conventions

- Support Python 3.11 and newer.
- Use type hints for application code.
- Follow Ruff and PEP 8; line length is 100.
- Keep CLI handlers thin. Business logic belongs in `services/`.
- Keep provider access behind `RepositoryProvider`.
- Keep SQL parameterized and confined to `storage/`.
- Keep untrusted model metadata out of Rich markup and out of any execution path.
- Add tests for behavior changes, not only for happy paths.
- Do not introduce a dependency without a clear need.
- Prefer clear error codes in `errors.py` over ad hoc strings.

## Tests

- Automated tests must not require live network access.
- Use `tests/conftest.py` or local fakes for provider behavior.
- Mark any test that contacts the Hub with `@pytest.mark.live` and guard it with
  `ETHI_ASR_LIVE=1`.
- Add regression tests for bugs.
- For parser changes, include malicious and lookalike inputs.
- For schema changes, update `docs/metadata-schema.md` and add serialization/round-trip
  tests.

## Pull requests

A good pull request:

1. Explains the user or maintainer problem.
2. Stays within the current stage or documents why an extension point is needed.
3. Includes tests.
4. Updates documentation when behavior, commands, schema, or security boundaries
   change.
5. Reports the commands that were actually run and their results.
6. Does not include local databases, exports, caches, tokens, or generated model
   files.

The pull request template includes a checklist. Keep commits reviewable; there is no
strict commit-message format yet.

## Schema and migration changes

Stored records are a public contract. If you change `InspectionRecord`:

- read the versioning policy in `docs/metadata-schema.md`;
- add fields with safe defaults where possible;
- update the schema documentation and example;
- add a migration plan for incompatible changes;
- ensure old valid records still load, or provide a clear storage error and migration
  path.

## Security issues

Do not report vulnerabilities in a public issue. Follow [SECURITY.md](SECURITY.md).

## Community

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Please be
respectful, specific, and generous with context, especially when discussing language
data, naming, and community priorities.

## Reporting ordinary bugs

Use the issue templates and include:

- the command or API call;
- input with tokens and private repository names removed;
- expected and actual behavior;
- exit code;
- relevant warnings or error codes;
- Python version, operating system, and project version.

Never paste a Hugging Face token or private model metadata into an issue.
