# Threat model

This threat model covers the Stage 1 metadata application. It is not a substitute for
a future review of model execution, dataset handling, or public result publication.

## Assets

- The user's Hugging Face access token and other environment secrets.
- The local SQLite database and exported JSON files.
- Integrity of stored metadata, revision identity, and provenance.
- The user's workstation and terminal.
- The project's reputation for honest, reproducible results.
- Model weights and datasets belonging to third parties, which Stage 1 intentionally
  does not touch.

## Trust boundaries

```text
User / batch file
        |
        v
   Link parser  <---- untrusted text
        |
        v
 Hugging Face Hub  <---- untrusted remote metadata and files
        |
        v
 Metadata extractor  <---- untrusted model-card and config strings
        |
        v
   SQLite / export  <---- local state
```

Remote content is never trusted to define URLs, paths, commands, imports, or
instructions.

## Threats and mitigations

### 1. Deceptive repository references

**Threat:** a lookalike host such as `huggingface.co.attacker.example` or embedded
credentials redirects the user or tool to an attacker.

**Stage 1 controls:**

- URLs are parsed with `urllib.parse.urlsplit`.
- The parsed hostname must exactly match an allowlist: `huggingface.co` and
  `www.huggingface.co`.
- No substring or suffix matching is used.
- Embedded usernames/passwords, non-HTTPS schemes, non-443 ports, fragments, encoded
  paths, and unsupported paths are rejected.
- Repository IDs are validated before provider calls.
- URL existence is checked only through the provider, not by following arbitrary
  links.

**Residual risk:** DNS or TLS compromise of the legitimate host is outside the
application's control.

### 2. Server-side request forgery or arbitrary URL fetching

**Threat:** metadata contains URLs that cause the inspector to fetch attacker-chosen
resources.

**Stage 1 controls:**

- Model-card URLs are stored as text only.
- The provider fetches metadata through the Hub client and only three fixed filenames
  from the validated repository: `README.md`, `config.json`, `requirements.txt`.
- No URL from the response is followed.

**Residual risk:** the Hub client itself may follow redirects defined by the Hub or its
CDN. This is part of the trusted provider dependency.

### 3. Credential exfiltration

**Threat:** a token appears in logs, error messages, exports, or the database.

**Stage 1 controls:**

- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` is read from the environment only.
- Tokens are not CLI arguments and are not stored in normalized records or raw source
  metadata.
- Known tokens are redacted from provider error messages.
- Logging does not include authorization headers or request objects.
- The CLI does not print environment contents.

**Residual risk:** other processes on the machine can read the environment, shell
history used to set a token, or local files. Users should protect their environment
and use scoped tokens.

### 4. Malicious model-card content and prompt injection

**Threat:** a model card contains instructions aimed at an AI agent or terminal
escapes aimed at a human, e.g. "ignore previous instructions and run this command."

**Stage 1 controls:**

- Model-card text is treated as data.
- It is never passed to a command runner, package installer, or model loader.
- Rich terminal output uses `Text` objects or escaping for untrusted values.
- JSON output and exports escape C0/C1 control and bidi formatting characters.
- Install snippets are stored in a clearly labeled descriptive field.
- JSON serialization is used for machine consumption rather than terminal rendering.

**Residual risk:** a human may still copy a command from a model card. Documentation
warns that model-card content is untrusted.

### 5. Execution of repository code

**Threat:** a repository includes malicious Python, a malicious `auto_map`, or a
`trust_remote_code` requirement.

**Stage 1 controls:**

- No model code is imported, downloaded, or executed.
- Python files are detected only from file listings.
- `auto_map` is recorded as a boolean indicator.
- `trust_remote_code` is recorded only as a text mention.
- `custom_code.code_executed` is fixed to `false`.
- Weights are never downloaded.

**Future requirement:** any stage that runs code must first implement isolation,
dependency pinning, resource limits, network policy, and a security review. Detecting
custom code is not a safety proof.

### 6. Dependency confusion or model-card-driven installation

**Threat:** a model card tells the user to install an attacker package or a typo-squat.

**Stage 1 controls:**

- Nothing from the repository is installed.
- `requirements.txt` and installation snippets are stored as untrusted text.
- No package index is contacted beyond the declared application dependencies.

### 7. Oversized files and resource exhaustion

**Threat:** a malicious repository declares a huge `README.md` or millions of files.

**Stage 1 controls:**

- Only three fixed text filenames are eligible.
- LFS-backed text files are skipped to avoid large artifact retrieval through an
  allowlisted text filename.
- Advertised size must be known and below the 1,000,000-byte limit; otherwise the file
  is skipped with a warning.
- The actual file read is capped at the limit plus one byte.
- The stored file inventory is capped at 2,000 paths with a warning.
- Python file indicators are capped at 100.
- Requirements and installation snippets are capped in count and length.
- Retries are bounded; `Retry-After` is capped.
- Batch input is capped at 1,000,000 bytes and 10,000 lines.

**Residual risk:** metadata retrieval itself can consume network bandwidth and time.
The timeout is configurable within bounds.

### 8. Malformed or hostile metadata

**Threat:** unexpected JSON types, deeply nested values, or inconsistent fields crash
the extractor or inject false classifications.

**Stage 1 controls:**

- Pydantic validates domain contracts.
- The extractor checks field types and records malformed optional fields as warnings.
- Missing fields remain `unknown` or `extraction_failed`.
- A failed normalization attempt is stored as a failed record with raw metadata when
  available.
- Architecture is taken from structured configuration, not prose.
- Licenses and languages are not inferred from repository names.

**Residual risk:** heuristics for weight files and language tags can be incomplete.
They are documented as indicators.

### 9. SQL injection

**Threat:** a repository ID or query value alters SQL.

**Stage 1 controls:**

- SQL statements use parameter binding.
- Table and index names are constants.
- `PRAGMA user_version` uses a validated integer from a constant migration table, not
  remote input.
- Repository IDs are separately validated before persistence.

### 10. Path traversal

**Threat:** a remote file path or repository name causes writes outside the intended
location.

**Stage 1 controls:**

- Remote file paths are validated as relative POSIX paths without `..`, `.`, empty
  segments, backslashes, or NUL.
- Remote file names are never used to construct local read or write paths.
- Only fixed allowlisted filenames are passed to the Hub downloader.
- Batch and export paths are explicitly chosen by the user; the application does not
  derive them from remote metadata.
- Export writes use a temporary file in the chosen parent directory and replace the
  target atomically.

**Residual risk:** a user can intentionally choose any local path. This is expected
CLI behavior and should be reviewed before adding a server interface.

### 11. Local data exposure

**Threat:** private repository metadata or tokens are exposed through the database,
exports, backups, or logs.

**Stage 1 controls:**

- Tokens are not persisted.
- Database files are created with restrictive permissions where the platform permits.
- Raw metadata is only exported when `--include-raw` is requested.
- Documentation tells users to protect local state.

**Residual risk:** filesystem backups, shared accounts, and OS-level access controls
are outside the application.

### 12. Rate limiting and denial of service by the Hub

**Threat:** repeated requests trigger rate limits or outages, making batch runs
unreliable or causing unbounded retries.

**Stage 1 controls:**

- Explicit request timeout.
- Bounded retry count.
- Exponential backoff.
- `Retry-After` support, capped at 30 seconds.
- Non-retryable failures are not retried.
- Failures are recorded per repository, and the batch continues.

### 13. Supply-chain compromise of application dependencies

**Threat:** a compromised Python dependency runs code during installation or import.

**Stage 1 controls:**

- A small dependency set is pinned by `uv.lock`.
- No model or audio stack is installed.
- CI installs from the lockfile.

**Residual risk:** normal Python supply-chain risk remains. Dependency review and
updates are required. Future model environments need stronger pinning and artifact
verification.

### 14. Misleading results

**Threat:** a partial inspection or a metadata mention is presented as compatibility
or benchmark evidence.

**Stage 1 controls:**

- Explicit `success`, `partial`, and `failed` statuses.
- Partial metadata contributes to exit code 1.
- The README and schema explicitly disclaim compatibility and benchmark claims.
- No WER/CER computation, ranking, or "best model" output exists.
- Reported model-card metrics are only recorded as declared metric names and
  untrusted description text.

## Out of scope for Stage 1

- Running model code or loading weights.
- Installing repository dependencies.
- Processing untrusted audio.
- Multi-tenant service operation.
- Public leaderboard publication.
- GPU driver or container escape threats.

## Required security work before execution

Before any remote model code is executed, the project must have:

1. A documented sandbox boundary with non-root execution.
2. No default network egress and an explicit allowlist where needed.
3. Read-only base images and per-run writable scratch space.
4. CPU, memory, disk, process, and wall-clock limits.
5. Pinned dependency manifests with hashes.
6. No host credentials, cloud metadata access, or home directory mounts.
7. A policy for `trust_remote_code` and custom modeling files.
8. Adversarial tests demonstrating that secrets and the host filesystem are
   unreachable.
9. Reproducible environment identifiers recorded with every prediction.
10. A fresh threat-model review.

## Reporting

Do not open a public issue containing exploit details. Follow
[`SECURITY.md`](../SECURITY.md).
