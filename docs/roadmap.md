# Roadmap

This roadmap is a plan, not a statement of existing capability. Stage 1 is the only
implemented stage. Completion criteria are intentionally concrete so progress can be
reviewed and so later stages do not inherit vague claims.

## Guiding rules

1. Preserve provenance and revision identity at every stage.
2. Separate metadata classification from execution.
3. Never execute third-party model code without a designed isolation boundary.
4. Never present a partial or unverified result as a benchmark result.
5. Prefer small, reviewable increments over speculative infrastructure.

## Stage 1 — Metadata foundation (current)

**Goal:** discover and inspect Hugging Face model repositories safely; normalize and
persist metadata without downloading weights or running code.

Implemented:

- [x] Exact-host link parsing for model IDs and HTTPS URLs.
- [x] Revision capture and URL/CLI conflict detection.
- [x] Batch parsing, per-line errors, deduplication, and continuation after failures.
- [x] Replaceable provider interface with a Hugging Face implementation.
- [x] Bounded retries, timeouts, rate-limit handling, and stable provider errors.
- [x] Small text-file allowlist (`README.md`, `config.json`, `requirements.txt`) with
  size caps.
- [x] Versioned normalized Pydantic schema with epistemic states and provenance.
- [x] File inventory, weight indicators, tokenizer/config presence, and custom-code
  indicators.
- [x] SQLite persistence with revision history, failed attempts, and idempotent
  reinspection.
- [x] JSON export (latest or full history; optional raw source).
- [x] Human and JSON CLI output with documented exit codes.
- [x] Unit and integration tests without live network dependence.
- [x] Ruff, mypy, optional live smoke test, and GitHub Actions CI.
- [x] Architecture, schema, roadmap, threat-model, contribution, and security docs.

Completion criteria:

- [x] A clean installation works from the documented commands.
- [x] Tests pass offline.
- [x] Linting and type checks pass.
- [x] A live inspection of the reference repository succeeds or is honestly reported
  as unavailable.
- [x] No model weights are downloaded and no repository code is executed.

Known Stage 1 boundaries are listed in the root README.

## Stage 2 — Compatibility classification

**Goal:** turn structured metadata into explicit, evidence-backed compatibility classes
without running models.

Planned work:

- Define a separate, versioned `CompatibilityAssessment` contract.
- Classify only from structured configuration, known library declarations, file
  formats, and explicit rules.
- Record rule identifiers and evidence for every class.
- Represent `unknown` and `unsupported` distinctly.
- Add fixtures for Whisper, other ASR families, non-ASR models, malformed configs, and
  custom-code repositories.
- Add regression tests for repositories whose names or descriptions mention a family
  that their configuration does not declare.
- Expose `ethi-asr classify` or an extension of `show`, without rewriting stored
  metadata.

Completion criteria:

- Every verdict has machine-readable evidence.
- No verdict relies on model-card prose alone.
- Classification remains offline-testable.
- Custom code is still never downloaded or executed.
- Existing Stage 1 records remain readable.

## Stage 3 — Isolated execution environments

**Goal:** design and prove isolation before any model code runs.

Required design work:

- Container or equivalent sandbox with a non-root user and read-only base image.
- Explicit network policy, including denial of arbitrary egress by default.
- CPU/memory/disk/time limits and process supervision.
- Pinned, hash-verified dependencies and artifact provenance.
- No host credentials, SSH agents, cloud metadata access, or user home mounts.
- Explicit policy for custom modeling code and `trust_remote_code`.
- Audit logging and reproducible environment manifests.
- Threat-model review and adversarial tests.

Completion criteria:

- A deliberately malicious fixture cannot read host secrets, write outside its
  workspace, or reach unapproved network destinations.
- Every environment is reproducible from a manifest.
- Failure cleanup is tested.

## Stage 4 — Standard model runner

**Goal:** produce a common prediction contract across model families.

Planned work:

- Define a versioned prediction contract: audio reference, hypothesis, timing,
  metadata, and error fields.
- Implement a Whisper fine-tune adapter first.
- Add additional ASR families only after the contract is stable.
- Pin model revisions and environment images.
- Record decoding parameters and runtime versions.
- Handle model-loading failures as first-class results.

Completion criteria:

- At least two compatible models can be run through the same interface.
- Predictions are reproducible for a fixed input, environment, and revision.
- Failures are recorded, not discarded.
- No path bypasses the sandbox.

## Stage 5 — Dataset and audio pipeline

**Goal:** make evaluation inputs controlled and reproducible.

Planned work:

- Versioned dataset manifests with checksums and licensing.
- Audio validation, decoding, channel handling, resampling, and duration limits.
- Deterministic splits and transcript normalization rules.
- Dataset access controls and provenance.
- Fixtures for silence, clipping, multiple channels, and malformed audio.

Completion criteria:

- A dataset revision can be identified by checksum.
- Preprocessing is deterministic and documented.
- Dataset license and access constraints are recorded.
- Evaluation does not silently download unapproved data.

## Stage 6 — Evaluation

**Goal:** calculate transparent error metrics with reproducible configurations.

Planned work:

- WER and CER with explicit normalization and tokenization rules.
- Per-sample and aggregate outputs.
- Confidence intervals and handling of empty references.
- Additional metrics where justified.
- Failure accounting for empty or invalid predictions.
- Versioned evaluation configuration.

Completion criteria:

- Metrics are reproducible from stored predictions and configuration.
- Normalization choices are visible in reports.
- Partial runs are labeled partial.
- No leaderboard is published without revision, dataset, configuration, and run
  provenance.

## Stage 7 — Publication

**Goal:** publish reproducible comparisons and a trustworthy public interface.

Planned work:

- Versioned result artifacts and report generation.
- Immutable run identifiers.
- A public comparison interface that links every score to its evidence.
- Correction and deprecation workflows.
- Governance for model/dataset inclusion and disputes.

Completion criteria:

- Every published score is traceable to a model revision, environment, dataset
  revision, configuration, predictions, and evaluation code.
- Corrections are visible rather than silently rewritten.
- The interface distinguishes unavailable, failed, partial, and scored runs.

## Non-goals

The roadmap does not promise:

- a state-of-the-art Amharic model;
- automatic trust in model-card metrics;
- execution of arbitrary repository code without isolation;
- a hosted service in Stage 1;
- benchmark rankings before evaluation infrastructure is trustworthy.

## How to propose a change

Open an issue describing the stage, the concrete user need, the evidence or fixture,
and the smallest testable increment. Proposals that skip isolation or provenance
requirements should expect review before implementation.
