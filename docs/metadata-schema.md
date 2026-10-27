# Metadata schema

- **Schema version:** `1.0.0`
- **Application status:** Stages 1–2
- **Contract location:** `src/ethi_asr_bench/models/repository.py`
- **Persistence:** normalized JSON in SQLite (`inspections.record_json`)
- **Export:** JSON records wrapped in a versioned export envelope

This document describes the normalized record produced by inspecting one repository
revision. It is a data contract, not a compatibility or accuracy claim.

Compatibility classification is a separate derived contract in
`src/ethi_asr_bench/models/compatibility.py`; it does not change this record schema.
See [`compatibility.md`](compatibility.md).

## Versioning policy

`InspectionRecord.schema_version` identifies the record contract.

- Adding optional fields with safe defaults is a **minor** schema change.
- Removing or renaming fields, changing meaning, or changing required types is a
  **major** schema change.
- A major change requires a migration or an explicit reader for older records.
- Stored records are validated against the current model when read. A record that
  does not match is reported as a storage error rather than silently repaired.
- Stage 1 only supports `1.0.0`; future versions must not reuse this value with
  different semantics.

The export envelope has its own `export_format_version`, currently `1.0.0`.

## Separate compatibility assessment

Stage 2 adds a versioned `CompatibilityAssessment` contract with its own
`classifier_version` (`1.0.0`). It is derived from a validated record and contains:

- `classification`
- `repo_id`, `resolved_commit`, `inspection_schema_version`, and
  `inspection_status`
- `reasons`
- `evidence[]`, with `kind` set to `observed` or `assumption`
- `warnings[]`

The assessment is not embedded in `InspectionRecord`, is not stored in SQLite, and is
not added to the default export. This is intentional: classification opinions can
change independently of historical facts. Stage 2 therefore requires no database
migration and preserves the readability of all Stage 1 records.

## Top-level structure

```json
{
  "schema_version": "1.0.0",
  "identity": {},
  "status": "success",
  "model": {},
  "contents": {},
  "dependencies": {},
  "custom_code": {},
  "warnings": [],
  "errors": [],
  "provenance": {}
}
```

## Unknown-value handling

Classification-relevant fields use `ObservedValue<T>`:

```json
{"state": "known", "value": "whisper", "source": "config.json.model_type"}
{"state": "unknown"}
{"state": "extraction_failed", "note": "config.architectures had an unexpected type"}
{"state": "not_applicable"}
```

Rules enforced by the model:

- `known` requires a non-empty value.
- `unknown`, `extraction_failed`, and `not_applicable` must not carry a value.
- `source` is required for extracted known values in practice.
- `note` explains malformed input or truncation.
- Exports omit null `value`, `source`, and `note` fields.

This is intentional. A missing language tag is stored as `unknown`; it is not stored
as an empty list, an inferred language, or a claim that the model cannot handle
Amharic.

## Enumerations

### `status`

| Value | Meaning |
| ----- | ------- |
| `success` | Retrieval and normalization completed without warning-severity issues. |
| `partial` | Useful metadata was stored, but one or more warning-severity issues occurred. |
| `failed` | No usable normalized metadata; `errors` is non-empty. |

Informational warnings do not change `success` to `partial`.

### `ObservedValue.state`

`known`, `unknown`, `extraction_failed`, `not_applicable`.

### `warnings[].severity`

- `warning`: contributes to `partial`.
- `info`: descriptive/security note; does not make the record partial.

### `repo_type`

Only `model` is accepted in Stage 1. Dataset and space URLs are rejected before any
provider call.

## Fields

### `identity`

| Field | Type | Semantics |
| ----- | ---- | --------- |
| `repo_id` | string | Canonical `owner/model` or single-segment model ID. |
| `owner` | string or null | Namespace, when the ID is namespaced. |
| `name` | string | Repository name. |
| `repo_type` | `"model"` | Repository category. |
| `canonical_url` | string | Provider-canonical HTTPS URL (Hugging Face for the current provider). |
| `requested_revision` | string or null | Branch, tag, or commit requested by the user. |
| `resolved_commit` | string or null | Commit SHA returned by the provider, if available. |
| `retrieved_at` | string | UTC ISO 8601 timestamp ending in `Z`. |

`requested_revision` and `resolved_commit` are distinct. A branch name is not a stable
identity; the commit SHA is. If the provider cannot resolve a commit, the record is
marked partial.

### `model`

| Field | Type | Semantics |
| ----- | ---- | --------- |
| `display_name` | `ObservedValue<string>` | Explicit model-card display name. |
| `description` | `ObservedValue<string>` | First readable model-card paragraph, capped at 2,000 characters. |
| `architectures` | `ObservedValue<string[]>` | Structured architecture classes from configuration. |
| `model_type` | `ObservedValue<string>` | Structured model type from configuration. |
| `library_name` | `ObservedValue<string>` | Hub/library declaration. |
| `pipeline_tag` | `ObservedValue<string>` | Hub task declaration. |
| `languages` | `ObservedValue<string[]>` | Declared language codes, lowercased and deduplicated. |
| `licenses` | `ObservedValue<string[]>` | Explicit license identifiers, deduplicated case-insensitively. |
| `task_tags` | string[] | Pipeline tag plus recognized task tags. |
| `tags` | string[] | Hub tags, preserving provider order and deduplicating nothing newly. |
| `declared_metrics` | string[] | Metric names declared in model-card metadata, e.g. `wer`. |

`declared_metrics` is not a result. Stage 1 does not verify or reproduce reported
metrics.

### `contents`

| Field | Type | Semantics |
| ----- | ---- | --------- |
| `files` | object[] | Up to 2,000 provider-advertised files, sorted by path. |
| `files_truncated` | boolean | Whether the file cap was applied. |
| `known_size_bytes` | integer | Sum of advertised sizes where sizes are known. |
| `unknown_size_file_count` | integer | Files with no advertised size. |
| `has_readme` | boolean | Root `README.md` is present. |
| `readme_filename` | string or null | Actual README path if present. |
| `has_config_json` | boolean | Root `config.json` is present. |
| `config_files` | string[] | Recognized configuration basenames present. |
| `tokenizer_or_processor_files` | string[] | Recognized tokenizer/processor basenames present. |
| `has_tokenizer_or_processor` | boolean | Any recognized tokenizer/processor file is present. |
| `weight_files` | string[] | Weight-like paths detected by filename heuristic. |
| `has_weight_files` | boolean | At least one weight-like path was detected. |
| `has_custom_code` | boolean | At least one `.py` path is present. |
| `generation_config_present` | boolean | Root `generation_config.json` is present. |

Each file entry:

```json
{"path": "model.safetensors", "size_bytes": 3000000000, "is_lfs": true}
```

File paths are relative POSIX paths. Absolute paths, backslashes, NUL bytes, and
empty/`.`/`..` segments are rejected before storage.

The provider fetches only root `README.md`, `config.json`, and `requirements.txt`,
within a byte limit. LFS-backed text files are not fetched. Other files are inventory
entries only and are never downloaded.

Weight detection is a **heuristic** based on extensions such as `.safetensors`,
`.onnx`, `.gguf`, `.msgpack`, `.h5`, `.tflite`, `.ckpt`, `.pth`, and `.bin`, with
known non-weight files such as `training_args.bin` excluded. Presence does not prove
loadability. Names are never fetched.

### `dependencies`

All strings here are untrusted descriptive metadata.

| Field | Type | Semantics |
| ----- | ---- | --------- |
| `requirements_files` | string[] | Recognized `requirements*.txt` basenames. |
| `declared_requirements` | string[] | Non-option lines from root `requirements.txt`, capped at 50. |
| `install_instructions` | string[] | Installation-looking model-card lines, capped at 5. |
| `compatibility_clues` | string[] | Descriptive clues such as declared library, pipeline, auto-model, weight extensions, and custom code. |

Nothing in this section is installed or executed. Clues are not compatibility
verdicts.

### `custom_code`

| Field | Type | Semantics |
| ----- | ---- | --------- |
| `python_files` | string[] | Up to 100 `.py` paths from the inventory. |
| `auto_map_declared` | `ObservedValue<boolean>` | Whether `config.auto_map` is non-empty. |
| `trust_remote_code_mentioned` | `ObservedValue<boolean>` | Whether the model card contains the string `trust_remote_code`. |
| `code_executed` | `false` | Fixed literal. Stage 1 never executes repository code. |

`trust_remote_code_mentioned=true` is a hint, not a validated requirement.

### `warnings`

Each warning is structured:

```json
{"code": "provider_text_file_error", "message": "Could not retrieve README.md: timed out", "severity": "warning"}
```

Warning codes are stable strings but are not yet a closed enum. Common values include
`provider_text_file_error`, `provider_note`, `commit_unresolved`,
`malformed_metadata`, `config_json_malformed`, `unsafe_file_path`,
`file_inventory_truncated`, `auto_map_declared`, `trust_remote_code_mentioned`,
`custom_code_present`, `install_instructions_present`, and
`weights_not_downloaded`.

### `errors`

Failed records include at least one error:

```json
{
  "code": "repository_not_found",
  "message": "404 Client Error ...",
  "retryable": false,
  "http_status": 404
}
```

Error codes come from `ErrorCode` and include validation, provider, extraction, and
storage categories. Provider messages are truncated and token-redacted.

### `provenance`

| Field | Type | Semantics |
| ----- | ---- | --------- |
| `provider` | string | Provider identifier, e.g. `huggingface`. |
| `provider_version` | string or null | Client version when available. |
| `sources` | string[] | Source labels used, e.g. `hub.model_info`, `README.md`. |
| `field_sources` | object | Composite fields mapped to source descriptions. |

Examples of `field_sources`:

```json
{
  "files": {"source": "hub.model_info.siblings"},
  "languages": {"source": "model_card.cardData.language + hub.model_info.tags"},
  "description": {"source": "model_card.README.md", "note": "untrusted descriptive text"}
}
```

## Persistence and export

The SQLite `inspections` table stores:

- indexed identity/revision/commit columns;
- `status` and `schema_version`;
- the normalized record JSON;
- optional raw provider JSON;
- provider name/version;
- first error code/message for diagnostics.

Raw source metadata is not part of the normalized record. It is included in an export
only with `--include-raw`. Compatibility assessments are not part of the Stage 1
export; use `ethi-asr classify --json` to produce one.

The export envelope:

```json
{
  "export_format_version": "1.0.0",
  "schema_version": "1.0.0",
  "generated_by": "ethi-asr-bench/0.1.0",
  "exported_at": "2026-10-27T12:48:10Z",
  "count": 1,
  "records": []
}
```

## Representative record (abridged, live)

This is an abridged real record for
`b1n1yam/shook-medium-amharic-2k`, inspected on 2026-10-27. File entries, tags, and
provenance have been shortened. Language and license remain unknown because the
repository did not declare them in structured metadata. The model-card WER mention is
unverified source text.

```json
{
  "schema_version": "1.0.0",
  "identity": {
    "repo_id": "b1n1yam/shook-medium-amharic-2k",
    "owner": "b1n1yam",
    "name": "shook-medium-amharic-2k",
    "repo_type": "model",
    "canonical_url": "https://huggingface.co/b1n1yam/shook-medium-amharic-2k",
    "resolved_commit": "60b7ae174266ed42362ee02e636aea144011378d",
    "retrieved_at": "2026-10-27T12:48:03Z"
  },
  "status": "success",
  "model": {
    "display_name": {"state": "unknown"},
    "description": {
      "state": "known",
      "value": "This model was trained from scratch on the None dataset. It achieves the following results on the evaluation set: - Loss: 0.0532 - Wer: 9.1091",
      "source": "model_card.README.md"
    },
    "architectures": {
      "state": "known",
      "value": ["WhisperForConditionalGeneration"],
      "source": "config.json.architectures"
    },
    "model_type": {"state": "known", "value": "whisper", "source": "config.json.model_type"},
    "library_name": {"state": "known", "value": "transformers", "source": "hub.model_info.library_name"},
    "pipeline_tag": {"state": "known", "value": "automatic-speech-recognition", "source": "hub.model_info.pipeline_tag"},
    "languages": {"state": "unknown"},
    "licenses": {"state": "unknown"},
    "task_tags": ["automatic-speech-recognition"],
    "tags": ["whisper", "automatic-speech-recognition"],
    "declared_metrics": ["wer"]
  },
  "contents": {
    "files": [
      {"path": "README.md", "size_bytes": 3019, "is_lfs": false},
      {"path": "config.json", "size_bytes": 1328, "is_lfs": false}
    ],
    "known_size_bytes": 3061192445,
    "unknown_size_file_count": 0,
    "has_readme": true,
    "readme_filename": "README.md",
    "has_config_json": true,
    "config_files": ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"],
    "tokenizer_or_processor_files": ["tokenizer.json", "tokenizer_config.json", "vocab.json"],
    "has_tokenizer_or_processor": true,
    "weight_files": ["model.safetensors"],
    "has_weight_files": true,
    "has_custom_code": false,
    "generation_config_present": true
  },
  "dependencies": {
    "requirements_files": [],
    "declared_requirements": [],
    "install_instructions": [],
    "compatibility_clues": [
      "Declared library: transformers",
      "Declared pipeline: automatic-speech-recognition",
      "Transformers auto_model: AutoModelForSpeechSeq2Seq",
      "Weight-like files present (.safetensors); not downloaded."
    ]
  },
  "custom_code": {
    "python_files": [],
    "auto_map_declared": {"state": "known", "value": false, "source": "config.auto_map"},
    "trust_remote_code_mentioned": {"state": "known", "value": false, "source": "model_card.README.md"},
    "code_executed": false
  },
  "warnings": [
    {"code": "weights_not_downloaded", "message": "Model weight-like files were detected from the file listing and were not downloaded.", "severity": "info"}
  ],
  "errors": [],
  "provenance": {
    "provider": "huggingface",
    "provider_version": "1.33.0",
    "sources": ["hub.model_info", "hub.file_listing", "README.md", "config.json"],
    "field_sources": {
      "files": {"source": "hub.model_info.siblings"},
      "languages": {"source": "model_card.cardData.language + hub.model_info.tags"}
    }
  }
}
```

## Security interpretation

Every string derived from a model repository is data, not instructions:

- Markdown/HTML is not executed.
- URLs in model cards are not fetched.
- Installation snippets are not run.
- `auto_map` is recorded, not resolved.
- Python files are listed, not downloaded.
- Raw metadata is stored for debugging, not trusted as executable configuration.
