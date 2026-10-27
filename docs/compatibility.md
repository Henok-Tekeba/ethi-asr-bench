# Compatibility classification

- **Classifier version:** `1.0.0`
- **Contract:** `src/ethi_asr_bench/models/compatibility.py`
- **Rules:** `src/ethi_asr_bench/services/classifier.py`
- **Input:** one validated Stage 1 `InspectionRecord`
- **Persistence:** none; assessments are derived on demand
- **Execution:** none; no weights are downloaded and no repository code is run

Stage 2 answers a narrow question: **given the structured metadata that Stage 1
stored, which inference path is the most honest candidate?** It does not prove that a
model loads, that audio preprocessing is compatible, or that transcripts are accurate.

## Why this is separate from the metadata schema

The Stage 1 record is a historical statement of what the Hub declared at a commit.
A compatibility assessment is an opinion derived from that record by a particular
classifier version. Keeping the two separate means:

- classification cannot rewrite historical metadata;
- the Stage 1 schema remains `1.0.0`;
- no database table or migration is required;
- a later classifier version can be compared with an earlier one without changing
  stored inspections.

The classifier is deterministic for the same record and classifier version.

## Classes

| Class | Meaning |
| ----- | ------- |
| `standard_whisper_candidate` | Structured metadata matches a standard Transformers Whisper ASR configuration, with no `auto_map` declaration. A future standard Whisper path is a candidate. |
| `known_variation` | Structured metadata matches a recognized non-Whisper ASR family, or a Whisper record declares a non-Transformers library. A family-specific adapter would be needed. |
| `custom_implementation_required` | `auto_map` is declared, or the structured architecture is outside the initial ASR allowlist. The standard Transformers Whisper path is not assumed. |
| `insufficient_information` | Structured architecture/model metadata is missing, malformed, or conflicting, including an ASR family paired with an explicitly non-ASR pipeline. No compatibility conclusion is asserted. |

`custom_implementation_required` does **not** mean the repository is malicious. It
means that using it for ASR requires an implementation outside the current standard
Whisper path.

## Evidence model

Every assessment contains at least one `observed` evidence item. Each item has:

```json
{
  "rule": "model.model_type",
  "kind": "observed",
  "signal": "model.model_type",
  "value": "whisper",
  "source": "config.json.model_type",
  "message": "A structured model_type was observed."
}
```

`kind` is one of:

- `observed`: read from the normalized record, with the original source when known;
- `assumption`: added by a classifier rule, such as assuming a standard Transformers
  loader when no library was declared.

Assumptions are never presented as observed facts. The assessment also includes
human-readable `reasons`, structured `warnings`, and `classifier_version`.

## Inputs

The classifier uses these structured fields:

| Signal | Use |
| ------ | --- |
| `model.architectures` | Primary family evidence. |
| `model.model_type` | Secondary family evidence. |
| `model.pipeline_tag` / `model.task_tags` | ASR task evidence. |
| `model.library_name` | Standard Transformers versus another declared runtime. |
| `custom_code.auto_map_declared` | Strong custom-code signal. |
| `contents.weight_files` | Warning only; absence is not incompatibility. |
| `contents.tokenizer_or_processor_files` | Warning only; absence is not incompatibility. |
| `contents.files_truncated` | Warning that absent files are inconclusive. |
| `custom_code.python_files` | Warning only unless `auto_map` is also declared. |
| `custom_code.trust_remote_code_mentioned` | Warning only; prose is not proof. |
| `inspection.status`, `inspection.errors` | Failed inspection handling. |

It does **not** use repository names, model-card prose, declared metrics, languages,
licenses, download counts, or `transformersInfo` to decide a family. Task tags are used
only to establish an ASR task declaration.

## Rule order

The first applicable rule wins.

1. **Failed inspection → `insufficient_information`.**
   No usable structured inference metadata was stored. A retrieval failure is not
   evidence of incompatibility.

2. **`auto_map` declared → `custom_implementation_required`.**
   Repository-provided code may be required. The classifier does not resolve,
   download, or execute it.

3. **Architecture list contains multiple families → `insufficient_information`.**
   For example, a list containing both Whisper and Wav2Vec2 classes is contradictory.

4. **Architecture and `model_type` resolve to different families → `insufficient_information`.**
   Conflicting structured signals are not resolved by guessing.

5. **No recognized architecture or `model_type` → `insufficient_information`.**
   Missing information is never treated as proof of incompatibility.

6. **Architecture/model type outside the allowlist → `custom_implementation_required`.**
   The standard Whisper path cannot be assumed for an unknown or non-ASR structured
   architecture.

7. **ASR architecture with an explicitly non-ASR pipeline → `insufficient_information`.**
   The architecture says ASR while the task declaration says otherwise; the conflict
   is reported rather than hidden.

8. **Whisper family with a non-Transformers library → `known_variation`.**
   The model may use a known runtime, but the standard Transformers Whisper path is
   not established.

9. **Whisper family with Transformers or no library declaration → `standard_whisper_candidate`.**
   The result is a candidate, not a loadability claim. A missing library is recorded as
   an assumption.

10. **Recognized non-Whisper ASR family → `known_variation`.**
    A family-specific adapter would be required; none is implemented in Stage 2.

## Initial allowlist

Whisper:

- architecture: `WhisperForConditionalGeneration`
- model type: `whisper`

Known variations:

- architectures: `Data2VecAudioForCTC`, `HubertForCTC`, `MCTCTForCTC`,
  `SeamlessM4TForSpeechToText`, `SeamlessM4Tv2ForSpeechToText`,
  `SpeechEncoderDecoderModel`, `Speech2TextForConditionalGeneration`,
  `UniSpeechForCTC`, `UniSpeechSatForCTC`, `Wav2Vec2ConformerForCTC`,
  `Wav2Vec2ForCTC`, `WavLMForCTC`
- model types: `data2vec-audio`, `hubert`, `mctct`, `mms`, `seamless_m4t`,
  `seamless_m4t_v2`, `speech_to_text`, `speech_to_text_2`, `unispeech`,
  `unispeech-sat`, `wav2vec2`, `wav2vec2-bert`, `wav2vec2-conformer`, `wavlm`

The allowlist is deliberately small. An unlisted architecture returns a custom or
insufficient verdict rather than an invented compatibility claim. Extending the list
requires a fixture and a rule review.

## Examples

### Standard Whisper candidate

Input facts:

```json
{
  "architectures": ["WhisperForConditionalGeneration"],
  "model_type": "whisper",
  "pipeline_tag": "automatic-speech-recognition",
  "library_name": "transformers",
  "auto_map_declared": false
}
```

Output:

```json
{
  "classifier_version": "1.0.0",
  "classification": "standard_whisper_candidate",
  "reasons": [
    "Structured metadata declares a Whisper architecture or Whisper model_type.",
    "No auto_map custom loader declaration was observed.",
    "This is a candidate for a standard Whisper path; compatibility is not proven."
  ],
  "warnings": []
}
```

### Known ASR variation

`Wav2Vec2ForCTC` with `model_type: wav2vec2` becomes `known_variation`. The evidence
records the structured architecture and a warning explains that a family-specific
adapter is not implemented.

### Custom implementation

A repository with `auto_map` becomes `custom_implementation_required`, even when its
architecture is Whisper. The code is not fetched or executed.

### Insufficient information

A record with no architecture and no `model_type` becomes
`insufficient_information`. The reasons explicitly state that missing metadata is not
evidence of incompatibility.

## Limitations

- The classifier is rule-based and intentionally incomplete. A model can be useful
  despite an `insufficient_information` or `custom_implementation_required` verdict.
- `standard_whisper_candidate` does not verify tokenizer assets, feature extraction,
  sampling rate, decoding, language tokens, or output quality.
- A declared pipeline can be wrong or stale. The classifier reports conflicts instead
  of silently preferring one field.
- A non-Transformers library produces `known_variation` for Whisper because a known
  runtime may exist, but no adapter is implemented or tested.
- `auto_map` is the only custom-code signal that forces a custom verdict on its own.
  Python files can be training scripts, and `trust_remote_code` can be a model-card
  recommendation rather than a requirement.
- File inventory is an advertisement, not proof. Missing weight or tokenizer files
  produce warnings, not a negative compatibility class.
- No weight format, quantization, ONNX, GGUF, or TensorFlow loading semantics are
  interpreted in Stage 2.
- The classifier does not contact a provider. It classifies the stored revision; run
  `inspect` again if a repository has moved.

## Extension policy

To add a recognized family:

1. Add a fixture with structured architecture/model metadata.
2. Add the exact architecture and/or model type to the allowlist.
3. Add a test for the expected class and evidence.
4. Document the family and its limitations.
5. Bump `classifier_version` if the meaning of an existing class changes or a rule
   reorders existing outcomes.

To add an evidence field, keep observed facts and assumptions separate and preserve
deterministic ordering.
