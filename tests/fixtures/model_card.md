---
language:
  - am
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
metrics:
  - wer
model_name: Shook Medium Amharic
---

# Shook Medium Amharic

A fine-tuned speech-recognition model for Amharic. This model card is a synthetic
fixture for tests; its reported metrics are not reproduced by Stage 1.

## Usage

```bash
pip install transformers
```

The fixture also mentions trust_remote_code=True so tests can verify that mentions are
recorded as untrusted hints and never executed.

## Evaluation

The model card may report a WER value. EthiASR Bench does not verify it.
