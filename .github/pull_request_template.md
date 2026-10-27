## Summary

What problem does this change solve, and what is the user-visible effect?

## Scope

- [ ] Stage 1 metadata foundation
- [ ] Documentation or tooling only
- [ ] Extension point for a later stage (explain below)

## Changes

-
-

## Tests and checks run

Paste the actual commands and results:

```text
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Security and data handling

- [ ] No tokens, private repository metadata, local databases, exports, or caches are
  included.
- [ ] Untrusted model metadata remains non-executable and is escaped in terminal
  output.
- [ ] No model weights are downloaded and no repository code is executed.
- [ ] SQL remains parameterized and paths are not derived from remote metadata.
- [ ] If behavior, commands, schema, or security boundaries changed, documentation
  was updated.

## Checklist

- [ ] Tests were added or updated.
- [ ] `docs/metadata-schema.md` was updated for schema changes.
- [ ] The change is focused and does not include unrelated refactors.
- [ ] I have read CONTRIBUTING.md and CODE_OF_CONDUCT.md.
