# Security policy

## Supported versions

Security fixes are applied to the current `0.1.x` Stage 1 line. Earlier development
snapshots are not supported.

## Reporting a vulnerability

Do **not** open a public issue containing exploit details, tokens, private repository
metadata, or a working proof of concept.

Use the repository host's private vulnerability reporting feature if it is enabled.
If it is unavailable, contact a project maintainer privately through the contact
information on their repository-host profile and ask for a secure channel.

Include, where possible:

- affected version or commit;
- operating system and Python version;
- the command or API call involved;
- a minimal reproduction with secrets and private model data removed;
- impact assessment;
- any suggested mitigation.

## What to expect

Maintainers will acknowledge the report, investigate, and coordinate disclosure. Do
not publish details until a fix or mitigation has been discussed. Timelines depend on
severity, reproducibility, and maintainer availability.

## Scope

In scope:

- host-validation bypasses or lookalike-domain acceptance;
- credential leakage in logs, errors, records, or exports;
- path traversal or arbitrary file write/read;
- SQL injection;
- unintended code execution or dependency installation;
- downloading model weights when the design says they are not downloaded;
- denial of service caused directly by untrusted metadata handling;
- unsafe terminal rendering of model-card content.

Out of scope for Stage 1:

- vulnerabilities in Hugging Face services or model repositories themselves;
- risks from running a model or executing repository code (not implemented);
- local attackers with read access to your filesystem or environment;
- model quality, bias, or benchmark claims;
- dependency vulnerabilities with no demonstrated impact on this project.

## Safe handling of credentials

- Use a scoped Hugging Face token where possible.
- Do not place tokens on the command line.
- Do not include tokens in bug reports.
- Rotate a token immediately if you believe it was exposed.

## Security design

The implemented controls and residual risks are documented in
[`docs/threat-model.md`](docs/threat-model.md).
