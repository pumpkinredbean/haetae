# Changelog

All notable project changes are recorded here. The project follows semantic
versioning once a package release exists.

## Unreleased

### Added

- Portable shared-v1 architecture, strict safetensors bundle verifier, local
  runtime, CLI, and HTTP API.
- `/v1/systemone` compatibility endpoint alongside the explicit
  `/v1/decide` endpoint.
- Hash-only evidence registry with immutable provenance verification.
- Shared-v1 generation 79 model card, research status, reproduction guide, and
  license ledger.
- Contribution, security, conduct, citation, notice, and issue guidance.

### Changed

- Prepared package version `0.2.0a1` with separate serving, research, and
  development dependency groups.
- Made shared-v1 the current research model and classified the
  separate-question ModernBERT implementation as the historical baseline.
- Replaced general probability and performance claims with evidence-bound
  public-development and host-specific benchmark results.

### Known limitations

- Generation 79 weights are not distributed while training-source terms remain
  unresolved.
- The portable runtime parity and clean-wheel release gates are still open.
- English transfer development results regress relative to the historical
  baseline.
