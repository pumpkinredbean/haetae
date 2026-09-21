# T00 evidence-boundary handoff

Status: ready for M0 review.

Base commit: `4828f9c69ba0f5b1f64fd72aa35fa3ce7269d294`.

The detached reference checkout is read-only and reproduces the completed shared run's training-code fingerprint:

- source identity: `ff54ea26399fd9ad335eabba9d6f6a85a49880d8ac23b5217de665b10c2cc3fe`;
- files in source identity: 11;
- reference commit: exact base commit.

The evidence registry verifies 28 original artifacts. The set includes both completed checkpoints and run identities, the frozen comparison plan, public-development populations and prediction reports, the paired comparison, the original execution protocol and result, the six-process confirmation protocol and result, and the qualified Laya calibration and weighting evidence.

- registry canonical SHA-256: `943c9510769dfc184ff43a8c1883e53a5d7347ca0de975b7d2e32eafc0af2774`;
- local verification report SHA-256: `243f1df886292ce3b4f8d3f086ac689bc83fbd45a71d83b976d06f711f494882`;
- verified artifacts: 28;
- unavailable optional artifacts: 0;
- locked-test artifacts registered or opened: 0;
- model forward passes: 0;
- optimizer updates: 0.

The local path map and verification report remain outside Git. Committed files contain only evidence IDs, hashes, sizes, claims, ownership boundaries, and the read-only reference identity.

The self-test covers valid verification, tampered digests, missing required and optional artifacts, unsupported schema versions, root escapes, placeholder digests, private-path leakage, explicit derivative provenance, and protected-evidence overwrite attempts.
