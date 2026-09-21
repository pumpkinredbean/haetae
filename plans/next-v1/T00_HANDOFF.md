# T00 evidence-boundary handoff

Status: ready for M0 re-review after the initial HOLD findings were resolved.

Base commit: `4828f9c69ba0f5b1f64fd72aa35fa3ce7269d294`.

The detached reference checkout is read-only and reproduces the completed shared run's training-code fingerprint:

- source identity: `ff54ea26399fd9ad335eabba9d6f6a85a49880d8ac23b5217de665b10c2cc3fe`;
- files in source identity: 11;
- reference commit: exact base commit.

The evidence registry verifies 28 original artifacts. The set includes both completed checkpoints and run identities, the frozen comparison plan, public-development populations and prediction reports, the paired comparison, the original execution protocol and result, the six-process confirmation protocol and result, and the qualified Laya calibration and weighting evidence.

- registry canonical SHA-256: `943c9510769dfc184ff43a8c1883e53a5d7347ca0de975b7d2e32eafc0af2774`;
- local verification report SHA-256: `ec11430222dcba056da43729bb9789640d469d81003676c7e90356932bce2ace`;
- verified artifacts: 28;
- unavailable optional artifacts: 0;
- locked-test artifacts registered or opened: 0;
- model forward passes: 0;
- optimizer updates: 0.

The local path map and verification report remain outside Git. Committed files contain only evidence IDs, hashes, sizes, claims, ownership boundaries, and the read-only reference identity.

The initial M0 review found unsafe output replacement, incomplete canonical-path and derivative checks, schema-validation gaps, and an insufficient reference-worktree guard. The implementation now uses atomic no-clobber output, rejects output destinations in evidence and reference locations, rejects symlink traversal and invalid optional mappings, validates exact schema fields and duplicate keys, checks derivative content and transitive provenance against current source bytes, and requires detached HEAD with nonwritable source files and parent directories.

The 35 self-tests cover the accepted path, digest and availability failures, malformed schemas and duplicate keys, lexical and symbolic-link escapes, private content, derivative source and provenance failures, output no-clobber behavior, provenance cycles, and reference-worktree mutation guards. Separate CLI probes confirm that attempts to write a report over registered evidence, a reference source file, or an existing report fail without changing the destination bytes.
