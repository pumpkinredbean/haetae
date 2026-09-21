# Haetae next development cycle

Base commit: `4828f9c69ba0f5b1f64fd72aa35fa3ce7269d294`.

ChatGPT 6 Pro inspected the exact base commit and produced the plan recorded in this directory. The model is a validated research prototype. The next scientific uncertainty is whether the emotion and offensive-tweet transfer losses mainly reflect missing supervised coverage or task/readout binding, rather than inaccessible representations that require a different architecture.

The work is split into three independent tracks:

1. `next/science-coverage` verifies cached evidence, measures supervised exposure, runs fixed-weight diagnostics, and only then may run one bounded paired replay experiment.
2. `next/runtime-alpha` exports the existing shared-state generation 79 checkpoint into a data-only inference bundle and builds a portable runtime without changing historical research code.
3. `next/docs-alpha` replaces stale claims with evidence-linked documentation, a model card, provenance and licensing records, and contributor and security guidance.

`next/integration` owns the plan, evidence registry, and reviewed cherry-picks. The research alpha remains generation 79 regardless of whether the replay experiment succeeds.

## Dependency order

```text
T00 evidence boundary
 ├─ T01 cached coverage audit → T04 fixed-weight diagnostics
 │                          → T05 paired replay implementation
 │                          → T06 bounded paired runs → T09 decision
 ├─ T02 documentation and license ledger ──────────────────────┐
 └─ T03 portable runtime and original-checkpoint export        │
                            → T07 packaging and CI             │
                            → T08 public evidence/model card ──┴─ T10 alpha release
```

T02 and T03 may proceed after T00 while T01 is reviewed. Model inference starts only after the T04 protocol is frozen and reviewed. Optimizer updates start only after the T05 implementation and replay plan are reviewed. One process owns MPS at a time.

## Fixed decisions

- Keep the read-only shared state, isolated bidirectional question branches, restarted positions, and contextual listwise pointer scoring for this cycle.
- Do not run a loss ablation, architecture sweep, or long continuation before the coverage and readout diagnostics.
- Use public training and development data only. Do not open or enumerate a locked test payload.
- Do not mutate qualified evidence or generation 79.
- Do not use Jev or Laya outputs as training labels.
- Keep calibration, representation, supervised coverage, and architecture conclusions separate.
- Publish negative and inconclusive outcomes under the same frozen gates as positive outcomes.

The full ordered gates are in [CHECKLIST.md](CHECKLIST.md). Machine-readable task ownership and acceptance criteria are in [implementation-plan.json](implementation-plan.json).

## Advisor artifact identity

The original machine plan and checklist returned through Aside REPL are retained outside Git. Their SHA-256 digests are:

- machine plan: `874bc1fb2dda3d231f9317d18cc5b21a90f1cadd1649e11f3abfd58c91e566d2`;
- checklist: `9f31986ab3ad15d95f95b53b261d5f90598f7678b7e6dd22db8820c209700b8d`.

This directory is a concise, repository-safe transcription of those artifacts. It excludes local paths and conversation links.
