# Reproduction guide

## Scope

The current checkout contains historical research producers. It does not yet
contain the portable generation 79 runtime. Reproduction therefore has two
levels:

1. verify the existing evidence and model identity without running inference;
2. rerun a producer only after reconstructing its exact pinned environment and
   public data.

Locked tests are outside both workflows.

## Environment

Use the lock file and avoid dependency upgrades:

```bash
uv sync --frozen
uv run python --version
uv run python tools/evidence_registry.py self-test
```

Generation 79 was produced with Python 3.13.2, PyTorch 2.14.0, Transformers
5.17.0, and the Apple MPS backend. These versions describe provenance; they do
not imply that other compatible environments will reproduce training
bit-for-bit. Evidence: `shared_run_spec`.

## Artifact verification

Copy no artifact into the repository. Create the complete structured path map
shown in `docs/evidence.md`, including `schema_version`, the detached
`reference_repository`, allowlisted `roots`, and an `artifacts` mapping for
every registered evidence ID. Verify the registry into a new output path:

```bash
uv run python tools/evidence_registry.py verify \
  --paths /path/to/local-evidence-paths.json \
  --out /path/to/new-verification-report.json
```

The report's `status` must be `verified`; `verified_count` counts captured
artifacts and `unavailable_optional_count` counts valid optional mappings whose
files or ancestors are unavailable. Every artifact in the checked-in registry
attests `contains_locked_data: false`; the verifier validates those registered
attestations and hashes rather than inspecting arbitrary files for examples.
Compare `registry_sha256` with the SHA-256 of the registry's canonical JSON,
`base_commit` with the registry's `base_commit`, and
`reference.source_sha256` with `reference_source_identity.sha256` in
`research/evidence/index.json`.

## Historical producer

The historical producer checkout is exact commit
`4828f9c69ba0f5b1f64fd72aa35fa3ce7269d294`, with training-source identity
`ff54ea26399fd9ad335eabba9d6f6a85a49880d8ac23b5217de665b10c2cc3fe`.
The frozen shared suite contains 15,572 requests and 23,068 questions. The
generation 79 training command was:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.train_shared \
  --suite evaluations/shared-v1 \
  --steps 3894 \
  --batch 8 \
  --microbatch 2 \
  --max-len 2048 \
  --device mps \
  --local-files-only \
  --out runs/shared-v1 \
  --resume none
```

The output directory is immutable. Do not point this command at an existing
run. Dataset files, tokenizer files, and the pinned mmBERT-small revision must
already be present in the local cache for offline execution.

## Public development evaluation

The evaluators accept only the frozen comparison and suite bindings. They do
not expose a locked-test option:

```bash
uv run python -m experiments.evaluate_shared \
  --run runs/shared-v1 \
  --comparison-plan evaluations/comparison-v1/plan.json \
  --decision-suite evaluations/shared-v1 \
  --transfer-suite evaluations/kev-transfer-v4 \
  --korean-suite evaluations/korean-v1 \
  --device mps \
  --batch 2 \
  --out /path/to/new-shared-development.json
```

Write new results to a fresh path. Do not overwrite the qualified report. The
original execution timing benchmark is historical evidence and should not be
rerun as a routine test because timing results depend on host state. Use its
protocol and result through `execution_protocol`, `execution_summary`, and
`confirmation_summary`.

## Expected identities

| Item | SHA-256 |
| --- | --- |
| Generation 79 checkpoint | `e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c` |
| Shared suite manifest | `89b02fe5e76bdfae6ca5930d1e9b15b31dd61188330a68454cc951ac26c24d9c` |
| Shared train population | `e6170e0f5b1920ca56f74a56f341d7fdf3472d70bbb8fe73ab72eb19dfacf22f` |
| Shared development report | `960349119103216be9b0945565e578ca1a5303afd70b0338ef2c64b92e4afd23` |
| Paired development comparison | `4d81804932a96afadd568ff4668c44c7cd993bd9ba8587ddc6042e015908b343` |

These values are evidence identities, not download links.
