# T05 implementation checklist

Basis: `9c4e0e8227f0f3299472a1b195cbbe98280491dc` on `next/science-coverage`.

**This is implementation authorization only. Real model loading, MPS and optimizer updates require an exact-commit M4 START.**

Specification self SHA-256: `a414f002e2b742c94c34f08ef22de4d04ee6cf8ad2b28e237371481272fe2354`.

## 1. T05 contract and immutable input audit

Input, schema, leakage and identity tests pass; literal training rubrics match the accepted native templates.

Paths: `experiments/coverage_v1/replay_v1/__init__.py`, `experiments/coverage_v1/replay_v1/contracts.py`, `experiments/coverage_v1/replay_v1/inputs.py`, `plans/next-v1/t05-v1/implementation-spec.json`, `plans/next-v1/t05-v1/artifact-schemas.json`, `tests/coverage_v1/replay_v1/test_inputs.py`

## 2. T05 deterministic paired tapes and freeze

Both seed pairs regenerate identical tapes; all alignment, counts, budgets and schedules validate.

Paths: `experiments/coverage_v1/replay_v1/tapes.py`, `experiments/coverage_v1/replay_v1/freeze.py`, `tests/coverage_v1/replay_v1/test_tapes.py`, `tests/coverage_v1/replay_v1/test_freeze.py`

## 3. T05 versioned initialization and update-boundary resume

CPU loss/gradient references, spy update engine, initialization, resume, signal, failure and locking tests pass.

Paths: `experiments/coverage_v1/replay_v1/state.py`, `experiments/coverage_v1/replay_v1/train.py`, `tests/coverage_v1/replay_v1/test_state.py`, `tests/coverage_v1/replay_v1/test_train.py`

## 4. T05 endpoint evaluation and paired gates

Independent metrics, whole-parent bootstrap, exact inventories, gate boundaries and seed-23 rejection tests pass.

Paths: `experiments/coverage_v1/replay_v1/evaluate.py`, `experiments/coverage_v1/replay_v1/analysis.py`, `tests/coverage_v1/replay_v1/test_evaluate.py`, `tests/coverage_v1/replay_v1/test_analysis.py`

## 5. T05 review package and guarded CLI

All focused checks and historical-source diff checks pass; commit the producer, then freeze twice and package.

Paths: `experiments/coverage_v1/replay_v1/__main__.py`, `experiments/coverage_v1/replay_v1/package.py`, `tests/coverage_v1/replay_v1/test_cli.py`, `tests/coverage_v1/replay_v1/test_package.py`, `plans/next-v1/t05-v1/CHECKLIST.md`

## 6. T05 M4 handoff metadata

Record exact producer and artifact identities in an append-only handoff. The actual run checks out the reviewed producer commit.

Paths: `plans/next-v1/t05-v1/M4_HANDOFF.md`, `RESEARCH_STATE.md`

## Fixed decisions

Both arms start at generation 79 with fresh state. Seed 17 comes first; seed 23 requires the verified pilot gate. Updates=512, effective batch=8, microbatch=2, max_len=2048, learning rates=5e-6/5e-5, warmup=30, cosine schedule, global clip=1. Objective is unchanged request-mean CE + 0.5 Brier + 0.25 Score RPS.

All pilot deltas are targeted minus the same-seed continuation control. Target populations are 92 native emotion questions and 80 offensive questions. Decision/Korean accuracy uses equal-task-source macro; other-transfer NLL uses 568 questions; Score RPS uses the 280-question four-source population. Probability gates use T=1.5197255188671874; raw results and generation-79 differences are mandatory descriptive outputs.

## Before M4 commands

```bash
test "$(git branch --show-current)" = next/science-coverage
git merge-base --is-ancestor 9c4e0e8227f0f3299472a1b195cbbe98280491dc HEAD
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --with pytest pytest tests/coverage_v1/replay_v1 -q
uv run --frozen --with ruff ruff check experiments/coverage_v1/replay_v1 tests/coverage_v1/replay_v1
python -m compileall -q experiments/coverage_v1/replay_v1 tests/coverage_v1/replay_v1
git diff --exit-code 9c4e0e8227f0f3299472a1b195cbbe98280491dc -- src/haetae experiments/train_shared.py experiments/shared_state.py experiments/kev_adapter.py experiments/coverage_v1/diagnostic_v1.py experiments/coverage_v1/run_diagnostic_v1.py experiments/coverage_v1/semantic_output_amendment_v1.py tools/evidence_registry.py research/evidence pyproject.toml uv.lock
git diff --check
git commit -m "Freeze T05 paired replay implementation" # after reviewed staged allowed files only
export T05_PRODUCER_COMMIT="$(git rev-parse HEAD)"
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen python -m experiments.coverage_v1.replay_v1 freeze --paths "$T05_PATHS" --registry research/evidence/index.json --out "$T05_FREEZE_A"
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen python -m experiments.coverage_v1.replay_v1 freeze --paths "$T05_PATHS" --registry research/evidence/index.json --out "$T05_FREEZE_B"
diff -r "$T05_FREEZE_A" "$T05_FREEZE_B"
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen python -m experiments.coverage_v1.replay_v1 verify-plan --plan "$T05_FREEZE_A" --paths "$T05_PATHS" --registry research/evidence/index.json
PYTHONPATH=.:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen python -m experiments.coverage_v1.replay_v1 package-review --plan "$T05_FREEZE_A" --paths "$T05_PATHS" --registry research/evidence/index.json --out "$T05_REVIEW_ZIP"
STOP: request exact-commit M4 review. No initialize, train, or evaluate command before explicit START.
```

## M4 terminal decision

`M4 PAIRED REPLAY START` or `M4 PAIRED REPLAY HOLD`.

The review package must contain exact input/plan/tape/code hashes and public-only private evidence, with zero real model loads, zero MPS operations and zero optimizer.step calls before review.
