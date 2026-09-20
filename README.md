# haetae

A local System One decision model. State and typed questions in,
calibrated probability distributions out — one forward pass per
question, no text generated, nothing to parse.

Haetae (해태) is the Korean mythical creature that judges right
from wrong.

This is a clean-room reproduction of the *interface* of TypeSafe's
Jev: same request shape (POST /v1/systemone), same three
primitives (choice / noul / score), same kind of output (a
probability per option plus a distribution-derived confidence).
It shares no weights, no training data, and no outputs with Jev.

## How it differs from other local attempts

- **Packed single forward pass.** The state is encoded once and every
  option of a question is scored from its own marker position in the
  same sequence: [CLS] state [SEP] question [SEP] opt1 [SEP] opt2 ...
  Other encoder attempts (Von, Laya) run one premise-hypothesis pair
  per forward — an NLI classifier, not a decision model.
- **Trained, not read out.** Probabilities come from a head trained
  with a proper scoring rule (cross-entropy + multi-class Brier),
  not from a chat model's next-token distribution.
- **Conformal layer.** Beyond temperature scaling, split conformal
  gives a distribution-free coverage guarantee on prediction sets —
  a stronger claim than empirical calibration.
- **Korean in the mixture.** KLUE-YNAT and NSMC are first-class
  training sources, not an afterthought.

## Layout

    src/haetae/data.py       public datasets -> normalized primitive records
    src/haetae/model.py      ModernBERT + packed option-marker head
    src/haetae/train.py      CE + Brier training loop
    src/haetae/checkpoint.py run identity + verified checkpoint generations
    src/haetae/eval.py       accuracy, ECE, latency per source
    src/haetae/calibrate.py  temperature scaling + split conformal
    src/haetae/certify.py    calibration + selective-risk certification harness
    src/haetae/measure.py    frozen A/B/C evaluation and signed result artifacts
    src/haetae/serve.py      /v1/systemone FastAPI
    experiments/shared_state.py       shared-state question-branch prototype
    experiments/train_shared.py       immutable shared-state trainer
    experiments/evaluate_shared.py    development-suite evaluator
    experiments/freeze_korean_suite.py Korean development/test freezer

## Contract notes

- Packing: [CLS] state [SEP] instructions [SEP] opt1 [SEP] opt2 ...
  Each option is scored from the hidden state of the [SEP] that
  precedes it (ModernBERT is bidirectional, so the marker sees the
  option). All options must fit whole — the state yields its token
  budget to the candidate set, never the reverse.
- Choice records get candidate-order augmentation at train time:
  positions are not permutation-equivariant, and fixed option order
  would let the model learn slot indexes instead of reading options.
- Score keeps canonical ordinal order; noul keeps "yes" first.
- Records that cannot fit their full candidate set are skipped and
  counted, never silently truncated.
- The act/abstain guarantee is a certified selective-risk bound
  (one-sided binomial on accepted examples), not just empirical ECE.

## Train

    uv sync
    uv run python -m haetae.train --sources ag_news,boolq,sst5,banking77,klue_ynat,nsmc --out runs/baseline --resume none

When the effective batch does not fit accelerator memory, split one
optimizer update into smaller forward and backward passes. For example,
`--batch 8 --microbatch 4` preserves the eight-question mean loss while
holding activations for at most four questions at a time. Both values are
part of the immutable run identity.

Resume the same immutable run after an interruption:

    uv run python -m haetae.train --sources ag_news,boolq,sst5,banking77,klue_ynat,nsmc --out runs/baseline --resume auto

The output directory is bound to one run identity. Each save writes a
new immutable checkpoint generation, publishes its size and SHA-256 in
`latest.json`, and retains the previous validated generation. A fresh
run refuses to overwrite an existing run directory. Serving and
certification accept only a manifest whose status is `completed`.

## Serve

    HAETAE_CHECKPOINT=runs/baseline uv run uvicorn haetae.serve:app

The server accepts only a completed generation. Without a calibration
artifact it reports `calibrated: false`. A `calibration.json` must be
bound to the same run ID, checkpoint generation and SHA-256, tokenizer,
inference policy, and named calibration data.

## Freeze and measure a result

Freeze the evaluation population before inspecting model results. The
preparation command verifies the training-data fingerprints, pins source
revisions, removes consumed record and state overlap, and writes
parent-disjoint A, B, and C role files. Replace the example run ID with
the value in the run's `run.json`.

    uv run python -m haetae.measure prepare --run-dir runs/baseline --expect-run-id RUN_ID --out evaluations/baseline

After training completes, role A fits one source-balanced temperature and
role B freezes conformal and selective-risk policy. Neither command reads
role C.

    uv run python -m haetae.measure fit --plan evaluations/baseline --run-dir runs/baseline --device mps --batch 4

The final command requires an explicit gate before reading role C. It
writes digest-bound raw and calibrated metrics, conformal coverage,
simultaneous selective-risk bounds, controlled Choice stress results, and
synchronized latency distributions.

    uv run python -m haetae.measure evaluate --plan evaluations/baseline --run-dir runs/baseline --device mps --batch 4 --allow-certification

## Shared-state experiment

The experimental model encodes a state once, then runs isolated bidirectional
question branches with restarted logical positions. A pointer head scores each
question's dynamic options. Its trainer uses digest-bound frozen suites and the
same immutable-generation recovery rules as the baseline.

Freeze a Korean development and locked test suite while excluding every state
used by the baseline train, internal validation, and A/B/C populations:

    uv run python -m experiments.freeze_korean_suite --run-dir runs/baseline --baseline-plan evaluations/baseline --out evaluations/korean-v1

The freezer removes exact duplicates with conflicting labels and balances each
source by its primary label. Development data may be used for model selection.
The adapter rejects the locked test unless its explicit test gate is enabled.
