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
    src/haetae/serve.py      /v1/systemone FastAPI

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
inference policy, and named calibration data. The certification command
can create that artifact for one evaluated source:

    uv run python -m haetae.certify --ckpt runs/baseline --sources sst5 --save-calibration-source sst5
