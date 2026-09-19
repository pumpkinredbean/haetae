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
    src/haetae/eval.py       accuracy, ECE, latency per source
    src/haetae/calibrate.py  temperature scaling + split conformal
    src/haetae/serve.py      /v1/systemone FastAPI

## Train

    uv sync
    uv run python -m haetae.train --sources ag_news,boolq,sst5,banking77,klue_ynat,nsmc

## Serve

    uv run uvicorn haetae.serve:app   # then POST /v1/systemone
