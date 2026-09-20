"""End-to-end evaluation of a checkpoint.

Data roles are kept separate: the validation split selects the
model, a held-out calibration split fits temperature, and the
certification split is touched only by the frozen act/abstain
rule. Reported: accuracy, NLL, Brier, ECE before/after
temperature, per-source breakdowns, permutation and option-drop
stress, and a binomial upper bound on P(wrong | accepted).
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .calibrate import certify_selective_risk, fit_temperature
from .checkpoint import (
    CheckpointError,
    load_completed_checkpoint,
    model_config_fingerprint,
    tokenizer_fingerprint,
)
from .data import LOADERS
from .eval import ece, predict, stress_option_perturbation, stress_permutation
from .model import HaetaeModel


def load_model(ckpt_dir, device):
    ck, run, _ = load_completed_checkpoint(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer_fingerprint(tok) != run["spec"]["tokenizer_fingerprint"]:
        raise CheckpointError("certification tokenizer does not match the run")
    m = HaetaeModel(ck["config"]["backbone"])
    if model_config_fingerprint(m) != run["spec"]["model_config_fingerprint"]:
        raise CheckpointError("certification model configuration does not match the run")
    m.backbone.load_state_dict(ck["backbone"])
    m.head.load_state_dict(ck["head"])
    return m.to(device).eval(), tok


def collect_logits(model, tok, records, device, max_len, batch):
    from .model import collate

    out = []  # (original_index, logits) — dropped records keep alignment
    for i in range(0, len(records), batch):
        recs = records[i:i + batch]
        b = collate(tok, recs, max_len)
        keep = [k for k, d in enumerate(b["dropped_options"]) if not d]
        if len(keep) < len(recs):
            recs = [recs[k] for k in keep]
            if not recs:
                continue
            b = collate(tok, recs, max_len)
        lg = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                   b["option_pos"].to(device), b["group_ptr"].to(device))
        out += [(i + k, l.cpu()) for k, l in zip(keep, lg)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v1")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--per-source", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--thresholds", default="0.5,0.7,0.9")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok = load_model(args.ckpt, device)
    rng = random.Random(args.seed)

    for name in args.sources.split(","):
        name = name.strip()
        rows = list(LOADERS[name](split="test" if name != "anli" else "dev_r1",
                                limit=args.per_source * 2))
        rng.shuffle(rows)
        calib, cert = rows[: args.per_source], rows[args.per_source:]
        if not rows:
            continue

        cal_pairs = collect_logits(model, tok, calib, device, args.max_len, args.batch)
        cal_logits = [l for _, l in cal_pairs]
        cal_labels = [calib[i]["label"] for i, _ in cal_pairs]
        T = fit_temperature(cal_logits, cal_labels)

        cert_probs = predict(model, tok, cert, device, args.max_len,
                             args.batch, temperature=T)
        kept = [(r, p) for r, p in zip(cert, cert_probs) if p is not None]
        cert_k = [r for r, _ in kept]
        cert_p = [p for _, p in kept]
        cert_labels = [r["label"] for r in cert_k]
        acc = sum(int(p.argmax()) == y for p, y in zip(cert_p, cert_labels)) / max(1, len(cert_p))
        nll = sum(-float(p[y].clamp_min(1e-9).log())
                  for p, y in zip(cert_p, cert_labels)) / max(1, len(cert_p))
        brier = sum(float(((p - F.one_hot(torch.tensor(y), p.numel()).float()) ** 2).sum())
                    for p, y in zip(cert_p, cert_labels)) / max(1, len(cert_p))
        confs = [float(p.max()) for p in cert_p]
        corr = [int(p.argmax()) == y for p, y in zip(cert_p, cert_labels)]

        print(f"\n== {name} (n={len(cert_p)}/{len(cert)}, T={T:.3f})")
        print(f"acc {acc:.3f}  nll {nll:.3f}  brier {brier:.3f}  ece {ece(confs, corr):.3f}")
        for t in [float(x) for x in args.thresholds.split(",")]:
            c = certify_selective_risk(cert_p, cert_labels, t)
            print(f"  t={t}: accepted {c['accepted']} ({c['coverage']:.0%}), "
                  f"errors {c['errors']}, risk<= {c['risk_upper_bound']:.3f} @95%")

    print("\n== stress (on last source's cert split)")
    stress_permutation(model, tok, cert[:100], device, args.max_len, args.batch)
    stress_option_perturbation(model, tok, cert[:100], device, args.max_len, args.batch)


if __name__ == "__main__":
    main()
