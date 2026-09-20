"""Accuracy, expected calibration error, and latency."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .model import collate


def ece(confs, correct, n_bins=15):
    e = 0.0
    n = len(confs)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        e += len(idx) / n * abs(acc - conf)
    return e


@torch.no_grad()
def predict(model, tok, records, device, max_len=1024, batch=16, temperature=1.0):
    """Return per-record probability tensors.

    Records whose candidate set cannot fit max_len are returned as
    None at their original index; callers must handle them.
    """
    probs = [None] * len(records)
    for i in range(0, len(records), batch):
        recs = records[i:i + batch]
        b = collate(tok, recs, max_len)
        keep = [k for k, d in enumerate(b["dropped_options"]) if not d]
        if not keep:
            continue
        recs_k = [recs[k] for k in keep]
        b = collate(tok, recs_k, max_len)
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["option_pos"].to(device), b["group_ptr"].to(device))
        for k, lg in zip(keep, logits):
            probs[i + k] = F.softmax(lg / temperature, dim=-1).cpu()
    return probs


@torch.no_grad()
def evaluate(model, tok, records, device, max_len=1024, batch=16, temperature=1.0):
    model.eval()
    t0 = time.time()
    probs = predict(model, tok, records, device, max_len, batch, temperature)
    dt = time.time() - t0
    per_source = {}
    confs, correct = [], []
    excluded = 0
    for r, p in zip(records, probs):
        if p is None:
            excluded += 1
            continue
        pred = int(p.argmax())
        ok = int(pred == r["label"]) if r["label"] is not None else 0
        conf = float(p.max())
        confs.append(conf)
        correct.append(ok)
        s = per_source.setdefault(r["source"], [0, 0])
        s[0] += ok
        s[1] += 1
    acc = sum(correct) / max(1, len(correct))
    print(f"accuracy {acc:.3f}  ece {ece(confs, correct):.3f}  "
          f"{dt:.1f}s for {len(records)} ({1000 * dt / max(1, len(records)):.0f}ms/rec)"
          + (f"  excluded {excluded}" if excluded else ""))
    for k, (c, n) in sorted(per_source.items()):
        print(f"  {k}: {c}/{n} = {c / n:.3f}")
    return {"accuracy": acc, "ece": ece(confs, correct)}


@torch.no_grad()
def stress_permutation(model, tok, records, device, max_len=1024, batch=16,
                       n_perms=3, seed=0):
    """Option-order invariance: does the predicted option TEXT survive
    shuffling, and does the distribution over shared options stay close?

    A decision model should not change its answer because the candidate
    list was reordered.
    """
    import random

    rng = random.Random(seed)
    base = predict(model, tok, records, device, max_len, batch)
    same_answer, mean_tv, compared = 0, 0.0, 0
    for _ in range(n_perms):
        shuffled, remap = [], []
        for r in records:
            idx = list(range(len(r["options"])))
            rng.shuffle(idx)
            rec = dict(r)
            rec["options"] = [r["options"][j] for j in idx]
            rec["label"] = idx.index(r["label"]) if r["label"] is not None else None
            shuffled.append(rec)
            remap.append(idx)
        perm = predict(model, tok, shuffled, device, max_len, batch)
        for r, p0, p1, idx in zip(records, base, perm, remap):
            if p0 is None or p1 is None:
                continue
            compared += 1
            orig_answer = int(p0.argmax())
            perm_answer_orig_idx = idx[int(p1.argmax())]
            same_answer += int(orig_answer == perm_answer_orig_idx)
            tv = 0.5 * sum(abs(float(p0[j]) - float(p1[idx.index(j)]))
                           for j in range(len(idx)))
            mean_tv += tv
    n = max(1, compared)
    print(f"permutation: same answer {same_answer}/{n} = {same_answer / n:.3f}  "
          f"mean TV distance {mean_tv / n:.4f}")
    return {"same_answer": same_answer / n, "mean_tv": mean_tv / n}


@torch.no_grad()
def stress_option_perturbation(model, tok, records, device, max_len=1024,
                               batch=16, seed=0):
    """Drop a random non-gold option: does the distribution over the
    surviving options stay close (renormalized)?"""
    import random

    rng = random.Random(seed)
    base = predict(model, tok, records, device, max_len, batch)
    perturbed, dropped = [], []
    for r in records:
        k = len(r["options"])
        if k < 3:
            continue
        choices = [j for j in range(k) if j != r["label"]]
        d = rng.choice(choices)
        rec = dict(r)
        rec["options"] = [o for j, o in enumerate(r["options"]) if j != d]
        rec["label"] = r["label"] - (1 if r["label"] is not None and d < r["label"] else 0)
        perturbed.append(rec)
        dropped.append(d)
    pert = predict(model, tok, perturbed, device, max_len, batch)
    same_answer, mean_tv, n = 0, 0.0, 0
    i = 0
    for r, p0 in zip(records, base):
        if len(r["options"]) < 3:
            continue
        d = dropped[i]
        p1 = pert[i]
        i += 1
        if p0 is None or p1 is None:
            continue
        n += 1
        keep = [j for j in range(len(r["options"])) if j != d]
        p0r = torch.tensor([float(p0[j]) for j in keep])
        p0r = p0r / p0r.sum()
        same_answer += int(keep[int(p1.argmax())] == int(p0.argmax()))
        mean_tv += float(0.5 * (p0r - p1).abs().sum())
    print(f"option-drop: same answer {same_answer}/{n} = {same_answer / n:.3f}  "
          f"mean TV distance {mean_tv / n:.4f}")
    return {"same_answer": same_answer / max(1, n), "mean_tv": mean_tv / max(1, n)}
