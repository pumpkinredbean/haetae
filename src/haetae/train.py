"""Train with cross-entropy + Brier (proper scoring rule) loss.

Soft labels (annotator disagreement) are used where a dataset
provides them; both CE and Brier are minimized only by the true
probability, which is the supervised form of RLCD-style training.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import LOADERS
from .model import HaetaeModel, collate


def brier(logits, target):
    p = F.softmax(logits, dim=-1)
    return ((p - target) ** 2).sum(-1).mean()


def batch_loss(logit_list, targets, brier_w=0.5):
    """targets: list of soft distributions (hard labels already one-hot)."""
    ce, br = 0.0, 0.0
    for logits, t in zip(logit_list, targets):
        t = t.to(logits.device)
        logp = F.log_softmax(logits, dim=-1)
        ce = ce - (t * logp).sum()
        br = br + ((F.softmax(logits, -1) - t) ** 2).sum()
    n = len(logit_list)
    return ce / n + brier_w * br / n


def to_target(rec, n):
    if rec["soft"] is not None:
        return torch.tensor(rec["soft"], dtype=torch.float)
    t = torch.zeros(n)
    t[rec["label"]] = 1.0
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--sources", default="ag_news,boolq,sst5,banking77")
    ap.add_argument("--per-source", type=int, default=2000)
    ap.add_argument("--eval-per-source", type=int, default=300)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--brier-w", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = HaetaeModel(args.backbone).to(device)

    train, val = [], []
    for name in args.sources.split(","):
        name = name.strip()
        rows = list(LOADERS[name](split="train", limit=args.per_source + args.eval_per_source))
        random.shuffle(rows)
        val += rows[: args.eval_per_source]
        train += rows[args.eval_per_source:]
        print(f"{name}: {len(rows) - min(len(rows), args.eval_per_source)} train / "
              f"{min(len(rows), args.eval_per_source)} val")
    random.shuffle(train)

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "head" in n_ else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "lr": args.lr, "weight_decay": 0.01},
         {"params": model.head.parameters(), "lr": args.head_lr, "weight_decay": 0.0}],
        lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * args.steps), args.steps)

    step = 0
    t0 = time.time()
    while step < args.steps:
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            if step >= args.steps:
                break
            recs = train[i:i + args.batch]
            batch = collate(tok, recs, args.max_len)
            if any(batch["dropped_options"]):
                keep = [r for r, d in zip(recs, batch["dropped_options"]) if not d]
                if not keep:
                    continue
                recs = keep
                batch = collate(tok, recs, args.max_len)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["option_pos"].to(device),
                batch["group_ptr"].to(device),
            )
            targets = [to_target(r, n) for r, n in zip(recs, batch["n_options"])]
            loss = batch_loss(logits, targets, args.brier_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                      f"({(time.time() - t0) / step:.2f}s/step)")

    import os
    os.makedirs(args.out, exist_ok=True)
    torch.save({"head": model.head.state_dict(),
                "backbone": model.backbone.state_dict(),
                "config": vars(args)}, f"{args.out}/model.pt")
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")

    # quick val pass: accuracy + ECE before temperature
    from .eval import evaluate
    evaluate(model, tok, val, device, args.max_len, batch=args.batch)


if __name__ == "__main__":
    main()
