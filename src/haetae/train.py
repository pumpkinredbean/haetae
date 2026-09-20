"""Train with cross-entropy + Brier (proper scoring rule) loss.

Soft labels (annotator disagreement) are used where a dataset
provides them; both CE and Brier are minimized only by the true
probability, which is the supervised form of RLCD-style training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import LOADERS
from .model import HaetaeModel, collate


def atomic_torch_save(payload, path):
    """Write a checkpoint without ever exposing a partial target file."""
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload, path):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def save_checkpoint(args, model, opt, sched, step, skipped):
    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, "checkpoint.pt")
    rng = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        rng["mps"] = torch.mps.get_rng_state()
    atomic_torch_save({
        "version": 1,
        "head": model.head.state_dict(),
        "backbone": model.backbone.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "skipped": skipped,
        "config": vars(args),
        "rng": rng,
    }, checkpoint_path)
    atomic_json_save({
        "step": step,
        "steps": args.steps,
        "skipped": skipped,
        "checkpoint": checkpoint_path,
        "updated_at_unix": time.time(),
    }, os.path.join(args.out, "progress.json"))
    print(f"checkpoint -> {checkpoint_path} (step {step})", flush=True)


def load_checkpoint(args, model, opt, sched, device):
    if args.resume == "none":
        return 0, 0
    path = (os.path.join(args.out, "checkpoint.pt")
            if args.resume == "auto" else args.resume)
    if not os.path.exists(path):
        if args.resume == "auto":
            print(f"no checkpoint at {path}; starting from step 0", flush=True)
            return 0, 0
        raise FileNotFoundError(path)

    ck = torch.load(path, map_location="cpu", weights_only=False)
    required = ("backbone", "sources", "per_source", "eval_per_source", "steps",
                "batch", "lr", "head_lr", "brier_w", "max_len", "seed")
    mismatches = {
        key: (ck["config"].get(key), getattr(args, key))
        for key in required
        if ck["config"].get(key) != getattr(args, key)
    }
    if mismatches:
        raise ValueError(f"checkpoint configuration mismatch: {mismatches}")

    model.backbone.load_state_dict(ck["backbone"])
    model.head.load_state_dict(ck["head"])
    opt.load_state_dict(ck["optimizer"])
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    sched.load_state_dict(ck["scheduler"])
    random.setstate(ck["rng"]["python"])
    torch.set_rng_state(ck["rng"]["torch"])
    if ("mps" in ck["rng"] and torch.backends.mps.is_available()
            and hasattr(torch.mps, "set_rng_state")):
        torch.mps.set_rng_state(ck["rng"]["mps"])
    print(f"resumed {path} at step {ck['step']}", flush=True)
    return int(ck["step"]), int(ck.get("skipped", 0))


def brier(logits, target):
    p = F.softmax(logits, dim=-1)
    return ((p - target) ** 2).sum(-1).mean()


def batch_loss(logit_list, targets, brier_w=0.5):
    """targets: list of soft distributions (hard labels already one-hot)."""
    assert len(logit_list) == len(targets) and len(logit_list) > 0
    ce, br = 0.0, 0.0
    for logits, t in zip(logit_list, targets):
        # validate on CPU — every .item()/.all() on an accelerator
        # tensor forces a device sync inside the training loop
        assert logits.ndim == 1 and t.ndim == 1 and logits.shape == t.shape
        assert logits.numel() >= 2, "need >=2 options for a decision"
        assert torch.isfinite(t).all() and (t >= 0).all()
        assert abs(float(t.sum()) - 1.0) < 1e-3
        t = t.to(logits.device)
        logp = F.log_softmax(logits, dim=-1)
        ce = ce - (t * logp).sum()
        br = br + ((F.softmax(logits, -1) - t) ** 2).sum()
    n = len(logit_list)
    return ce / n + brier_w * br / n


def to_target(rec, n):
    if rec["soft"] is not None:
        t = torch.tensor(rec["soft"], dtype=torch.float)
        assert t.numel() == n
        return t
    assert rec["label"] is not None and 0 <= rec["label"] < n
    t = torch.zeros(n)
    t[rec["label"]] = 1.0
    return t


def permute_choice(rec, rng):
    """Candidate-order augmentation for choice records.

    The packed model is not permutation-equivariant (rotary positions,
    local-attention neighborhoods), and several sources always use the
    same option order — without this the model can learn 'this task's
    second slot'. Score keeps ordinal order; noul keeps 'yes' first.
    """
    if rec["type"] != "choice" or len(rec["options"]) < 2:
        return rec
    perm = list(range(len(rec["options"])))
    rng.shuffle(perm)
    out = dict(rec)
    out["options"] = [rec["options"][j] for j in perm]
    if rec["soft"] is not None:
        out["soft"] = [rec["soft"][j] for j in perm]
    if rec["label"] is not None:
        out["label"] = perm.index(rec["label"])
    return out


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
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", default="none",
                    help="none, auto, or a checkpoint path")
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
        # Split by parent state, not by emitted question: expanded
        # records (e.g. helpsteer2's five scores per prompt) must not
        # straddle the split.
        parents = {}
        for r in rows:
            parents.setdefault(r.get("parent", id(r)), []).append(r)
        keys = list(parents)
        random.shuffle(keys)
        n_val_parents = max(1, int(len(keys) * args.eval_per_source /
                                   (args.per_source + args.eval_per_source)))
        for k in keys[:n_val_parents]:
            val += parents[k]
        for k in keys[n_val_parents:]:
            train += parents[k]
        print(f"{name}: {len(rows) - min(len(rows), args.eval_per_source)} train / "
              f"{min(len(rows), args.eval_per_source)} val")
    random.shuffle(train)

    backbone_decay, backbone_no_decay = [], []
    for p in model.backbone.parameters():
        if p.requires_grad:
            (backbone_no_decay if p.ndim <= 1 else backbone_decay).append(p)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": backbone_decay, "lr": args.lr, "weight_decay": 0.01},
         {"params": backbone_no_decay, "lr": args.lr, "weight_decay": 0.0},
         {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}],
        lr=args.lr)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(actual) == expected and len(actual) == len(expected)
    model.train()
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * args.steps), args.steps)

    step, skipped = load_checkpoint(args, model, opt, sched, device)
    os.makedirs(args.out, exist_ok=True)
    tok.save_pretrained(args.out)

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; saving after current update", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    t0 = time.time()
    while step < args.steps:
        random.shuffle(train)
        updates_this_pass = 0
        for i in range(0, len(train), args.batch):
            if step >= args.steps:
                break
            recs = [permute_choice(r, random) for r in train[i:i + args.batch]]
            batch = collate(tok, recs, args.max_len)
            if any(batch["dropped_options"]):
                keep = [r for r, d in zip(recs, batch["dropped_options"]) if not d]
                skipped += sum(batch["dropped_options"])
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
            updates_this_pass += 1
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(args, model, opt, sched, step, skipped)
            if step % 25 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                      f"({(time.time() - t0) / step:.2f}s/step, skipped {skipped})")
            if stop_requested:
                save_checkpoint(args, model, opt, sched, step, skipped)
                print("stopped with resumable checkpoint", flush=True)
                return
        if updates_this_pass == 0:
            print("no trainable records survived a full pass — stopping")
            break

    os.makedirs(args.out, exist_ok=True)
    atomic_torch_save({"head": model.head.state_dict(),
                       "backbone": model.backbone.state_dict(),
                       "config": vars(args),
                       "step": step}, f"{args.out}/model.pt")
    save_checkpoint(args, model, opt, sched, step, skipped)
    print(f"saved -> {args.out}")

    # quick val pass: accuracy + ECE before temperature
    from .eval import evaluate
    evaluate(model, tok, val, device, args.max_len, batch=args.batch)


if __name__ == "__main__":
    main()
