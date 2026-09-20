"""Train with cross-entropy + Brier (proper scoring rule) loss.

Soft labels (annotator disagreement) are used where a dataset
provides them; both CE and Brier are minimized only by the true
probability, which is the supervised form of RLCD-style training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .checkpoint import (
    CheckpointError,
    CheckpointStore,
    RunDirectoryLock,
    checkpoint_payload,
    make_run_spec,
    producer_identity,
    restore_checkpoint,
    validate_checkpoint,
)
from .data import LOADERS
from .model import HaetaeModel, collate


def dataset_fingerprint(records):
    """Stable digest of the exact ordered training records."""
    digest = hashlib.sha256()
    fields = ("state", "type", "instructions", "options", "label",
              "soft", "source", "parent")
    for record in records:
        payload = {key: record.get(key) for key in fields}
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()




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
    ap.add_argument(
        "--microbatch", type=int,
        help="questions per forward/backward pass; defaults to --batch",
    )
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--brier-w", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", choices=("none", "auto"), default="none",
                    help="start a new run or resume its validated manifest")
    args = ap.parse_args()
    if args.microbatch is None:
        args.microbatch = args.batch

    with RunDirectoryLock(args.out):
        train(args)


def train(args):

    if not hasattr(args, "microbatch") or args.microbatch is None:
        args.microbatch = args.batch
    if (isinstance(args.batch, bool) or not isinstance(args.batch, int)
            or args.batch <= 0):
        raise ValueError("batch must be a positive integer")
    if (isinstance(args.microbatch, bool)
            or not isinstance(args.microbatch, int)
            or not 1 <= args.microbatch <= args.batch):
        raise ValueError("microbatch must be a positive integer no larger than batch")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer_source = args.out if args.resume == "auto" else args.backbone
    tok = AutoTokenizer.from_pretrained(tokenizer_source)
    model = HaetaeModel(args.backbone).to(device)

    train, val = [], []
    for name in args.sources.split(","):
        name = name.strip()
        train_before, val_before = len(train), len(val)
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
        print(
            f"{name}: {len(train) - train_before} train questions / "
            f"{len(val) - val_before} validation questions; "
            f"{len(keys) - n_val_parents} train parents / "
            f"{n_val_parents} validation parents"
        )
    train_fingerprint = dataset_fingerprint(train)
    validation_fingerprint = dataset_fingerprint(val)

    backbone_decay, backbone_no_decay = [], []
    for p in model.backbone.parameters():
        if p.requires_grad:
            (backbone_no_decay if p.ndim <= 1 else backbone_decay).append(p)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"name": "backbone_decay", "params": backbone_decay,
          "lr": args.lr, "weight_decay": 0.01},
         {"name": "backbone_no_decay", "params": backbone_no_decay,
          "lr": args.lr, "weight_decay": 0.0},
         {"name": "head", "params": head_params,
          "lr": args.head_lr, "weight_decay": 0.0}],
        lr=args.lr)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(actual) == expected and len(actual) == len(expected)
    model.train()
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * args.steps), args.steps)

    repository = Path(__file__).resolve().parents[2]
    producer = producer_identity(repository)
    spec = make_run_spec(
        args, model, opt, tok, device, train_fingerprint,
        validation_fingerprint, repository,
    )
    store = CheckpointStore(args.out)
    if args.resume == "none":
        run = store.start(spec, producer)
        step, skipped, data_state = 0, 0, None
    else:
        run = store.open(spec)
        payload, role, failures = store.load(
            lambda candidate: validate_checkpoint(
                candidate, run, args, model, opt, sched, len(train), device
            )
        )
        if payload["status"] == "failed_no_progress":
            raise CheckpointError(
                "the latest valid checkpoint failed with no progress; "
                "start a corrected recipe in a new output directory"
            )
        step, skipped, data_state = restore_checkpoint(
            payload, model, opt, sched, device
        )
        if failures:
            print("checkpoint recovery warnings: " + "; ".join(failures), flush=True)
        print(
            f"resumed {role} generation {store.active_descriptor['generation']} "
            f"at step {step}", flush=True
        )
    if args.resume == "none":
        tok.save_pretrained(args.out)

    fresh_run = data_state is None
    if fresh_run:
        order = list(range(len(train)))
        random.shuffle(order)
        epoch = 0
        cursor = 0
    else:
        order = list(data_state["order"])
        epoch = int(data_state["epoch"])
        cursor = int(data_state["cursor"])

    stop_requested = False
    stop_signal = None

    def request_stop(signum, frame):
        nonlocal stop_requested, stop_signal
        stop_requested = True
        stop_signal = signum

    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    t0 = time.time()
    start_step = step
    current_data_state = {"epoch": epoch, "order": order, "cursor": cursor}

    def publish(status):
        payload = checkpoint_payload(
            run, args, model, opt, sched, step, skipped,
            current_data_state, status, producer,
        )
        validate_checkpoint(
            payload, run, args, model, opt, sched, len(train), device
        )
        descriptor = store.publish(
            payload, status,
            validator=lambda saved: validate_checkpoint(
                saved, run, args, model, opt, sched, len(train), device
            ),
        )
        print(
            f"checkpoint generation {descriptor['generation']} -> "
            f"{descriptor['path']} (step {step}, {status})", flush=True
        )

    def publish_interrupted():
        print(
            f"received signal {stop_signal}; saving at update {step}",
            flush=True,
        )
        publish("interrupted")
        print("stopped with a validated resumable checkpoint", flush=True)

    try:
        if fresh_run:
            publish("running")
        if not train:
            publish("failed_no_progress")
            raise CheckpointError("training split is empty")
        full_epoch_observed = cursor == 0
        updates_in_epoch = 0
        while step < args.steps:
            if stop_requested:
                publish_interrupted()
                return
            if cursor >= len(order):
                if full_epoch_observed and updates_in_epoch == 0:
                    publish("failed_no_progress")
                    raise CheckpointError(
                        "no trainable records survived a full epoch"
                    )
                epoch += 1
                order = list(range(len(train)))
                random.shuffle(order)
                cursor = 0
                full_epoch_observed = True
                updates_in_epoch = 0
                current_data_state = {
                    "epoch": epoch, "order": order, "cursor": cursor,
                }
            batch_indices = order[cursor:cursor + args.batch]
            cursor += len(batch_indices)
            recs = [permute_choice(train[i], random) for i in batch_indices]
            batch = collate(tok, recs, args.max_len)
            if any(batch["dropped_options"]):
                keep = [
                    record for record, dropped
                    in zip(recs, batch["dropped_options"]) if not dropped
                ]
                skipped += sum(batch["dropped_options"])
                if not keep:
                    current_data_state = {
                        "epoch": epoch, "order": order, "cursor": cursor,
                    }
                    if stop_requested:
                        publish_interrupted()
                        return
                    continue
                recs = keep
                batch = collate(tok, recs, args.max_len)
            log_update = (step + 1) % 25 == 0
            logged_loss = 0.0
            for micro_start in range(0, len(recs), args.microbatch):
                micro_records = recs[micro_start:micro_start + args.microbatch]
                if micro_start == 0 and len(micro_records) == len(recs):
                    micro = batch
                else:
                    micro = collate(tok, micro_records, args.max_len)
                logits = model(
                    micro["input_ids"].to(device),
                    micro["attention_mask"].to(device),
                    micro["option_pos"].to(device),
                    micro["group_ptr"].to(device),
                )
                targets = [
                    to_target(record, count)
                    for record, count in zip(micro_records, micro["n_options"])
                ]
                micro_loss = batch_loss(logits, targets, args.brier_w)
                weight = len(micro_records) / len(recs)
                (micro_loss * weight).backward()
                if log_update:
                    logged_loss += float(micro_loss.detach()) * weight
                del logits, targets, micro_loss
                if device == "mps":
                    torch.mps.empty_cache()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if device == "mps":
                torch.mps.empty_cache()
            step += 1
            updates_in_epoch += 1
            current_data_state = {
                "epoch": epoch,
                "order": order,
                "cursor": cursor,
            }
            if args.save_every > 0 and step % args.save_every == 0:
                publish("completed" if step == args.steps else "running")
            if step % 25 == 0:
                completed = max(1, step - start_step)
                print(
                    f"step {step}/{args.steps} loss {logged_loss:.4f} "
                    f"({(time.time() - t0) / completed:.2f}s/step since start, "
                    f"skipped {skipped})",
                    flush=True,
                )
            if stop_requested:
                if step < args.steps:
                    publish_interrupted()
                    return
                break

        if store.active_descriptor is None or step != store.active_descriptor["step"] \
                or store.active_descriptor["status"] != "completed":
            publish("completed")
        print(f"completed -> {args.out}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)

    if stop_requested:
        print("evaluation skipped after stop request", flush=True)
        return

    # quick val pass: accuracy + ECE before temperature
    from .eval import evaluate
    evaluate(model, tok, val, device, args.max_len, batch=args.microbatch)


if __name__ == "__main__":
    main()
