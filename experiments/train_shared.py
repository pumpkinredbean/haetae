"""Train the shared-state experiment on a digest-bound frozen suite."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from haetae.checkpoint import (
    CheckpointError,
    CheckpointStore,
    RunDirectoryLock,
    checkpoint_payload,
    make_run_spec,
    restore_checkpoint,
    validate_checkpoint,
)
from experiments.audit_shared_suite import load_rendered_state_audit
from experiments.kev_adapter import file_sha256, load_frozen_split
from experiments.shared_state import encode_request, load_shared_state_model


DEFAULT_BACKBONE = "jhu-clsp/mmBERT-small"
DEFAULT_BACKBONE_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"
EXPERIMENT_VERSION = 1


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_identity(repository: Path) -> dict:
    files = [
        repository / "experiments" / "audit_shared_suite.py",
        repository / "experiments" / "comparison_protocol.py",
        repository / "experiments" / "evaluate_baseline.py",
        repository / "experiments" / "evaluate_shared.py",
        repository / "experiments" / "freeze_comparison.py",
        repository / "experiments" / "freeze_shared_suite.py",
        repository / "experiments" / "kev_adapter.py",
        repository / "experiments" / "shared_state.py",
        repository / "experiments" / "train_shared.py",
        repository / "src" / "haetae" / "checkpoint.py",
        repository / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    names = []
    for source in files:
        relative = str(source.relative_to(repository))
        names.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": names}


def permute_choices(request: dict, rng) -> dict:
    result = copy.deepcopy(request)
    for question in result["questions"]:
        if question["type"] != "choice":
            continue
        size = len(question["options"])
        permutation = list(range(size))
        rng.shuffle(permutation)
        old_label = question["label"]
        question["options"] = [
            question["options"][index] for index in permutation
        ]
        question["option_keys"] = [
            question["option_keys"][index] for index in permutation
        ]
        if question["soft"] is not None:
            question["soft"] = [
                question["soft"][index] for index in permutation
            ]
        question["label"] = permutation.index(old_label)
    return result


def target_distribution(question: dict, logits: torch.Tensor) -> torch.Tensor:
    if question["soft"] is None:
        target = torch.zeros_like(logits)
        target[question["label"]] = 1.0
        return target
    target = torch.tensor(
        question["soft"], dtype=logits.dtype, device=logits.device,
    )
    if target.shape != logits.shape:
        raise ValueError("soft target shape differs from option logits")
    if not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError("soft target is not a finite distribution")
    if not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5):
        raise ValueError("soft target does not sum to one")
    return target


def question_loss(
    logits: torch.Tensor,
    question: dict,
    brier_weight: float,
    ordinal_weight: float,
) -> torch.Tensor:
    if logits.ndim != 1 or logits.numel() != len(question["options"]):
        raise ValueError("question logits do not match its options")
    target = target_distribution(question, logits)
    log_probabilities = F.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    loss = -(target * log_probabilities).sum()
    loss = loss + brier_weight * ((probabilities - target) ** 2).sum()
    if question["type"] == "score" and ordinal_weight:
        difference = (
            probabilities.cumsum(0)[:-1] - target.cumsum(0)[:-1]
        )
        loss = loss + ordinal_weight * difference.square().mean()
    return loss


def request_batch_loss(
    outputs: list[list[torch.Tensor]],
    requests: list[dict],
    brier_weight: float,
    ordinal_weight: float,
) -> torch.Tensor:
    if len(outputs) != len(requests) or not requests:
        raise ValueError("model output count differs from request count")
    request_losses = []
    for question_logits, request in zip(outputs, requests):
        questions = request["questions"]
        if len(question_logits) != len(questions):
            raise ValueError("model question count differs from request")
        losses = [
            question_loss(logits, question, brier_weight, ordinal_weight)
            for logits, question in zip(question_logits, questions)
        ]
        request_losses.append(torch.stack(losses).mean())
    return torch.stack(request_losses).mean()


def validate_context_policy(
    tokenizer,
    delimiters: dict[str, int],
    requests: list[dict],
    max_length: int,
) -> dict:
    if not requests:
        raise ValueError("frozen training partition is empty")
    lengths = []
    for request in requests:
        encoding = encode_request(
            tokenizer,
            request["state"],
            request["questions"],
            max_length,
            delimiters,
        )
        if encoding is None:
            raise ValueError("frozen training request exceeds the fixed content budget")
        if encoding.retained_state_tokens != encoding.state_tokens:
            raise ValueError("frozen training request would truncate its state")
        lengths.append(len(encoding.input_ids))
    ordered = sorted(lengths)
    return {
        "requests": len(lengths),
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
        "maximum": ordered[-1],
    }


def encode_requests(
    tokenizer,
    delimiters: dict[str, int],
    requests: list[dict],
    max_length: int,
):
    encodings = [
        encode_request(
            tokenizer,
            request["state"],
            request["questions"],
            max_length,
            delimiters,
        )
        for request in requests
    ]
    if any(encoding is None for encoding in encodings):
        raise ValueError("a training request exceeded the validated context policy")
    return encodings


def build_optimizer(model, backbone_lr: float, head_lr: float):
    backbone_decay, backbone_no_decay = [], []
    for parameter in model.backbone.parameters():
        if parameter.requires_grad:
            target = backbone_no_decay if parameter.ndim <= 1 else backbone_decay
            target.append(parameter)
    head = [parameter for parameter in model.head.parameters()
            if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            "name": "backbone_decay",
            "params": backbone_decay,
            "lr": backbone_lr,
            "weight_decay": 0.01,
        },
        {
            "name": "backbone_no_decay",
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "head",
            "params": head,
            "lr": head_lr,
            "weight_decay": 0.0,
        },
    ], lr=backbone_lr)
    expected = {id(parameter) for parameter in model.parameters()
                if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups
              for parameter in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer does not cover trainable parameters exactly once")
    return optimizer


def prepare_checkpoint_arguments(args, manifest: dict) -> None:
    args.sources = ",".join(manifest.get("trainable_sources", []))
    args.per_source = 0
    args.eval_per_source = 0
    args.lr = args.backbone_lr
    args.head_lr = args.pointer_lr


def train(args) -> None:
    if not 1 <= args.microbatch <= args.batch:
        raise ValueError("microbatch must be between one and batch")
    if args.steps <= 0 or args.save_every <= 0:
        raise ValueError("steps and save_every must be positive")
    if args.log_every <= 0 or args.pointer_size <= 0 or args.max_len <= 0:
        raise ValueError("log interval, pointer size, and maximum length must be positive")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup fraction must be in [0, 1)")
    if min(args.brier_w, args.ordinal_w) < 0:
        raise ValueError("loss weights must be nonnegative")

    repository = Path(__file__).resolve().parents[1]
    suite = Path(args.suite).resolve()
    requests, manifest = load_frozen_split(suite, "train")
    _, calibration_manifest = load_frozen_split(suite, "calibration")
    if calibration_manifest != manifest:
        raise ValueError("suite manifest changed while loading partitions")
    if not manifest.get("trainable_sources"):
        raise ValueError("suite does not declare trainable sources")
    rendered_state_audit = load_rendered_state_audit(suite)
    prepare_checkpoint_arguments(args, manifest)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    output = Path(args.out)
    tokenizer_source = output if args.resume == "auto" else args.backbone
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if args.resume == "auto" else args.backbone_revision,
        local_files_only=args.local_files_only,
    )
    model, delimiters = load_shared_state_model(
        args.backbone,
        tokenizer,
        device,
        revision=args.backbone_revision,
        pointer_size=args.pointer_size,
        attention_implementation="eager",
        local_files_only=args.local_files_only,
    )
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
    context_report = validate_context_policy(
        tokenizer, delimiters, requests, args.max_len,
    )
    print("context " + json.dumps(context_report, sort_keys=True), flush=True)

    optimizer = build_optimizer(model, args.backbone_lr, args.pointer_lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.steps * args.warmup_fraction), args.steps,
    )
    producer = source_identity(repository)
    train_descriptor = manifest["files"]["train.jsonl"]
    held_descriptors = {
        name: manifest["files"][name]
        for name in ("calibration.jsonl", "development.jsonl")
    }
    spec = make_run_spec(
        args,
        model,
        optimizer,
        tokenizer,
        device,
        train_descriptor["sha256"],
        canonical_sha256(held_descriptors),
        repository,
    )
    spec["training_code"] = producer
    spec["experiment"] = {
        "version": EXPERIMENT_VERSION,
        "suite_manifest_sha256": file_sha256(suite / "manifest.json"),
        "train_descriptor": train_descriptor,
        "held_descriptors": held_descriptors,
        "backbone_revision": args.backbone_revision,
        "pointer_size": args.pointer_size,
        "ordinal_weight": args.ordinal_w,
        "warmup_fraction": args.warmup_fraction,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "attention_implementation": "eager",
        "request_objective": "mean question loss, then mean request loss",
        "context_report": context_report,
        "rendered_state_audit": {
            "artifact_sha256": rendered_state_audit["artifact_sha256"],
            "file_sha256": file_sha256(
                suite / "rendered-state-audit.json"
            ),
        },
    }

    store = CheckpointStore(output)
    if args.resume == "none":
        run = store.start(spec, producer)
        tokenizer.save_pretrained(output)
        step, skipped, data_state = 0, 0, None
    else:
        run = store.open(spec)
        payload, role, failures = store.load(
            lambda candidate: validate_checkpoint(
                candidate,
                run,
                args,
                model,
                optimizer,
                scheduler,
                len(requests),
                device,
            )
        )
        step, skipped, data_state = restore_checkpoint(
            payload, model, optimizer, scheduler, device,
        )
        if failures:
            print("checkpoint recovery warnings: " + "; ".join(failures), flush=True)
        print(
            f"resumed {role} generation {store.active_descriptor['generation']} "
            f"at step {step}",
            flush=True,
        )

    if data_state is None:
        order = list(range(len(requests)))
        random.shuffle(order)
        epoch, cursor = 0, 0
    else:
        order = list(data_state["order"])
        epoch = int(data_state["epoch"])
        cursor = int(data_state["cursor"])
    current_data_state = {"epoch": epoch, "order": order, "cursor": cursor}

    stop_requested = False
    stop_signal = None

    def request_stop(signum, frame):
        nonlocal stop_requested, stop_signal
        stop_requested = True
        stop_signal = signum

    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sighup = (
        signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None
    )
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_stop)

    def publish(status: str) -> None:
        payload = checkpoint_payload(
            run,
            args,
            model,
            optimizer,
            scheduler,
            step,
            skipped,
            current_data_state,
            status,
            producer,
        )
        validator = lambda saved: validate_checkpoint(
            saved,
            run,
            args,
            model,
            optimizer,
            scheduler,
            len(requests),
            device,
        )
        validator(payload)
        descriptor = store.publish(payload, status, validator)
        print(
            f"checkpoint generation {descriptor['generation']} -> "
            f"{descriptor['path']} (step {step}, {status})",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    start_step = step
    try:
        if data_state is None:
            publish("running")
        while step < args.steps:
            if stop_requested:
                print(f"received signal {stop_signal}; saving at update {step}", flush=True)
                publish("interrupted")
                return
            if cursor >= len(order):
                epoch += 1
                order = list(range(len(requests)))
                random.shuffle(order)
                cursor = 0
                current_data_state = {
                    "epoch": epoch, "order": order, "cursor": cursor,
                }

            indexes = order[cursor:cursor + args.batch]
            cursor += len(indexes)
            batch_requests = [
                permute_choices(requests[index], random) for index in indexes
            ]
            logged_loss = 0.0
            for start in range(0, len(batch_requests), args.microbatch):
                micro_requests = batch_requests[start:start + args.microbatch]
                encodings = encode_requests(
                    tokenizer, delimiters, micro_requests, args.max_len,
                )
                outputs = model(encodings)
                micro_loss = request_batch_loss(
                    outputs,
                    micro_requests,
                    args.brier_w,
                    args.ordinal_w,
                )
                weight = len(micro_requests) / len(batch_requests)
                (micro_loss * weight).backward()
                if (step + 1) % args.log_every == 0:
                    logged_loss += float(micro_loss.detach()) * weight
                del encodings, outputs, micro_loss
                if device == "mps":
                    torch.mps.empty_cache()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device == "mps":
                torch.mps.empty_cache()
            step += 1
            current_data_state = {
                "epoch": epoch, "order": order, "cursor": cursor,
            }
            if step % args.save_every == 0:
                publish("completed" if step == args.steps else "running")
            if step % args.log_every == 0:
                completed = max(1, step - start_step)
                print(
                    f"step {step}/{args.steps} loss {logged_loss:.4f} "
                    f"({(time.time() - started) / completed:.2f}s/step)",
                    flush=True,
                )
            if stop_requested and step < args.steps:
                print(f"received signal {stop_signal}; saving at update {step}", flush=True)
                publish("interrupted")
                return
        if (store.active_descriptor is None
                or store.active_descriptor["step"] != step
                or store.active_descriptor["status"] != "completed"):
            publish("completed")
        print(f"completed -> {output}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, original_sighup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--backbone-revision", default=DEFAULT_BACKBONE_REVISION)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--pointer-lr", type=float, default=2e-4)
    parser.add_argument("--brier-w", type=float, default=0.5)
    parser.add_argument("--ordinal-w", type=float, default=0.25)
    parser.add_argument("--warmup-fraction", type=float, default=0.06)
    parser.add_argument("--pointer-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--out", default="runs/shared-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    args = parser.parse_args()
    with RunDirectoryLock(args.out):
        train(args)


if __name__ == "__main__":
    main()
