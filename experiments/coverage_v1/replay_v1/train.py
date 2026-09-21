"""Guarded T05 initialization and matched-tape training engine."""

from __future__ import annotations

import copy
import random
import signal
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.state import tensor_state_identity
from experiments.train_shared import build_optimizer, request_batch_loss

TOTAL_UPDATES = 512
WARMUP_UPDATES = 30
EFFECTIVE_BATCH = 8
MICROBATCH = 2
MEMORY_LIMIT_BYTES = 12 * 1024**3

_STOP_REQUESTED = False


def request_stop(signum=None, frame=None) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


@contextmanager
def _signal_handlers():
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    prior = {item: signal.getsignal(item) for item in signals}
    try:
        for item in signals:
            signal.signal(item, request_stop)
        yield
    finally:
        for item, handler in prior.items():
            signal.signal(item, handler)


def build_scheduler(optimizer):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_UPDATES,
        num_training_steps=TOTAL_UPDATES,
    )


def _validate_optimizer(model, optimizer) -> None:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen = []
    expected_groups = (
        ("backbone_decay", 5e-6, 0.01),
        ("backbone_no_decay", 5e-6, 0.0),
        ("head", 5e-5, 0.0),
    )
    if len(optimizer.param_groups) != 3:
        raise ContractError("T05 optimizer must have exactly three groups")
    for group, (name, learning_rate, weight_decay) in zip(
        optimizer.param_groups, expected_groups,
    ):
        if group.get("name") != name:
            raise ContractError("T05 optimizer group order or name differs")
        settings = {
            "lr": learning_rate, "weight_decay": weight_decay,
            "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False,
            "maximize": False, "foreach": None, "capturable": False,
            "differentiable": False, "fused": None,
        }
        for key, expected in settings.items():
            if group.get(key) != expected:
                raise ContractError(f"T05 optimizer setting differs: {name}.{key}")
        for parameter in group["params"]:
            if id(parameter) not in names:
                raise ContractError("optimizer contains a foreign parameter")
            seen.append(id(parameter))
            if name == "backbone_decay" and parameter.ndim <= 1:
                raise ContractError("one-dimensional backbone parameter has decay")
            if name == "backbone_no_decay" and parameter.ndim > 1:
                raise ContractError("multi-dimensional backbone parameter lacks decay")
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(seen) != len(set(seen)) or set(seen) != set(expected):
        raise ContractError("optimizer parameter coverage differs from the model")


def initialize_arm(
    source_state: Mapping[str, torch.Tensor],
    model,
    *,
    seed: int,
    arm: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if seed not in {17, 23} or arm not in {"control", "targeted"}:
        raise ContractError("invalid T05 initialization identity")
    current = model.state_dict()
    if set(source_state) != set(current):
        raise ContractError("source checkpoint tensor names differ from the new model")
    copied = {}
    for name, expected in current.items():
        value = source_state[name]
        if not torch.is_tensor(value) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ContractError(f"source tensor differs in shape or dtype: {name}")
        if not torch.isfinite(value).all():
            raise ContractError(f"source tensor is nonfinite: {name}")
        copied[name] = value.detach().clone()
    model.load_state_dict(copied, strict=True)
    optimizer = build_optimizer(model, backbone_lr=5e-6, head_lr=5e-5)
    _validate_optimizer(model, optimizer)
    scheduler = build_scheduler(optimizer)
    if optimizer.state:
        raise ContractError("fresh optimizer unexpectedly has moments")
    identity = tensor_state_identity(model)
    return optimizer, scheduler, {
        "seed": seed,
        "arm": arm,
        "tensor_state_identity": identity,
        "reset_state": {
            "optimizer_moments": 0,
            "scheduler_completed_updates": 0,
            "source_optimizer_inherited": False,
            "source_rng_inherited": False,
        },
    }


def _seed_update(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def validate_batch(requests: Sequence[Mapping[str, Any]], expected: int = EFFECTIVE_BATCH) -> None:
    if len(requests) != expected:
        raise ContractError(f"T05 update requires exactly {expected} requests")
    for request in requests:
        if not request.get("questions"):
            raise ContractError("T05 update contains an empty request")


def apply_tape_slot(
    request: Mapping[str, Any], slot: Mapping[str, Any],
) -> dict[str, Any]:
    if request["request_uid"] != slot["request_uid"]:
        raise ContractError("tape slot request UID mismatch")
    augmented = copy.deepcopy(dict(request))
    permutations = slot["question_permutations"]
    if len(permutations) != len(augmented["questions"]):
        raise ContractError("tape question permutation count mismatch")
    for question, stored in zip(augmented["questions"], permutations):
        if stored.get("question_id") != question["id"]:
            raise ContractError("tape question permutation identity mismatch")
        permutation = stored.get("new_to_old")
        size = len(question["options"])
        if (
            not isinstance(permutation, list)
            or any(type(index) is not int for index in permutation)
            or sorted(permutation) != list(range(size))
        ):
            raise ContractError("tape question permutation is invalid")
        old_label = question["label"]
        question["options"] = [question["options"][index] for index in permutation]
        question["option_keys"] = [question["option_keys"][index] for index in permutation]
        if question["soft"] is not None:
            question["soft"] = [question["soft"][index] for index in permutation]
        question["label"] = permutation.index(old_label)
    if digest_object(augmented) != slot["augmented_request_sha256"]:
        raise ContractError("runtime augmented request differs from the frozen tape")
    return augmented


def microbatch_coefficients(total: int, sizes: Sequence[int]) -> list[float]:
    if type(total) is not int or total <= 0 or any(type(size) is not int or size <= 0 for size in sizes):
        raise ContractError("microbatch sizes must be positive plain integers")
    if sum(sizes) != total:
        raise ContractError("microbatch sizes do not sum to the effective batch")
    return [size / total for size in sizes]


def run_update(
    *,
    model,
    optimizer,
    scheduler,
    requests: Sequence[Mapping[str, Any]],
    encodings: Sequence[Any],
    model_rng_seed: int,
    clip_grad: Callable[..., Any] = torch.nn.utils.clip_grad_norm_,
) -> dict[str, Any]:
    validate_batch(requests)
    if len(encodings) != EFFECTIVE_BATCH:
        raise ContractError("T05 encoding count differs from the update batch")
    _seed_update(model_rng_seed)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for start in range(0, EFFECTIVE_BATCH, MICROBATCH):
        batch_requests = list(requests[start:start + MICROBATCH])
        outputs = model(list(encodings[start:start + MICROBATCH]))
        loss = request_batch_loss(outputs, batch_requests, 0.5, 0.25)
        if not torch.isfinite(loss):
            raise ContractError("T05 update produced a nonfinite loss")
        (loss * (len(batch_requests) / EFFECTIVE_BATCH)).backward()
        losses.append(float(loss.detach().cpu()))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ContractError(f"T05 gradient is missing or nonfinite: {name}")
    norm = clip_grad(model.parameters(), 1.0)
    if torch.is_tensor(norm) and not torch.isfinite(norm):
        raise ContractError("T05 gradient norm is nonfinite")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "microbatch_losses": losses,
        "mean_request_loss": sum(losses) / len(losses),
        "model_rng_seed": model_rng_seed,
        "optimizer_updates": 1,
        "scheduler_updates": 1,
        "model_forward_calls": 4,
    }


def _memory_bytes() -> int:
    if not torch.backends.mps.is_available():
        return 0
    return int(torch.mps.driver_allocated_memory())


def train_arm(
    *,
    model,
    optimizer,
    scheduler,
    tape: Sequence[Mapping[str, Any]],
    requests_by_uid: Mapping[str, Mapping[str, Any]],
    encode: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
    start_step: int = 0,
    on_commit: Callable[[int, str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if len(tape) != TOTAL_UPDATES or not 0 <= start_step <= TOTAL_UPDATES:
        raise ContractError("T05 train cursor differs from the frozen tape")
    committed = start_step
    with _signal_handlers():
        for update_index in range(start_step, TOTAL_UPDATES):
            if _memory_bytes() > MEMORY_LIMIT_BYTES:
                raise ContractError("T05 MPS allocation exceeds the frozen limit")
            row = tape[update_index]
            requests = [
                apply_tape_slot(requests_by_uid[slot["request_uid"]], slot)
                for slot in row["slots"]
            ]
            encodings = [encode(request, slot) for request, slot in zip(requests, row["slots"])]
            result = run_update(
                model=model, optimizer=optimizer, scheduler=scheduler,
                requests=requests, encodings=encodings,
                model_rng_seed=row["model_rng_seed"],
            )
            committed = update_index + 1
            status = "completed" if committed == TOTAL_UPDATES else "running"
            if _STOP_REQUESTED and committed < TOTAL_UPDATES:
                status = "interrupted"
            if on_commit is not None:
                on_commit(committed, status, result)
            if status == "interrupted":
                break
    return {
        "completed_updates": committed,
        "status": "completed" if committed == TOTAL_UPDATES else "interrupted",
        "committed_prefix_sha256": digest_object([
            row["row_sha256"] for row in tape[:committed]
        ]),
    }
