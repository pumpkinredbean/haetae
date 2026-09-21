"""Guarded T05 initialization and matched-tape training engine."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import signal
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup

from experiments.coverage_v1.replay_v1.contracts import (
    ContractError,
    assert_reviewed_identity,
    canonical_bytes,
    digest_object,
    strict_json_loads,
    write_exclusive_json,
)
from experiments.coverage_v1.replay_v1.evaluate import (
    evaluate_endpoint,
    replay_metrics,
    run_initial_parity,
)
from experiments.coverage_v1.replay_v1.freeze import (
    _packed_identity,
    materialize_freeze,
    verify_plan,
)
from experiments.coverage_v1.replay_v1.inputs import _resolve_artifact
from experiments.coverage_v1.replay_v1.state import (
    CHECKPOINT_SCHEDULE,
    MpsOwnerLock,
    ReplayCheckpointStore,
    RunDirectoryLock,
    append_execution_event,
    checkpoint_payload,
    load_current_only,
    make_t05_run_spec,
    optimizer_signature,
    publish_and_pin,
    restore_into_fresh_objects,
    tensor_state_identity,
    validate_t05_checkpoint,
    verify_initial_pair,
)
from experiments.shared_state import SharedStateDecisionModel, encode_request
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


def _load_json(path: str | Path) -> dict[str, Any]:
    value = strict_json_loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_bytes().splitlines():
        if line:
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"expected JSONL objects: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    data = b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "rows": len(rows)}


def _reviewed_plan(args, repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_plan(args.plan, expected_producer_commit=args.reviewed_commit)
    protocol = _load_json(Path(args.plan) / "protocol.json")
    allow = bool(
        getattr(args, "allow_reviewed_training", False)
        or getattr(args, "allow_reviewed_inference", False)
    )
    assert_reviewed_identity(
        repository,
        reviewed_commit=args.reviewed_commit,
        actual_protocol_sha256=protocol["protocol_sha256"],
        reviewed_protocol_sha256=args.reviewed_protocol_sha256,
        actual_manifest_sha256=manifest["manifest_sha256"],
        reviewed_manifest_sha256=args.reviewed_manifest_sha256,
        allow_reviewed_execution=allow,
    )
    return manifest, protocol


def _seed23_guard(args, protocol: Mapping[str, Any]) -> None:
    if args.seed == 17:
        return
    if not getattr(args, "pilot_gate", None):
        raise ContractError("seed 23 requires the seed-17 pilot gate")
    gate = _load_json(args.pilot_gate)
    if gate.get("gate_sha256") != digest_object(gate, "gate_sha256"):
        raise ContractError("seed-23 pilot gate self-digest mismatch")
    if (
        gate.get("seed") != 17
        or gate.get("protocol_sha256") != protocol["protocol_sha256"]
        or gate.get("all_requirements_passed") is not True
        or gate.get("seed23_eligible") is not True
        or not all(row.get("passed") is True for row in gate.get("requirements", []))
        or len(gate.get("requirements", [])) != 6
    ):
        raise ContractError("seed-17 pilot gate does not authorize seed 23")


def _owner_path(material: Mapping[str, Any]) -> Path:
    paths = material["private_material"]["paths"]
    location = paths["mps_owner_lock"]
    if set(location) != {"root", "relative_path"}:
        raise ContractError("MPS owner lock mapping has the wrong fields")
    root = paths["roots"][location["root"]]
    relative = location["relative_path"]
    if not any(
        relative == prefix or relative.startswith(prefix.rstrip("/") + "/")
        for prefix in root["allow_prefixes"]
    ):
        raise ContractError("MPS owner lock is outside its allowlist")
    return Path(root["path"]) / relative


def _ensure_output_path(material: Mapping[str, Any], value: str | Path) -> Path:
    paths = material["private_material"]["paths"]
    root = Path(paths["roots"][paths["output_root"]]["path"]).resolve()
    output = Path(value).absolute()
    if root != output and root not in output.parents:
        raise ContractError("T05 output is outside the configured output root")
    return output


def _construct_model(material: Mapping[str, Any]):
    config_path = material["private_material"]["captured"]["backbone_config.json"].path
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    backbone = AutoModel.from_config(config, attn_implementation="eager")
    tokenizer = material["private_material"]["tokenizer"]
    backbone.resize_token_embeddings(len(tokenizer))
    model = SharedStateDecisionModel(backbone, tokenizer.pad_token_id, pointer_size=128)
    if getattr(model.backbone, "is_gradient_checkpointing", False):
        model.backbone.gradient_checkpointing_disable()
    return model


def _source_checkpoint(material: Mapping[str, Any], model) -> dict[str, Any]:
    paths = material["private_material"]["paths"]
    root, relative = _resolve_artifact(paths, "shared_checkpoint_g79")
    payload = torch.load(Path(root) / relative, map_location="cpu", weights_only=False)
    binding = material["spec"]["identity_rules"]["checkpoint_binding"]
    required = {"run_id", "spec_sha256", "step", "status", "backbone", "head"}
    if not isinstance(payload, dict) or required - set(payload):
        raise ContractError("generation-79 checkpoint is incomplete")
    if (
        payload["run_id"] != binding["run_id"]
        or payload["spec_sha256"] != binding["spec_sha256"]
        or payload["step"] != material["spec"]["identity_rules"]["source_generation_step"]
        or payload["status"] != "completed"
    ):
        raise ContractError("generation-79 checkpoint identity differs")
    model.backbone.load_state_dict(payload["backbone"], strict=True)
    model.head.load_state_dict(payload["head"], strict=True)
    for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all():
            raise ContractError(f"generation-79 model tensor is nonfinite: {name}")
    return payload


def _tape(plan: Path, seed: int, arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = plan / f"tape-s{seed}-{arm}.jsonl"
    rows = _load_jsonl(path)
    manifest = _load_json(plan / "manifest.json")
    return rows, manifest["files"][path.name]


def _prefixes(tape: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        digest_object([row["row_sha256"] for row in tape[:step]])
        for step in range(len(tape) + 1)
    ]


def _training_args(
    seed: int, arm: str, tape: Sequence[Mapping[str, Any]], requests: Mapping[str, Mapping[str, Any]],
) -> SimpleNamespace:
    sources = sorted({
        question["source"]
        for row in tape for slot in row["slots"]
        for question in requests[slot["request_uid"]]["questions"]
    })
    return SimpleNamespace(
        backbone="jhu-clsp/mmBERT-small", sources=",".join(sources),
        per_source=0, eval_per_source=0, steps=512, batch=8, microbatch=2,
        lr=5e-6, head_lr=5e-5, brier_w=0.5, max_len=2048, seed=seed,
        arm=arm,
    )


def _run_spec(
    *, model, optimizer, material, args, plan: Path, tape_descriptor, owner,
) -> dict[str, Any]:
    plan_manifest = _load_json(plan / "manifest.json")
    protocol = _load_json(plan / "protocol.json")
    identity = {
        "version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "plan_manifest_sha256": plan_manifest["manifest_sha256"],
        "initial_source_checkpoint": material["input_bindings"]["shared_checkpoint_g79"],
        "initial_source_run_id": material["spec"]["identity_rules"]["checkpoint_binding"]["run_id"],
        "initial_source_spec_sha256": material["spec"]["identity_rules"]["checkpoint_binding"]["spec_sha256"],
        "seed": args.seed, "arm": args.arm, "ordinal_weight": 0.25,
        "optimizer_settings": material["spec"]["training"]["optimizer"],
        "schedule_table_descriptor": plan_manifest["files"]["lr-table.json"],
        "model_rng_policy": material["spec"]["tapes"]["execution_randomness"],
        "maximum_length": 2048, "microbatch": 2,
        "owner_lock_identity": owner.identity,
        "runtime": material["runtime"],
    }
    return make_t05_run_spec(
        args, model, optimizer, material["private_material"]["tokenizer"], "mps",
        tape_descriptor=tape_descriptor,
        evaluation_inventory_descriptor=plan_manifest["files"]["evaluation-inventory.jsonl"],
        t05_identity=identity,
        repository=material["repository"],
        source_closure=protocol["producer_code"],
    )


def _rng_identity() -> dict[str, Any]:
    value = {
        "python": repr(random.getstate()),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        value["mps"] = torch.mps.get_rng_state().tolist()
    return {"sha256": digest_object(value), "source_rng_inherited": False}


def _step_zero_parity(model, material, plan: Path) -> dict[str, Any]:
    inventory = _load_jsonl(plan / "evaluation-inventory.jsonl")
    selected = [
        row for row in inventory
        if row["track"] == "transfer"
        and (
            "emotion_native" in row["population_ids"]
            or "offensive_native" in row["population_ids"]
            or "augmented_emotion" in row["population_ids"]
        )
    ]
    if len(selected) != 196:
        raise ContractError("step-zero parity inventory is not 196 rows")
    by_uid = {row["request_uid"]: [] for row in selected}
    for row in selected:
        by_uid[row["request_uid"]].append(row)
    uids = list(by_uid)
    schedule = [
        {"track": "transfer", "batch_index": index // 2, "request_uids": uids[index:index + 2]}
        for index in range(0, len(uids), 2)
    ]
    requests = material["private_material"]["evaluation_requests"]
    tokenizer = material["private_material"]["tokenizer"]
    delimiters = material["private_material"]["delimiters"]

    def encode(request):
        packed = encode_request(tokenizer, request["state"], request["questions"], 2048, delimiters)
        if packed is None or packed.retained_state_tokens != packed.state_tokens:
            raise ContractError("step-zero parity encoding differs")
        identity = _packed_identity(packed)
        return packed, digest_object(identity), len(packed.input_ids)

    observations = evaluate_endpoint(
        model=model, requests_by_uid=requests, inventory_by_uid=by_uid,
        schedule=schedule, encode=encode,
    )
    zero = {
        tuple(row["key"]): row for row in _load_jsonl(plan / "zero-reference.jsonl")
    }
    references = [zero[tuple(row["key"])] for row in observations]
    return run_initial_parity(observations, references)


def _initialize_cli(args, repository: Path) -> None:
    _manifest, protocol = _reviewed_plan(args, repository)
    _seed23_guard(args, protocol)
    material = materialize_freeze(
        repository=repository, paths_file=args.paths,
        registry_file=repository / "research/evidence/index.json",
    )
    output = _ensure_output_path(material, args.run)
    checkpoint_size = material["input_bindings"]["shared_checkpoint_g79"]["size_bytes"]
    if shutil.disk_usage(output.parent).free < 32 * checkpoint_size:
        raise ContractError("T05 run volume lacks the frozen checkpoint reserve")
    tape, tape_descriptor = _tape(Path(args.plan), args.seed, args.arm)
    requests = {
        row["request_uid"]: row
        for row in [*material["original_pool"], *material["target_pool"]]
    }
    config = _training_args(args.seed, args.arm, tape, requests)
    initialization_seed = int(digest_object(["t05-initialization-v1", args.seed])[:16], 16) % (2**63 - 1)
    random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    owner_path = _owner_path(material)
    with MpsOwnerLock(owner_path, protocol["protocol_sha256"]) as owner:
        if not torch.backends.mps.is_available():
            raise ContractError("reviewed T05 initialization requires MPS")
        torch.mps.manual_seed(initialization_seed)
        model = _construct_model(material)
        _source_checkpoint(material, model)
        source_identity = tensor_state_identity(model)
        model.to("mps")
        optimizer, scheduler, reset = initialize_arm(
            {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            model, seed=args.seed, arm=args.arm,
        )
        spec = _run_spec(
            model=model, optimizer=optimizer, material=material, args=config,
            plan=Path(args.plan), tape_descriptor=tape_descriptor, owner=owner,
        )
        with RunDirectoryLock(output):
            store = ReplayCheckpointStore(output)
            run = store.start(spec, protocol["producer_code"])
            prefixes = _prefixes(tape)
            payload = checkpoint_payload(
                run, config, model, optimizer, scheduler, step=0, skipped=0,
                data_state={"epoch": 0, "order": list(range(512)), "cursor": 0},
                status="running", producer=protocol["producer_code"],
            )
            payload["replay_state"] = {
                "protocol_sha256": protocol["protocol_sha256"],
                "tape_sha256": tape_descriptor["sha256"], "seed": args.seed,
                "arm": args.arm, "next_update": 0,
                "committed_tape_prefix_sha256": prefixes[0],
                "committed_model_forward_calls": 0,
                "committed_optimizer_updates": 0,
            }
            validator = lambda value: validate_t05_checkpoint(
                value, run, config, model, optimizer, scheduler, device="mps",
                protocol_sha256=protocol["protocol_sha256"],
                tape_sha256=tape_descriptor["sha256"], seed=args.seed,
                arm=args.arm, tape_prefixes=prefixes,
            )
            descriptor, pin = publish_and_pin(store, payload, "running", validator)
            model.eval()
            parity = _step_zero_parity(model, material, Path(args.plan))
            if not parity["passed"]:
                raise ContractError("step-zero generation-79 parity failed")
            receipt = {
                "schema_version": 1, "status": "complete",
                "protocol_sha256": protocol["protocol_sha256"],
                "seed": args.seed, "arm": args.arm, "run_id": run["run_id"],
                "source_checkpoint": material["input_bindings"]["shared_checkpoint_g79"],
                "step_zero_checkpoint": {**descriptor, "pin": pin},
                "tensor_state_identity": source_identity,
                "optimizer_signature": optimizer_signature(model, optimizer),
                "reset_state": reset["reset_state"], "rng_identity": _rng_identity(),
                "parity_report": parity, "runtime": material["runtime"],
                "code": protocol["producer_code"],
            }
            receipt["initialization_sha256"] = digest_object(receipt)
            write_exclusive_json(output / "initialization.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


def _verify_initial_pair_cli(args) -> None:
    plan = Path(args.plan)
    protocol = _load_json(plan / "protocol.json")
    receipts = {}
    for arm, directory in (("control", args.control_run), ("targeted", args.targeted_run)):
        receipt = _load_json(Path(directory) / "initialization.json")
        if receipt["initialization_sha256"] != digest_object(receipt, "initialization_sha256"):
            raise ContractError("initialization receipt self-digest mismatch")
        latest = _load_json(Path(directory) / "latest.json")
        if latest["current"]["step"] != 0 or latest["current"]["status"] != "running":
            raise ContractError("initial pair checkpoint is not step zero")
        if receipt["step_zero_checkpoint"]["sha256"] != latest["current"]["sha256"]:
            raise ContractError("initial receipt checkpoint binding mismatch")
        receipts[arm] = receipt
    result = verify_initial_pair(
        receipts["control"], receipts["targeted"],
        protocol_sha256=protocol["protocol_sha256"], seed=args.seed,
    )
    write_exclusive_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _load_runtime_run(args, repository: Path, material, manifest, protocol, owner):
    plan = Path(args.plan)
    tape, tape_descriptor = _tape(plan, args.seed, args.arm)
    requests = {
        row["request_uid"]: row
        for row in [*material["original_pool"], *material["target_pool"]]
    }
    config = _training_args(args.seed, args.arm, tape, requests)
    model = _construct_model(material).to("mps")
    optimizer = build_optimizer(model, 5e-6, 5e-5)
    scheduler = build_scheduler(optimizer)
    expected_spec = _run_spec(
        model=model, optimizer=optimizer, material=material, args=config,
        plan=plan, tape_descriptor=tape_descriptor, owner=owner,
    )
    store = ReplayCheckpointStore(args.run)
    run = store.open(expected_spec)
    prefixes = _prefixes(tape)
    validator = lambda value: validate_t05_checkpoint(
        value, run, config, model, optimizer, scheduler, device="mps",
        protocol_sha256=protocol["protocol_sha256"],
        tape_sha256=tape_descriptor["sha256"], seed=args.seed, arm=args.arm,
        tape_prefixes=prefixes,
    )
    payload, descriptor = load_current_only(store, validator)
    restore_into_fresh_objects(payload, model, optimizer, scheduler, "mps")
    return model, optimizer, scheduler, store, run, payload, descriptor, tape, requests, config, prefixes, tape_descriptor


def _train_cli(args, repository: Path) -> None:
    manifest, protocol = _reviewed_plan(args, repository)
    _seed23_guard(args, protocol)
    pair = _load_json(args.initial_pair)
    if (
        pair.get("pair_initialization_sha256") != digest_object(pair, "pair_initialization_sha256")
        or pair.get("protocol_sha256") != protocol["protocol_sha256"]
        or pair.get("seed") != args.seed
        or not all(pair.get(field) is True for field in (
            "all_tensors_equal", "all_resets_valid", "all_parity_passed",
        ))
    ):
        raise ContractError("initial pair receipt does not authorize training")
    material = materialize_freeze(
        repository=repository, paths_file=args.paths,
        registry_file=repository / "research/evidence/index.json",
    )
    owner_path = _owner_path(material)
    run_dir = _ensure_output_path(material, args.run)
    with MpsOwnerLock(owner_path, protocol["protocol_sha256"]) as owner, RunDirectoryLock(run_dir):
        if not torch.backends.mps.is_available():
            raise ContractError("reviewed T05 training requires MPS")
        (
            model, optimizer, scheduler, store, run, payload, _descriptor, tape,
            requests, config, prefixes, tape_descriptor,
        ) = _load_runtime_run(args, repository, material, manifest, protocol, owner)
        if payload["status"] == "completed":
            raise ContractError("completed T05 arm cannot train again")
        if payload["step"] > 0 and not args.resume_current:
            raise ContractError("continuing an interrupted arm requires --resume-current")
        tokenizer = material["private_material"]["tokenizer"]
        delimiters = material["private_material"]["delimiters"]

        def encode(request, slot):
            packed = encode_request(tokenizer, request["state"], request["questions"], 2048, delimiters)
            if packed is None or packed.retained_state_tokens != packed.state_tokens:
                raise ContractError("runtime training occurrence truncates")
            identity = _packed_identity(packed)
            if digest_object(identity) != slot["encoding_sha256"] or len(packed.input_ids) != slot["tokens"]:
                raise ContractError("runtime training encoding differs from the frozen tape")
            return packed

        receipt_path = run_dir / "execution-events.jsonl"

        def on_commit(step, status, update_result):
            if status == "running" and step not in CHECKPOINT_SCHEDULE:
                return
            checkpoint = checkpoint_payload(
                run, config, model, optimizer, scheduler, step=step, skipped=0,
                data_state={"epoch": 0, "order": list(range(512)), "cursor": step},
                status=status, producer=protocol["producer_code"],
            )
            checkpoint["replay_state"] = {
                "protocol_sha256": protocol["protocol_sha256"],
                "tape_sha256": tape_descriptor["sha256"], "seed": args.seed,
                "arm": args.arm, "next_update": step,
                "committed_tape_prefix_sha256": prefixes[step],
                "committed_model_forward_calls": step * 4,
                "committed_optimizer_updates": step,
            }
            validator = lambda value: validate_t05_checkpoint(
                value, run, config, model, optimizer, scheduler, device="mps",
                protocol_sha256=protocol["protocol_sha256"],
                tape_sha256=tape_descriptor["sha256"], seed=args.seed,
                arm=args.arm, tape_prefixes=prefixes,
            )
            published, pin = publish_and_pin(store, checkpoint, status, validator)
            append_execution_event(receipt_path, {
                "schema_version": 1, "status": status, "step": step,
                "checkpoint": published, "pin": pin, "update": update_result,
            })

        result = train_arm(
            model=model, optimizer=optimizer, scheduler=scheduler, tape=tape,
            requests_by_uid=requests, encode=encode, start_step=payload["step"],
            on_commit=on_commit,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _report_directory(value: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(value)
    report = _load_json(root / "report.json")
    if report["report_sha256"] != digest_object(report, "report_sha256"):
        raise ContractError("prediction report self-digest mismatch")
    observations = _load_jsonl(root / "observations.jsonl")
    data = (root / "observations.jsonl").read_bytes()
    descriptor = report["observations"]
    if (
        descriptor["sha256"] != hashlib.sha256(data).hexdigest()
        or descriptor["size_bytes"] != len(data)
        or descriptor["rows"] != len(observations)
    ):
        raise ContractError("prediction observation descriptor mismatch")
    return report, observations


def _evaluate_cli(args, repository: Path) -> None:
    manifest, protocol = _reviewed_plan(args, repository)
    _seed23_guard(args, protocol)
    binding = _load_json(args.pair_terminal_bindings)
    if set(binding.get("runs", {})) != {"control", "targeted"}:
        raise ContractError("pair terminal binding must name both arms")
    for arm, item in binding["runs"].items():
        latest = _load_json(Path(item["run"]) / "latest.json")
        if latest["status"] != "completed" or latest["current"]["step"] != 512:
            raise ContractError(f"{arm} arm is not terminal")
        if latest["current"]["sha256"] != item["checkpoint_sha256"]:
            raise ContractError("pair terminal checkpoint binding mismatch")
    material = materialize_freeze(
        repository=repository, paths_file=args.paths,
        registry_file=repository / "research/evidence/index.json",
    )
    output = _ensure_output_path(material, args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    owner_path = _owner_path(material)
    run_dir = _ensure_output_path(material, args.run)
    with MpsOwnerLock(owner_path, protocol["protocol_sha256"]) as owner, RunDirectoryLock(run_dir):
        (
            model, optimizer, scheduler, store, run, payload, descriptor, tape,
            requests, config, prefixes, tape_descriptor,
        ) = _load_runtime_run(args, repository, material, manifest, protocol, owner)
        del optimizer, scheduler, store, tape, requests, config, prefixes, tape_descriptor
        if payload["status"] != "completed" or payload["step"] != 512:
            raise ContractError("evaluation requires the terminal checkpoint")
        inventory = _load_jsonl(Path(args.plan) / "evaluation-inventory.jsonl")
        by_uid = {}
        for row in inventory:
            by_uid.setdefault(row["request_uid"], []).append(row)
        schedule = _load_jsonl(Path(args.plan) / "evaluation-schedule.jsonl")
        tokenizer = material["private_material"]["tokenizer"]
        delimiters = material["private_material"]["delimiters"]

        def encode(request):
            packed = encode_request(tokenizer, request["state"], request["questions"], 2048, delimiters)
            if packed is None or packed.retained_state_tokens != packed.state_tokens:
                raise ContractError("endpoint evaluation truncates")
            identity = _packed_identity(packed)
            return packed, digest_object(identity), len(packed.input_ids)

        model.eval()
        observations = evaluate_endpoint(
            model=model,
            requests_by_uid=material["private_material"]["evaluation_requests"],
            inventory_by_uid=by_uid, schedule=schedule, encode=encode,
        )
    observation_descriptor = _write_jsonl(output / "observations.jsonl", observations)
    report = {
        "schema_version": 1, "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"], "seed": args.seed,
        "arm": args.arm, "run_id": run["run_id"], "checkpoint": descriptor,
        "evaluation_inventory": manifest["files"]["evaluation-inventory.jsonl"],
        "observations": observation_descriptor, "metrics": replay_metrics(observations),
        "temperature_contract": {"raw": 1.0, "fixed": 1.5197255188671874, "fitted": False},
        "counters": {
            "prediction_vectors": len(observations), "requests_encoded": 4867,
            "model_forward_calls": 2434, "optimizer_updates": 0,
        },
        "runtime": material["runtime"], "code": protocol["producer_code"],
    }
    report["report_sha256"] = digest_object(report)
    write_exclusive_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _analyze_pair(plan: Path, seed: int, control_path: str, targeted_path: str):
    from experiments.coverage_v1.replay_v1.analysis import (
        bootstrap_parent_differences,
        compute_pilot_gate,
        per_question_metrics,
    )

    protocol = _load_json(plan / "protocol.json")
    control_report, control = _report_directory(control_path)
    targeted_report, targeted = _report_directory(targeted_path)
    if [row["key"] for row in control] != [row["key"] for row in targeted]:
        raise ContractError("paired endpoint memberships differ")
    control_metrics = replay_metrics(control)
    targeted_metrics = replay_metrics(targeted)
    bootstrap = {}
    streams = {
        row["population"]: row for row in protocol["bootstrap_contract"]["stream_seed_table"]
        if row["experiment_seed"] == seed
    }
    for population, stream in streams.items():
        control_rows = [row for row in control if population in row["population_ids"]]
        targeted_rows = [row for row in targeted if population in row["population_ids"]]
        requirement = next(
            row for row in protocol["gate_contract"]["requirements"]
            if row["population"] == population
        )
        temperature = requirement.get("temperature", 1.0)
        metric = requirement["metric"]
        contributions = []
        for left, right in zip(control_rows, targeted_rows):
            left_metric = per_question_metrics(left, temperature)
            right_metric = per_question_metrics(right, temperature)
            value_name = "correct" if metric == "accuracy" else metric
            contributions.append({
                "origin_source": left["origin_source"],
                "parent_id": left["parent_id"], "task_source": left["task_source"],
                "control": left_metric[value_name], "targeted": right_metric[value_name],
            })
        bootstrap[population] = bootstrap_parent_differences(
            contributions, metric=metric, rng_seed=stream["rng_seed"],
            aggregation=requirement["aggregation"],
        )
    gate = compute_pilot_gate(
        seed=seed, protocol_sha256=protocol["protocol_sha256"],
        producer_code=protocol["producer_code"], control_metrics=control_metrics,
        targeted_metrics=targeted_metrics,
        requirements=protocol["gate_contract"]["requirements"],
        arm_checkpoints={
            "control": control_report["checkpoint"],
            "targeted": targeted_report["checkpoint"],
        },
        arm_reports={
            "control": control_report["report_sha256"],
            "targeted": targeted_report["report_sha256"],
        },
    )
    result = {
        "schema_version": 1, "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"], "seed": seed,
        "zero_reference": _load_json(plan / "zero-reference-metrics.json"),
        "arm_reports": {"control": control_report, "targeted": targeted_report},
        "paired_memberships": {"rows": len(control), "keys_sha256": digest_object([row["key"] for row in control])},
        "point_metrics": {"control": control_metrics, "targeted": targeted_metrics},
        "contrasts": {
            row["id"]: next(item for item in gate["requirements"] if item["id"] == row["id"])["delta"]
            for row in protocol["gate_contract"]["requirements"]
        },
        "bootstrap": bootstrap, "gate": gate,
        "limitations": [
            "Public-development exploratory result only.",
            "No checkpoint promotion or release decision is implied.",
        ],
    }
    result["result_sha256"] = digest_object(result)
    return result, gate


def _analyze_cli(args) -> None:
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result, gate = _analyze_pair(
        Path(args.plan), args.seed, args.control_report, args.targeted_report,
    )
    write_exclusive_json(output / "paired-result.json", result)
    write_exclusive_json(output / "pilot-gate.json", gate)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _verify_result_cli(args) -> None:
    from experiments.coverage_v1.replay_v1.analysis import verify_result

    actual = _load_json(args.result)
    expected, _ = _analyze_pair(
        Path(args.plan), args.seed, args.control_report, args.targeted_report,
    )
    verify_result(actual, expected)
    print(json.dumps({"status": "verified", "result_sha256": actual["result_sha256"]}))


def dispatch_reviewed_command(args, repository: Path) -> None:
    if args.command == "initialize":
        _initialize_cli(args, repository)
    elif args.command == "verify-initial-pair":
        _verify_initial_pair_cli(args)
    elif args.command == "train":
        _train_cli(args, repository)
    elif args.command == "evaluate":
        _evaluate_cli(args, repository)
    elif args.command == "analyze":
        _analyze_cli(args)
    elif args.command == "verify-result":
        _verify_result_cli(args)
    else:
        raise ContractError(f"unsupported reviewed command: {args.command}")
