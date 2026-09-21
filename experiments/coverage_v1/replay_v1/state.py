"""T05 checkpoint, ownership and initialization state wrappers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from experiments.coverage_v1.replay_v1.contracts import (
    ContractError,
    canonical_bytes,
    digest_object,
)
from haetae.checkpoint import (
    CheckpointError,
    CheckpointStore,
    RunDirectoryLock,
    checkpoint_payload,
    make_run_spec,
    optimizer_signature,
    restore_checkpoint,
    validate_checkpoint,
)

CHECKPOINT_SCHEDULE = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 512)


class MpsOwnerLock:
    """Hold the single project-wide accelerator ownership lock."""

    def __init__(self, path: str | Path, protocol_sha256: str):
        self.path = Path(path)
        self.protocol_sha256 = protocol_sha256
        self.handle = None
        self.owner_id = str(uuid.uuid4())

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "path_sha256": hashlib.sha256(str(self.path).encode("utf-8")).hexdigest(),
            "protocol_sha256": self.protocol_sha256,
        }

    def __enter__(self):
        if self.path.is_symlink():
            raise ContractError("MPS owner lock may not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise CheckpointError("another process owns the Haetae MPS lock") from error
        self.handle.seek(0)
        self.handle.truncate()
        json.dump({
            "schema_version": 1,
            "owner_id": self.owner_id,
            "pid": os.getpid(),
            "protocol_sha256": self.protocol_sha256,
            "acquired_at_unix": time.time(),
        }, self.handle, ensure_ascii=False, sort_keys=True)
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class ReplayCheckpointStore(CheckpointStore):
    """Existing format-3 store with current-only T05 recovery semantics."""

    def load_current(self, validator):
        manifest = self._read_manifest()
        descriptor = manifest.get("current")
        if descriptor is None:
            raise CheckpointError("T05 manifest has no current checkpoint")
        payload = self._load_descriptor(descriptor)
        validator(payload)
        self.active_descriptor = descriptor
        return payload, descriptor


def make_t05_run_spec(
    args,
    model,
    optimizer,
    tokenizer,
    device: str,
    *,
    tape_descriptor: dict[str, Any],
    evaluation_inventory_descriptor: dict[str, Any],
    t05_identity: dict[str, Any],
    repository: str | Path,
    source_closure: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "version", "protocol_sha256", "plan_manifest_sha256",
        "initial_source_checkpoint", "initial_source_run_id",
        "initial_source_spec_sha256", "seed", "arm", "ordinal_weight",
        "optimizer_settings", "schedule_table_descriptor", "model_rng_policy",
        "maximum_length", "microbatch", "owner_lock_identity", "runtime",
    }
    missing = required - set(t05_identity)
    if missing:
        raise ContractError(f"T05 run identity is missing fields: {sorted(missing)}")
    spec = make_run_spec(
        args, model, optimizer, tokenizer, device,
        train_fingerprint=tape_descriptor["sha256"],
        validation_fingerprint=evaluation_inventory_descriptor["sha256"],
        repository=repository,
    )
    identity = dict(t05_identity)
    identity["tape_descriptor"] = dict(tape_descriptor)
    identity["evaluation_inventory_descriptor"] = dict(evaluation_inventory_descriptor)
    identity["source_closure"] = dict(source_closure)
    spec["training_code"] = dict(source_closure)
    spec["t05"] = identity
    return spec


def _plain_int(value: Any) -> bool:
    return type(value) is int


def validate_t05_checkpoint(
    payload: dict[str, Any],
    run: dict[str, Any],
    args,
    model,
    optimizer,
    scheduler,
    *,
    device: str,
    protocol_sha256: str,
    tape_sha256: str,
    seed: int,
    arm: str,
    tape_prefixes: list[str],
) -> None:
    validate_checkpoint(payload, run, args, model, optimizer, scheduler, 512, device)
    if set(payload) != {
        "version", "run_id", "spec_sha256", "producer", "status", "head",
        "backbone", "optimizer", "scheduler", "step", "skipped", "config",
        "rng", "data_state", "replay_state",
    }:
        raise CheckpointError("T05 checkpoint has unexpected or missing fields")
    replay = payload["replay_state"]
    required = {
        "protocol_sha256", "tape_sha256", "seed", "arm", "next_update",
        "committed_tape_prefix_sha256", "committed_model_forward_calls",
        "committed_optimizer_updates",
    }
    if not isinstance(replay, dict) or set(replay) != required:
        raise CheckpointError("T05 replay state has the wrong fields")
    step = payload["step"]
    expected = {
        "protocol_sha256": protocol_sha256,
        "tape_sha256": tape_sha256,
        "seed": seed,
        "arm": arm,
        "next_update": step,
        "committed_tape_prefix_sha256": tape_prefixes[step],
        "committed_model_forward_calls": step * 4,
        "committed_optimizer_updates": step,
    }
    if replay != expected:
        raise CheckpointError("T05 replay state differs from the committed tape prefix")
    data = payload["data_state"]
    if data != {"epoch": 0, "order": list(range(512)), "cursor": step}:
        raise CheckpointError("T05 checkpoint data cursor is not the update boundary")
    if payload["skipped"] != 0:
        raise CheckpointError("T05 checkpoint may not skip an update")
    if payload["status"] == "completed" and step != 512:
        raise CheckpointError("T05 completed checkpoint is not terminal")
    if payload["status"] == "interrupted" and not step < 512:
        raise CheckpointError("T05 interrupted checkpoint is terminal")


def _pin_descriptor(run_dir: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    pins = run_dir / "pins"
    pins.mkdir(exist_ok=True)
    source = run_dir / descriptor["path"]
    destination = pins / descriptor["path"]
    if destination.exists() or destination.is_symlink():
        if destination.stat().st_ino != source.stat().st_ino:
            raise CheckpointError("checkpoint pin exists with a different inode")
    else:
        os.link(source, destination)
    pin = {
        "generation": descriptor["generation"],
        "step": descriptor["step"],
        "path": f"pins/{descriptor['path']}",
        "sha256": descriptor["sha256"],
        "size": descriptor["size"],
    }
    pin_path = pins / f"generation-{descriptor['generation']:08d}.json"
    data = canonical_bytes(pin) + b"\n"
    if pin_path.exists():
        if pin_path.read_bytes() != data:
            raise CheckpointError("checkpoint pin descriptor mismatch")
    else:
        with pin_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return pin


def publish_and_pin(
    store: ReplayCheckpointStore,
    payload: dict[str, Any],
    status: str,
    validator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = store.publish(payload, status, validator)
    pin = _pin_descriptor(store.output, descriptor)
    return descriptor, pin


def load_current_only(store: ReplayCheckpointStore, validator):
    return store.load_current(validator)


def restore_into_fresh_objects(
    payload: dict[str, Any],
    model,
    optimizer,
    scheduler,
    device: str,
):
    return restore_checkpoint(payload, model, optimizer, scheduler, device)


def tensor_state_identity(model) -> dict[str, Any]:
    tensors = []
    for name, tensor in model.state_dict().items():
        if not torch.is_tensor(tensor):
            raise ContractError(f"model state {name} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ContractError(f"model state {name} is nonfinite")
        raw = value.view(torch.uint8).numpy().tobytes()
        tensors.append({
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return {"tensors": tensors, "sha256": digest_object(tensors)}


def verify_initial_pair(
    control: dict[str, Any], targeted: dict[str, Any], *, protocol_sha256: str, seed: int,
) -> dict[str, Any]:
    for arm, receipt in (("control", control), ("targeted", targeted)):
        if receipt.get("protocol_sha256") != protocol_sha256:
            raise ContractError("initialization protocol mismatch")
        if receipt.get("seed") != seed or receipt.get("arm") != arm:
            raise ContractError("initialization arm identity mismatch")
        if receipt.get("reset_state") != {
            "optimizer_moments": 0, "scheduler_completed_updates": 0,
            "source_optimizer_inherited": False, "source_rng_inherited": False,
        }:
            raise ContractError("initialization state was not reset")
        if receipt.get("parity_report", {}).get("passed") is not True:
            raise ContractError("step-zero output parity did not pass")
    equal = control["tensor_state_identity"] == targeted["tensor_state_identity"]
    if not equal:
        raise ContractError("step-zero arm tensors differ")
    result = {
        "schema_version": 1,
        "status": "passed",
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "arms": {
            "control": control["run_id"], "targeted": targeted["run_id"],
        },
        "all_tensors_equal": True,
        "all_resets_valid": True,
        "all_parity_passed": True,
    }
    result["pair_initialization_sha256"] = digest_object(result)
    return result


def append_execution_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ContractError("execution event log may not be a symbolic link")
    record = dict(event)
    record["event_sha256"] = digest_object(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


__all__ = [
    "CHECKPOINT_SCHEDULE",
    "CheckpointStore",
    "RunDirectoryLock",
    "checkpoint_payload",
    "optimizer_signature",
]
