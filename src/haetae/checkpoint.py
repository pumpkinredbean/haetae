"""Durable run identity and checkpoint publication."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import re
import tempfile
import time
import uuid
import warnings
from importlib import metadata
from pathlib import Path

import torch


CHECKPOINT_VERSION = 3
MANIFEST_VERSION = 1
RUN_VERSION = 1
TRAINING_CONFIG_FIELDS = (
    "backbone", "sources", "per_source", "eval_per_source", "steps",
    "batch", "lr", "head_lr", "brier_w", "max_len", "seed",
)
RUN_STATUSES = {"running", "interrupted", "completed", "failed_no_progress"}
UNKNOWN_PRODUCER = hashlib.sha256(b"unknown producer").hexdigest()


class CheckpointError(RuntimeError):
    pass


class RecoverableCheckpointError(CheckpointError):
    """Physical artifact failure that permits rollback to a published predecessor."""


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    if platform.system() == "Darwin" and hasattr(fcntl, "F_FULLFSYNC"):
        fcntl.fcntl(handle.fileno(), fcntl.F_FULLFSYNC)


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            _sync_file(handle)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


class RunDirectoryLock:
    """Hold one nonblocking writer lock for a run directory."""

    def __init__(self, output: str | Path):
        self.output = Path(output)
        self.handle = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        lock_path = self.output / ".train.lock"
        self.handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise CheckpointError(
                f"another training process holds {lock_path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        json.dump({"pid": os.getpid(), "acquired_at_unix": time.time()}, self.handle)
        self.handle.write("\n")
        _sync_file(self.handle)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def _source_identity(directory: Path, files: list[Path]) -> dict:
    digest = hashlib.sha256()
    relative_paths = []
    for file_path in files:
        relative = str(file_path.relative_to(directory))
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": relative_paths}


def training_code_identity(directory: str | Path) -> dict:
    directory = Path(directory)
    files = [directory / "src" / "haetae" / name
             for name in ("data.py", "model.py", "train.py")]
    return _source_identity(directory, files)


def producer_identity(directory: str | Path) -> dict:
    directory = Path(directory)
    files = sorted((directory / "src" / "haetae").glob("*.py"))
    files.append(directory / "pyproject.toml")
    return _source_identity(directory, files)


def optimizer_signature(model, optimizer) -> list[dict]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    signature = []
    seen = set()
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = []
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise CheckpointError("optimizer contains a duplicate parameter")
            seen.add(id(parameter))
            name = names.get(id(parameter))
            if name is None:
                raise CheckpointError("optimizer contains a parameter outside the model")
            parameters.append({
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            })
        signature.append({
            "group": group.get("name", str(group_index)),
            "parameters": parameters,
        })
    return signature


def tokenizer_fingerprint(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        tokenization_definition = json.loads(backend.to_str())
    else:
        tokenization_definition = {"vocab": tokenizer.get_vocab()}
    payload = {
        "class": tokenizer.__class__.__name__,
        "tokenization_definition": tokenization_definition,
        "special_tokens_map": tokenizer.special_tokens_map,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    return _sha256_bytes(_canonical_bytes(payload))


def model_config_fingerprint(model) -> str:
    return _sha256_bytes(_canonical_bytes(model.backbone.config.to_dict()))


def make_run_spec(args, model, optimizer, tokenizer, device: str,
                  train_fingerprint: str, validation_fingerprint: str,
                  repository: str | Path) -> dict:
    return {
        "config": {key: getattr(args, key) for key in TRAINING_CONFIG_FIELDS},
        "train_fingerprint": train_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "backend": device,
        "optimizer_class": (
            f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
        ),
        "optimizer_signature": optimizer_signature(model, optimizer),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "model_config_fingerprint": model_config_fingerprint(model),
        "training_code": training_code_identity(repository),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
        },
    }


def _require_exact_run_spec(actual: dict, expected: dict) -> None:
    if actual != expected:
        differences = sorted(
            key for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise CheckpointError(f"run identity mismatch in: {differences}")


class CheckpointStore:
    """Publish immutable generations through one authoritative manifest."""

    def __init__(self, output: str | Path):
        self.output = Path(output)
        self.run = None
        self.active_descriptor = None

    @property
    def run_path(self) -> Path:
        return self.output / "run.json"

    @property
    def manifest_path(self) -> Path:
        return self.output / "latest.json"

    def start(self, spec: dict, producer: dict | None = None) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        existing = [path for path in self.output.iterdir()
                    if path.name != ".train.lock"]
        if existing:
            names = sorted(path.name for path in existing)
            raise CheckpointError(
                f"refusing a fresh run in a populated output directory: {names}"
            )
        spec_sha256 = _sha256_bytes(_canonical_bytes(spec))
        self.run = {
            "version": RUN_VERSION,
            "run_id": str(uuid.uuid4()),
            "created_at_unix": time.time(),
            "spec": spec,
            "spec_sha256": spec_sha256,
            "producer": producer or {"sha256": UNKNOWN_PRODUCER, "files": []},
        }
        atomic_json_save(self.run, self.run_path)
        return self.run

    def open(self, expected_spec: dict) -> dict:
        if not self.run_path.exists():
            raise CheckpointError(
                f"auto resume requires {self.run_path}; use a new output directory"
            )
        with self.run_path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        if run.get("version") != RUN_VERSION:
            raise CheckpointError(f"unsupported run version: {run.get('version')}")
        if not isinstance(run.get("run_id"), str) or not run["run_id"]:
            raise CheckpointError("run identity is missing")
        expected_digest = _sha256_bytes(_canonical_bytes(run.get("spec")))
        if run.get("spec_sha256") != expected_digest:
            raise CheckpointError("run identity digest mismatch")
        _require_exact_run_spec(run["spec"], expected_spec)
        self.run = run
        return run

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            raise CheckpointError(
                f"run exists but has no checkpoint manifest: {self.manifest_path}"
            )
        with self.manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("version") != MANIFEST_VERSION:
            raise CheckpointError(
                f"unsupported manifest version: {manifest.get('version')}"
            )
        if self.run is None or manifest.get("run_id") != self.run["run_id"]:
            raise CheckpointError("manifest run identity mismatch")
        if manifest.get("spec_sha256") != self.run["spec_sha256"]:
            raise CheckpointError("manifest run specification mismatch")
        return manifest

    def _load_descriptor(self, descriptor: dict):
        required = {
            "generation", "path", "sha256", "size", "step", "status",
            "checkpoint_version", "producer_sha256",
        }
        if not isinstance(descriptor, dict) or required - set(descriptor):
            raise CheckpointError("checkpoint descriptor is incomplete")
        if (not _is_plain_int(descriptor["generation"])
                or descriptor["generation"] <= 0):
            raise CheckpointError("checkpoint descriptor generation is invalid")
        if not _is_plain_int(descriptor["size"]) or descriptor["size"] < 0:
            raise CheckpointError("checkpoint descriptor size is invalid")
        if not _is_plain_int(descriptor["step"]) or descriptor["step"] < 0:
            raise CheckpointError("checkpoint descriptor step is invalid")
        if descriptor["status"] not in RUN_STATUSES:
            raise CheckpointError("checkpoint descriptor status is invalid")
        if (not _is_plain_int(descriptor["checkpoint_version"])
                or descriptor["checkpoint_version"] != CHECKPOINT_VERSION):
            raise CheckpointError("checkpoint descriptor version mismatch")
        filename = descriptor["path"]
        match = (re.fullmatch(
            r"checkpoint-g([0-9]{8})-s([0-9]{9})-([0-9a-f]{8})\.pt",
            filename,
        ) if isinstance(filename, str) else None)
        if (match is None or Path(filename).name != filename
                or int(match.group(1)) != descriptor["generation"]
                or int(match.group(2)) != descriptor["step"]):
            raise CheckpointError("unsafe checkpoint descriptor path")
        if not _is_sha256(descriptor["sha256"]):
            raise CheckpointError("checkpoint descriptor digest is invalid")
        if not _is_sha256(descriptor["producer_sha256"]):
            raise CheckpointError("checkpoint producer digest is invalid")
        path = self.output / filename
        try:
            handle = path.open("rb")
        except FileNotFoundError as exc:
            raise RecoverableCheckpointError(
                f"checkpoint file is missing: {filename}"
            ) from exc
        with handle:
            if os.fstat(handle.fileno()).st_size != descriptor["size"]:
                raise RecoverableCheckpointError(
                    f"checkpoint size mismatch: {filename}"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != descriptor["sha256"]:
                raise RecoverableCheckpointError(
                    f"checkpoint digest mismatch: {filename}"
                )
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload is not a mapping")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise CheckpointError("checkpoint payload version mismatch")
        if payload.get("run_id") != self.run["run_id"]:
            raise CheckpointError("checkpoint payload run identity mismatch")
        if payload.get("spec_sha256") != self.run["spec_sha256"]:
            raise CheckpointError("checkpoint payload specification mismatch")
        if payload.get("step") != descriptor["step"]:
            raise CheckpointError("checkpoint payload step mismatch")
        if payload.get("status") != descriptor["status"]:
            raise CheckpointError("checkpoint payload status mismatch")
        if payload.get("producer", {}).get("sha256") != descriptor["producer_sha256"]:
            raise CheckpointError("checkpoint producer identity mismatch")
        return payload

    def load(self, validator):
        manifest = self._read_manifest()
        failures = []
        for role in ("current", "previous"):
            descriptor = manifest.get(role)
            if descriptor is None:
                continue
            try:
                payload = self._load_descriptor(descriptor)
                validator(payload)
                self.active_descriptor = descriptor
                if role == "previous":
                    self._record_recovery(
                        manifest["current"], descriptor, failures, payload
                    )
                return payload, role, failures
            except RecoverableCheckpointError as exc:
                failures.append(f"{role}: {exc}")
        raise CheckpointError(
            "no valid checkpoint generation; " + "; ".join(failures)
        )

    def _record_recovery(self, rejected: dict, selected: dict,
                         failures: list[str], payload: dict) -> None:
        path = self.output / "recovery_events.json"
        events = []
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, list):
                events = stored
        events.append({
            "recorded_at_unix": time.time(),
            "rejected_generation": rejected.get("generation"),
            "rejected_step": rejected.get("step"),
            "selected_generation": selected.get("generation"),
            "selected_step": selected.get("step"),
            "lost_updates": max(0, rejected.get("step", 0) - selected.get("step", 0)),
            "reason": failures[-1],
        })
        atomic_json_save(events, path)
        try:
            atomic_json_save({
                "run_id": self.run["run_id"],
                "status": payload["status"],
                "step": payload["step"],
                "steps": self.run["spec"]["config"]["steps"],
                "skipped": payload["skipped"],
                "epoch": payload["data_state"]["epoch"],
                "cursor": payload["data_state"]["cursor"],
                "generation": selected["generation"],
                "checkpoint": selected["path"],
                "sha256": selected["sha256"],
                "recovered_from_generation": rejected.get("generation"),
                "recovery_reason": failures[-1],
                "updated_at_unix": time.time(),
            }, self.output / "progress.json")
        except OSError as exc:
            warnings.warn(
                f"recovery succeeded but progress display update failed: {exc}",
                RuntimeWarning,
            )

    def publish(self, payload: dict, status: str, validator) -> dict:
        if self.run is None:
            raise CheckpointError("run store is not initialized")
        if status not in RUN_STATUSES:
            raise CheckpointError(f"invalid run status: {status}")
        if payload.get("run_id") != self.run["run_id"]:
            raise CheckpointError("refusing to publish a foreign run")
        if not callable(validator):
            raise CheckpointError("checkpoint publication requires a validator")
        old_manifest = self._read_manifest() if self.manifest_path.exists() else None
        if old_manifest is not None and self.active_descriptor is None:
            raise CheckpointError(
                "load and validate the active generation before publishing another"
            )
        prior_generation = 0
        if old_manifest is not None:
            prior_generation = max(
                item.get("generation", 0)
                for item in (old_manifest.get("current"), old_manifest.get("previous"))
                if item is not None
            )
        generation = prior_generation + 1
        filename = (
            f"checkpoint-g{generation:08d}-s{payload['step']:09d}-"
            f"{uuid.uuid4().hex[:8]}.pt"
        )
        final_path = self.output / filename
        fd, temporary = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=self.output
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                _sync_file(handle)
            os.replace(temporary, final_path)
            _sync_directory(self.output)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        descriptor = {
            "generation": generation,
            "path": filename,
            "sha256": file_sha256(final_path),
            "size": final_path.stat().st_size,
            "step": payload["step"],
            "status": status,
            "checkpoint_version": CHECKPOINT_VERSION,
            "producer_sha256": payload["producer"]["sha256"],
        }
        serialized_payload = self._load_descriptor(descriptor)
        validator(serialized_payload)
        manifest = {
            "version": MANIFEST_VERSION,
            "run_id": self.run["run_id"],
            "spec_sha256": self.run["spec_sha256"],
            "status": status,
            "current": descriptor,
            "previous": copy.deepcopy(self.active_descriptor),
            "updated_at_unix": time.time(),
        }
        atomic_json_save(manifest, self.manifest_path)
        self.active_descriptor = descriptor
        try:
            atomic_json_save({
                "run_id": self.run["run_id"],
                "status": status,
                "step": payload["step"],
                "steps": self.run["spec"]["config"]["steps"],
                "skipped": payload["skipped"],
                "epoch": payload["data_state"]["epoch"],
                "cursor": payload["data_state"]["cursor"],
                "generation": generation,
                "checkpoint": filename,
                "sha256": descriptor["sha256"],
                "updated_at_unix": manifest["updated_at_unix"],
            }, self.output / "progress.json")
        except OSError as exc:
            warnings.warn(
                f"checkpoint published but progress display update failed: {exc}",
                RuntimeWarning,
            )
        keep = {
            item["path"] for item in (manifest["current"], manifest["previous"])
            if item is not None
        }
        try:
            removed = False
            for path in self.output.glob("checkpoint-g*.pt"):
                if path.name not in keep:
                    path.unlink()
                    removed = True
            if removed:
                _sync_directory(self.output)
        except OSError as exc:
            warnings.warn(
                f"checkpoint published but old-generation cleanup failed: {exc}",
                RuntimeWarning,
            )
        return descriptor


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_finite_tensor(value: torch.Tensor, label: str) -> None:
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
        raise CheckpointError(f"nonfinite tensor in {label}")


def _validate_model_state(saved: dict, current: dict, label: str) -> None:
    if set(saved) != set(current):
        raise CheckpointError(f"{label} state keys mismatch")
    for name, value in saved.items():
        expected = current[name]
        if not torch.is_tensor(value):
            raise CheckpointError(f"{label}.{name} is not a tensor")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise CheckpointError(f"{label}.{name} shape or dtype mismatch")
        _require_finite_tensor(value, f"{label}.{name}")


def _validate_rng_state(rng: dict, device: str) -> None:
    if not isinstance(rng, dict) or "python" not in rng or "torch" not in rng:
        raise CheckpointError("checkpoint RNG state is incomplete")
    try:
        random.Random().setstate(rng["python"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint Python RNG state is invalid") from exc
    cpu_state = rng["torch"]
    if (not torch.is_tensor(cpu_state) or cpu_state.device.type != "cpu"
            or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1):
        raise CheckpointError("checkpoint Torch RNG state is malformed")
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
    except RuntimeError as exc:
        raise CheckpointError("checkpoint Torch RNG state is invalid") from exc
    if device == "mps":
        mps_state = rng.get("mps")
        original = torch.mps.get_rng_state()
        if (not torch.is_tensor(mps_state)
                or mps_state.device.type != "cpu"
                or mps_state.dtype != original.dtype
                or mps_state.shape != original.shape):
            raise CheckpointError("checkpoint MPS RNG state is malformed")
        try:
            torch.mps.set_rng_state(mps_state)
        except RuntimeError as exc:
            raise CheckpointError("checkpoint MPS RNG state is invalid") from exc
        finally:
            torch.mps.set_rng_state(original)


def _validate_optimizer_state(saved: dict, optimizer, scheduler, step: int) -> None:
    current = optimizer.state_dict()
    saved_groups = saved.get("param_groups")
    saved_state = saved.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise CheckpointError("optimizer state is malformed")
    if len(saved_groups) != len(current["param_groups"]):
        raise CheckpointError("optimizer group count mismatch")
    parameter_by_id = {}
    settings_by_id = {}
    for saved_group, current_group, live_group in zip(
            saved_groups, current["param_groups"], optimizer.param_groups):
        if saved_group.get("params") != current_group["params"]:
            raise CheckpointError("optimizer parameter order mismatch")
        if len(saved_group["params"]) != len(live_group["params"]):
            raise CheckpointError("optimizer live parameter count mismatch")
        for parameter_id, parameter in zip(saved_group["params"], live_group["params"]):
            parameter_by_id[parameter_id] = parameter
            settings_by_id[parameter_id] = saved_group
        for key in ("name", "betas", "eps", "weight_decay", "amsgrad",
                    "maximize", "foreach", "capturable", "differentiable",
                    "fused", "initial_lr"):
            if saved_group.get(key) != current_group.get(key):
                raise CheckpointError(f"optimizer setting mismatch: {key}")
    expected_ids = set(parameter_by_id)
    if step == 0:
        if saved_state:
            raise CheckpointError("step-zero optimizer unexpectedly has moments")
    else:
        if set(saved_state) != expected_ids:
            raise CheckpointError("optimizer moment coverage mismatch")
        for parameter_id, parameter in parameter_by_id.items():
            state = saved_state[parameter_id]
            required = {"step", "exp_avg", "exp_avg_sq"}
            if required - set(state):
                raise CheckpointError("optimizer moment fields are missing")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if (not torch.is_tensor(value) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype):
                    raise CheckpointError(f"optimizer {key} shape or dtype mismatch")
                _require_finite_tensor(value, f"optimizer.{key}")
            if torch.any(state["exp_avg_sq"] < 0):
                raise CheckpointError("optimizer second moment is negative")
            if settings_by_id[parameter_id].get("amsgrad"):
                maximum = state.get("max_exp_avg_sq")
                if (not torch.is_tensor(maximum)
                        or maximum.shape != parameter.shape
                        or maximum.dtype != parameter.dtype):
                    raise CheckpointError("optimizer AMSGrad state is malformed")
                _require_finite_tensor(maximum, "optimizer.max_exp_avg_sq")
                if torch.any(maximum < state["exp_avg_sq"]):
                    raise CheckpointError("optimizer AMSGrad maximum is inconsistent")
            elif "max_exp_avg_sq" in state:
                raise CheckpointError("unexpected optimizer AMSGrad state")
            counter = state["step"]
            if (not torch.is_tensor(counter) or counter.shape != torch.Size([])
                    or not counter.is_floating_point()):
                raise CheckpointError("optimizer step counter is malformed")
            _require_finite_tensor(counter, "optimizer.step")
            if float(counter.item()) != float(step):
                raise CheckpointError("optimizer step counter mismatch")
    saved_scheduler = scheduler.state_dict()
    base_lrs = saved_scheduler["base_lrs"]
    for index, group in enumerate(saved_groups):
        expected_lr = base_lrs[index] * scheduler.lr_lambdas[index](step)
        if not math.isclose(float(group["lr"]), float(expected_lr), rel_tol=1e-9,
                            abs_tol=1e-12):
            raise CheckpointError("optimizer learning rate does not match schedule")


def validate_checkpoint(payload: dict, run: dict, args, model, optimizer,
                        scheduler, train_size: int, device: str) -> None:
    required = {
        "version", "run_id", "spec_sha256", "producer", "status", "head",
        "backbone", "optimizer", "scheduler", "step", "skipped", "config",
        "rng", "data_state",
    }
    if required - set(payload):
        raise CheckpointError(
            f"checkpoint missing keys: {sorted(required - set(payload))}"
        )
    if payload["version"] != CHECKPOINT_VERSION:
        raise CheckpointError("checkpoint version mismatch")
    if payload["run_id"] != run["run_id"]:
        raise CheckpointError("checkpoint run identity mismatch")
    if payload["spec_sha256"] != run["spec_sha256"]:
        raise CheckpointError("checkpoint run specification mismatch")
    if payload["status"] not in RUN_STATUSES:
        raise CheckpointError("checkpoint status is invalid")
    step = payload["step"]
    skipped = payload["skipped"]
    if not _is_plain_int(step) or not 0 <= step <= args.steps:
        raise CheckpointError("checkpoint step is invalid")
    if not _is_plain_int(skipped) or skipped < 0:
        raise CheckpointError("checkpoint skipped count is invalid")
    expected_config = {key: getattr(args, key) for key in TRAINING_CONFIG_FIELDS}
    if payload["config"] != expected_config:
        raise CheckpointError("checkpoint configuration mismatch")
    _validate_model_state(payload["backbone"], model.backbone.state_dict(), "backbone")
    _validate_model_state(payload["head"], model.head.state_dict(), "head")
    _validate_optimizer_state(payload["optimizer"], optimizer, scheduler, step)
    scheduler_state = payload["scheduler"]
    expected_scheduler = scheduler.state_dict()
    if (not isinstance(scheduler_state, dict)
            or set(scheduler_state) != set(expected_scheduler)):
        raise CheckpointError("scheduler state structure mismatch")
    if scheduler_state.get("last_epoch") != step:
        raise CheckpointError("scheduler step mismatch")
    if scheduler_state.get("_step_count") != step + 1:
        raise CheckpointError("scheduler update count mismatch")
    if scheduler_state.get("base_lrs") != expected_scheduler.get("base_lrs"):
        raise CheckpointError("scheduler base learning rates mismatch")
    if scheduler_state.get("lr_lambdas") != expected_scheduler.get("lr_lambdas"):
        raise CheckpointError("scheduler function state mismatch")
    for key in ("_is_initial", "_get_lr_called_within_step"):
        if scheduler_state.get(key) != expected_scheduler.get(key):
            raise CheckpointError(f"scheduler setting mismatch: {key}")
    saved_lrs = scheduler_state.get("_last_lr")
    optimizer_lrs = [group["lr"] for group in payload["optimizer"]["param_groups"]]
    if (not isinstance(saved_lrs, list)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in saved_lrs)
            or saved_lrs != optimizer_lrs):
        raise CheckpointError("scheduler and optimizer learning rates differ")
    data_state = payload["data_state"]
    if not isinstance(data_state, dict):
        raise CheckpointError("checkpoint data state is malformed")
    if not _is_plain_int(data_state.get("epoch")) or data_state["epoch"] < 0:
        raise CheckpointError("checkpoint epoch is invalid")
    order = data_state.get("order")
    if (not isinstance(order, list)
            or any(not _is_plain_int(index) for index in order)
            or sorted(order) != list(range(train_size))):
        raise CheckpointError("checkpoint data order is not a training-row permutation")
    cursor = data_state.get("cursor")
    if not _is_plain_int(cursor) or not 0 <= cursor <= train_size:
        raise CheckpointError("checkpoint data cursor is invalid")
    if run["spec"]["backend"] != device:
        raise CheckpointError("checkpoint backend mismatch")
    _validate_rng_state(payload["rng"], device)


def checkpoint_payload(run: dict, args, model, optimizer, scheduler, step: int,
                       skipped: int, data_state: dict, status: str,
                       producer: dict | None = None) -> dict:
    rng = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        rng["mps"] = torch.mps.get_rng_state()
    return {
        "version": CHECKPOINT_VERSION,
        "run_id": run["run_id"],
        "spec_sha256": run["spec_sha256"],
        "producer": producer or run.get("producer", {
            "sha256": UNKNOWN_PRODUCER, "files": [],
        }),
        "status": status,
        "head": model.head.state_dict(),
        "backbone": model.backbone.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "skipped": skipped,
        "config": {key: getattr(args, key) for key in TRAINING_CONFIG_FIELDS},
        "rng": rng,
        "data_state": data_state,
    }


def restore_checkpoint(payload: dict, model, optimizer, scheduler, device: str):
    model.backbone.load_state_dict(payload["backbone"])
    model.head.load_state_dict(payload["head"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and state[key].device != parameter.device:
                    raise CheckpointError(
                        f"optimizer {key} was restored on {state[key].device}, "
                        f"expected {parameter.device}"
                    )
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch"])
    if device == "mps" and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(payload["rng"]["mps"])
    return payload["step"], payload["skipped"], payload["data_state"]


def load_completed_checkpoint(output: str | Path) -> tuple[dict, dict, dict]:
    store = CheckpointStore(output)
    with store.run_path.open(encoding="utf-8") as handle:
        store.run = json.load(handle)
    if store.run.get("version") != RUN_VERSION:
        raise CheckpointError("completed run metadata has an unsupported version")
    run_spec_digest = _sha256_bytes(_canonical_bytes(store.run.get("spec")))
    if store.run.get("spec_sha256") != run_spec_digest:
        raise CheckpointError("completed run metadata digest mismatch")
    manifest = store._read_manifest()
    if manifest.get("status") != "completed":
        raise CheckpointError(
            f"run is not completed: {manifest.get('status')}"
        )
    payload = store._load_descriptor(manifest["current"])
    if payload.get("status") != "completed":
        raise CheckpointError("completed manifest points to a noncompleted checkpoint")
    run_config = store.run.get("spec", {}).get("config")
    if not isinstance(run_config, dict) or payload.get("config") != run_config:
        raise CheckpointError("completed checkpoint configuration differs from its run")
    target = run_config.get("steps")
    if (not _is_plain_int(target) or target < 0
            or not _is_plain_int(payload.get("step"))
            or payload["step"] != target):
        raise CheckpointError("completed checkpoint has not reached its target")
    for label in ("backbone", "head"):
        state = payload.get(label)
        if not isinstance(state, dict):
            raise CheckpointError(f"completed checkpoint has no {label} state")
        for name, value in state.items():
            if not torch.is_tensor(value):
                raise CheckpointError(f"{label}.{name} is not a tensor")
            _require_finite_tensor(value, f"{label}.{name}")
    return payload, store.run, manifest
