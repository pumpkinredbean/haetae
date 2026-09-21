"""Model-free freeze and verification for T05 paired replay."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.coverage_v1.replay_v1.contracts import (
    ContractError,
    canonical_bytes,
    digest_object,
    producer_identity,
    strict_json_loads,
    write_exclusive_json,
)
from experiments.coverage_v1.replay_v1.inputs import TARGET_RENDERING
from experiments.coverage_v1.replay_v1.tapes import (
    build_pair_tapes,
    exposure_report,
    validate_pair_alignment,
)
from experiments.shared_state import encode_request

EXPERIMENT_ID = "haetae-t05-paired-replay-v1"
FROZEN_FILES = (
    "protocol.json",
    "input-bindings.json",
    "original-pool.jsonl",
    "target-pool.jsonl",
    "target-rendering.json",
    "exclusion-audit.json",
    "evaluation-inventory.jsonl",
    "evaluation-schedule.jsonl",
    "zero-reference.jsonl",
    "zero-reference-metrics.json",
    "tape-s17-control.jsonl",
    "tape-s17-targeted.jsonl",
    "tape-s23-control.jsonl",
    "tape-s23-targeted.jsonl",
    "tape-alignment.json",
    "exposure.json",
    "lr-table.json",
    "bootstrap-strata.json",
    "bootstrap-seeds.json",
    "encoding-budget.json",
    "historical-source-digests.json",
)
SOURCE_CLOSURE = (
    "experiments/coverage_v1/replay_v1/__init__.py",
    "experiments/coverage_v1/replay_v1/contracts.py",
    "experiments/coverage_v1/replay_v1/inputs.py",
    "experiments/coverage_v1/replay_v1/tapes.py",
    "experiments/coverage_v1/replay_v1/freeze.py",
    "experiments/kev_adapter.py",
    "experiments/shared_state.py",
    "experiments/train_shared.py",
    "src/haetae/checkpoint.py",
    "pyproject.toml",
    "uv.lock",
)


def _descriptor(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    result = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    if path.suffix == ".jsonl":
        result["rows"] = sum(bool(line.strip()) for line in data.splitlines())
    return result


def _write_exclusive_bytes(path: Path, data: bytes) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _descriptor(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    data = b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows)
    return _write_exclusive_bytes(path, data)


def _head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def freeze_source_identity(repository: str | Path) -> dict[str, Any]:
    return producer_identity(repository, SOURCE_CLOSURE)


def _packed_identity(packed) -> dict[str, Any]:
    value = asdict(packed)
    return {
        "input_ids": value["input_ids"],
        "segments": value["segments"],
        "position_ids": value["position_ids"],
        "decision_indices": value["decision_indices"],
        "option_indices": value["option_indices"],
        "labels": value["labels"],
        "state_tokens": value["state_tokens"],
        "retained_state_tokens": value["retained_state_tokens"],
    }


def token_budget_preflight(
    requests: Iterable[Mapping[str, Any]],
    tokenizer,
    delimiters: Mapping[str, int],
    *,
    maximum_length: int = 2048,
) -> dict[str, Any]:
    identities = []
    for request in requests:
        packed = encode_request(
            tokenizer, request["state"], request["questions"], maximum_length,
            dict(delimiters),
        )
        if packed is None:
            raise ContractError("request candidates exceed the frozen token budget")
        identity = _packed_identity(packed)
        if identity["state_tokens"] != identity["retained_state_tokens"]:
            raise ContractError("request state would be truncated")
        identities.append({
            "request_uid": request["request_uid"],
            "encoding_sha256": digest_object(identity),
            "tokens": len(identity["input_ids"]),
            "segments": max(identity["segments"]) + 1,
        })
    lengths = sorted(row["tokens"] for row in identities)
    return {
        "maximum_length": maximum_length,
        "requests": len(identities),
        "minimum_tokens": lengths[0] if lengths else 0,
        "maximum_tokens": lengths[-1] if lengths else 0,
        "identities": identities,
        "identities_sha256": digest_object(identities),
    }


def _evaluation_schedule(inventory: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[str, list[str]] = {"decision": [], "transfer": [], "korean": [], "secondary": []}
    seen_requests: dict[str, set[str]] = {key: set() for key in by_track}
    for row in inventory:
        track = row["track"]
        if row["request_uid"] not in seen_requests[track]:
            seen_requests[track].add(row["request_uid"])
            by_track[track].append(row["request_uid"])
    expected = {"decision": 1199, "transfer": 764, "korean": 2000, "secondary": 904}
    if {key: len(value) for key, value in by_track.items()} != expected:
        raise ContractError("evaluation request counts differ from the frozen protocol")
    schedule = []
    for track in ("decision", "transfer", "korean", "secondary"):
        requests = by_track[track]
        for offset in range(0, len(requests), 2):
            schedule.append({
                "track": track,
                "batch_index": offset // 2,
                "request_uids": requests[offset:offset + 2],
            })
    if len(schedule) != 2434:
        raise ContractError("evaluation forward schedule is not 2,434 batches")
    return schedule


def _protocol(
    *,
    spec: Mapping[str, Any],
    producer_commit: str,
    code: Mapping[str, Any],
    runtime: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_pre_m4",
        "basis_commit": spec["basis_commit"],
        "producer_commit": producer_commit,
        "producer_code": dict(code),
        "runtime": dict(runtime),
        "input_bindings": dict(input_bindings),
        "source_checkpoint": spec["identity_rules"]["checkpoint_binding"],
        "data_contract": spec["data"],
        "tape_contract": spec["tapes"],
        "training_contract": spec["training"],
        "evaluation_contract": spec["evaluation"],
        "bootstrap_contract": spec["uncertainty"],
        "gate_contract": spec["gates"],
        "checkpoint_contract": spec["checkpointing"],
        "compute_contract": spec["compute_budget"],
        "prohibited": spec["prohibited_actions"],
    }
    protocol["protocol_sha256"] = digest_object(protocol)
    return protocol


def freeze(
    *,
    repository: str | Path,
    out: str | Path,
    spec: Mapping[str, Any],
    runtime: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
    original_pool: list[Mapping[str, Any]],
    target_pool: list[Mapping[str, Any]],
    exclusion_audit: Mapping[str, Any],
    evaluation_inventory: list[Mapping[str, Any]],
    zero_reference: list[Mapping[str, Any]],
    zero_reference_metrics: Mapping[str, Any],
    bootstrap_strata: Mapping[str, Any],
    encoding_budget: Mapping[str, Any],
    historical_source_digests: Mapping[str, Any],
    encoder,
) -> dict[str, Any]:
    repository = Path(repository)
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    try:
        producer_commit = _head(repository)
        code = freeze_source_identity(repository)
        protocol = _protocol(
            spec=spec, producer_commit=producer_commit, code=code,
            runtime=runtime, input_bindings=input_bindings,
        )
        write_exclusive_json(out / "protocol.json", protocol)
        write_exclusive_json(out / "input-bindings.json", dict(input_bindings))
        _write_jsonl(out / "original-pool.jsonl", original_pool)
        _write_jsonl(out / "target-pool.jsonl", target_pool)
        write_exclusive_json(out / "target-rendering.json", TARGET_RENDERING)
        write_exclusive_json(out / "exclusion-audit.json", dict(exclusion_audit))
        _write_jsonl(out / "evaluation-inventory.jsonl", evaluation_inventory)
        _write_jsonl(out / "evaluation-schedule.jsonl", _evaluation_schedule(evaluation_inventory))
        _write_jsonl(out / "zero-reference.jsonl", zero_reference)
        write_exclusive_json(out / "zero-reference-metrics.json", dict(zero_reference_metrics))

        all_tapes = {}
        alignment = {}
        for seed in (17, 23):
            pair = build_pair_tapes(original_pool, target_pool, seed, encoder)
            all_tapes.update({f"s{seed}-{arm}": rows for arm, rows in pair.items()})
            for arm, rows in pair.items():
                _write_jsonl(out / f"tape-s{seed}-{arm}.jsonl", rows)
            alignment[str(seed)] = validate_pair_alignment(
                pair["control"], pair["targeted"], seed,
            )
        write_exclusive_json(out / "tape-alignment.json", alignment)
        requests = {row["request_uid"]: row for row in [*original_pool, *target_pool]}
        exposure = {
            str(seed): exposure_report(
                {arm: all_tapes[f"s{seed}-{arm}"] for arm in ("control", "targeted")},
                requests,
            )
            for seed in (17, 23)
        }
        write_exclusive_json(out / "exposure.json", exposure)
        write_exclusive_json(
            out / "lr-table.json", {"rows": spec["training"]["scheduler"]["lr_table_values"]},
        )
        write_exclusive_json(out / "bootstrap-strata.json", dict(bootstrap_strata))
        write_exclusive_json(
            out / "bootstrap-seeds.json", {"streams": spec["uncertainty"]["stream_seed_table"]},
        )
        write_exclusive_json(out / "encoding-budget.json", dict(encoding_budget))
        write_exclusive_json(
            out / "historical-source-digests.json", dict(historical_source_digests),
        )
        if tuple(path.name for path in sorted(out.iterdir())) != tuple(sorted(FROZEN_FILES)):
            raise ContractError("freeze payload inventory differs from the protocol")
        files = {name: _descriptor(out / name) for name in FROZEN_FILES}
        manifest = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "complete",
            "protocol_sha256": protocol["protocol_sha256"],
            "files": files,
            "code": code,
            "runtime": dict(runtime),
            "population_counts": {
                "original_requests": len(original_pool),
                "original_questions": sum(len(row["questions"]) for row in original_pool),
                "target_requests": len(target_pool),
                "evaluation_rows": len(evaluation_inventory),
                "zero_reference_rows": len(zero_reference),
            },
            "tape_counts": {"seeds": 2, "arms": 4, "rows": 2048, "slots": 16384},
            "pre_m4_counters": {
                "checkpoint_deserializations": 0,
                "model_forwards": 0,
                "mps_operations": 0,
                "optimizer_updates": 0,
                "locked_test_accesses": 0,
            },
        }
        manifest["manifest_sha256"] = digest_object(manifest)
        write_exclusive_json(out / "manifest.json", manifest)
        return manifest
    except Exception:
        if not (out / "manifest.json").exists():
            shutil.rmtree(out, ignore_errors=True)
        raise


def verify_plan(
    plan: str | Path,
    *,
    expected_producer_commit: str | None = None,
) -> dict[str, Any]:
    plan = Path(plan)
    if plan.is_symlink() or not plan.is_dir():
        raise ContractError("freeze root must be a real directory")
    names = sorted(path.name for path in plan.iterdir())
    if names != sorted([*FROZEN_FILES, "manifest.json"]):
        raise ContractError("freeze directory has unexpected files")
    manifest = strict_json_loads((plan / "manifest.json").read_bytes())
    if manifest["manifest_sha256"] != digest_object(manifest, "manifest_sha256"):
        raise ContractError("freeze manifest self-digest mismatch")
    protocol = strict_json_loads((plan / "protocol.json").read_bytes())
    if protocol["protocol_sha256"] != digest_object(protocol, "protocol_sha256"):
        raise ContractError("freeze protocol self-digest mismatch")
    if manifest["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ContractError("manifest protocol binding mismatch")
    if expected_producer_commit is not None and protocol["producer_commit"] != expected_producer_commit:
        raise ContractError("freeze producer commit mismatch")
    for name in FROZEN_FILES:
        if (plan / name).is_symlink() or _descriptor(plan / name) != manifest["files"][name]:
            raise ContractError(f"freeze file descriptor mismatch: {name}")
    for seed in (17, 23):
        tapes = {}
        for arm in ("control", "targeted"):
            data = (plan / f"tape-s{seed}-{arm}.jsonl").read_bytes()
            tapes[arm] = [strict_json_loads(line) for line in data.splitlines() if line]
        validate_pair_alignment(tapes["control"], tapes["targeted"], seed)
    if manifest["pre_m4_counters"] != {
        "checkpoint_deserializations": 0,
        "model_forwards": 0,
        "mps_operations": 0,
        "optimizer_updates": 0,
        "locked_test_accesses": 0,
    }:
        raise ContractError("pre-M4 activity counters are not zero")
    return manifest
