"""Model-free freeze and verification for T05 paired replay."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.coverage_v1.replay_v1.contracts import (
    CapturedFile,
    ContractError,
    canonical_bytes,
    digest_object,
    producer_identity,
    strict_json_loads,
    write_exclusive_json,
)
from experiments.coverage_v1.replay_v1.evaluate import (
    replay_metrics,
    run_initial_parity,
)
from experiments.coverage_v1.replay_v1.inputs import (
    TARGET_RENDERING,
    _resolve_artifact,
    normalize_original_pool,
    validate_target_support,
)
from experiments.coverage_v1.replay_v1.tapes import (
    build_pair_tapes,
    exposure_report,
    validate_pair_alignment,
)
from experiments.shared_state import add_delimiters, encode_request

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
    "experiments/coverage_v1/replay_v1/__main__.py",
    "experiments/coverage_v1/replay_v1/analysis.py",
    "experiments/coverage_v1/replay_v1/contracts.py",
    "experiments/coverage_v1/replay_v1/evaluate.py",
    "experiments/coverage_v1/replay_v1/inputs.py",
    "experiments/coverage_v1/replay_v1/tapes.py",
    "experiments/coverage_v1/replay_v1/freeze.py",
    "experiments/coverage_v1/replay_v1/package.py",
    "experiments/coverage_v1/replay_v1/state.py",
    "experiments/coverage_v1/replay_v1/train.py",
    "experiments/coverage_v1/__init__.py",
    "experiments/__init__.py",
    "experiments/kev_adapter.py",
    "experiments/shared_state.py",
    "experiments/train_shared.py",
    "src/haetae/__init__.py",
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


def _path_map(path: str | Path) -> dict[str, Any]:
    value = strict_json_loads(Path(path).read_bytes())
    required = {"schema_version", "roots", "artifacts", "output_root", "mps_owner_lock"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1:
        raise ContractError("T05 private path map has the wrong schema")
    return value


def _capture_artifact(
    paths: Mapping[str, Any], artifact_id: str, expected: Mapping[str, Any] | None,
) -> CapturedFile:
    root, relative = _resolve_artifact(paths, artifact_id)
    descriptor = None
    if expected is not None and "sha256" in expected and "size_bytes" in expected:
        descriptor = {
            "artifact_id": artifact_id,
            "sha256": expected["sha256"],
            "size_bytes": expected["size_bytes"],
        }
    from experiments.coverage_v1.replay_v1.contracts import capture_allowed_file

    return capture_allowed_file(root, relative, descriptor)


def _stream_artifact_descriptor(
    paths: Mapping[str, Any], artifact_id: str, expected: Mapping[str, Any],
) -> dict[str, Any]:
    root, relative = _resolve_artifact(paths, artifact_id)
    candidate = Path(root).absolute() / relative
    current = Path(root).absolute()
    if current.is_symlink() or not current.is_dir():
        raise ContractError("streaming artifact root is unsafe")
    for part in Path(relative).parts:
        if part in ("", ".", ".."):
            raise ContractError("streaming artifact path is unsafe")
        current /= part
        if current.is_symlink():
            raise ContractError("streaming artifact path contains a symbolic link")
    if not candidate.is_file():
        raise ContractError("streaming artifact is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with candidate.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
    ):
        raise ContractError("streaming artifact changed during hashing")
    if digest.hexdigest() != expected["sha256"] or size != expected["size_bytes"]:
        raise ContractError("streaming artifact descriptor mismatch")
    return {"artifact_id": artifact_id, "sha256": digest.hexdigest(), "size_bytes": size}


def _rows(captured: CapturedFile) -> list[dict[str, Any]]:
    result = []
    for line in captured.data.splitlines():
        if line.strip():
            row = strict_json_loads(line)
            if not isinstance(row, dict):
                raise ContractError("bound JSONL contains a non-object row")
            result.append(row)
    return result


def _self_digest(value: Mapping[str, Any], expected: str) -> str:
    fields = (
        "manifest_sha256", "result_sha256", "protocol_sha256", "artifact_sha256",
        "plan_sha256", "summary_sha256",
    )
    matches = [field for field in fields if value.get(field) == expected]
    if not matches:
        raise ContractError("bound object does not contain its accepted self-digest")
    field = matches[0]
    if digest_object(value, field) != expected:
        raise ContractError("bound object self-digest does not reproduce")
    return field


def _normal_evaluation_requests(
    captured: CapturedFile,
    *,
    excluded_request_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_request_ids = excluded_request_ids or set()
    requests = [
        normalize_original_pool([raw])[0] for raw in _rows(captured)
    ]
    retained = [request for request in requests if request["request_id"] not in excluded_request_ids]
    return sorted(
        retained,
        key=lambda request: (
            request["origin_source"], request["parent_id"], request["request_id"],
        ),
    )


def _secondary_request(variant: Mapping[str, Any]) -> dict[str, Any]:
    question = dict(variant["question"])
    question["soft"] = None
    question["source"] = variant["target"]
    core = {
        "origin_source": variant["source"],
        "parent_id": variant["parent_id"],
        "request_id": variant["development_id"],
        "state": variant["state"],
        "questions": [question],
        "original_record_sha256": variant["input_sha256"],
    }
    return {
        "request_uid": variant["variant_id"],
        **core,
        "state_sha256": variant["state_sha256"],
        "request_sha256": digest_object(core),
        "family": variant["family"],
        "population": variant["population"],
    }


def _inventory_row(
    track: str,
    request: Mapping[str, Any],
    question: Mapping[str, Any],
    family: str,
    populations: list[str],
) -> dict[str, Any]:
    return {
        "track": track,
        "key": [track, request["request_id"], question["id"], family],
        "request_uid": request["request_uid"],
        "request_id": request["request_id"],
        "question_id": question["id"],
        "origin_source": request["origin_source"],
        "parent_id": request["parent_id"],
        "task_source": question["source"],
        "primitive": question["type"],
        "population_ids": populations,
        "state_sha256": request["state_sha256"],
        "question_sha256": digest_object(question),
        "ordered_option_keys": question["option_keys"],
        "ordered_option_descriptions": question["options"],
        "label": question["label"],
        "soft": question["soft"],
    }


def _evaluation_material(
    decision: list[dict[str, Any]],
    transfer: list[dict[str, Any]],
    korean: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    t04_development: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    t04 = {
        (row["development_id"], row["question"]["id"]): row
        for row in t04_development
    }
    inventory = []
    requests = {}
    for track, track_requests in (
        ("decision", decision), ("transfer", transfer), ("korean", korean),
    ):
        for request in track_requests:
            requests[request["request_uid"]] = request
            for question in request["questions"]:
                populations = [f"{track}_full"]
                if track == "decision":
                    populations.append("decision_common_clean")
                elif track == "korean":
                    populations.append("korean_common_clean")
                else:
                    t04_row = t04.get((request["request_id"], question["id"]))
                    if t04_row is not None:
                        if t04_row["population"] == "native_six_class":
                            populations.append("emotion_native")
                        elif t04_row["population"] == "native_binary":
                            populations.append("offensive_native")
                        else:
                            populations.append("augmented_emotion")
                    elif question["source"] not in {"emotion", "tweet_offensive"}:
                        populations.append("other_transfer")
                if question["type"] == "score":
                    populations.append("score_common_clean")
                inventory.append(_inventory_row(
                    track, request, question, "original", populations,
                ))
    for variant in variants:
        if variant["family"] == "original":
            continue
        request = _secondary_request(variant)
        if request["request_uid"] in requests:
            raise ContractError("secondary variant UID collides with an evaluation request")
        requests[request["request_uid"]] = request
        inventory.append(_inventory_row(
            "secondary", request, request["questions"][0], variant["family"],
            [f"secondary/{variant['population']}/{variant['family']}"],
        ))
    counts = {
        "decision": sum(row["track"] == "decision" for row in inventory),
        "transfer": sum(row["track"] == "transfer" for row in inventory),
        "korean": sum(row["track"] == "korean" for row in inventory),
        "secondary": sum(row["track"] == "secondary" for row in inventory),
    }
    if counts != {"decision": 1463, "transfer": 764, "korean": 5000, "secondary": 904}:
        raise ContractError(f"evaluation question inventory differs: {counts}")
    population_counts = Counter(
        population for row in inventory for population in row["population_ids"]
    )
    expected = {
        "decision_common_clean": 1463, "korean_common_clean": 5000,
        "emotion_native": 92, "offensive_native": 80,
        "other_transfer": 568, "score_common_clean": 280,
    }
    if any(population_counts[key] != value for key, value in expected.items()):
        raise ContractError("primary or guardrail population counts differ")
    return inventory, requests


def _add_encoding_identities(
    inventory: list[dict[str, Any]],
    requests: Mapping[str, Mapping[str, Any]],
    tokenizer,
    delimiters: Mapping[str, int],
) -> dict[str, Any]:
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in inventory:
        by_uid.setdefault(row["request_uid"], []).append(row)
    identities = []
    for uid in dict.fromkeys(row["request_uid"] for row in inventory):
        request = requests[uid]
        packed = encode_request(
            tokenizer, request["state"], request["questions"], 2048, dict(delimiters),
        )
        if packed is None or packed.retained_state_tokens != packed.state_tokens:
            raise ContractError("evaluation request truncates under the frozen tokenizer")
        identity = _packed_identity(packed)
        encoded = digest_object(identity)
        tokens = len(identity["input_ids"])
        for row in by_uid[uid]:
            row["encoding_sha256"] = encoded
            row["packed_tokens"] = tokens
        identities.append({"request_uid": uid, "encoding_sha256": encoded, "tokens": tokens})
    return {
        "requests": len(identities),
        "identities_sha256": digest_object(identities),
        "minimum_tokens": min(row["tokens"] for row in identities),
        "maximum_tokens": max(row["tokens"] for row in identities),
    }


def _zero_references(
    inventory: list[Mapping[str, Any]],
    shared_report: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    semantic_observations: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    report_rows = {
        (track, row["request_id"], row["question_id"]): row
        for name, track in (
            ("decision_development", "decision"),
            ("transfer_development", "transfer"),
            ("korean_development", "korean"),
        )
        for row in shared_report["predictions"][name]
    }
    variant_by_id = {row["variant_id"]: row for row in variants}
    semantic_by_id = {row["variant_id"]: row for row in semantic_observations}
    if len(variant_by_id) != 1100 or len(semantic_by_id) != 1100:
        raise ContractError("T04 semantic inventory or observations differ")
    result = []
    parity_observations = []
    parity_references = []
    for expected in inventory:
        if expected["track"] == "secondary":
            semantic = semantic_by_id[expected["request_uid"]]
            values = semantic["logits"]
        else:
            report = report_rows.get((
                expected["track"], expected["request_id"], expected["question_id"],
            ))
            if report is None:
                raise ContractError("cached generation-79 prediction is missing")
            values = report["logits"]
        row = {
            **{key: expected[key] for key in expected if key != "track"},
            "cpu_float32": values,
            "logits": values,
        }
        result.append(row)
    original_variants = [row for row in variants if row["family"] == "original"]
    original_inventory = {
        (row["request_id"], row["question_id"]): row
        for row in inventory if row["track"] == "transfer"
    }
    for variant in original_variants:
        expected = original_inventory[(variant["development_id"], variant["question"]["id"])]
        semantic = semantic_by_id[variant["variant_id"]]
        report = report_rows[("transfer", expected["request_id"], expected["question_id"])]
        common = {
            "key": expected["key"], "state_sha256": expected["state_sha256"],
            "question_sha256": expected["question_sha256"],
        }
        parity_observations.append({**common, "logits": semantic["logits"]})
        parity_references.append({**common, "logits": report["logits"]})
    parity = run_initial_parity(parity_observations, parity_references)
    if not parity["passed"]:
        raise ContractError("accepted T04 originals fail generation-79 parity")
    return result


def _bootstrap_strata(inventory: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for population in (
        "decision_common_clean", "emotion_native", "korean_common_clean",
        "offensive_native", "other_transfer", "score_common_clean",
    ):
        rows = [row for row in inventory if population in row["population_ids"]]
        parents: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            parents.setdefault((row["origin_source"], row["parent_id"]), set()).add(
                row["task_source"],
            )
        strata: dict[tuple[str, tuple[str, ...]], list[list[str]]] = {}
        for parent, sources in parents.items():
            key = (parent[0], tuple(sorted(sources)))
            strata.setdefault(key, []).append([parent[0], parent[1]])
        if any(len(value) < 2 for value in strata.values()):
            raise ContractError(f"bootstrap singleton stratum: {population}")
        result[population] = [
            {
                "origin_source": key[0], "task_source_incidence": list(key[1]),
                "parents": sorted(value), "count": len(value),
            }
            for key, value in sorted(strata.items())
        ]
    return {"populations": result}


def _runtime_metadata(shared_run: Mapping[str, Any]) -> dict[str, Any]:
    package = lambda name: metadata.version(name)
    return {
        "device": "mps",
        "parameter_dtype": "float32",
        "attention_implementation": "eager",
        "gradient_checkpointing": False,
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": package("transformers"),
        "datasets": shared_run["spec"]["software"]["datasets"],
        "tokenizers": package("tokenizers"),
        "safetensors": package("safetensors"),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "model_rng_policy": "seed Python, Torch CPU, and Torch MPS once per frozen update",
    }


def materialize_freeze(
    *, repository: str | Path, paths_file: str | Path, registry_file: str | Path,
) -> dict[str, Any]:
    repository = Path(repository)
    spec = strict_json_loads(
        (repository / "plans/next-v1/t05-v1/implementation-spec.json").read_bytes(),
    )
    if spec["spec_sha256"] != digest_object(spec, "spec_sha256"):
        raise ContractError("T05 implementation specification self-digest mismatch")
    paths = _path_map(paths_file)
    registry = strict_json_loads(Path(registry_file).read_bytes())
    if digest_object(registry) != spec["identity_rules"]["registry_canonical_sha256"]:
        raise ContractError("evidence registry canonical digest differs from the specification")
    captured = {}
    bindings = {}
    for artifact_id, expected in spec["bound_inputs"].items():
        if artifact_id == "shared_checkpoint_g79":
            bindings[artifact_id] = _stream_artifact_descriptor(paths, artifact_id, expected)
            continue
        item = _capture_artifact(paths, artifact_id, expected)
        captured[artifact_id] = item
        binding = {"artifact_id": artifact_id, "sha256": item.sha256, "size_bytes": item.size_bytes}
        if "rows" in expected:
            binding["rows"] = expected["rows"]
        bindings[artifact_id] = binding
        if "self_sha256" in expected:
            value = item.json()
            _self_digest(value, expected["self_sha256"])
            binding["self_sha256"] = expected["self_sha256"]
    original_pool = normalize_original_pool(_rows(captured["shared_suite_train"]))
    if len(original_pool) != 15572 or sum(len(row["questions"]) for row in original_pool) != 23068:
        raise ContractError("generation-79 original training inventory differs")
    exclusions = captured["t04_plan/exclusions.json"].json()
    forbidden_states = {row["sha256"] for row in exclusions["state_sha256"]}
    forbidden_parents = {(row["source"], row["parent_id"]) for row in exclusions["source_parents"]}
    target_pool = validate_target_support(
        _rows(captured["t04_plan/support-fit.jsonl"]),
        forbidden_state_sha256=forbidden_states,
        forbidden_parents=forbidden_parents,
    )
    comparison = captured["comparison_plan"].json()
    excluded_decision = {
        row["request_id"] for row in comparison["excluded_development_requests"]["decision"]
    }
    if len(excluded_decision) != 5:
        raise ContractError("comparison plan does not contain the five BoolQ exclusions")
    decision = _normal_evaluation_requests(
        captured["shared_suite_development"], excluded_request_ids=excluded_decision,
    )
    transfer = _normal_evaluation_requests(captured["transfer_suite_development"])
    korean = _normal_evaluation_requests(captured["korean_suite_development"])
    variants = _rows(captured["t04_plan/semantic-variants.jsonl"])
    t04_development = _rows(captured["t04_plan/development.jsonl"])
    inventory, evaluation_requests = _evaluation_material(
        decision, transfer, korean, variants, t04_development,
    )
    tokenizer_root, tokenizer_relative = _resolve_artifact(paths, "g79_tokenizer/tokenizer.json")
    tokenizer_directory = Path(tokenizer_root) / Path(tokenizer_relative).parent
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, use_fast=True,
    )
    delimiters = add_delimiters(tokenizer)
    evaluation_budget = _add_encoding_identities(
        inventory, evaluation_requests, tokenizer, delimiters,
    )

    def encoder(request):
        packed = encode_request(
            tokenizer, request["state"], request["questions"], 2048, dict(delimiters),
        )
        if packed is None:
            raise ContractError("training occurrence exceeds the frozen token budget")
        return _packed_identity(packed)

    training_budget = token_budget_preflight(
        [*original_pool, *target_pool], tokenizer, delimiters,
    )
    shared_report = captured["shared_development_report"].json()
    semantic_observations = _rows(captured["t04_corrected_semantic_observations"])
    zero_reference = _zero_references(
        inventory, shared_report, variants, semantic_observations,
    )
    zero_reference_metrics = replay_metrics(zero_reference)
    shared_run = captured["shared_run_spec"].json()
    if shared_run["run_id"] != spec["identity_rules"]["checkpoint_binding"]["run_id"]:
        raise ContractError("generation-79 run identity differs")
    runtime = _runtime_metadata(shared_run)
    exclusion_audit = {
        "support_requests": len(target_pool),
        "forbidden_states": len(forbidden_states),
        "forbidden_parents": len(forbidden_parents),
        "state_overlap": 0,
        "parent_overlap": 0,
        "support_calibration_excluded": len(_rows(captured["t04_plan/support-calibration.jsonl"])),
        "locked_test_opened": False,
    }
    source_digests = {
        "input_bindings": bindings,
        "registry_sha256": hashlib.sha256(Path(registry_file).read_bytes()).hexdigest(),
        "python_executable_sha256": hashlib.sha256(sys.executable.encode("utf-8")).hexdigest(),
    }
    return {
        "repository": repository,
        "spec": spec,
        "runtime": runtime,
        "input_bindings": bindings,
        "original_pool": original_pool,
        "target_pool": target_pool,
        "exclusion_audit": exclusion_audit,
        "evaluation_inventory": inventory,
        "zero_reference": zero_reference,
        "zero_reference_metrics": zero_reference_metrics,
        "bootstrap_strata": _bootstrap_strata(inventory),
        "encoding_budget": {
            "maximum_length": 2048,
            "training_distinct": training_budget,
            "evaluation": evaluation_budget,
            "training_occurrences": 16384,
        },
        "historical_source_digests": source_digests,
        "encoder": encoder,
        "private_material": {
            "paths": paths,
            "captured": captured,
            "evaluation_requests": evaluation_requests,
            "tokenizer": tokenizer,
            "delimiters": delimiters,
        },
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
