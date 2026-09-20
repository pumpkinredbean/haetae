"""Benchmark equivalent execution paths for the completed shared-state model.

The runner reads public development splits only. It freezes all population and
timing schedules before model inference, then compares one packed request with
batched and serial single-question encodings using the same trained weights.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

from haetae.checkpoint import tokenizer_fingerprint
from experiments.comparison_protocol import (
    load_comparison_plan,
    request_state_sha256,
    validate_suite_binding,
)
from experiments.evaluate_shared import load_model
from experiments.kev_adapter import file_sha256, load_frozen_split
from experiments.shared_state import PackedRequest, add_delimiters, encode_request


DESIGN_PATH = Path(__file__).with_name("shared_execution_protocol.json")
ARMS = ("P", "B", "S")
SUITES = ("decision", "transfer", "korean")
SCOPES = ("pretokenized_forward", "complete_request")
PROCESS_PID = os.getpid()
PROCESS_STARTED_UNIX_NS = time.time_ns()
PROCESS_INSTANCE_ID = uuid.uuid4().hex


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_write(path, canonical_bytes(value) + b"\n")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def runtime_identity() -> dict:
    identity = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode == 0:
            identity["cpu_brand"] = result.stdout.strip()
    return identity


def runner_identity(repository: Path) -> dict:
    paths = [Path(__file__).resolve(), DESIGN_PATH.resolve()]
    files = {}
    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.relative_to(repository))
        content = path.read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"sha256": digest.hexdigest(), "files": files, "git_commit": commit}


def load_design() -> dict:
    design = json.loads(DESIGN_PATH.read_text())
    if design.get("version") != 1:
        raise ValueError("unsupported shared execution protocol version")
    if tuple(design.get("arms", {})) != ARMS:
        raise ValueError("protocol arms differ from the implemented arms")
    return design


def validate_checkpoint_descriptor(run_dir: Path, design: dict) -> tuple[dict, dict]:
    run = json.loads((run_dir / "run.json").read_text())
    manifest = json.loads((run_dir / "latest.json").read_text())
    expected = design["checkpoint"]
    current = manifest["current"]
    checks = {
        "run ID": (run["run_id"], expected["run_id"]),
        "run specification": (run["spec_sha256"], expected["spec_sha256"]),
        "manifest run ID": (manifest["run_id"], expected["run_id"]),
        "manifest specification": (
            manifest["spec_sha256"], expected["spec_sha256"],
        ),
        "generation": (current["generation"], expected["generation"]),
        "step": (current["step"], expected["step"]),
        "checkpoint digest": (
            current["sha256"], expected["checkpoint_sha256"],
        ),
        "maximum length": (
            run["spec"]["config"]["max_len"], expected["max_length"],
        ),
        "attention implementation": (
            run["spec"]["experiment"]["attention_implementation"],
            expected["attention_implementation"],
        ),
    }
    for name, (actual, wanted) in checks.items():
        if actual != wanted:
            raise ValueError(f"{name} differs from the frozen protocol")
    if current.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("shared-state run is not completed")
    checkpoint = run_dir / current["path"]
    if checkpoint.stat().st_size != current["size"]:
        raise ValueError("checkpoint size differs from the completed manifest")
    if file_sha256(checkpoint) != current["sha256"]:
        raise ValueError("checkpoint bytes differ from the completed manifest")
    return run, manifest


def validate_development_report(
    path: Path, design: dict, comparison_plan: dict,
) -> dict:
    """Bind the fixed temperature to the accepted public-development report."""
    expected = design["checkpoint"]["development_report"]
    if file_sha256(path) != expected["file_sha256"]:
        raise ValueError("development report file digest differs from the protocol")
    report = json.loads(path.read_text())
    unsigned = dict(report)
    recorded_sha = unsigned.pop("report_sha256", None)
    if recorded_sha != expected["report_sha256"]:
        raise ValueError("development report digest differs from the protocol")
    if value_sha256(unsigned) != recorded_sha:
        raise ValueError("development report content digest is invalid")
    if report.get("status") != "evaluated":
        raise ValueError("development report is not complete")
    if report.get("locked_test_opened") is not False:
        raise ValueError("development report does not record a closed locked test")
    run = report.get("run", {})
    for key in ("run_id", "spec_sha256", "generation", "checkpoint_sha256"):
        if run.get(key) != design["checkpoint"][key]:
            raise ValueError(f"development report {key} differs from the protocol")
    if report.get("comparison_plan_sha256") != comparison_plan["plan_sha256"]:
        raise ValueError("development report comparison plan differs")
    temperature = report.get("temperature", {})
    if temperature.get("value") != design["checkpoint"]["temperature"]:
        raise ValueError("development report temperature differs from the protocol")
    return {
        "path": str(path.resolve()),
        "file_sha256": expected["file_sha256"],
        "report_sha256": recorded_sha,
        "temperature": temperature["value"],
    }


def request_identity(suite: str, request: dict) -> str:
    return value_sha256({
        "suite": suite,
        "request_id": request["meta"].get("id"),
        "group_id": request["meta"].get("group_id"),
        "source": request["meta"].get("source"),
        "state_sha256": request_state_sha256(request),
        "question_ids": [question["id"] for question in request["questions"]],
    })


def parent_identity(suite: str, request: dict) -> str:
    source = request["meta"].get("source") or suite
    group = request["meta"].get("group_id") or request["meta"].get("id")
    if group is None:
        group = request_state_sha256(request)
    return f"{source}:{group}"


def load_common_clean_requests(
    comparison_plan_path: Path,
    suite_paths: dict[str, Path],
) -> tuple[dict[str, list[dict]], dict, dict]:
    plan = load_comparison_plan(comparison_plan_path)
    result, manifests = {}, {}
    for suite in SUITES:
        validate_suite_binding(plan, suite, suite_paths[suite])
        requests, manifest = load_frozen_split(suite_paths[suite], "development")
        excluded = {
            (
                item.get("request_id"), item.get("group_id"),
                item["state_sha256"],
            )
            for item in plan["excluded_development_requests"][suite]
        }
        clean = [
            request for request in requests
            if (
                request["meta"].get("id"),
                request["meta"].get("group_id"),
                request_state_sha256(request),
            ) not in excluded
        ]
        expected = (
            plan["audit"]["development_requests"][suite]
            - plan["audit"]["excluded_requests"][suite]
        )
        if len(clean) != expected:
            raise ValueError(f"{suite} common-clean request count differs from plan")
        result[suite] = clean
        manifests[suite] = manifest
    if plan.get("locked_test_opened") is not False:
        raise ValueError("comparison plan does not record a closed locked test")
    return result, manifests, plan


def encoding_digest(encoding: PackedRequest) -> str:
    return value_sha256({
        "input_ids": encoding.input_ids,
        "segments": encoding.segments,
        "position_ids": encoding.position_ids,
        "decision_indices": encoding.decision_indices,
        "option_indices": encoding.option_indices,
    })


def encode_and_validate(tokenizer, delimiters, request: dict, max_length: int):
    packed = encode_request(
        tokenizer, request["state"], request["questions"],
        max_length, delimiters,
    )
    singles = [
        encode_request(
            tokenizer, request["state"], [question], max_length, delimiters,
        )
        for question in request["questions"]
    ]
    if packed is None or any(single is None for single in singles):
        raise ValueError("public development request exceeds the fixed content budget")
    if packed.retained_state_tokens != packed.state_tokens or any(
        single.retained_state_tokens != single.state_tokens for single in singles
    ):
        raise ValueError("public development request would truncate its state")
    packed_state_ids = [
        token for token, segment in zip(packed.input_ids, packed.segments)
        if segment == 0
    ]
    packed_state_positions = [
        position for position, segment in zip(packed.position_ids, packed.segments)
        if segment == 0
    ]
    for index, single in enumerate(singles, start=1):
        single_state_ids = [
            token for token, segment in zip(single.input_ids, single.segments)
            if segment == 0
        ]
        single_state_positions = [
            position for position, segment in zip(
                single.position_ids, single.segments,
            ) if segment == 0
        ]
        packed_branch_ids = [
            token for token, segment in zip(packed.input_ids, packed.segments)
            if segment == index
        ]
        packed_branch_positions = [
            position for position, segment in zip(
                packed.position_ids, packed.segments,
            ) if segment == index
        ]
        single_branch_ids = [
            token for token, segment in zip(single.input_ids, single.segments)
            if segment == 1
        ]
        single_branch_positions = [
            position for position, segment in zip(
                single.position_ids, single.segments,
            ) if segment == 1
        ]
        if packed_state_ids != single_state_ids:
            raise ValueError("packed and separate state token IDs differ")
        if packed_state_positions != single_state_positions:
            raise ValueError("packed and separate state positions differ")
        if packed_branch_ids != single_branch_ids:
            raise ValueError("packed and separate branch token IDs differ")
        if packed_branch_positions != single_branch_positions:
            raise ValueError("packed and separate branch positions differ")
    return packed, singles


def workload_row(
    suite: str,
    request: dict,
    packed: PackedRequest,
    singles: list[PackedRequest],
) -> dict:
    state_length = sum(segment == 0 for segment in packed.segments)
    branch_lengths = [
        sum(segment == index for segment in packed.segments)
        for index in range(1, len(request["questions"]) + 1)
    ]
    return {
        "workload_id": request_identity(suite, request),
        "parent_id": parent_identity(suite, request),
        "suite": suite,
        "request_id": request["meta"].get("id"),
        "group_id": request["meta"].get("group_id"),
        "source": request["meta"].get("source"),
        "state_sha256": request_state_sha256(request),
        "question_ids": [question["id"] for question in request["questions"]],
        "question_count": len(request["questions"]),
        "candidate_counts": [
            len(question["options"]) for question in request["questions"]
        ],
        "state_length": state_length,
        "branch_lengths": branch_lengths,
        "packed_length": len(packed.input_ids),
        "separate_lengths": [len(single.input_ids) for single in singles],
        "separate_padded_tokens": (
            max(len(single.input_ids) for single in singles) * len(singles)
        ),
        "packed_dense_attention_elements": len(packed.input_ids) ** 2,
        "separate_dense_attention_elements": sum(
            len(single.input_ids) ** 2 for single in singles
        ),
        "packed_token_sha256": encoding_digest(packed),
        "separate_token_sha256": [
            encoding_digest(single) for single in singles
        ],
        "equivalence": True,
        "performance_panels": [],
    }


def selection_rank(row: dict, seed: int, panel: str) -> str:
    return value_sha256({
        "seed": seed, "panel": panel, "workload_id": row["workload_id"],
    })


def distinct_parent_rows(
    rows: list[dict], limit: int, seed: int, panel: str,
) -> list[dict]:
    """Select at most one deterministic workload from each evaluation parent."""
    by_parent = defaultdict(list)
    for row in rows:
        by_parent[row["parent_id"]].append(row)
    representatives = [
        min(members, key=lambda row: selection_rank(row, seed, panel + "-member"))
        for members in by_parent.values()
    ]
    representatives.sort(key=lambda row: selection_rank(row, seed, panel))
    if len(representatives) < limit:
        raise ValueError(
            f"performance cell {panel} has {len(representatives)} distinct parents; "
            f"the protocol requires {limit}"
        )
    return representatives[:limit]


def longest_diagnostic_rows(
    rows: list[dict], limit: int, seed: int, panel: str,
) -> list[dict]:
    """Cover the highest-option case, then fill by packed length using distinct parents."""
    if not rows or limit <= 0:
        return []
    highest_option = max(
        rows,
        key=lambda row: (
            max(row["candidate_counts"]), row["packed_length"],
            selection_rank(row, seed, panel + "-option"),
        ),
    )
    by_parent = defaultdict(list)
    for row in rows:
        if row["parent_id"] == highest_option["parent_id"]:
            continue
        by_parent[row["parent_id"]].append(row)
    representatives = [
        max(
            members,
            key=lambda row: (
                row["packed_length"], max(row["candidate_counts"]),
                selection_rank(row, seed, panel + "-member"),
            ),
        )
        for members in by_parent.values()
    ]
    ordered = sorted(
        representatives,
        key=lambda row: (
            -row["packed_length"], -max(row["candidate_counts"]),
            row["workload_id"],
        ),
    )
    return [highest_option, *ordered[: max(0, limit - 1)]]


def assign_performance_panels(rows: list[dict], design: dict) -> None:
    selection = design["workload_selection"]
    seed = selection["seed"]
    cells = defaultdict(list)
    for row in rows:
        cells[(row["suite"], row["question_count"])].append(row)
    for (suite, count), members in cells.items():
        if count > 1 and suite in selection["primary_multi_question_suites"]:
            ordered = distinct_parent_rows(
                members,
                selection["hash_selected_per_multi_question_cell"],
                seed,
                f"multi-primary:{suite}:{count}",
            )
            for row in ordered:
                row["performance_panels"].append("multi_primary_pool")
        if count == 1 and suite in selection["single_question_control_suites"]:
            ordered = distinct_parent_rows(
                members,
                selection["hash_selected_single_question_controls_per_suite"],
                seed,
                f"single-control:{suite}",
            )
            for row in ordered:
                row["performance_panels"].append("single_control_pool")
        longest = longest_diagnostic_rows(
            members,
            min(selection["longest_diagnostics_per_cell"], len({
                row["parent_id"] for row in members
            })),
            seed,
            f"longest:{suite}:{count}",
        )
        for row in longest:
            row["performance_panels"].append("longest_diagnostic")
    for row in rows:
        row["performance_panels"].sort()


def balanced_sequence(
    rows: list[dict], total: int, seed: int, label: str,
    strata: tuple[str, ...],
) -> list[str]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in strata)].append(row)
    if not grouped:
        raise ValueError(f"schedule population {label} is empty")
    for key in grouped:
        grouped[key].sort(key=lambda row: selection_rank(row, seed, label))
    keys = sorted(grouped)
    offsets = {key: 0 for key in keys}
    sequence = []
    for index in range(total):
        key = keys[index % len(keys)]
        members = grouped[key]
        member = members[offsets[key] % len(members)]
        offsets[key] += 1
        sequence.append(member["workload_id"])
    return sequence


def batch_schedule(ids: list[str], rows_by_id: dict[str, dict], batch: int) -> list[dict]:
    if batch <= 0:
        raise ValueError("request batch size must be positive")
    grouped = defaultdict(list)
    for workload_id in ids:
        grouped[rows_by_id[workload_id]["suite"]].append(workload_id)
    result = []
    for suite in sorted(grouped):
        values = grouped[suite]
        for start in range(0, len(values), batch):
            chunk = values[start:start + batch]
            result.append({
                "suite": suite,
                "workload_ids": chunk,
                "parent_ids": [rows_by_id[value]["parent_id"] for value in chunk],
            })
    return result


def build_schedules(rows: list[dict], design: dict) -> dict:
    timing = design["timing"]
    seed = design["workload_selection"]["seed"]
    rows_by_id = {row["workload_id"]: row for row in rows}
    multi = [
        row for row in rows
        if "multi_primary_pool" in row["performance_panels"]
    ]
    controls = [
        row for row in rows
        if "single_control_pool" in row["performance_panels"]
    ]
    populations = {}
    for name, pool, warmup, measured, strata in (
        (
            "multi_primary", multi,
            timing["primary_warmup_requests"],
            timing["primary_measured_requests"],
            ("suite", "question_count"),
        ),
        (
            "single_control", controls,
            timing["single_control_warmup_requests"],
            timing["single_control_measured_requests"],
            ("suite",),
        ),
    ):
        warmup_ids = balanced_sequence(
            pool, warmup, seed, f"{name}-warmup", strata,
        )
        measured_ids = balanced_sequence(
            pool, measured, seed, f"{name}-measured", strata,
        )
        populations[name] = {}
        for batch in timing["request_batch_sizes"]:
            populations[name][str(batch)] = {
                "warmup": batch_schedule(warmup_ids, rows_by_id, batch),
                "measured": batch_schedule(measured_ids, rows_by_id, batch),
            }
    repetitions = {
        str(repetition): copy.deepcopy(populations)
        for repetition in range(timing["fresh_process_repetitions"])
    }
    memory_design = design["memory"]
    memory_ids = balanced_sequence(
        multi,
        memory_design["steady_state_requests"],
        seed,
        "memory-steady-state",
        ("suite", "question_count"),
    )
    schedules = {
        "version": 1,
        "selection_seed": seed,
        "repetitions": repetitions,
        "memory": {
            "requests": len(memory_ids),
            "request_batch_size": memory_design["request_batch_size"],
            "batches": batch_schedule(
                memory_ids, rows_by_id, memory_design["request_batch_size"],
            ),
        },
    }
    schedules["schedules_sha256"] = value_sha256(schedules)
    return schedules


def suite_descriptor(path: Path, manifest: dict) -> dict:
    development = manifest["files"]["development.jsonl"]
    return {
        "path": str(path.resolve()),
        "manifest_file_sha256": file_sha256(path / "manifest.json"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "development_sha256": development["sha256"],
        "development_records": development["records"],
        "development_questions": development["questions"],
    }


def validate_schedule_batches(
    batches: list[dict], rows_by_id: dict[str, dict], expected_requests: int,
    maximum_batch_size: int,
) -> None:
    observed = 0
    for batch in batches:
        workload_ids = batch.get("workload_ids")
        parent_ids = batch.get("parent_ids")
        if (
            not isinstance(workload_ids, list) or not workload_ids
            or len(workload_ids) > maximum_batch_size
        ):
            raise ValueError("frozen schedule has an invalid request batch")
        if any(workload_id not in rows_by_id for workload_id in workload_ids):
            raise ValueError("frozen schedule contains an unknown workload")
        if batch.get("suite") is None or any(
            rows_by_id[workload_id]["suite"] != batch["suite"]
            for workload_id in workload_ids
        ):
            raise ValueError("frozen schedule mixes workload suites")
        expected_parents = [
            rows_by_id[workload_id]["parent_id"] for workload_id in workload_ids
        ]
        if parent_ids != expected_parents:
            raise ValueError("frozen schedule parent identities differ")
        observed += len(workload_ids)
    if observed != expected_requests:
        raise ValueError("frozen schedule request count differs")


def validate_frozen_schedules(schedules: dict, rows: list[dict], design: dict) -> None:
    rows_by_id = {row["workload_id"]: row for row in rows}
    timing = design["timing"]
    repetitions = schedules.get("repetitions", {})
    expected_keys = {
        str(index) for index in range(timing["fresh_process_repetitions"])
    }
    if set(repetitions) != expected_keys:
        raise ValueError("frozen timing repetition inventory differs")
    reference = repetitions["0"]
    if any(value != reference for value in repetitions.values()):
        raise ValueError("frozen timing request schedules differ across repetitions")
    for population, warmup, measured in (
        ("multi_primary", timing["primary_warmup_requests"], timing["primary_measured_requests"]),
        ("single_control", timing["single_control_warmup_requests"], timing["single_control_measured_requests"]),
    ):
        by_batch = reference.get(population, {})
        if set(by_batch) != {str(value) for value in timing["request_batch_sizes"]}:
            raise ValueError("frozen timing batch inventory differs")
        for batch_size in timing["request_batch_sizes"]:
            schedule = by_batch[str(batch_size)]
            validate_schedule_batches(
                schedule["warmup"], rows_by_id, warmup, batch_size,
            )
            validate_schedule_batches(
                schedule["measured"], rows_by_id, measured, batch_size,
            )
    memory = schedules.get("memory", {})
    memory_design = design["memory"]
    if memory.get("requests") != memory_design["steady_state_requests"]:
        raise ValueError("frozen memory request count differs")
    if memory.get("request_batch_size") != memory_design["request_batch_size"]:
        raise ValueError("frozen memory batch size differs")
    validate_schedule_batches(
        memory.get("batches", []), rows_by_id,
        memory_design["steady_state_requests"],
        memory_design["request_batch_size"],
    )


def freeze(args) -> dict:
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = [
        output / "protocol.json", output / "workloads.jsonl",
        output / "workloads.meta.json", output / "schedules.json",
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("refusing to overwrite a frozen execution protocol")
    repository = Path(__file__).resolve().parents[1]
    design = load_design()
    run_dir = Path(args.run).resolve()
    run, manifest = validate_checkpoint_descriptor(run_dir, design)
    suite_paths = {
        "decision": Path(args.decision_suite).resolve(),
        "transfer": Path(args.transfer_suite).resolve(),
        "korean": Path(args.korean_suite).resolve(),
    }
    requests, manifests, comparison_plan = load_common_clean_requests(
        Path(args.comparison_plan).resolve(), suite_paths,
    )
    development_report = validate_development_report(
        Path(args.development_report).resolve(), design, comparison_plan,
    )
    tokenizer = AutoTokenizer.from_pretrained(run_dir, local_files_only=True)
    delimiters = add_delimiters(tokenizer)
    if tokenizer_fingerprint(tokenizer) != run["spec"]["tokenizer_fingerprint"]:
        raise ValueError("saved tokenizer differs from the completed run")
    rows = []
    for suite in SUITES:
        for request in requests[suite]:
            packed, singles = encode_and_validate(
                tokenizer, delimiters, request, design["checkpoint"]["max_length"],
            )
            rows.append(workload_row(suite, request, packed, singles))
    rows.sort(key=lambda row: (row["suite"], row["workload_id"]))
    if len({row["workload_id"] for row in rows}) != len(rows):
        raise ValueError("workload identities are not unique")
    assign_performance_panels(rows, design)
    schedules = build_schedules(rows, design)
    validate_frozen_schedules(schedules, rows, design)
    workload_data = jsonl_bytes(rows)
    protocol = {
        "version": 1,
        "status": "frozen",
        "design": design,
        "design_file_sha256": file_sha256(DESIGN_PATH),
        "runner": runner_identity(repository),
        "runtime_at_freeze": runtime_identity(),
        "run": {
            "path": str(run_dir),
            "run_id": run["run_id"],
            "spec_sha256": run["spec_sha256"],
            "generation": manifest["current"]["generation"],
            "checkpoint_sha256": manifest["current"]["sha256"],
            "training_code_sha256": run["spec"]["training_code"]["sha256"],
            "tokenizer_fingerprint": run["spec"]["tokenizer_fingerprint"],
            "model_config_fingerprint": run["spec"]["model_config_fingerprint"],
        },
        "comparison_plan": {
            "path": str(Path(args.comparison_plan).resolve()),
            "file_sha256": file_sha256(args.comparison_plan),
            "plan_sha256": comparison_plan["plan_sha256"],
        },
        "development_report": development_report,
        "suites": {
            suite: suite_descriptor(suite_paths[suite], manifests[suite])
            for suite in SUITES
        },
        "locked_test_opened": False,
    }
    protocol["protocol_sha256"] = value_sha256(protocol)
    workload_meta = {
        "version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "workloads_file_sha256": hashlib.sha256(workload_data).hexdigest(),
        "requests": len(rows),
        "questions": sum(row["question_count"] for row in rows),
        "suite_requests": dict(Counter(row["suite"] for row in rows)),
        "cells": {
            f"{suite}:{count}": sum(
                row["suite"] == suite and row["question_count"] == count
                for row in rows
            )
            for suite, count in sorted({
                (row["suite"], row["question_count"]) for row in rows
            })
        },
        "panel_requests": {
            panel: sum(panel in row["performance_panels"] for row in rows)
            for panel in (
                "multi_primary_pool", "single_control_pool",
                "longest_diagnostic",
            )
        },
        "schedules_sha256": schedules["schedules_sha256"],
        "locked_test_opened": False,
    }
    atomic_json(output / "protocol.json", protocol)
    atomic_write(output / "workloads.jsonl", workload_data)
    atomic_json(output / "workloads.meta.json", workload_meta)
    atomic_json(output / "schedules.json", schedules)
    return {"protocol": protocol, "workloads": workload_meta}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_frozen(output: Path) -> tuple[dict, list[dict], dict, dict]:
    protocol = json.loads((output / "protocol.json").read_text())
    unsigned = dict(protocol)
    recorded_protocol_sha = unsigned.pop("protocol_sha256")
    if value_sha256(unsigned) != recorded_protocol_sha:
        raise ValueError("protocol digest mismatch")
    meta = json.loads((output / "workloads.meta.json").read_text())
    if meta["protocol_sha256"] != recorded_protocol_sha:
        raise ValueError("workload metadata refers to a different protocol")
    workload_path = output / "workloads.jsonl"
    if file_sha256(workload_path) != meta["workloads_file_sha256"]:
        raise ValueError("workload file digest mismatch")
    rows = load_jsonl(workload_path)
    if len(rows) != meta["requests"]:
        raise ValueError("workload request count mismatch")
    schedules = json.loads((output / "schedules.json").read_text())
    schedule_unsigned = dict(schedules)
    schedule_sha = schedule_unsigned.pop("schedules_sha256")
    if value_sha256(schedule_unsigned) != schedule_sha:
        raise ValueError("schedule digest mismatch")
    if schedule_sha != meta["schedules_sha256"]:
        raise ValueError("workload metadata refers to different schedules")
    validate_frozen_schedules(schedules, rows, protocol["design"])
    repository = Path(__file__).resolve().parents[1]
    current_runner = runner_identity(repository)
    if current_runner["sha256"] != protocol["runner"]["sha256"]:
        raise ValueError("benchmark source differs from the frozen protocol")
    return protocol, rows, meta, schedules


def reload_requests(protocol: dict, rows: list[dict]) -> dict[str, dict]:
    plan_path = Path(protocol["comparison_plan"]["path"])
    suite_paths = {
        suite: Path(protocol["suites"][suite]["path"]) for suite in SUITES
    }
    requests, _, plan = load_common_clean_requests(plan_path, suite_paths)
    if plan["plan_sha256"] != protocol["comparison_plan"]["plan_sha256"]:
        raise ValueError("comparison plan content changed")
    mapped = {}
    for suite in SUITES:
        for request in requests[suite]:
            identity = request_identity(suite, request)
            if identity in mapped:
                raise ValueError("reloaded workload identity is duplicated")
            mapped[identity] = request
    expected = {row["workload_id"] for row in rows}
    if set(mapped) != expected:
        raise ValueError("reloaded public development population changed")
    return mapped


class RequestTokenizerCache:
    """Cache repeated state tokenization within one request execution."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, *, add_special_tokens=False):
        key = (text, add_special_tokens)
        if key not in self.cache:
            self.cache[key] = self.tokenizer(
                text, add_special_tokens=add_special_tokens,
            )
        return copy.deepcopy(self.cache[key])


def validate_runtime_request(request: dict) -> dict:
    if not isinstance(request.get("questions"), list) or not request["questions"]:
        raise ValueError("runtime request has no questions")
    questions = []
    for question in request["questions"]:
        if question.get("type") not in {"choice", "noul", "score"}:
            raise ValueError("runtime request has an unsupported question type")
        options = question.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("runtime request has invalid options")
        questions.append({
            "id": question["id"],
            "type": question["type"],
            "instructions": str(question["instructions"]),
            "options": [str(option) for option in options],
            "option_keys": list(question["option_keys"]),
            "label": question["label"],
            "soft": question["soft"],
            "source": question["source"],
        })
    return {"state": request["state"], "questions": questions, "meta": request["meta"]}


def prepare_request(tokenizer, delimiters, request: dict, max_length: int) -> dict:
    packed, singles = encode_and_validate(
        tokenizer, delimiters, request, max_length,
    )
    return {"request": request, "packed": packed, "singles": singles}


def prepare_for_arm(tokenizer, delimiters, request: dict, max_length: int, arm: str):
    request = validate_runtime_request(request)
    cached = RequestTokenizerCache(tokenizer)
    if arm == "P":
        packed = encode_request(
            cached, request["state"], request["questions"],
            max_length, delimiters,
        )
        if packed is None or packed.retained_state_tokens != packed.state_tokens:
            raise ValueError("complete request rejected a frozen packed request")
        return {"request": request, "packed": packed, "singles": None}
    singles = [
        encode_request(
            cached, request["state"], [question],
            max_length, delimiters,
        )
        for question in request["questions"]
    ]
    if any(single is None for single in singles) or any(
        single.retained_state_tokens != single.state_tokens for single in singles
    ):
        raise ValueError("complete request rejected a frozen separate question")
    return {"request": request, "packed": None, "singles": singles}


def execute_prepared(model, prepared: list[dict], arm: str):
    if arm == "P":
        return model([item["packed"] for item in prepared])
    if arm == "B":
        flat = [single for item in prepared for single in item["singles"]]
        flat_outputs = model(flat)
        result, cursor = [], 0
        for item in prepared:
            count = len(item["singles"])
            result.append([flat_outputs[cursor + offset][0] for offset in range(count)])
            cursor += count
        return result
    if arm == "S":
        return [
            [model([single])[0][0] for single in item["singles"]]
            for item in prepared
        ]
    raise ValueError(f"unknown execution arm: {arm}")


def materialize_responses(outputs, prepared: list[dict], temperature: float):
    responses = []
    for request_outputs, item in zip(outputs, prepared):
        questions = item["request"]["questions"]
        if len(request_outputs) != len(questions):
            raise RuntimeError("execution arm returned the wrong question count")
        response = []
        for question, logits in zip(questions, request_outputs):
            probabilities = F.softmax(logits / temperature, dim=-1)
            values = probabilities.detach().cpu().to(torch.float64).tolist()
            response.append({
                "question_id": question["id"],
                "probabilities": values,
                "selected": max(range(len(values)), key=values.__getitem__),
            })
        responses.append(response)
    return responses


def execute_complete(
    model, tokenizer, delimiters, requests: list[dict],
    max_length: int, arm: str, temperature: float,
):
    prepared = [
        prepare_for_arm(tokenizer, delimiters, request, max_length, arm)
        for request in requests
    ]
    outputs = execute_prepared(model, prepared, arm)
    return materialize_responses(outputs, prepared, temperature)


def logits_cpu(outputs) -> list[list[list[float]]]:
    return [
        [logits.detach().cpu().to(torch.float64).tolist() for logits in request]
        for request in outputs
    ]


def comparison_metrics(reference: list[float], candidate: list[float], temperature: float):
    reference_tensor = torch.tensor(reference, dtype=torch.float64)
    candidate_tensor = torch.tensor(candidate, dtype=torch.float64)
    if reference_tensor.shape != candidate_tensor.shape:
        raise ValueError("comparison logits have different shapes")
    reference_log = F.log_softmax(reference_tensor / temperature, dim=-1)
    candidate_log = F.log_softmax(candidate_tensor / temperature, dim=-1)
    reference_prob = reference_log.exp()
    candidate_prob = candidate_log.exp()
    top_values = reference_prob.topk(min(2, reference_prob.numel())).values
    margin = float(top_values[0] - top_values[1])
    return {
        "temperature": temperature,
        "total_variation": float(
            0.5 * (reference_prob - candidate_prob).abs().sum()
        ),
        "max_absolute_log_probability_difference": float(
            (reference_log - candidate_log).abs().max()
        ),
        "reference_probabilities": reference_prob.tolist(),
        "candidate_probabilities": candidate_prob.tolist(),
        "reference_action": int(reference_prob.argmax()),
        "candidate_action": int(candidate_prob.argmax()),
        "reference_top_two_margin": margin,
        "action_changed": bool(reference_prob.argmax() != candidate_prob.argmax()),
    }


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def verify_prepared_against_row(prepared: dict, row: dict) -> None:
    packed, singles = prepared["packed"], prepared["singles"]
    if encoding_digest(packed) != row["packed_token_sha256"]:
        raise ValueError("packed token array differs from the frozen workload")
    if [encoding_digest(value) for value in singles] != row["separate_token_sha256"]:
        raise ValueError("separate token array differs from the frozen workload")


def artifact_row_binding(
    protocol: dict, workload_meta: dict, device: str,
    schedules: dict | None = None,
) -> dict:
    binding = {
        "version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "workloads_sha256": workload_meta["workloads_file_sha256"],
        "checkpoint_sha256": protocol["run"]["checkpoint_sha256"],
        "device": device,
    }
    if schedules is not None:
        binding["schedules_sha256"] = schedules["schedules_sha256"]
    return binding


def validate_row_binding(
    row: dict, protocol: dict, workload_meta: dict, device: str,
    schedules: dict | None = None,
) -> None:
    expected = artifact_row_binding(protocol, workload_meta, device, schedules)
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"artifact row {key} binding differs")


def expected_equivalence_inventory(rows: list[dict]) -> dict[tuple, dict]:
    by_suite = defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row)
    expected = {}
    for row in rows:
        workload_id = row["workload_id"]
        common = {"parent_id": row["parent_id"], "suite": row["suite"]}
        for index, question_id in enumerate(row["question_ids"]):
            expected[("base_equivalence", workload_id, question_id)] = {
                **common, "question_index": index,
            }
            if row["question_count"] > 1:
                for kind in ("question_order_reversal", "sibling_removal"):
                    expected[(kind, workload_id, question_id)] = common
        if "longest_diagnostic" in row["performance_panels"]:
            peers = [
                candidate for candidate in by_suite[row["suite"]]
                if candidate["workload_id"] != workload_id
            ]
            if peers:
                replacement = max(
                    peers,
                    key=lambda candidate: (
                        abs(candidate["packed_length"] - row["packed_length"]),
                        candidate["workload_id"],
                    ),
                )
                expected[(
                    "unrelated_sibling_replacement", workload_id,
                    row["question_ids"][0],
                )] = {
                    **common,
                    "neighbor_workload_id": replacement["workload_id"],
                }
            for kind, predicate in (
                ("batch_neighbor_shorter", lambda value: value < row["packed_length"]),
                ("batch_neighbor_longer", lambda value: value > row["packed_length"]),
            ):
                if any(predicate(candidate["packed_length"]) for candidate in peers):
                    candidates = [
                        candidate for candidate in peers
                        if predicate(candidate["packed_length"])
                    ]
                    neighbor = (
                        min(candidates, key=lambda candidate: (
                            candidate["packed_length"], candidate["workload_id"],
                        ))
                        if kind.endswith("shorter")
                        else max(candidates, key=lambda candidate: (
                            candidate["packed_length"], candidate["workload_id"],
                        ))
                    )
                    for question_id in row["question_ids"]:
                        expected[(kind, workload_id, question_id)] = {
                            **common,
                            "neighbor_workload_id": neighbor["workload_id"],
                        }
        expected[("request_complete", workload_id, None)] = common
    return expected


def equivalence_record_key(row: dict) -> tuple:
    return (
        row.get("kind"), row.get("workload_id"),
        None if row.get("kind") == "request_complete" else row.get("question_id"),
    )


def validate_equivalence_transaction(
    records: list[dict], expected: dict[tuple, dict], protocol: dict,
    workload_meta: dict, device: str,
) -> str:
    if not records or records[-1].get("kind") != "request_complete":
        raise ValueError("equivalence transaction lacks a completion record")
    workload_id = records[-1].get("workload_id")
    if not isinstance(workload_id, str):
        raise ValueError("equivalence completion lacks a workload identity")
    expected_keys = {key for key in expected if key[1] == workload_id}
    actual_keys = []
    for row in records:
        validate_row_binding(row, protocol, workload_meta, device)
        key = equivalence_record_key(row)
        if key not in expected:
            raise ValueError("equivalence transaction contains a non-frozen identity")
        details = expected[key]
        if row.get("parent_id") != details["parent_id"]:
            raise ValueError("equivalence parent identity differs from the workload")
        if row.get("suite") != details["suite"]:
            raise ValueError("equivalence suite differs from the workload")
        for field in ("question_index", "neighbor_workload_id"):
            if field in details and row.get(field) != details[field]:
                raise ValueError(f"equivalence {field} differs from the workload")
        actual_keys.append(key)
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("equivalence transaction contains duplicate observations")
    if set(actual_keys) != expected_keys:
        raise ValueError("equivalence transaction is incomplete")
    return workload_id


def clean_incomplete_equivalence(
    path: Path, protocol: dict, workload_meta: dict,
    workload_rows: list[dict], device: str,
) -> set[str]:
    header = {
        "kind": "journal_header",
        **artifact_row_binding(protocol, workload_meta, device),
    }
    if not path.exists():
        atomic_write(path, jsonl_bytes([header]))
        return set()
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        boundary = data.rfind(b"\n")
        if boundary < 0:
            raise ValueError("equivalence journal has no committed header")
        data = data[:boundary + 1]
    parsed = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        try:
            parsed.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"equivalence journal is corrupt at committed line {line_number}"
            ) from error
    if not parsed or parsed[0] != header:
        raise ValueError("equivalence journal header binding differs")
    if any(row.get("kind") == "journal_header" for row in parsed[1:]):
        raise ValueError("equivalence journal contains a duplicate header")
    expected = expected_equivalence_inventory(workload_rows)
    retained = [header]
    transaction = []
    completed = set()
    for row in parsed[1:]:
        transaction.append(row)
        if row.get("kind") != "request_complete":
            continue
        workload_id = validate_equivalence_transaction(
            transaction, expected, protocol, workload_meta, device,
        )
        if workload_id in completed:
            raise ValueError("equivalence journal repeats a completed workload")
        completed.add(workload_id)
        retained.extend(transaction)
        transaction = []
    atomic_write(path, jsonl_bytes(retained))
    return completed


def equivalence(args) -> dict:
    output = Path(args.out).resolve()
    protocol, rows, meta, _ = load_frozen(output)
    if args.batch <= 0:
        raise ValueError("equivalence request batch must be positive")
    if args.device not in protocol["design"]["timing"]["devices"]:
        raise ValueError("device is not declared by the protocol")
    result_path = output / f"equivalence-{args.device}.jsonl"
    meta_path = output / f"equivalence-{args.device}.meta.json"
    if meta_path.exists():
        raise FileExistsError("equivalence artifact is already complete")
    completed = clean_incomplete_equivalence(
        result_path, protocol, meta, rows, args.device,
    )
    binding = artifact_row_binding(protocol, meta, args.device)
    requests = reload_requests(protocol, rows)
    model, tokenizer, delimiters, _, run, manifest = load_model(
        protocol["run"]["path"], args.device,
    )
    if manifest["current"]["sha256"] != protocol["run"]["checkpoint_sha256"]:
        raise ValueError("loaded checkpoint differs from the protocol")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("loaded model precision differs from the protocol")
    max_length = protocol["design"]["checkpoint"]["max_length"]
    temperatures = protocol["design"]["equivalence"]["temperatures"]
    diagnostic_ids = {
        row["workload_id"] for row in rows
        if "longest_diagnostic" in row["performance_panels"]
    }
    rows_by_id = {row["workload_id"]: row for row in rows}
    pending = [row for row in rows if row["workload_id"] not in completed]
    handle = result_path.open("ab")
    try:
        with torch.no_grad():
            for start in range(0, len(pending), args.batch):
                selected = pending[start:start + args.batch]
                prepared = [
                    prepare_request(
                        tokenizer, delimiters, requests[row["workload_id"]],
                        max_length,
                    )
                    for row in selected
                ]
                for item, row in zip(prepared, selected):
                    verify_prepared_against_row(item, row)
                packed_logits = logits_cpu(execute_prepared(model, prepared, "P"))
                batched_logits = logits_cpu(execute_prepared(model, prepared, "B"))
                serial_logits = logits_cpu(execute_prepared(model, prepared, "S"))
                synchronize(args.device)
                for row, item, p_values, b_values, s_values in zip(
                    selected, prepared, packed_logits, batched_logits, serial_logits,
                ):
                    request = item["request"]
                    records = []
                    for index, question in enumerate(request["questions"]):
                        records.append({
                            "kind": "base_equivalence",
                            **binding,
                            "workload_id": row["workload_id"],
                            "parent_id": row["parent_id"],
                            "suite": row["suite"],
                            "question_id": question["id"],
                            "question_index": index,
                            "packed_logits": p_values[index],
                            "comparisons": {
                                arm: [
                                    comparison_metrics(
                                        reference[index], p_values[index], temperature,
                                    )
                                    for temperature in temperatures
                                ]
                                for arm, reference in (
                                    ("B", b_values), ("S", s_values)
                                )
                            },
                        })
                    alone_logits = None
                    if len(request["questions"]) > 1 or (
                        row["workload_id"] in diagnostic_ids
                    ):
                        alone_logits = logits_cpu(model([item["packed"]]))[0]
                    if len(request["questions"]) > 1:
                        reversed_questions = list(reversed(request["questions"]))
                        reversed_encoding = encode_request(
                            tokenizer, request["state"], reversed_questions,
                            max_length, delimiters,
                        )
                        if reversed_encoding is None or (
                            reversed_encoding.retained_state_tokens
                            != reversed_encoding.state_tokens
                        ):
                            raise ValueError("question-order diagnostic truncated")
                        reversed_output = logits_cpu(model([reversed_encoding]))[0]
                        reversed_output.reverse()
                        for index, question in enumerate(request["questions"]):
                            for kind, reference, candidate in (
                                (
                                    "question_order_reversal",
                                    alone_logits[index], reversed_output[index],
                                ),
                                (
                                    "sibling_removal",
                                    s_values[index], alone_logits[index],
                                ),
                            ):
                                records.append({
                                    "kind": kind,
                                    **binding,
                                    "workload_id": row["workload_id"],
                                    "parent_id": row["parent_id"],
                                    "suite": row["suite"],
                                    "question_id": question["id"],
                                    "comparisons": [
                                        comparison_metrics(
                                            reference, candidate, temperature,
                                        ) for temperature in temperatures
                                    ],
                                })
                    if row["workload_id"] in diagnostic_ids:
                        suite_rows = [
                            candidate for candidate in rows
                            if candidate["suite"] == row["suite"]
                            and candidate["workload_id"] != row["workload_id"]
                        ]
                        if suite_rows:
                            neighbor_row = max(
                                suite_rows,
                                key=lambda candidate: (
                                    abs(candidate["packed_length"] - row["packed_length"]),
                                    candidate["workload_id"],
                                ),
                            )
                            neighbor = requests[neighbor_row["workload_id"]]
                            replacement_questions = [
                                request["questions"][0], neighbor["questions"][0],
                            ]
                            replacement = encode_request(
                                tokenizer, request["state"], replacement_questions,
                                max_length, delimiters,
                            )
                            if replacement is None or (
                                replacement.retained_state_tokens
                                != replacement.state_tokens
                            ):
                                raise ValueError("diagnostic sibling replacement truncated")
                            replacement_logits = logits_cpu(model([replacement]))[0][0]
                            records.append({
                                "kind": "unrelated_sibling_replacement",
                                **binding,
                                "workload_id": row["workload_id"],
                                "parent_id": row["parent_id"],
                                "suite": row["suite"],
                                "question_id": request["questions"][0]["id"],
                                "neighbor_workload_id": neighbor_row["workload_id"],
                                "comparisons": [
                                    comparison_metrics(
                                        alone_logits[0], replacement_logits, temperature,
                                    ) for temperature in temperatures
                                ],
                            })
                        neighbor_groups = {
                            "batch_neighbor_shorter": [
                                candidate for candidate in suite_rows
                                if candidate["packed_length"] < row["packed_length"]
                            ],
                            "batch_neighbor_longer": [
                                candidate for candidate in suite_rows
                                if candidate["packed_length"] > row["packed_length"]
                            ],
                        }
                        for kind, candidates in neighbor_groups.items():
                            if not candidates:
                                continue
                            neighbor_row = (
                                min(candidates, key=lambda candidate: (
                                    candidate["packed_length"],
                                    candidate["workload_id"],
                                ))
                                if kind.endswith("shorter")
                                else max(candidates, key=lambda candidate: (
                                    candidate["packed_length"],
                                    candidate["workload_id"],
                                ))
                            )
                            neighbor = requests[neighbor_row["workload_id"]]
                            neighbor_prepared = prepare_request(
                                tokenizer, delimiters, neighbor, max_length,
                            )
                            beside_logits = logits_cpu(model([
                                item["packed"], neighbor_prepared["packed"],
                            ]))[0]
                            for index, question in enumerate(request["questions"]):
                                records.append({
                                    "kind": kind,
                                    **binding,
                                    "workload_id": row["workload_id"],
                                    "parent_id": row["parent_id"],
                                    "suite": row["suite"],
                                    "question_id": question["id"],
                                    "neighbor_workload_id": neighbor_row["workload_id"],
                                    "comparisons": [
                                        comparison_metrics(
                                            alone_logits[index], beside_logits[index],
                                            temperature,
                                        ) for temperature in temperatures
                                    ],
                                })
                    records.append({
                        "kind": "request_complete",
                        **binding,
                        "workload_id": row["workload_id"],
                        "parent_id": row["parent_id"],
                        "suite": row["suite"],
                    })
                    for record in records:
                        handle.write(canonical_bytes(record) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
    finally:
        handle.close()
    result_rows = load_jsonl(result_path)
    completed_requests = [
        row["workload_id"] for row in result_rows
        if row["kind"] == "request_complete"
    ]
    if len(completed_requests) != len(rows) or (
        len(set(completed_requests)) != len(rows)
    ):
        raise RuntimeError("equivalence output does not cover each workload once")
    final_meta = {
        "version": 1,
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "workloads_sha256": meta["workloads_file_sha256"],
        "checkpoint_sha256": protocol["run"]["checkpoint_sha256"],
        "device": args.device,
        "request_batch": args.batch,
        "requests": len(completed_requests),
        "records": len(result_rows),
        "kind_counts": dict(Counter(row["kind"] for row in result_rows)),
        "file_sha256": file_sha256(result_path),
        "runtime": runtime_identity(),
        "locked_test_opened": False,
    }
    atomic_json(meta_path, final_meta)
    return final_meta


def operation_for_batch(
    model, tokenizer, delimiters, prepared_by_id: dict,
    requests: dict, workload_ids: list[str], max_length: int,
    arm: str, scope: str, temperature: float,
):
    if scope == "pretokenized_forward":
        prepared = [prepared_by_id[workload_id] for workload_id in workload_ids]
        return lambda: execute_prepared(model, prepared, arm)
    if scope == "complete_request":
        selected = [requests[workload_id] for workload_id in workload_ids]
        return lambda: execute_complete(
            model, tokenizer, delimiters, selected,
            max_length, arm, temperature,
        )
    raise ValueError(f"unknown timing scope: {scope}")


def ordered_conditions(repetition: int, populations: dict) -> list[tuple]:
    population_names = list(populations)
    if repetition % 2:
        population_names.reverse()
    scopes = list(SCOPES)
    batches = sorted({
        int(batch) for by_batch in populations.values() for batch in by_batch
    })
    if repetition % 2:
        scopes.reverse()
    if repetition == 2:
        batches.reverse()
    return [
        (population, batch, scope)
        for population in population_names
        for batch in batches
        for scope in scopes
    ]


def canonical_conditions(populations: dict) -> list[tuple]:
    return [
        (population, int(batch), scope)
        for population in populations
        for batch in sorted(populations[population], key=int)
        for scope in SCOPES
    ]


def arm_order(repetition: int, condition_index: int) -> list[str]:
    offset = (repetition + condition_index) % len(ARMS)
    return list(ARMS[offset:] + ARMS[:offset])


def execution_shape(
    rows_by_id: dict[str, dict], workload_ids: list[str], arm: str,
) -> dict:
    selected = [rows_by_id[workload_id] for workload_id in workload_ids]
    if arm == "P":
        lengths = [row["packed_length"] for row in selected]
        padded = max(lengths) * len(lengths)
    else:
        lengths = [
            length for row in selected for length in row["separate_lengths"]
        ]
        padded = sum(lengths) if arm == "S" else max(lengths) * len(lengths)
    return {
        "input_sequences": len(lengths),
        "maximum_sequence_length": max(lengths),
        "unpadded_tokens": sum(lengths),
        "padded_tokens": padded,
        "padding_tokens": padded - sum(lengths),
    }


def timing(args) -> dict:
    output = Path(args.out).resolve()
    protocol, rows, meta, schedules = load_frozen(output)
    repetitions = protocol["design"]["timing"]["fresh_process_repetitions"]
    if not 0 <= args.repetition < repetitions:
        raise ValueError("timing repetition is outside the frozen protocol")
    if args.device not in protocol["design"]["timing"]["devices"]:
        raise ValueError("device is not declared by the timing protocol")
    result_path = output / f"timings-{args.device}-r{args.repetition}.jsonl"
    meta_path = output / f"timings-{args.device}-r{args.repetition}.meta.json"
    if result_path.exists() or meta_path.exists():
        raise FileExistsError("refusing to overwrite a timing repetition")
    partial = result_path.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    requests = reload_requests(protocol, rows)
    model, tokenizer, delimiters, _, _, manifest = load_model(
        protocol["run"]["path"], args.device,
    )
    if manifest["current"]["sha256"] != protocol["run"]["checkpoint_sha256"]:
        raise ValueError("loaded checkpoint differs from the protocol")
    max_length = protocol["design"]["checkpoint"]["max_length"]
    temperature = protocol["design"]["checkpoint"]["temperature"]
    selected_ids = {
        workload_id
        for population in schedules["repetitions"][str(args.repetition)].values()
        for schedule in population.values()
        for phase in ("warmup", "measured")
        for batch in schedule[phase]
        for workload_id in batch["workload_ids"]
    }
    rows_by_id = {row["workload_id"]: row for row in rows}
    prepared_by_id = {}
    for workload_id in selected_ids:
        prepared = prepare_request(
            tokenizer, delimiters, requests[workload_id], max_length,
        )
        verify_prepared_against_row(prepared, rows_by_id[workload_id])
        prepared_by_id[workload_id] = prepared
    if args.device == "mps":
        torch.mps.empty_cache()
        synchronize(args.device)
    process_id = PROCESS_INSTANCE_ID
    process_started = time.time()
    raw_rows = []
    populations = schedules["repetitions"][str(args.repetition)]
    conditions = ordered_conditions(args.repetition, populations)
    canonical = canonical_conditions(populations)
    binding = artifact_row_binding(protocol, meta, args.device, schedules)
    with torch.no_grad():
        for condition in conditions:
            population, batch_size, scope = condition
            condition_index = canonical.index(condition)
            schedule = populations[population][str(batch_size)]
            for arm in arm_order(args.repetition, condition_index):
                gc.collect()
                for phase in ("warmup", "measured"):
                    for schedule_index, batch in enumerate(schedule[phase]):
                        operation = operation_for_batch(
                            model, tokenizer, delimiters, prepared_by_id,
                            requests, batch["workload_ids"], max_length,
                            arm, scope, temperature,
                        )
                        synchronize(args.device)
                        started = time.perf_counter_ns()
                        result = operation()
                        synchronize(args.device)
                        duration = time.perf_counter_ns() - started
                        del result
                        if phase == "measured":
                            shape = execution_shape(
                                rows_by_id, batch["workload_ids"], arm,
                            )
                            raw_rows.append({
                                **binding,
                                "process_id": process_id,
                                "repetition": args.repetition,
                                "device": args.device,
                                "population": population,
                                "suite": batch["suite"],
                                "request_batch_size": batch_size,
                                "actual_requests": len(batch["workload_ids"]),
                                "scope": scope,
                                "arm": arm,
                                "schedule_index": schedule_index,
                                "workload_ids": batch["workload_ids"],
                                "parent_ids": batch["parent_ids"],
                                "duration_ns": duration,
                                "ns_per_request": duration / len(batch["workload_ids"]),
                                **shape,
                            })
    atomic_write(partial, jsonl_bytes(raw_rows))
    os.replace(partial, result_path)
    final_meta = {
        "version": 1,
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "workloads_sha256": meta["workloads_file_sha256"],
        "schedules_sha256": schedules["schedules_sha256"],
        "checkpoint_sha256": protocol["run"]["checkpoint_sha256"],
        "process_id": process_id,
        "process_pid": PROCESS_PID,
        "process_started_unix_ns": PROCESS_STARTED_UNIX_NS,
        "process_started_unix": process_started,
        "process_finished_unix": time.time(),
        "repetition": args.repetition,
        "device": args.device,
        "records": len(raw_rows),
        "file_sha256": file_sha256(result_path),
        "runtime": runtime_identity(),
        "load_average_at_finish": list(os.getloadavg()),
        "synchronization": (
            "before and after every timed operation"
            if args.device == "mps" else "CPU operations are synchronous"
        ),
        "locked_test_opened": False,
    }
    atomic_json(meta_path, final_meta)
    return final_meta


def resident_high_water_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def memory_snapshot(device: str) -> dict:
    result = {"rss_high_water_bytes": resident_high_water_bytes()}
    if device == "mps":
        synchronize(device)
        result.update({
            "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "mps_recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        })
    return result


def memory(args) -> dict:
    output = Path(args.out).resolve()
    protocol, rows, meta, schedules = load_frozen(output)
    if args.arm not in ARMS:
        raise ValueError("unknown memory execution arm")
    if args.device not in protocol["design"]["memory"]["devices"]:
        raise ValueError("device is not declared by the memory protocol")
    result_path = output / f"memory-{args.device}-{args.arm}.jsonl"
    meta_path = output / f"memory-{args.device}-{args.arm}.meta.json"
    if result_path.exists() or meta_path.exists():
        raise FileExistsError("refusing to overwrite a memory pass")
    requests = reload_requests(protocol, rows)
    model, tokenizer, delimiters, _, _, manifest = load_model(
        protocol["run"]["path"], args.device,
    )
    if manifest["current"]["sha256"] != protocol["run"]["checkpoint_sha256"]:
        raise ValueError("loaded checkpoint differs from the protocol")
    max_length = protocol["design"]["checkpoint"]["max_length"]
    temperature = protocol["design"]["checkpoint"]["temperature"]
    memory_design = protocol["design"]["memory"]
    memory_schedule = schedules["memory"]
    total = memory_schedule["requests"]
    batch_size = memory_schedule["request_batch_size"]
    batches = memory_schedule["batches"]
    binding = artifact_row_binding(protocol, meta, args.device, schedules)
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
        synchronize(args.device)
    baseline = memory_snapshot(args.device)
    observations = []
    with torch.no_grad():
        for index, batch in enumerate(batches):
            ids = batch["workload_ids"]
            result = execute_complete(
                model, tokenizer, delimiters,
                [requests[workload_id] for workload_id in ids],
                max_length, args.arm, temperature,
            )
            del result
            snapshot = memory_snapshot(args.device)
            observations.append({
                **binding,
                "arm": args.arm,
                "batch_index": index,
                "suite": batch["suite"],
                "workload_ids": ids,
                "parent_ids": batch["parent_ids"],
                **snapshot,
            })
    synchronize(args.device)
    gc.collect()
    before_cache_release = memory_snapshot(args.device)
    if args.device == "mps":
        torch.mps.empty_cache()
        synchronize(args.device)
    after_cache_release = memory_snapshot(args.device)
    residual = None
    if args.device == "mps":
        residual = (
            after_cache_release["mps_current_allocated_bytes"]
            - baseline["mps_current_allocated_bytes"]
        )
    atomic_write(result_path, jsonl_bytes(observations))
    final_meta = {
        "version": 1,
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "workloads_sha256": meta["workloads_file_sha256"],
        "schedules_sha256": schedules["schedules_sha256"],
        "checkpoint_sha256": protocol["run"]["checkpoint_sha256"],
        "device": args.device,
        "arm": args.arm,
        "scope": memory_design["scope"],
        "requests": total,
        "request_batch_size": batch_size,
        "baseline": baseline,
        "observed_maximum": {
            key: max(row[key] for row in observations)
            for key in observations[0]
            if key.endswith("_bytes")
        },
        "before_cache_release": before_cache_release,
        "after_cache_release": after_cache_release,
        "residual_live_tensor_increase_bytes": residual,
        "residual_flag": bool(
            residual is not None and residual
            > memory_design["maximum_residual_live_tensor_increase_bytes"]
        ),
        "file_sha256": file_sha256(result_path),
        "runtime": runtime_identity(),
        "locked_test_opened": False,
    }
    atomic_json(meta_path, final_meta)
    return final_meta


def aggregate_equivalence(path: Path, design: dict) -> dict:
    rows = [
        row for row in load_jsonl(path)
        if row["kind"] not in {"journal_header", "request_complete"}
    ]
    threshold = design["equivalence"]
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] == "base_equivalence":
            for arm, comparisons in row["comparisons"].items():
                for value in comparisons:
                    groups[(row["kind"], arm, str(value["temperature"]))].append(value)
        else:
            for value in row["comparisons"]:
                groups[(row["kind"], "P", str(value["temperature"]))].append(value)
    report = {}
    for key, values in sorted(groups.items()):
        total_variation = [value["total_variation"] for value in values]
        log_difference = [
            value["max_absolute_log_probability_difference"] for value in values
        ]
        material_action_changes = sum(
            value["action_changed"] and value["reference_top_two_margin"]
            > threshold["minimum_reference_top_two_margin"]
            for value in values
        )
        near_tie_action_changes = sum(
            value["action_changed"] and value["reference_top_two_margin"]
            <= threshold["minimum_reference_top_two_margin"]
            for value in values
        )
        metrics = {
            "n": len(values),
            "maximum_total_variation": max(total_variation),
            "p99_total_variation": quantile(total_variation, 0.99),
            "maximum_absolute_log_probability_difference": max(log_difference),
            "material_action_changes": material_action_changes,
            "near_tie_action_changes": near_tie_action_changes,
        }
        metrics["passed"] = bool(
            metrics["maximum_total_variation"] <= threshold["max_total_variation"]
            and metrics["p99_total_variation"] <= threshold["p99_total_variation"]
            and metrics["maximum_absolute_log_probability_difference"]
            <= threshold["max_absolute_log_probability_difference"]
            and material_action_changes
            <= threshold["allowed_action_changes_above_margin"]
        )
        report["/".join(key)] = metrics
    return report


def cross_device_equivalence(cpu_path: Path, mps_path: Path, design: dict) -> dict:
    def packed_outputs(path: Path) -> dict:
        result = {}
        for row in load_jsonl(path):
            if row["kind"] != "base_equivalence":
                continue
            key = (row["workload_id"], row["question_id"])
            if key in result:
                raise ValueError("cross-device equivalence key is duplicated")
            result[key] = row["packed_logits"]
        return result

    cpu = packed_outputs(cpu_path)
    mps = packed_outputs(mps_path)
    if set(cpu) != set(mps):
        raise ValueError("CPU and MPS equivalence populations differ")
    report = {}
    for temperature in design["equivalence"]["temperatures"]:
        values = [
            comparison_metrics(cpu[key], mps[key], temperature)
            for key in sorted(cpu)
        ]
        total_variation = [value["total_variation"] for value in values]
        log_difference = [
            value["max_absolute_log_probability_difference"] for value in values
        ]
        report[str(temperature)] = {
            "n": len(values),
            "reference": "CPU packed execution",
            "candidate": "MPS packed execution",
            "maximum_total_variation": max(total_variation),
            "p99_total_variation": quantile(total_variation, 0.99),
            "maximum_absolute_log_probability_difference": max(log_difference),
            "action_changes": sum(value["action_changed"] for value in values),
            "scope": "descriptive only; same-device substitution gates remain primary",
        }
    return report


def validate_artifact_binding(
    path: Path, meta_path: Path, protocol: dict, workload_meta: dict,
    schedules: dict | None = None,
) -> dict:
    metadata = json.loads(meta_path.read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"artifact is not completed: {path.name}")
    if metadata.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError(f"artifact protocol binding differs: {path.name}")
    if metadata.get("workloads_sha256") != workload_meta["workloads_file_sha256"]:
        raise ValueError(f"artifact workload binding differs: {path.name}")
    if metadata.get("checkpoint_sha256") != protocol["run"]["checkpoint_sha256"]:
        raise ValueError(f"artifact checkpoint binding differs: {path.name}")
    if metadata.get("file_sha256") != file_sha256(path):
        raise ValueError(f"artifact file digest differs: {path.name}")
    if schedules is not None and (
        metadata.get("schedules_sha256") != schedules["schedules_sha256"]
    ):
        raise ValueError(f"artifact schedule binding differs: {path.name}")
    if metadata.get("locked_test_opened") is not False:
        raise ValueError(f"artifact does not record a closed locked test: {path.name}")
    return metadata


def expected_equivalence_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(key[0] for key in expected_equivalence_inventory(rows))
    counts["journal_header"] = 1
    return dict(counts)


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_comparison_value(value: dict, temperature: float) -> None:
    if value.get("temperature") != temperature:
        raise ValueError("equivalence comparison temperature differs")
    reference = value.get("reference_probabilities")
    candidate = value.get("candidate_probabilities")
    if (
        not isinstance(reference, list) or not isinstance(candidate, list)
        or len(reference) < 2 or len(reference) != len(candidate)
        or any(not finite_number(item) or item < 0 for item in reference + candidate)
    ):
        raise ValueError("equivalence comparison probabilities are invalid")
    if not math.isclose(sum(reference), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("equivalence reference probabilities do not sum to one")
    if not math.isclose(sum(candidate), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("equivalence candidate probabilities do not sum to one")
    total_variation = 0.5 * sum(
        abs(left - right) for left, right in zip(reference, candidate)
    )
    reference_action = max(range(len(reference)), key=reference.__getitem__)
    candidate_action = max(range(len(candidate)), key=candidate.__getitem__)
    top = sorted(reference, reverse=True)[:2]
    checks = {
        "total_variation": total_variation,
        "reference_top_two_margin": top[0] - top[1],
    }
    for key, expected in checks.items():
        if not finite_number(value.get(key)) or not math.isclose(
            value[key], expected, rel_tol=1e-9, abs_tol=1e-12,
        ):
            raise ValueError(f"equivalence comparison {key} is inconsistent")
    if value.get("reference_action") != reference_action:
        raise ValueError("equivalence reference action is inconsistent")
    if value.get("candidate_action") != candidate_action:
        raise ValueError("equivalence candidate action is inconsistent")
    if not isinstance(value.get("action_changed"), bool) or (
        value["action_changed"] != (reference_action != candidate_action)
    ):
        raise ValueError("equivalence action-change flag is inconsistent")
    log_difference = value.get("max_absolute_log_probability_difference")
    if not finite_number(log_difference) or log_difference < 0:
        raise ValueError("equivalence log-probability difference is invalid")
    if min(reference + candidate) > 0:
        expected_log_difference = max(
            abs(math.log(left) - math.log(right))
            for left, right in zip(reference, candidate)
        )
        if not math.isclose(
            log_difference, expected_log_difference, rel_tol=1e-8, abs_tol=1e-11,
        ):
            raise ValueError("equivalence log-probability difference is inconsistent")


def validate_equivalence_coverage(
    path: Path, metadata: dict, workload_rows: list[dict], design: dict,
    protocol: dict, workload_meta: dict,
) -> None:
    rows = load_jsonl(path)
    actual = Counter(row["kind"] for row in rows)
    expected = expected_equivalence_counts(workload_rows)
    if dict(actual) != expected:
        raise ValueError(f"equivalence coverage differs on {metadata['device']}")
    if metadata.get("kind_counts") != expected:
        raise ValueError(f"equivalence metadata coverage differs on {metadata['device']}")
    if metadata.get("records") != len(rows):
        raise ValueError("equivalence metadata record count is inconsistent")
    if metadata.get("requests") != len(workload_rows):
        raise ValueError("equivalence metadata request count is inconsistent")
    expected = expected_equivalence_inventory(workload_rows)
    header = rows[0] if rows else {}
    if header.get("kind") != "journal_header":
        raise ValueError("equivalence artifact lacks its journal header")
    validate_row_binding(
        header, protocol, workload_meta, metadata["device"],
    )
    keys = set()
    temperatures = design["equivalence"]["temperatures"]
    for row in rows[1:]:
        validate_row_binding(
            row, protocol, workload_meta, metadata["device"],
        )
        key = equivalence_record_key(row)
        if key not in expected:
            raise ValueError("equivalence result identity is not frozen")
        if key in keys:
            raise ValueError("equivalence result key is duplicated")
        keys.add(key)
        details = expected[key]
        if row.get("parent_id") != details["parent_id"]:
            raise ValueError("equivalence result parent differs from the workload")
        if row.get("suite") != details["suite"]:
            raise ValueError("equivalence result suite differs from the workload")
        for field in ("question_index", "neighbor_workload_id"):
            if field in details and row.get(field) != details[field]:
                raise ValueError(f"equivalence result {field} differs")
        if row["kind"] == "request_complete":
            continue
        comparisons = row.get("comparisons")
        if row["kind"] == "base_equivalence":
            if not isinstance(comparisons, dict) or set(comparisons) != {"B", "S"}:
                raise ValueError("base equivalence omits an execution arm")
            if (
                not isinstance(row.get("packed_logits"), list)
                or len(row["packed_logits"]) < 2
                or any(not finite_number(value) for value in row["packed_logits"])
            ):
                raise ValueError("base equivalence packed logits are invalid")
            groups = comparisons.values()
        else:
            groups = [comparisons]
        for group in groups:
            if not isinstance(group, list) or len(group) != len(temperatures):
                raise ValueError("equivalence comparison count differs from protocol")
            for item, temperature in zip(group, temperatures):
                validate_comparison_value(item, temperature)
    if keys != set(expected):
        raise ValueError("equivalence artifact does not match the frozen inventory")


def validate_timing_coverage(
    path: Path, metadata: dict, schedules: dict, protocol: dict,
    workload_meta: dict, workload_rows: list[dict],
) -> None:
    rows = load_jsonl(path)
    repetition = str(metadata["repetition"])
    populations = schedules["repetitions"][repetition]
    canonical = canonical_conditions(populations)
    expected = []
    for condition in ordered_conditions(metadata["repetition"], populations):
        population, batch, scope = condition
        schedule = populations[population][str(batch)]["measured"]
        for arm in arm_order(metadata["repetition"], canonical.index(condition)):
            for index, item in enumerate(schedule):
                expected.append((population, batch, scope, arm, index, item))
    if len(rows) != len(expected):
        raise ValueError("timing row count differs from the frozen schedule")
    rows_by_id = {row["workload_id"]: row for row in workload_rows}
    seen = set()
    for row, (population, batch, scope, arm, index, item) in zip(rows, expected):
        validate_row_binding(
            row, protocol, workload_meta, metadata["device"], schedules,
        )
        key = (population, batch, scope, arm, index)
        actual_key = (
            row.get("population"), row.get("request_batch_size"), row.get("scope"),
            row.get("arm"), row.get("schedule_index"),
        )
        if actual_key != key:
            raise ValueError("timing rows are not in the frozen execution order")
        if key in seen:
            raise ValueError("timing condition row is duplicated")
        seen.add(key)
        for field in ("suite", "workload_ids", "parent_ids"):
            if row.get(field) != item[field]:
                raise ValueError(f"timing row {field} differs from the schedule")
        if row.get("repetition") != metadata["repetition"]:
            raise ValueError("timing row repetition differs from metadata")
        if row["process_id"] != metadata["process_id"]:
            raise ValueError("timing rows contain another process identity")
        actual_requests = len(item["workload_ids"])
        if row.get("actual_requests") != actual_requests:
            raise ValueError("timing row request count is inconsistent")
        duration = row.get("duration_ns")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
            raise ValueError("timing duration must be a positive integer")
        if not finite_number(row.get("ns_per_request")) or not math.isclose(
            row["ns_per_request"], duration / actual_requests,
            rel_tol=1e-12, abs_tol=0.0,
        ):
            raise ValueError("timing per-request duration is inconsistent")
        shape = execution_shape(rows_by_id, item["workload_ids"], arm)
        for field, value in shape.items():
            if row.get(field) != value:
                raise ValueError(f"timing execution shape {field} is inconsistent")
    if metadata.get("records") != len(rows):
        raise ValueError("timing metadata record count is inconsistent")


def validate_memory_snapshot(snapshot: dict, device: str) -> None:
    required = {"rss_high_water_bytes"}
    if device == "mps":
        required.update({
            "mps_current_allocated_bytes", "mps_driver_allocated_bytes",
            "mps_recommended_max_bytes",
        })
    if not required.issubset(snapshot):
        raise ValueError("memory snapshot omits a required counter")
    if any(
        not isinstance(snapshot[key], int) or isinstance(snapshot[key], bool)
        or snapshot[key] < 0
        for key in required
    ):
        raise ValueError("memory snapshot counter is invalid")


def validate_memory_coverage(
    path: Path, metadata: dict, schedules: dict, protocol: dict,
    workload_meta: dict,
) -> dict:
    rows = load_jsonl(path)
    design = protocol["design"]["memory"]
    memory_schedule = schedules["memory"]
    expected = memory_schedule["batches"]
    if metadata.get("requests") != design["steady_state_requests"]:
        raise ValueError("memory metadata request count differs from the protocol")
    if memory_schedule["requests"] != design["steady_state_requests"]:
        raise ValueError("memory schedule request count differs from the protocol")
    if metadata.get("request_batch_size") != design["request_batch_size"]:
        raise ValueError("memory metadata batch size differs from the protocol")
    if metadata.get("scope") != design["scope"]:
        raise ValueError("memory scope differs from the protocol")
    if len(rows) != len(expected):
        raise ValueError("memory observations do not cover the steady-state pass")
    for index, (row, batch) in enumerate(zip(rows, expected)):
        validate_row_binding(
            row, protocol, workload_meta, metadata["device"], schedules,
        )
        if row.get("batch_index") != index:
            raise ValueError("memory observation indexes are not contiguous")
        if row.get("arm") != metadata["arm"]:
            raise ValueError("memory observation arm differs from metadata")
        for field in ("suite", "workload_ids", "parent_ids"):
            if row.get(field) != batch[field]:
                raise ValueError(f"memory observation {field} differs from schedule")
        validate_memory_snapshot(row, metadata["device"])
    for name in ("baseline", "before_cache_release", "after_cache_release"):
        snapshot = metadata.get(name)
        if not isinstance(snapshot, dict):
            raise ValueError(f"memory metadata omits {name}")
        validate_memory_snapshot(snapshot, metadata["device"])
    observed_maximum = {
        key: max(row[key] for row in rows)
        for key in rows[0]
        if key.endswith("_bytes")
    }
    if metadata.get("observed_maximum") != observed_maximum:
        raise ValueError("memory observed maximum is inconsistent")
    residual = None
    if metadata["device"] == "mps":
        residual = (
            metadata["after_cache_release"]["mps_current_allocated_bytes"]
            - metadata["baseline"]["mps_current_allocated_bytes"]
        )
    if metadata.get("residual_live_tensor_increase_bytes") != residual:
        raise ValueError("memory residual is inconsistent")
    residual_flag = bool(
        residual is not None
        and residual > design["maximum_residual_live_tensor_increase_bytes"]
    )
    if not isinstance(metadata.get("residual_flag"), bool) or (
        metadata["residual_flag"] != residual_flag
    ):
        raise ValueError("memory residual flag is inconsistent")
    return {"residual": residual, "residual_flag": residual_flag}


def paired_ratio_bootstrap(
    records: list[dict], draws: int, seed: int,
) -> dict:
    import random

    paired = defaultdict(lambda: defaultdict(list))
    parent_by_workload = {}
    for record in records:
        if record["request_batch_size"] != 1:
            continue
        if len(record.get("workload_ids", [])) != 1 or len(record.get("parent_ids", [])) != 1:
            raise ValueError("timing bootstrap requires one workload and parent per record")
        workload_id = record["workload_ids"][0]
        parent_id = record["parent_ids"][0]
        previous = parent_by_workload.setdefault(workload_id, parent_id)
        if previous != parent_id:
            raise ValueError("timing workload maps to multiple parents")
        key = (record["repetition"], workload_id, parent_id)
        paired[key][record["arm"]].append(record["duration_ns"])
    per_repetition = {}
    for key, arms in paired.items():
        if set(arms) != {"P", "B"}:
            raise ValueError("timing bootstrap has an unpaired execution arm")
        if len(arms["P"]) != len(arms["B"]):
            raise ValueError("timing bootstrap observation counts differ by arm")
        per_repetition[key] = {
            arm: statistics.fmean(values) for arm, values in arms.items()
        }
    if not per_repetition:
        raise ValueError("timing bootstrap has no paired parents")
    repetitions_by_workload = defaultdict(set)
    for repetition, workload_id, _ in per_repetition:
        repetitions_by_workload[workload_id].add(repetition)
    repetition_sets = {tuple(sorted(value)) for value in repetitions_by_workload.values()}
    if len(repetition_sets) != 1:
        raise ValueError("timing bootstrap repetitions differ across workloads")
    by_workload = {}
    for workload_id, parent_id in sorted(parent_by_workload.items()):
        values = [
            arms for (repetition, identity, parent), arms in per_repetition.items()
            if identity == workload_id and parent == parent_id
        ]
        by_workload[workload_id] = {
            "parent_id": parent_id,
            "P": statistics.fmean(value["P"] for value in values),
            "B": statistics.fmean(value["B"] for value in values),
        }
    by_parent = defaultdict(list)
    for workload_id, value in by_workload.items():
        by_parent[value["parent_id"]].append(workload_id)
    point = sum(value["P"] for value in by_workload.values()) / sum(
        value["B"] for value in by_workload.values()
    )
    rng = random.Random(seed)
    identities = sorted(by_parent)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(identities) for _ in identities]
        numerator = denominator = 0.0
        for parent_id in selected:
            for workload_id in by_parent[parent_id]:
                numerator += by_workload[workload_id]["P"]
                denominator += by_workload[workload_id]["B"]
        samples.append(numerator / denominator)
    return {
        "parents": len(by_parent),
        "workloads": len(by_workload),
        "paired_observations": sum(len(arms["P"]) for arms in paired.values()),
        "ratio": point,
        "interval_95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "draws": draws,
        "seed": seed,
    }


def aggregate_timings(paths: list[Path], design: dict) -> dict:
    rows = [row for path in paths for row in load_jsonl(path)]
    gate = design["timing"]["primary_gate"]
    report = {}
    for suite in ("decision", "korean"):
        selected = [
            row for row in rows
            if row["device"] == "mps"
            and row["population"] == "multi_primary"
            and row["suite"] == suite
            and row["request_batch_size"] == 1
            and row["scope"] == "complete_request"
            and row["arm"] in {"P", "B"}
        ]
        bootstrap = paired_ratio_bootstrap(
            selected, gate["bootstrap_draws"], gate["bootstrap_seed"],
        )
        by_arm = {
            arm: [row["ns_per_request"] for row in selected if row["arm"] == arm]
            for arm in ("P", "B")
        }
        p95_ratio = quantile(by_arm["P"], 0.95) / quantile(by_arm["B"], 0.95)
        repetition_ratios = {}
        for repetition in range(design["timing"]["fresh_process_repetitions"]):
            values = {
                arm: [
                    row["ns_per_request"] for row in selected
                    if row["arm"] == arm and row["repetition"] == repetition
                ] for arm in ("P", "B")
            }
            repetition_ratios[str(repetition)] = (
                statistics.fmean(values["P"]) / statistics.fmean(values["B"])
            )
        report[suite] = {
            "paired_parent": bootstrap,
            "p95_latency_ratio": p95_ratio,
            "repetition_mean_ratios": repetition_ratios,
            "passed": bool(
                bootstrap["ratio"] <= gate["maximum_mean_latency_ratio"]
                and bootstrap["interval_95"][1]
                < gate["maximum_paired_parent_95_percent_upper_bound"]
                and p95_ratio <= gate["maximum_p95_latency_ratio"]
            ),
            "replicated_regression": all(
                ratio > gate["replicated_regression_ratio"]
                for ratio in repetition_ratios.values()
            ),
        }
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            row["device"], row["population"], row["suite"],
            str(row["request_batch_size"]), row["scope"], row["arm"],
        )].append(row["ns_per_request"] / 1_000_000)
    report["distributions"] = {
        "/".join(key): {
            "n": len(values),
            "mean_ms_per_request": statistics.fmean(values),
            "median_ms_per_request": statistics.median(values),
            "p95_ms_per_request": quantile(values, 0.95),
        }
        for key, values in sorted(grouped.items())
    }
    return report


def validate_runtime_compatibility(
    metadata: list[dict], expected: dict | None = None,
) -> None:
    if not metadata:
        raise ValueError("benchmark has no runtime metadata")
    reference = metadata[0].get("runtime")
    if not isinstance(reference, dict):
        raise ValueError("benchmark artifact omits runtime identity")
    if expected is not None and reference != expected:
        raise ValueError("benchmark runtime differs from the frozen runtime")
    for item in metadata[1:]:
        if item.get("runtime") != reference:
            raise ValueError("benchmark artifacts use inconsistent runtimes")


def summarize(args) -> dict:
    output = Path(args.out).resolve()
    protocol, workload_rows, meta, schedules = load_frozen(output)
    design = protocol["design"]
    equivalence_paths = {
        device: output / f"equivalence-{device}.jsonl"
        for device in design["timing"]["devices"]
    }
    timing_paths = [
        output / f"timings-{device}-r{repetition}.jsonl"
        for device in design["timing"]["devices"]
        for repetition in range(design["timing"]["fresh_process_repetitions"])
    ]
    equivalence_meta_paths = {
        device: output / f"equivalence-{device}.meta.json"
        for device in design["timing"]["devices"]
    }
    timing_meta_paths = [path.with_suffix(".meta.json") for path in timing_paths]
    memory_paths = [
        output / f"memory-{device}-{arm}.jsonl"
        for device in design["memory"]["devices"] for arm in ARMS
    ]
    memory_meta_paths = [
        output / f"memory-{device}-{arm}.meta.json"
        for device in design["memory"]["devices"] for arm in ARMS
    ]
    required = (
        list(equivalence_paths.values()) + list(equivalence_meta_paths.values())
        + timing_paths + timing_meta_paths + memory_paths + memory_meta_paths
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing benchmark artifacts: " + ", ".join(missing))
    equivalence_metadata = [
        validate_artifact_binding(
            equivalence_paths[device], equivalence_meta_paths[device],
            protocol, meta,
        )
        for device in design["timing"]["devices"]
    ]
    timing_metadata = [
        validate_artifact_binding(
            path, metadata, protocol, meta, schedules,
        )
        for path, metadata in zip(timing_paths, timing_meta_paths)
    ]
    memory_reports = [
        validate_artifact_binding(
            path, metadata, protocol, meta, schedules,
        )
        for path, metadata in zip(memory_paths, memory_meta_paths)
    ]
    for device, metadata in zip(
        design["timing"]["devices"], equivalence_metadata,
    ):
        if metadata.get("device") != device:
            raise ValueError("equivalence artifact device differs")
        validate_equivalence_coverage(
            equivalence_paths[device], metadata, workload_rows, design,
            protocol, meta,
        )
    timing_identities = [
        (device, repetition)
        for device in design["timing"]["devices"]
        for repetition in range(design["timing"]["fresh_process_repetitions"])
    ]
    for path, metadata, (device, repetition) in zip(
        timing_paths, timing_metadata, timing_identities,
    ):
        if metadata.get("device") != device or metadata.get("repetition") != repetition:
            raise ValueError("timing artifact identity differs")
        validate_timing_coverage(
            path, metadata, schedules, protocol, meta, workload_rows,
        )
    memory_identities = [
        (device, arm)
        for device in design["memory"]["devices"] for arm in ARMS
    ]
    validated_memory = []
    for path, metadata, (device, arm) in zip(
        memory_paths, memory_reports, memory_identities,
    ):
        if metadata.get("device") != device or metadata.get("arm") != arm:
            raise ValueError("memory artifact identity differs")
        validated_memory.append(validate_memory_coverage(
            path, metadata, schedules, protocol, meta,
        ))
    validate_runtime_compatibility(
        equivalence_metadata + timing_metadata + memory_reports,
        protocol["runtime_at_freeze"],
    )
    process_ids = {
        (
            item.get("process_id"), item.get("process_pid"),
            item.get("process_started_unix_ns"),
        )
        for item in timing_metadata
    }
    if any(None in identity for identity in process_ids):
        raise ValueError("timing artifact omits process identity")
    if len(process_ids) != len(timing_metadata):
        raise ValueError("timing repetitions did not use distinct fresh processes")
    equivalence_report = {
        device: aggregate_equivalence(path, design)
        for device, path in equivalence_paths.items()
    }
    cross_device = cross_device_equivalence(
        equivalence_paths["cpu"], equivalence_paths["mps"], design,
    )
    equivalence_passed = all(
        group["passed"]
        for device in equivalence_report.values() for group in device.values()
    )
    timing_report = aggregate_timings(timing_paths, design)
    memory_passed = all(
        not validation["residual_flag"]
        for report, validation in zip(memory_reports, validated_memory)
        if report["device"] == "mps"
    )
    timing_passed = all(timing_report[suite]["passed"] for suite in ("decision", "korean"))
    if not equivalence_passed:
        outcome = "equivalence_failure"
    elif timing_passed and memory_passed:
        outcome = "equivalent_and_faster"
    else:
        outcome = "equivalent_but_not_faster"
    combined_paths = {
        "equivalence": output / "equivalence.jsonl",
        "timings": output / "timings.jsonl",
        "memory": output / "memory.jsonl",
    }
    destination = output / "summary.json"
    if destination.exists() or any(path.exists() for path in combined_paths.values()):
        raise FileExistsError("refusing to overwrite benchmark summary artifacts")
    atomic_write(
        combined_paths["equivalence"],
        jsonl_bytes([
            row for path in equivalence_paths.values() for row in load_jsonl(path)
        ]),
    )
    atomic_write(
        combined_paths["timings"],
        jsonl_bytes([row for path in timing_paths for row in load_jsonl(path)]),
    )
    atomic_write(
        combined_paths["memory"],
        jsonl_bytes([row for path in memory_paths for row in load_jsonl(path)]),
    )
    source_files = required + list(combined_paths.values()) + [
        output / "protocol.json", output / "workloads.jsonl",
        output / "workloads.meta.json", output / "schedules.json",
    ]
    report = {
        "version": 1,
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "workloads_sha256": meta["workloads_file_sha256"],
        "schedules_sha256": schedules["schedules_sha256"],
        "checkpoint_sha256": protocol["run"]["checkpoint_sha256"],
        "equivalence": equivalence_report,
        "cross_device_packed": cross_device,
        "equivalence_artifacts": equivalence_metadata,
        "equivalence_passed": equivalence_passed,
        "timing": timing_report,
        "timing_passed": timing_passed,
        "memory": memory_reports,
        "memory_passed": memory_passed,
        "outcome": outcome,
        "following_experiment": design["decision"][outcome],
        "artifacts": {path.name: file_sha256(path) for path in source_files},
        "runtime": runtime_identity(),
        "locked_test_opened": False,
    }
    report["report_sha256"] = value_sha256(report)
    atomic_json(destination, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--run", required=True)
    freeze_parser.add_argument("--comparison-plan", required=True)
    freeze_parser.add_argument("--development-report", required=True)
    freeze_parser.add_argument("--decision-suite", required=True)
    freeze_parser.add_argument("--transfer-suite", required=True)
    freeze_parser.add_argument("--korean-suite", required=True)
    freeze_parser.add_argument("--out", required=True)
    equivalence_parser = commands.add_parser("equivalence")
    equivalence_parser.add_argument("--out", required=True)
    equivalence_parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    equivalence_parser.add_argument("--batch", type=int, default=2)
    timing_parser = commands.add_parser("timing")
    timing_parser.add_argument("--out", required=True)
    timing_parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    timing_parser.add_argument("--repetition", type=int, required=True)
    memory_parser = commands.add_parser("memory")
    memory_parser.add_argument("--out", required=True)
    memory_parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    memory_parser.add_argument("--arm", choices=ARMS, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--out", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "freeze":
        result = freeze(args)
    elif args.command == "equivalence":
        result = equivalence(args)
    elif args.command == "timing":
        result = timing(args)
    elif args.command == "memory":
        result = memory(args)
    else:
        result = summarize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
