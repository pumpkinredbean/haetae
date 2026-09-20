"""Confirm warmed MPS batch-one packed execution in six fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch

from experiments import benchmark_shared_execution as base
from experiments.evaluate_shared import load_model
from experiments.kev_adapter import file_sha256


DESIGN_PATH = Path(__file__).with_name(
    "shared_execution_confirmation_protocol.json"
)
RUN_SCRIPT_PATH = Path(__file__).with_name(
    "run_shared_execution_confirmation.zsh"
)
PROCESS_ID = uuid.uuid4().hex
PROCESS_PID = os.getpid()
PROCESS_STARTED_UNIX_NS = time.time_ns()


def load_design() -> dict:
    design = json.loads(DESIGN_PATH.read_text())
    if design.get("version") != 1:
        raise ValueError("unsupported confirmation protocol version")
    if design.get("device") != "mps" or design.get("arms") != ["P", "B"]:
        raise ValueError("confirmation protocol must compare P and B on MPS")
    if design.get("fresh_processes") != 6:
        raise ValueError("confirmation protocol requires six processes")
    if design.get("locked_test_opened") is not False:
        raise ValueError("confirmation protocol opened a locked test")
    return design


def runner_identity(repository: Path) -> dict:
    paths = [Path(__file__).resolve(), DESIGN_PATH.resolve(), RUN_SCRIPT_PATH.resolve()]
    digest = hashlib.sha256()
    files = {}
    for path in paths:
        relative = str(path.relative_to(repository))
        content = path.read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"sha256": digest.hexdigest(), "files": files, "git_commit": commit}


def verify_self_digest(value: dict, key: str, label: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not isinstance(recorded, str) or base.value_sha256(unsigned) != recorded:
        raise ValueError(f"{label} self-digest differs")


def original_summary_binding(path: Path) -> dict:
    summary = json.loads(path.read_text())
    verify_self_digest(summary, "report_sha256", "original benchmark summary")
    if summary.get("locked_test_opened") is not False:
        raise ValueError("original benchmark summary opened a locked test")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "report_sha256": summary["report_sha256"],
        "outcome": summary["outcome"],
    }


def complementary_triplets() -> list[tuple[int, ...]]:
    all_processes = set(range(6))
    result = []
    seen = set()
    for combination in itertools.combinations(range(6), 3):
        if combination in seen:
            continue
        complement = tuple(sorted(all_processes.difference(combination)))
        result.extend([combination, complement])
        seen.add(combination)
        seen.add(complement)
    if len(result) != 20:
        raise RuntimeError("failed to construct balanced process triplets")
    return result


def digest_rank(seed: int, label: str, workload_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{label}:{workload_id}".encode()
    ).hexdigest()


def selected_source_rows(
    source_schedules: dict, source_rows: list[dict], design: dict
) -> list[dict]:
    rows_by_id = {row["workload_id"]: row for row in source_rows}
    repetition = source_schedules["repetitions"]["0"]
    selected = []
    for batch in repetition["multi_primary"]["1"]["measured"]:
        if len(batch["workload_ids"]) != 1:
            raise ValueError("source primary schedule is not batch one")
        workload_id = batch["workload_ids"][0]
        suite = batch["suite"]
        if suite not in {"decision", "korean"}:
            raise ValueError("source primary schedule contains another suite")
        selected.append({
            "population": suite,
            "suite": suite,
            "workload_id": workload_id,
            "parent_id": batch["parent_ids"][0],
        })
    for batch in repetition["single_control"]["1"]["measured"]:
        if len(batch["workload_ids"]) != 1:
            raise ValueError("source control schedule is not batch one")
        selected.append({
            "population": "single_control",
            "suite": batch["suite"],
            "workload_id": batch["workload_ids"][0],
            "parent_id": batch["parent_ids"][0],
        })
    expected = design["populations"]
    observed = {
        population: sum(item["population"] == population for item in selected)
        for population in expected
    }
    if observed != expected:
        raise ValueError("source schedule population counts differ")
    if len({item["workload_id"] for item in selected}) != len(selected):
        raise ValueError("confirmation workload identities are not unique")
    for population in design["populations"]:
        parents = [
            item["parent_id"]
            for item in selected
            if item["population"] == population
        ]
        if len(set(parents)) != len(parents):
            raise ValueError("confirmation population repeats a parent")
    for item in selected:
        row = rows_by_id.get(item["workload_id"])
        if row is None or row["parent_id"] != item["parent_id"]:
            raise ValueError("source workload or parent binding differs")
        if item["population"] == "single_control" and row["question_count"] != 1:
            raise ValueError("negative control is not single-question")
        if item["population"] != "single_control" and row["question_count"] <= 1:
            raise ValueError("primary confirmation request is not multi-question")
    return selected


def build_schedule(selected: list[dict], design: dict) -> dict:
    seed = design["uncertainty"]["process_parent_bootstrap_seed"]
    triplets = complementary_triplets()
    assignment = {}
    for population in design["populations"]:
        members = sorted(
            (item for item in selected if item["population"] == population),
            key=lambda item: digest_rank(seed, f"assignment:{population}", item["workload_id"]),
        )
        for index, item in enumerate(members):
            assignment[item["workload_id"]] = triplets[index % len(triplets)]
    processes = {}
    for process_index in range(design["fresh_processes"]):
        ordered = sorted(
            selected,
            key=lambda item: digest_rank(
                seed, f"order:{process_index}", item["workload_id"]
            ),
        )
        processes[str(process_index)] = [
            {
                **item,
                "schedule_index": index,
                "first_arm": (
                    "P" if process_index in assignment[item["workload_id"]] else "B"
                ),
            }
            for index, item in enumerate(ordered)
        ]
    schedule = {
        "version": 1,
        "source_schedules_sha256": None,
        "processes": processes,
        "locked_test_opened": False,
    }
    return schedule


def validate_schedule(schedule: dict, protocol: dict) -> None:
    design = protocol["design"]
    if schedule.get("locked_test_opened") is not False:
        raise ValueError("confirmation schedule opened a locked test")
    if schedule.get("source_schedules_sha256") != protocol["source"][
        "schedules_sha256"
    ]:
        raise ValueError("confirmation schedule source binding differs")
    unsigned = dict(schedule)
    recorded_sha = unsigned.pop("schedule_sha256", None)
    if recorded_sha != protocol["schedule_sha256"] or base.value_sha256(unsigned) != recorded_sha:
        raise ValueError("confirmation schedule digest differs")
    processes = schedule.get("processes", {})
    expected_processes = {str(index) for index in range(design["fresh_processes"])}
    if set(processes) != expected_processes:
        raise ValueError("confirmation process inventory differs")
    reference_ids = None
    first_counts: dict[str, int] = defaultdict(int)
    for process_index, entries in processes.items():
        if len(entries) != sum(design["populations"].values()):
            raise ValueError("confirmation process request count differs")
        ids = {entry["workload_id"] for entry in entries}
        if len(ids) != len(entries):
            raise ValueError("confirmation process contains duplicate workloads")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("confirmation processes use different workloads")
        for index, entry in enumerate(entries):
            if entry.get("schedule_index") != index:
                raise ValueError("confirmation schedule indexes are not contiguous")
            if entry.get("first_arm") not in {"P", "B"}:
                raise ValueError("confirmation first arm is invalid")
            if entry["first_arm"] == "P":
                first_counts[entry["workload_id"]] += 1
        for population, expected in design["populations"].items():
            subset = [entry for entry in entries if entry["population"] == population]
            if len(subset) != expected:
                raise ValueError("confirmation population count differs")
            if sum(entry["first_arm"] == "P" for entry in subset) != expected // 2:
                raise ValueError("confirmation process arm order is not balanced")
    if set(first_counts) != reference_ids or any(
        count != design["fresh_processes"] // 2 for count in first_counts.values()
    ):
        raise ValueError("each request must be P-first in exactly three processes")


def freeze(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace confirmation directory {output}")
    output.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    design = load_design()
    source_dir = (repository / design["source_benchmark"]).resolve()
    source_protocol, source_rows, source_meta, source_schedules = base.load_frozen(
        source_dir
    )
    source_summary_path = source_dir / "summary.json"
    source_summary = original_summary_binding(source_summary_path)
    selected = selected_source_rows(source_schedules, source_rows, design)
    schedule = build_schedule(selected, design)
    schedule["source_schedules_sha256"] = source_schedules["schedules_sha256"]
    schedule["schedule_sha256"] = base.value_sha256(schedule)
    protocol = {
        "version": 1,
        "status": "frozen",
        "design": design,
        "design_file_sha256": file_sha256(DESIGN_PATH),
        "runner": runner_identity(repository),
        "runtime": base.runtime_identity(),
        "source": {
            "directory": str(source_dir),
            "protocol_file_sha256": file_sha256(source_dir / "protocol.json"),
            "protocol_sha256": source_protocol["protocol_sha256"],
            "workloads_file_sha256": source_meta["workloads_file_sha256"],
            "schedules_file_sha256": file_sha256(source_dir / "schedules.json"),
            "schedules_sha256": source_schedules["schedules_sha256"],
            "summary": source_summary,
            "checkpoint_sha256": source_protocol["run"]["checkpoint_sha256"],
            "temperature": source_protocol["design"]["checkpoint"]["temperature"],
            "max_length": source_protocol["design"]["checkpoint"]["max_length"],
            "precision": source_protocol["design"]["checkpoint"]["precision"],
        },
        "schedule_sha256": schedule["schedule_sha256"],
        "locked_test_opened": False,
    }
    protocol["protocol_sha256"] = base.value_sha256(protocol)
    validate_schedule(schedule, protocol)
    base.atomic_json(output / "protocol.json", protocol)
    base.atomic_json(output / "schedule.json", schedule)
    return protocol


def load_frozen(output: Path) -> tuple[dict, dict, dict, list[dict], dict, dict]:
    protocol = json.loads((output / "protocol.json").read_text())
    verify_self_digest(protocol, "protocol_sha256", "confirmation protocol")
    if protocol.get("status") != "frozen" or protocol.get("locked_test_opened") is not False:
        raise ValueError("confirmation protocol is not a closed-test freeze")
    repository = Path(__file__).resolve().parents[1]
    current = runner_identity(repository)
    if current["sha256"] != protocol["runner"]["sha256"]:
        raise ValueError("confirmation runner identity differs")
    if file_sha256(DESIGN_PATH) != protocol["design_file_sha256"]:
        raise ValueError("confirmation design bytes differ")
    if base.runtime_identity() != protocol["runtime"]:
        raise ValueError("confirmation runtime differs from the freeze")
    schedule = json.loads((output / "schedule.json").read_text())
    validate_schedule(schedule, protocol)
    source_dir = Path(protocol["source"]["directory"])
    source_protocol, source_rows, source_meta, source_schedules = base.load_frozen(
        source_dir
    )
    checks = {
        "source protocol file": (
            file_sha256(source_dir / "protocol.json"),
            protocol["source"]["protocol_file_sha256"],
        ),
        "source protocol": (
            source_protocol["protocol_sha256"],
            protocol["source"]["protocol_sha256"],
        ),
        "source workloads": (
            source_meta["workloads_file_sha256"],
            protocol["source"]["workloads_file_sha256"],
        ),
        "source schedule file": (
            file_sha256(source_dir / "schedules.json"),
            protocol["source"]["schedules_file_sha256"],
        ),
        "source schedule": (
            source_schedules["schedules_sha256"],
            protocol["source"]["schedules_sha256"],
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{label} differs")
    if original_summary_binding(source_dir / "summary.json") != protocol["source"]["summary"]:
        raise ValueError("original summary binding differs")
    return protocol, schedule, source_protocol, source_rows, source_meta, source_schedules


def arm_pair(first_arm: str) -> tuple[str, str]:
    if first_arm == "P":
        return "P", "B"
    if first_arm == "B":
        return "B", "P"
    raise ValueError("unknown first arm")


def measure(args) -> dict:
    output = Path(args.out).resolve()
    protocol, schedule, source_protocol, source_rows, _, _ = load_frozen(output)
    if not 0 <= args.process_index < protocol["design"]["fresh_processes"]:
        raise ValueError("process index is outside the frozen protocol")
    result_path = output / f"process-{args.process_index}.jsonl"
    meta_path = output / f"process-{args.process_index}.meta.json"
    partial_path = result_path.with_suffix(".partial")
    if meta_path.exists():
        raise FileExistsError("refusing to overwrite a confirmation process")
    if result_path.exists():
        incomplete = result_path.with_name(
            f"process-{args.process_index}.incomplete-"
            f"{file_sha256(result_path)[:12]}.jsonl"
        )
        if incomplete.exists():
            raise FileExistsError("an identical incomplete process is already preserved")
        os.replace(result_path, incomplete)
    partial_path.unlink(missing_ok=True)
    requests = base.reload_requests(source_protocol, source_rows)
    selected = schedule["processes"][str(args.process_index)]
    selected_ids = {entry["workload_id"] for entry in selected}
    if not selected_ids.issubset(requests):
        raise ValueError("confirmation request disappeared from source data")
    model, tokenizer, delimiters, _, _, manifest = load_model(
        source_protocol["run"]["path"], "mps"
    )
    if manifest["current"]["sha256"] != protocol["source"]["checkpoint_sha256"]:
        raise ValueError("loaded checkpoint differs from the confirmation protocol")
    max_length = protocol["source"]["max_length"]
    temperature = protocol["source"]["temperature"]
    torch.mps.empty_cache()
    base.synchronize("mps")
    started_unix = time.time()
    rows = []
    with torch.no_grad():
        for warmup_pass in range(protocol["design"]["warmup"]["complete_passes"]):
            for entry in selected:
                pair = arm_pair(entry["first_arm"])
                if warmup_pass % 2:
                    pair = tuple(reversed(pair))
                for arm in pair:
                    result = base.execute_complete(
                        model,
                        tokenizer,
                        delimiters,
                        [requests[entry["workload_id"]]],
                        max_length,
                        arm,
                        temperature,
                    )
                    base.synchronize("mps")
                    del result
        for entry in selected:
            for pair_position, arm in enumerate(arm_pair(entry["first_arm"])):
                base.synchronize("mps")
                started = time.perf_counter_ns()
                result = base.execute_complete(
                    model,
                    tokenizer,
                    delimiters,
                    [requests[entry["workload_id"]]],
                    max_length,
                    arm,
                    temperature,
                )
                base.synchronize("mps")
                duration = time.perf_counter_ns() - started
                del result
                rows.append({
                    "version": 1,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "schedule_sha256": schedule["schedule_sha256"],
                    "source_protocol_sha256": protocol["source"]["protocol_sha256"],
                    "source_workloads_sha256": protocol["source"]["workloads_file_sha256"],
                    "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
                    "device": "mps",
                    "scope": "complete_request",
                    "request_batch_size": 1,
                    "process_index": args.process_index,
                    "process_id": PROCESS_ID,
                    "schedule_index": entry["schedule_index"],
                    "population": entry["population"],
                    "suite": entry["suite"],
                    "workload_id": entry["workload_id"],
                    "parent_id": entry["parent_id"],
                    "first_arm": entry["first_arm"],
                    "pair_position": pair_position,
                    "arm": arm,
                    "duration_ns": duration,
                })
    base.atomic_write(partial_path, base.jsonl_bytes(rows))
    os.replace(partial_path, result_path)
    metadata = {
        "version": 1,
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "source_protocol_sha256": protocol["source"]["protocol_sha256"],
        "source_workloads_sha256": protocol["source"]["workloads_file_sha256"],
        "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
        "device": "mps",
        "scope": "complete_request",
        "request_batch_size": 1,
        "process_index": args.process_index,
        "process_id": PROCESS_ID,
        "process_pid": PROCESS_PID,
        "process_started_unix_ns": PROCESS_STARTED_UNIX_NS,
        "measurement_started_unix": started_unix,
        "process_finished_unix": time.time(),
        "warmup_complete_passes": protocol["design"]["warmup"]["complete_passes"],
        "warmup_operations": (
            len(selected)
            * len(protocol["design"]["arms"])
            * protocol["design"]["warmup"]["complete_passes"]
        ),
        "records": len(rows),
        "file_sha256": file_sha256(result_path),
        "runtime": base.runtime_identity(),
        "load_average_at_finish": list(os.getloadavg()),
        "locked_test_opened": False,
    }
    metadata["report_sha256"] = base.value_sha256(metadata)
    base.atomic_json(meta_path, metadata)
    return metadata


def validate_process(
    result_path: Path,
    meta_path: Path,
    process_index: int,
    protocol: dict,
    schedule: dict,
) -> tuple[list[dict], dict]:
    metadata = json.loads(meta_path.read_text())
    verify_self_digest(metadata, "report_sha256", "confirmation process metadata")
    expected_meta = {
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "source_protocol_sha256": protocol["source"]["protocol_sha256"],
        "source_workloads_sha256": protocol["source"]["workloads_file_sha256"],
        "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
        "device": "mps",
        "scope": "complete_request",
        "request_batch_size": 1,
        "process_index": process_index,
        "warmup_complete_passes": protocol["design"]["warmup"]["complete_passes"],
        "warmup_operations": (
            len(schedule["processes"][str(process_index)])
            * len(protocol["design"]["arms"])
            * protocol["design"]["warmup"]["complete_passes"]
        ),
        "runtime": protocol["runtime"],
        "locked_test_opened": False,
    }
    for field, expected in expected_meta.items():
        if metadata.get(field) != expected:
            raise ValueError(f"confirmation process metadata {field} differs")
    if metadata.get("version") != 1:
        raise ValueError("confirmation process metadata version differs")
    if not isinstance(metadata.get("process_id"), str) or not metadata["process_id"]:
        raise ValueError("confirmation process identity is invalid")
    for field in ("process_pid", "process_started_unix_ns"):
        value = metadata.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"confirmation process metadata {field} is invalid")
    started = metadata.get("measurement_started_unix")
    finished = metadata.get("process_finished_unix")
    if not base.finite_number(started) or not base.finite_number(finished) or finished < started:
        raise ValueError("confirmation process timestamps are invalid")
    load_average = metadata.get("load_average_at_finish")
    if (
        not isinstance(load_average, list)
        or len(load_average) != 3
        or any(not base.finite_number(value) or value < 0 for value in load_average)
    ):
        raise ValueError("confirmation process load average is invalid")
    if file_sha256(result_path) != metadata.get("file_sha256"):
        raise ValueError("confirmation process file digest differs")
    rows = base.load_jsonl(result_path)
    expected_entries = schedule["processes"][str(process_index)]
    if len(rows) != len(expected_entries) * 2 or metadata.get("records") != len(rows):
        raise ValueError("confirmation process row count differs")
    seen = set()
    for row_index, row in enumerate(rows):
        entry = expected_entries[row_index // 2]
        pair_position = row_index % 2
        expected_arm = arm_pair(entry["first_arm"])[pair_position]
        expected = {
            "version": 1,
            "protocol_sha256": protocol["protocol_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "source_protocol_sha256": protocol["source"]["protocol_sha256"],
            "source_workloads_sha256": protocol["source"]["workloads_file_sha256"],
            "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
            "device": "mps",
            "scope": "complete_request",
            "request_batch_size": 1,
            "process_index": process_index,
            "process_id": metadata["process_id"],
            "schedule_index": entry["schedule_index"],
            "population": entry["population"],
            "suite": entry["suite"],
            "workload_id": entry["workload_id"],
            "parent_id": entry["parent_id"],
            "first_arm": entry["first_arm"],
            "pair_position": pair_position,
            "arm": expected_arm,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"confirmation row {field} differs")
        duration = row.get("duration_ns")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
            raise ValueError("confirmation duration must be a positive integer")
        key = (process_index, entry["workload_id"], expected_arm)
        if key in seen:
            raise ValueError("confirmation observation is duplicated")
        seen.add(key)
    return rows, metadata


def verify_process(args) -> dict:
    output = Path(args.out).resolve()
    protocol, schedule, _, _, _, _ = load_frozen(output)
    if not 0 <= args.process_index < protocol["design"]["fresh_processes"]:
        raise ValueError("process index is outside the frozen protocol")
    result_path = output / f"process-{args.process_index}.jsonl"
    meta_path = output / f"process-{args.process_index}.meta.json"
    rows, metadata = validate_process(
        result_path, meta_path, args.process_index, protocol, schedule
    )
    return {
        "status": "validated",
        "process_index": args.process_index,
        "records": len(rows),
        "process_id": metadata["process_id"],
        "file_sha256": metadata["file_sha256"],
        "report_sha256": metadata["report_sha256"],
    }


def quantile(values: list[float], probability: float) -> float:
    return base.quantile(values, probability)


def population_statistics(
    rows: list[dict], population: str, design: dict
) -> dict:
    selected = [row for row in rows if row["population"] == population]
    processes = sorted({row["process_index"] for row in selected})
    workloads = sorted({row["workload_id"] for row in selected})
    by_key = {}
    parent_by_workload = {}
    first_by_key = {}
    for row in selected:
        key = (row["process_index"], row["workload_id"], row["arm"])
        if key in by_key:
            raise ValueError("population analysis has a duplicate observation")
        by_key[key] = float(row["duration_ns"])
        previous = parent_by_workload.setdefault(row["workload_id"], row["parent_id"])
        if previous != row["parent_id"]:
            raise ValueError("workload maps to multiple parents")
        pair_key = (row["process_index"], row["workload_id"])
        prior_first = first_by_key.setdefault(pair_key, row["first_arm"])
        if prior_first != row["first_arm"]:
            raise ValueError("paired observations disagree on first arm")
    expected = len(processes) * len(workloads) * 2
    if len(by_key) != expected:
        raise ValueError("population analysis is missing observations")
    per_process = {}
    for process_index in processes:
        sums = {
            arm: sum(by_key[(process_index, workload, arm)] for workload in workloads)
            for arm in ("P", "B")
        }
        per_process[str(process_index)] = sums["P"] / sums["B"]
    workload_means = {
        workload: {
            arm: statistics.fmean(
                by_key[(process_index, workload, arm)] for process_index in processes
            )
            for arm in ("P", "B")
        }
        for workload in workloads
    }
    equal_workload_ratio = sum(value["P"] for value in workload_means.values()) / sum(
        value["B"] for value in workload_means.values()
    )
    by_parent: dict[str, list[str]] = defaultdict(list)
    for workload, parent in parent_by_workload.items():
        by_parent[parent].append(workload)
    parents = sorted(by_parent)
    uncertainty = design["uncertainty"]
    rng = random.Random(uncertainty["parent_bootstrap_seed"])
    parent_samples = []
    for _ in range(uncertainty["parent_bootstrap_draws"]):
        numerator = denominator = 0.0
        for _ in parents:
            parent = rng.choice(parents)
            for workload in by_parent[parent]:
                numerator += workload_means[workload]["P"]
                denominator += workload_means[workload]["B"]
        parent_samples.append(numerator / denominator)
    rng = random.Random(uncertainty["process_parent_bootstrap_seed"])
    crossed_samples = []
    for _ in range(uncertainty["process_parent_bootstrap_draws"]):
        sampled_processes = [rng.choice(processes) for _ in processes]
        sampled_parents = [rng.choice(parents) for _ in parents]
        numerator = denominator = 0.0
        for process_index in sampled_processes:
            for parent in sampled_parents:
                for workload in by_parent[parent]:
                    numerator += by_key[(process_index, workload, "P")]
                    denominator += by_key[(process_index, workload, "B")]
        crossed_samples.append(numerator / denominator)
    order_strata = {}
    for first_arm in ("P", "B"):
        numerator = denominator = 0.0
        pairs = 0
        for process_index in processes:
            for workload in workloads:
                if first_by_key[(process_index, workload)] == first_arm:
                    numerator += by_key[(process_index, workload, "P")]
                    denominator += by_key[(process_index, workload, "B")]
                    pairs += 1
        order_strata[f"{first_arm}_first"] = {
            "pairs": pairs,
            "ratio": numerator / denominator,
        }
    durations = {
        arm: [
            by_key[(process_index, workload, arm)]
            for process_index in processes
            for workload in workloads
        ]
        for arm in ("P", "B")
    }
    return {
        "processes": len(processes),
        "parents": len(parents),
        "workloads": len(workloads),
        "observations": len(by_key),
        "equal_workload_ratio": equal_workload_ratio,
        "parent_bootstrap_interval_95": [
            quantile(parent_samples, 0.025),
            quantile(parent_samples, 0.975),
        ],
        "process_parent_bootstrap_interval_95": [
            quantile(crossed_samples, 0.025),
            quantile(crossed_samples, 0.975),
        ],
        "p95_latency_ratio": quantile(durations["P"], 0.95)
        / quantile(durations["B"], 0.95),
        "per_process_ratios": per_process,
        "processes_below_one": sum(value < 1.0 for value in per_process.values()),
        "order_strata": order_strata,
        "parent_bootstrap_draws": uncertainty["parent_bootstrap_draws"],
        "parent_bootstrap_seed": uncertainty["parent_bootstrap_seed"],
        "process_parent_bootstrap_draws": uncertainty[
            "process_parent_bootstrap_draws"
        ],
        "process_parent_bootstrap_seed": uncertainty[
            "process_parent_bootstrap_seed"
        ],
    }


def apply_gates(statistics_by_population: dict, design: dict) -> dict:
    control = statistics_by_population[design["negative_control"]["population"]]
    lower = design["negative_control"]["material_ratio_lower"]
    upper = design["negative_control"]["material_ratio_upper"]
    interval = control["process_parent_bootstrap_interval_95"]
    control_failure = bool(
        not lower <= control["equal_workload_ratio"] <= upper
        or not interval[0] <= 1.0 <= interval[1]
    )
    gate = design["primary_gate"]
    primary = {}
    for population in gate["populations"]:
        item = statistics_by_population[population]
        checks = {
            "mean_ratio": item["equal_workload_ratio"]
            <= gate["maximum_equal_workload_mean_ratio"],
            "parent_interval": item["parent_bootstrap_interval_95"][1]
            < gate["maximum_parent_bootstrap_95_percent_upper_bound"],
            "p95_ratio": item["p95_latency_ratio"]
            <= gate["maximum_p95_latency_ratio"],
            "process_replication": item["processes_below_one"]
            >= gate["minimum_processes_below_one"],
            "order_strata": (
                not gate["require_both_arm_order_strata_below_one"]
                or all(
                    value["ratio"] < 1.0
                    for value in item["order_strata"].values()
                )
            ),
            "negative_control": not control_failure,
        }
        primary[population] = {"checks": checks, "passed": all(checks.values())}
    return {
        "negative_control": {
            "failure": control_failure,
            "material_ratio_interval": [lower, upper],
            "ratio": control["equal_workload_ratio"],
            "process_parent_interval_95": interval,
        },
        "primary": primary,
    }


def collect_process_evidence(
    output: Path, protocol: dict, schedule: dict
) -> tuple[list[dict], list[dict], dict]:
    all_rows = []
    metadata = []
    artifacts = {}
    for process_index in range(protocol["design"]["fresh_processes"]):
        result_path = output / f"process-{process_index}.jsonl"
        meta_path = output / f"process-{process_index}.meta.json"
        if not result_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing process {process_index} artifacts")
        rows, meta = validate_process(
            result_path, meta_path, process_index, protocol, schedule
        )
        all_rows.extend(rows)
        metadata.append(meta)
        artifacts[result_path.name] = file_sha256(result_path)
        artifacts[meta_path.name] = file_sha256(meta_path)
    identities = {
        (item["process_id"], item["process_pid"], item["process_started_unix_ns"])
        for item in metadata
    }
    if len(identities) != protocol["design"]["fresh_processes"]:
        raise ValueError("confirmation did not use six distinct fresh processes")
    return all_rows, metadata, artifacts


def build_summary(
    protocol: dict,
    schedule: dict,
    all_rows: list[dict],
    metadata: list[dict],
    artifacts: dict,
) -> dict:
    identities = {
        (item["process_id"], item["process_pid"], item["process_started_unix_ns"])
        for item in metadata
    }
    statistics_by_population = {
        population: population_statistics(all_rows, population, protocol["design"])
        for population in protocol["design"]["populations"]
    }
    gates = apply_gates(statistics_by_population, protocol["design"])
    passed = all(item["passed"] for item in gates["primary"].values())
    summary = {
        "version": 1,
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "source_summary": protocol["source"]["summary"],
        "runtime": protocol["runtime"],
        "fresh_processes": len(identities),
        "warmup_complete_passes": protocol["design"]["warmup"][
            "complete_passes"
        ],
        "statistics": statistics_by_population,
        "gates": gates,
        "passed": passed,
        "outcome": "batch_one_speedup_confirmed" if passed else "batch_one_speedup_not_confirmed",
        "interpretation": protocol["design"]["interpretation"],
        "artifacts": dict(sorted(artifacts.items())),
        "locked_test_opened": False,
    }
    summary["report_sha256"] = base.value_sha256(summary)
    return summary


def summarize(args) -> dict:
    output = Path(args.out).resolve()
    protocol, schedule, _, _, _, _ = load_frozen(output)
    all_rows, metadata, artifacts = collect_process_evidence(
        output, protocol, schedule
    )
    destination = output / "summary.json"
    combined = output / "observations.jsonl"
    if destination.exists():
        raise FileExistsError("refusing to overwrite confirmation summary")
    combined_bytes = base.jsonl_bytes(all_rows)
    if combined.exists():
        if combined.read_bytes() != combined_bytes:
            raise ValueError("existing combined observations do not reproduce")
    else:
        base.atomic_write(combined, combined_bytes)
    artifacts[combined.name] = file_sha256(combined)
    artifacts["protocol.json"] = file_sha256(output / "protocol.json")
    artifacts["schedule.json"] = file_sha256(output / "schedule.json")
    summary = build_summary(protocol, schedule, all_rows, metadata, artifacts)
    base.atomic_json(destination, summary)
    return summary


def verify(args) -> dict:
    output = Path(args.out).resolve()
    protocol, schedule, _, _, _, _ = load_frozen(output)
    all_rows, metadata, artifacts = collect_process_evidence(
        output, protocol, schedule
    )
    combined = output / "observations.jsonl"
    summary_path = output / "summary.json"
    if not combined.is_file() or not summary_path.is_file():
        raise FileNotFoundError("confirmation summary artifacts are incomplete")
    if combined.read_bytes() != base.jsonl_bytes(all_rows):
        raise ValueError("combined confirmation observations do not reproduce")
    artifacts[combined.name] = file_sha256(combined)
    artifacts["protocol.json"] = file_sha256(output / "protocol.json")
    artifacts["schedule.json"] = file_sha256(output / "schedule.json")
    expected = build_summary(protocol, schedule, all_rows, metadata, artifacts)
    observed = json.loads(summary_path.read_text())
    verify_self_digest(observed, "report_sha256", "confirmation summary")
    if observed != expected:
        raise ValueError("confirmation summary does not reproduce exactly")
    return {
        "status": "validated",
        "report_sha256": observed["report_sha256"],
        "file_sha256": file_sha256(summary_path),
        "outcome": observed["outcome"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--out", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--out", required=True)
    measure_parser.add_argument("--process-index", required=True, type=int)
    process_verify_parser = commands.add_parser("verify-process")
    process_verify_parser.add_argument("--out", required=True)
    process_verify_parser.add_argument("--process-index", required=True, type=int)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--out", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args)
    elif args.command == "measure":
        result = measure(args)
    elif args.command == "verify-process":
        result = verify_process(args)
    elif args.command == "summarize":
        result = summarize(args)
    else:
        result = verify(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
