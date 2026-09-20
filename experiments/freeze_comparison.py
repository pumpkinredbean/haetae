"""Freeze a result-blind development comparison plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from haetae.checkpoint import atomic_json_save, load_completed_checkpoint
from haetae.measure import reconstruct_consumed
from experiments.comparison_protocol import (
    PLAN_VERSION,
    canonical_sha256,
    comparison_code_identity,
    request_state_sha256,
)
from experiments.freeze_shared_suite import reconstruct_partitions
from experiments.kev_adapter import file_sha256, load_frozen_split


def suite_binding(suite: Path, filenames: tuple[str, ...]) -> dict:
    manifest_path = suite / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "files": {name: manifest["files"][name] for name in filenames},
    }


def overlap_entries(requests: list[dict], consumed_states: set[str]) -> list[dict]:
    entries = []
    for request in requests:
        identity = request_state_sha256(request)
        if identity not in consumed_states:
            continue
        entries.append({
            "request_id": request["meta"].get("id"),
            "group_id": request["meta"].get("group_id"),
            "source": request["meta"].get("source"),
            "state_sha256": identity,
            "questions": len(request["questions"]),
        })
    return sorted(entries, key=lambda item: (
        str(item["source"]), str(item["request_id"]), item["state_sha256"],
    ))


def freeze(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    baseline_run = Path(args.baseline_run).resolve()
    _, run, manifest = load_completed_checkpoint(baseline_run)
    consumed = reconstruct_consumed(run)
    baseline_train, baseline_validation = reconstruct_partitions(run)

    suites = {
        "decision": Path(args.decision_suite).resolve(),
        "transfer": Path(args.transfer_suite).resolve(),
        "korean": Path(args.korean_suite).resolve(),
    }
    calibration, _ = load_frozen_split(suites["decision"], "calibration")
    development = {
        key: load_frozen_split(path, "development")[0]
        for key, path in suites.items()
    }
    consumed_states = set(consumed["state_digests"])
    training_states = {
        canonical_sha256(record["state"]) for record in baseline_train
    }
    validation_states = {
        canonical_sha256(record["state"]) for record in baseline_validation
    }
    exclusions = {
        key: overlap_entries(requests, consumed_states)
        for key, requests in development.items()
    }
    calibration_states = {
        request_state_sha256(request) for request in calibration
    }
    calibration_training_overlap = calibration_states & training_states
    calibration_exclusions = overlap_entries(calibration, training_states)

    repository = Path(__file__).resolve().parents[1]
    plan = {
        "version": PLAN_VERSION,
        "status": "frozen",
        "baseline_run": {
            "run_id": run["run_id"],
            "spec_sha256": run["spec_sha256"],
            "generation": manifest["current"]["generation"],
            "checkpoint_sha256": manifest["current"]["sha256"],
            "train_fingerprint": run["spec"]["train_fingerprint"],
            "validation_fingerprint": run["spec"]["validation_fingerprint"],
        },
        "suites": {
            "decision": suite_binding(
                suites["decision"],
                ("calibration.jsonl", "development.jsonl"),
            ),
            "transfer": suite_binding(
                suites["transfer"], ("development.jsonl",),
            ),
            "korean": suite_binding(
                suites["korean"], ("development.jsonl",),
            ),
        },
        "excluded_development_requests": exclusions,
        "excluded_calibration_requests": calibration_exclusions,
        "audit": {
            "baseline_consumed_states": len(consumed_states),
            "calibration_requests": len(calibration),
            "calibration_training_overlap_states": len(
                calibration_training_overlap
            ),
            "calibration_internal_validation_overlap_states": len(
                calibration_states & validation_states
            ),
            "calibration_retained_requests": (
                len(calibration) - len(calibration_exclusions)
            ),
            "development_requests": {
                key: len(requests) for key, requests in development.items()
            },
            "excluded_requests": {
                key: len(entries) for key, entries in exclusions.items()
            },
            "excluded_questions": {
                key: sum(item["questions"] for item in entries)
                for key, entries in exclusions.items()
            },
        },
        "selection": (
            "report every public development request and a common-clean result "
            "that removes any complete request whose exact state was consumed "
            "by the baseline train or internal-validation partition"
        ),
        "locked_test_opened": False,
        "code": comparison_code_identity(repository),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    output.mkdir(parents=True)
    atomic_json_save(plan, output / "plan.json")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--decision-suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--korean-suite", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = freeze(args)
    print(json.dumps({
        "plan_sha256": plan["plan_sha256"],
        "excluded_requests": plan["audit"]["excluded_requests"],
        "excluded_questions": plan["audit"]["excluded_questions"],
    }, indent=2))


if __name__ == "__main__":
    main()
