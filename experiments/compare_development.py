"""Compare two frozen development reports with paired parent uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from haetae.checkpoint import atomic_json_save
from experiments.comparison_protocol import (
    canonical_sha256,
    common_clean_predictions,
    load_comparison_plan,
    request_state_sha256,
    training_clean_calibration_predictions,
    validate_suite_binding,
)
from experiments.evaluate_shared import (
    canonical_report_sha256,
    fit_source_balanced_temperature,
)
from experiments.kev_adapter import file_sha256, load_frozen_split


SPLITS = {
    "decision": "decision_development",
    "transfer": "transfer_development",
    "korean": "korean_development",
}
SUPPORTED_TYPES = {"choice", "noul", "score"}
QUANTILE_METHOD = "inverse empirical CDF with nearest-rank endpoints"
BOOTSTRAP_DESIGN = (
    "resample whole request parents within fixed origin-source and "
    "task-source-incidence strata"
)


def load_report(path: str | Path) -> dict:
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get("status") != "evaluated":
        raise ValueError(f"development report is not evaluated: {path}")
    if report.get("locked_test_opened") is not False:
        raise ValueError(f"development report does not attest a closed test: {path}")
    if canonical_report_sha256(report) != report.get("report_sha256"):
        raise ValueError(f"development report digest is invalid: {path}")
    if not isinstance(report.get("predictions"), dict):
        raise ValueError(f"development report predictions are missing: {path}")
    return report


def prediction_key(prediction: dict) -> tuple:
    return (
        prediction["source"],
        prediction["request_id"],
        prediction["question_id"],
        prediction["state_sha256"],
    )


def index_predictions(predictions: list[dict]) -> dict[tuple, dict]:
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    indexed = {}
    for prediction in predictions:
        key = prediction_key(prediction)
        if key in indexed:
            raise ValueError(f"duplicate prediction identity: {key}")
        indexed[key] = prediction
    return indexed


def validate_prediction(prediction: dict, label: str) -> None:
    options = prediction.get("options")
    option_keys = prediction.get("option_keys")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"{label} needs at least two options")
    if not isinstance(option_keys, list) or len(option_keys) != len(options):
        raise ValueError(f"{label} option keys do not match its options")
    if prediction.get("type") not in SUPPORTED_TYPES:
        raise ValueError(f"{label} has an unsupported primitive")
    target_label = prediction.get("label")
    if (
        type(target_label) is not int
        or target_label < 0
        or target_label >= len(options)
    ):
        raise ValueError(f"{label} has an invalid hard label")
    logits = prediction.get("logits")
    if not isinstance(logits, list) or len(logits) != len(options):
        raise ValueError(f"{label} logits do not match the option count")
    try:
        finite_logits = all(math.isfinite(float(value)) for value in logits)
    except (TypeError, ValueError):
        finite_logits = False
    if not finite_logits:
        raise ValueError(f"{label} logits are not finite numbers")
    target_distribution(prediction)


def validate_pair(baseline: dict, shared: dict) -> None:
    fields = (
        "group_id", "source", "type", "options", "option_keys", "label", "soft",
    )
    for field in fields:
        if baseline.get(field) != shared.get(field):
            raise ValueError(f"paired predictions differ in {field}")
    validate_prediction(baseline, "baseline prediction")
    validate_prediction(shared, "shared prediction")


def target_distribution(prediction: dict) -> torch.Tensor:
    option_count = len(prediction["options"])
    label = prediction.get("label")
    if type(label) is not int or label < 0 or label >= option_count:
        raise ValueError("paired prediction has an invalid hard label")
    if prediction["soft"] is None:
        target = torch.zeros(option_count, dtype=torch.float64)
        target[label] = 1.0
        return target
    target = torch.tensor(prediction["soft"], dtype=torch.float64)
    if (
        target.ndim != 1
        or target.numel() != option_count
        or not torch.isfinite(target).all()
        or torch.any(target < 0)
        or not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-6)
    ):
        raise ValueError("paired prediction target is not a probability distribution")
    return target


def prediction_metrics(prediction: dict, temperature: float) -> dict[str, float]:
    logits = torch.tensor(prediction["logits"], dtype=torch.float64)
    target = target_distribution(prediction)
    log_probabilities = F.log_softmax(logits / temperature, dim=-1)
    probabilities = log_probabilities.exp()
    values = {
        "accuracy": float(int(probabilities.argmax()) == prediction["label"]),
        "nll": float(-(target * log_probabilities).sum()),
        "brier_histogram": float(((probabilities - target) ** 2).sum()),
        "brier_annotation": float(
            1.0 + probabilities.square().sum()
            - 2.0 * (probabilities * target).sum()
        ),
    }
    if prediction["type"] == "score":
        values["ranked_probability_score"] = float((
            probabilities.cumsum(0)[:-1] - target.cumsum(0)[:-1]
        ).square().mean())
    return values


def inventory_from_requests(requests: list[dict]) -> dict[tuple, dict]:
    inventory = {}
    for request in requests:
        meta = request.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("frozen request metadata is missing")
        origin_source = meta.get("source")
        request_id = meta.get("id")
        group_id = meta.get("group_id")
        parent_id = group_id or request_id
        if not all(
            isinstance(value, str) and value
            for value in (origin_source, request_id, parent_id)
        ):
            raise ValueError("frozen request parent metadata is incomplete")
        state_sha256 = request_state_sha256(request)
        for question in request["questions"]:
            expected = {
                "request_id": request_id,
                "group_id": group_id,
                "state_sha256": state_sha256,
                "question_id": question["id"],
                "source": question["source"],
                "type": question["type"],
                "options": question["options"],
                "option_keys": question["option_keys"],
                "label": question["label"],
                "soft": question["soft"],
            }
            key = prediction_key(expected)
            if key in inventory:
                raise ValueError(f"frozen population has a duplicate question: {key}")
            inventory[key] = {
                "expected": expected,
                "parent_namespace": origin_source,
                "parent": f"{origin_source}\0{parent_id}",
            }
    return inventory


def validate_population(
    predictions: list[dict], inventory: dict[tuple, dict], label: str,
) -> dict[tuple, dict]:
    indexed = index_predictions(predictions)
    if set(indexed) != set(inventory):
        missing = len(set(inventory) - set(indexed))
        extra = len(set(indexed) - set(inventory))
        raise ValueError(
            f"{label} differs from the frozen population: "
            f"missing={missing}, extra={extra}"
        )
    fields = (
        "request_id", "group_id", "state_sha256", "question_id", "source",
        "type", "options", "option_keys", "label", "soft",
    )
    for key, prediction in indexed.items():
        expected = inventory[key]["expected"]
        for field in fields:
            if prediction.get(field) != expected.get(field):
                raise ValueError(f"{label} differs from frozen {field}: {key}")
        validate_prediction(prediction, label)
    return indexed


def validate_split_population(
    suite: str | Path, split: str, descriptor: dict,
) -> tuple[list[dict], dict[tuple, dict]]:
    requests, _ = load_frozen_split(suite, split)
    inventory = inventory_from_requests(requests)
    if len(requests) != descriptor["records"]:
        raise ValueError(f"{split} request count differs from the frozen plan")
    if len(inventory) != descriptor["questions"]:
        raise ValueError(f"{split} question count differs from the frozen plan")
    return requests, inventory


def quantile_interval(values: list[float]) -> dict:
    ordered = sorted(values)
    samples = len(ordered)
    lower = max(0, math.ceil(0.025 * samples) - 1)
    upper = min(samples - 1, math.ceil(0.975 * samples) - 1)
    return {
        "lower": ordered[lower],
        "upper": ordered[upper],
        "samples": samples,
        "unit": "origin-source-namespaced request parent",
        "design": BOOTSTRAP_DESIGN,
        "quantile_method": QUANTILE_METHOD,
    }


def bootstrap_differences(
    rows: list[dict], metric: str, samples: int, seed: int,
) -> tuple[dict, dict, dict]:
    by_parent = defaultdict(list)
    parent_namespaces = {}
    for row in rows:
        if metric not in row["difference"]:
            continue
        parent = row["parent"]
        namespace = row["parent_namespace"]
        if parent in parent_namespaces and parent_namespaces[parent] != namespace:
            raise ValueError("one parent appears in multiple origin namespaces")
        parent_namespaces[parent] = namespace
        by_parent[parent].append(row)
    strata = defaultdict(dict)
    for parent, parent_rows in by_parent.items():
        incidence = tuple(sorted({row["source"] for row in parent_rows}))
        stratum = (parent_namespaces[parent], incidence)
        strata[stratum][parent] = parent_rows
    if not strata or any(len(parents) < 2 for parents in strata.values()):
        raise ValueError(
            f"{metric} needs at least two parents in every bootstrap stratum"
        )
    generator = random.Random(seed)
    weighted_estimates, source_macro_estimates = [], []
    expected_sources = {row["source"] for row in rows if metric in row["difference"]}
    for _ in range(samples):
        selected = []
        for parents in (strata[key] for key in sorted(strata)):
            keys = sorted(parents)
            for parent in generator.choices(keys, k=len(keys)):
                selected.extend(parents[parent])
        sampled_by_source = defaultdict(list)
        all_values = []
        for row in selected:
            value = row["difference"][metric]
            all_values.append(value)
            sampled_by_source[row["source"]].append(value)
        if set(sampled_by_source) != expected_sources:
            raise RuntimeError("fixed bootstrap strata lost a task source")
        weighted_estimates.append(sum(all_values) / len(all_values))
        source_macro_estimates.append(sum(
            sum(values) / len(values) for values in sampled_by_source.values()
        ) / len(sampled_by_source))
    return (
        quantile_interval(weighted_estimates),
        quantile_interval(source_macro_estimates),
        {
            "design": BOOTSTRAP_DESIGN,
            "strata": len(strata),
            "minimum_parents_per_stratum": min(map(len, strata.values())),
            "maximum_parents_per_stratum": max(map(len, strata.values())),
        },
    )


def paired_summary(
    baseline_predictions: list[dict],
    shared_predictions: list[dict],
    parent_bindings: dict[tuple, dict],
    baseline_temperature: float,
    shared_temperature: float,
    samples: int,
    seed: int,
) -> dict:
    if (
        not math.isfinite(baseline_temperature)
        or baseline_temperature <= 0
        or not math.isfinite(shared_temperature)
        or shared_temperature <= 0
    ):
        raise ValueError("comparison temperatures must be finite and positive")
    baseline_index = index_predictions(baseline_predictions)
    shared_index = index_predictions(shared_predictions)
    if set(baseline_index) != set(shared_index):
        missing_baseline = len(set(shared_index) - set(baseline_index))
        missing_shared = len(set(baseline_index) - set(shared_index))
        raise ValueError(
            "paired prediction membership differs: "
            f"missing_baseline={missing_baseline}, missing_shared={missing_shared}"
        )
    if set(parent_bindings) != set(baseline_index):
        raise ValueError("parent bindings differ from paired prediction membership")
    rows = []
    for key in sorted(baseline_index):
        baseline = baseline_index[key]
        shared = shared_index[key]
        validate_pair(baseline, shared)
        binding = parent_bindings[key]
        if not all(
            isinstance(binding.get(field), str) and binding[field]
            for field in ("parent_namespace", "parent")
        ):
            raise ValueError("paired prediction has an invalid parent binding")
        baseline_values = prediction_metrics(baseline, baseline_temperature)
        shared_values = prediction_metrics(shared, shared_temperature)
        rows.append({
            "source": baseline["source"],
            "parent_namespace": binding["parent_namespace"],
            "parent": binding["parent"],
            "difference": {
                metric: shared_values[metric] - value
                for metric, value in baseline_values.items()
                if metric in shared_values
            },
        })
    metrics = sorted({
        metric for row in rows for metric in row["difference"]
    })
    result = {}
    for index, metric in enumerate(metrics):
        metric_rows = [row for row in rows if metric in row["difference"]]
        source_values = defaultdict(list)
        for row in metric_rows:
            source_values[row["source"]].append(row["difference"][metric])
        weighted = sum(
            row["difference"][metric] for row in metric_rows
        ) / len(metric_rows)
        source_macro = sum(
            sum(values) / len(values) for values in source_values.values()
        ) / len(source_values)
        weighted_interval, source_macro_interval, bootstrap = bootstrap_differences(
            metric_rows, metric, samples, seed + index,
        )
        result[metric] = {
            "n": len(metric_rows),
            "parents": len({row["parent"] for row in metric_rows}),
            "parent_namespaces": len({
                row["parent_namespace"] for row in metric_rows
            }),
            "task_sources": len(source_values),
            "shared_minus_baseline": weighted,
            "source_macro_shared_minus_baseline": source_macro,
            "bootstrap": bootstrap,
            "parent_bootstrap_ci95": weighted_interval,
            "source_macro_parent_bootstrap_ci95": source_macro_interval,
        }
    return {
        "questions": len(rows),
        "parents": len({row["parent"] for row in rows}),
        "parent_namespaces": len({row["parent_namespace"] for row in rows}),
        "task_sources": len({row["source"] for row in rows}),
        "metrics": result,
    }


def validate_report_plan(report: dict, plan: dict, label: str) -> None:
    if report.get("comparison_plan_sha256") != plan["plan_sha256"]:
        raise ValueError(f"{label} report uses a different comparison plan")
    for track, split in SPLITS.items():
        section = report.get(split, {})
        binding = plan["suites"][track]
        if section.get("manifest_sha256") != binding["manifest_sha256"]:
            raise ValueError(f"{label} {track} manifest differs from the plan")
        expected_split = binding["files"]["development.jsonl"]["sha256"]
        if section.get("split_sha256") != expected_split:
            raise ValueError(f"{label} {track} split differs from the plan")
    temperature = report.get("temperature", {})
    calibration = plan["suites"]["decision"]
    if temperature.get("fit_manifest_sha256") != calibration["manifest_sha256"]:
        raise ValueError(f"{label} calibration manifest differs from the plan")
    expected = calibration["files"]["calibration.jsonl"]["sha256"]
    if temperature.get("fit_split_sha256") != expected:
        raise ValueError(f"{label} calibration split differs from the plan")


def validate_baseline_run(report: dict, plan: dict) -> None:
    fields = ("run_id", "spec_sha256", "generation", "checkpoint_sha256")
    actual = report.get("run", {})
    expected = plan["baseline_run"]
    if any(actual.get(field) != expected.get(field) for field in fields):
        raise ValueError("baseline report run differs from the frozen plan")
    if report.get("model") != "baseline-separate-question":
        raise ValueError("baseline report model identity is invalid")


def validate_completed_run(report: dict, run_directory: str | Path) -> dict:
    run_directory = Path(run_directory).resolve()
    run = json.loads((run_directory / "run.json").read_text())
    manifest = json.loads((run_directory / "latest.json").read_text())
    if run.get("spec_sha256") != canonical_sha256(run.get("spec")):
        raise ValueError("shared run specification digest is invalid")
    if (
        manifest.get("run_id") != run.get("run_id")
        or manifest.get("spec_sha256") != run.get("spec_sha256")
    ):
        raise ValueError("shared run manifest identity is invalid")
    current = manifest.get("current", {})
    if manifest.get("status") != "completed" or current.get("status") != "completed":
        raise ValueError("shared run is not completed")
    expected = {
        "run_id": run["run_id"],
        "spec_sha256": run["spec_sha256"],
        "generation": current["generation"],
        "checkpoint_sha256": current["sha256"],
    }
    if report.get("run") != expected:
        raise ValueError("shared report run differs from the completed run")
    return {"path": str(run_directory), **expected}


def validate_temperature(
    report: dict, calibration_predictions: list[dict], plan: dict, label: str,
) -> float:
    cleaned, audit = training_clean_calibration_predictions(
        calibration_predictions, plan,
    )
    if report["temperature"].get("common_clean") != audit:
        raise ValueError(f"{label} calibration clean audit is invalid")
    value = float(report["temperature"]["value"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} temperature is invalid")
    recomputed = fit_source_balanced_temperature(cleaned)
    if not math.isclose(value, recomputed, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"{label} temperature does not match its predictions")
    return value


def analyzer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "path": str(Path(__file__).resolve().relative_to(repository)),
        "sha256": file_sha256(Path(__file__)),
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "quantile_method": QUANTILE_METHOD,
        "bootstrap_design": BOOTSTRAP_DESIGN,
    }


def compare(args) -> dict:
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    baseline_path = Path(args.baseline).resolve()
    shared_path = Path(args.shared).resolve()
    baseline = load_report(baseline_path)
    shared = load_report(shared_path)
    plan = load_comparison_plan(args.comparison_plan)
    validate_report_plan(baseline, plan, "baseline")
    validate_report_plan(shared, plan, "shared")
    validate_baseline_run(baseline, plan)
    shared_run = validate_completed_run(shared, args.shared_run)

    suite_paths = {
        "decision": Path(args.decision_suite).resolve(),
        "transfer": Path(args.transfer_suite).resolve(),
        "korean": Path(args.korean_suite).resolve(),
    }
    for track, suite in suite_paths.items():
        validate_suite_binding(plan, track, suite)

    calibration_descriptor = plan["suites"]["decision"]["files"][
        "calibration.jsonl"
    ]
    _, calibration_inventory = validate_split_population(
        suite_paths["decision"], "calibration", calibration_descriptor,
    )
    baseline_calibration = baseline["predictions"].get("calibration")
    shared_calibration = shared["predictions"].get("calibration")
    validate_population(
        baseline_calibration, calibration_inventory, "baseline calibration",
    )
    validate_population(
        shared_calibration, calibration_inventory, "shared calibration",
    )
    baseline_temperature = validate_temperature(
        baseline, baseline_calibration, plan, "baseline",
    )
    shared_temperature = validate_temperature(
        shared, shared_calibration, plan, "shared",
    )

    baseline_digest = file_sha256(baseline_path)
    shared_digest = file_sha256(shared_path)
    seed_material = f'{plan["plan_sha256"]}:{baseline_digest}:{shared_digest}'
    root_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    comparisons = {}
    for index, (track, split) in enumerate(SPLITS.items()):
        descriptor = plan["suites"][track]["files"]["development.jsonl"]
        _, inventory = validate_split_population(
            suite_paths[track], "development", descriptor,
        )
        baseline_predictions = baseline["predictions"].get(split)
        shared_predictions = shared["predictions"].get(split)
        validate_population(
            baseline_predictions, inventory, f"baseline {track} development",
        )
        validate_population(
            shared_predictions, inventory, f"shared {track} development",
        )
        baseline_clean, baseline_audit = common_clean_predictions(
            baseline_predictions, plan, track,
        )
        shared_clean, shared_audit = common_clean_predictions(
            shared_predictions, plan, track,
        )
        if baseline_audit != shared_audit:
            raise ValueError(f"{track} reports have different clean-filter audits")
        for report, label in ((baseline, "baseline"), (shared, "shared")):
            section_audit = report[split].get("common_clean", {})
            reported = {
                field: section_audit.get(field)
                for field in (
                    "excluded_requests", "excluded_questions", "retained_questions",
                )
            }
            if reported != baseline_audit:
                raise ValueError(f"{label} {track} clean audit is invalid")
        clean_keys = {prediction_key(item) for item in baseline_clean}
        parent_bindings = {
            key: {
                "parent_namespace": inventory[key]["parent_namespace"],
                "parent": inventory[key]["parent"],
            }
            for key in clean_keys
        }
        comparisons[track] = {
            "manifest_sha256": plan["suites"][track]["manifest_sha256"],
            "split_sha256": descriptor["sha256"],
            "common_clean": baseline_audit,
            **paired_summary(
                baseline_clean,
                shared_clean,
                parent_bindings,
                baseline_temperature,
                shared_temperature,
                args.bootstrap_samples,
                root_seed + index * 100,
            ),
        }
    report = {
        "version": 1,
        "status": "compared",
        "comparison_plan_sha256": plan["plan_sha256"],
        "analyzer": analyzer_identity(),
        "baseline_report": {
            "path": str(baseline_path),
            "file_sha256": baseline_digest,
            "report_sha256": baseline["report_sha256"],
            "temperature": baseline_temperature,
        },
        "shared_report": {
            "path": str(shared_path),
            "file_sha256": shared_digest,
            "report_sha256": shared["report_sha256"],
            "temperature": shared_temperature,
            "run": shared_run,
        },
        "suites": {key: str(path) for key, path in suite_paths.items()},
        "difference_definition": (
            "shared minus baseline after each model's independently fitted "
            "temperature; positive favors shared for accuracy and negative "
            "favors shared for loss and Brier metrics"
        ),
        "uncertainty_scope": (
            "pointwise paired parent bootstrap for fixed models and fixed fitted "
            "temperatures; excludes training-seed and calibration-fit uncertainty"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "root_seed": root_seed,
        "locked_test_opened": False,
        "splits": comparisons,
    }
    report["report_sha256"] = canonical_report_sha256(report)
    atomic_json_save(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--shared-run", required=True)
    parser.add_argument("--comparison-plan", required=True)
    parser.add_argument("--decision-suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--korean-suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = compare(args)
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "difference_definition": report["difference_definition"],
        "splits": {
            split: values["metrics"]
            for split, values in report["splits"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
