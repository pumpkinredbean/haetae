"""Compare two frozen development reports with paired parent bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from haetae.checkpoint import atomic_json_save
from experiments.comparison_protocol import (
    common_clean_predictions,
    load_comparison_plan,
)
from experiments.evaluate_shared import canonical_report_sha256
from experiments.kev_adapter import file_sha256


SPLITS = {
    "decision": "decision_development",
    "transfer": "transfer_development",
    "korean": "korean_development",
}


def load_report(path: str | Path) -> dict:
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get("status") != "evaluated":
        raise ValueError(f"development report is not evaluated: {path}")
    if report.get("locked_test_opened") is not False:
        raise ValueError(f"development report does not attest a closed test: {path}")
    if canonical_report_sha256(report) != report.get("report_sha256"):
        raise ValueError(f"development report digest is invalid: {path}")
    return report


def prediction_key(prediction: dict) -> tuple:
    return (
        prediction["source"],
        prediction["request_id"],
        prediction["question_id"],
        prediction["state_sha256"],
    )


def parent_key(prediction: dict) -> str:
    parent = (
        prediction.get("group_id")
        or prediction.get("request_id")
        or "state:" + prediction["state_sha256"]
    )
    return f'{prediction["source"]}\0{parent}'


def index_predictions(predictions: list[dict]) -> dict[tuple, dict]:
    indexed = {}
    for prediction in predictions:
        key = prediction_key(prediction)
        if key in indexed:
            raise ValueError(f"duplicate prediction identity: {key}")
        indexed[key] = prediction
    return indexed


def validate_pair(baseline: dict, shared: dict) -> None:
    fields = (
        "group_id", "source", "type", "options", "option_keys", "label", "soft",
    )
    for field in fields:
        if baseline.get(field) != shared.get(field):
            raise ValueError(f"paired predictions differ in {field}")
    for name, prediction in (("baseline", baseline), ("shared", shared)):
        logits = prediction.get("logits")
        if not isinstance(logits, list) or len(logits) != len(prediction["options"]):
            raise ValueError(f"{name} logits do not match the option count")
        if not all(math.isfinite(float(value)) for value in logits):
            raise ValueError(f"{name} logits are not finite")


def target_distribution(prediction: dict) -> torch.Tensor:
    if prediction["soft"] is None:
        target = torch.zeros(len(prediction["options"]), dtype=torch.float64)
        target[prediction["label"]] = 1.0
        return target
    target = torch.tensor(prediction["soft"], dtype=torch.float64)
    if (
        target.ndim != 1
        or target.numel() != len(prediction["options"])
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


def quantile_interval(values: list[float]) -> dict:
    ordered = sorted(values)
    samples = len(ordered)
    return {
        "lower": ordered[max(0, math.floor(0.025 * samples) - 1)],
        "upper": ordered[min(samples - 1, math.ceil(0.975 * samples) - 1)],
        "samples": samples,
        "unit": "source-namespaced request parent",
    }


def bootstrap_differences(
    rows: list[dict], metric: str, samples: int, seed: int,
) -> tuple[dict, dict]:
    by_source_parent = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if metric in row["difference"]:
            by_source_parent[row["source"]][row["parent"]].append(
                row["difference"][metric]
            )
    if not by_source_parent or any(
        len(parents) < 2 for parents in by_source_parent.values()
    ):
        raise ValueError(f"{metric} needs at least two parents in every source")
    generator = random.Random(seed)
    weighted_estimates, source_macro_estimates = [], []
    for _ in range(samples):
        sampled_by_source = {}
        for source, parents in sorted(by_source_parent.items()):
            keys = sorted(parents)
            chosen = generator.choices(keys, k=len(keys))
            sampled_by_source[source] = [
                value for key in chosen for value in parents[key]
            ]
        all_values = [
            value
            for source_values in sampled_by_source.values()
            for value in source_values
        ]
        weighted_estimates.append(sum(all_values) / len(all_values))
        source_macro_estimates.append(sum(
            sum(source_values) / len(source_values)
            for source_values in sampled_by_source.values()
        ) / len(sampled_by_source))
    return (
        quantile_interval(weighted_estimates),
        quantile_interval(source_macro_estimates),
    )


def paired_summary(
    baseline_predictions: list[dict],
    shared_predictions: list[dict],
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
    rows = []
    for key in sorted(baseline_index):
        baseline = baseline_index[key]
        shared = shared_index[key]
        validate_pair(baseline, shared)
        baseline_values = prediction_metrics(baseline, baseline_temperature)
        shared_values = prediction_metrics(shared, shared_temperature)
        rows.append({
            "source": baseline["source"],
            "parent": parent_key(baseline),
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
        weighted_interval, source_macro_interval = bootstrap_differences(
            metric_rows, metric, samples, seed + index,
        )
        result[metric] = {
            "n": len(metric_rows),
            "shared_minus_baseline": weighted,
            "source_macro_shared_minus_baseline": source_macro,
            "parent_bootstrap_ci95": weighted_interval,
            "source_stratified_parent_bootstrap_ci95": source_macro_interval,
        }
    return {
        "questions": len(rows),
        "parents": len({row["parent"] for row in rows}),
        "sources": len({row["source"] for row in rows}),
        "metrics": result,
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
    plan_sha256 = plan["plan_sha256"]
    if baseline["comparison_plan_sha256"] != plan_sha256:
        raise ValueError("baseline report uses a different comparison plan")
    if shared["comparison_plan_sha256"] != plan_sha256:
        raise ValueError("shared report uses a different comparison plan")
    baseline_digest = file_sha256(baseline_path)
    shared_digest = file_sha256(shared_path)
    seed_material = f"{plan_sha256}:{baseline_digest}:{shared_digest}"
    root_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    comparisons = {}
    for index, (track, split) in enumerate(SPLITS.items()):
        baseline_section = baseline[split]
        shared_section = shared[split]
        for field in ("manifest_sha256", "split_sha256"):
            if baseline_section[field] != shared_section[field]:
                raise ValueError(f"{track} reports differ in {field}")
        baseline_clean, baseline_audit = common_clean_predictions(
            baseline["predictions"][split], plan, track,
        )
        shared_clean, shared_audit = common_clean_predictions(
            shared["predictions"][split], plan, track,
        )
        if baseline_audit != shared_audit:
            raise ValueError(f"{track} reports have different clean-filter audits")
        comparisons[track] = {
            "manifest_sha256": baseline_section["manifest_sha256"],
            "split_sha256": baseline_section["split_sha256"],
            "common_clean": baseline_audit,
            **paired_summary(
                baseline_clean,
                shared_clean,
                float(baseline["temperature"]["value"]),
                float(shared["temperature"]["value"]),
                args.bootstrap_samples,
                root_seed + index * 100,
            ),
        }
    report = {
        "version": 1,
        "status": "compared",
        "comparison_plan_sha256": plan_sha256,
        "baseline_report": {
            "path": str(baseline_path),
            "file_sha256": baseline_digest,
            "report_sha256": baseline["report_sha256"],
            "temperature": baseline["temperature"]["value"],
        },
        "shared_report": {
            "path": str(shared_path),
            "file_sha256": shared_digest,
            "report_sha256": shared["report_sha256"],
            "temperature": shared["temperature"]["value"],
        },
        "difference_definition": (
            "shared minus baseline; positive favors shared for accuracy and "
            "negative favors shared for loss and Brier metrics"
        ),
        "bootstrap_samples": args.bootstrap_samples,
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
    parser.add_argument("--comparison-plan", required=True)
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
