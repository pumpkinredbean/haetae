"""CPU-only paired metrics, bootstrap uncertainty and T05 gates."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object

FIXED_TEMPERATURE = 1.5197255188671874
BOOTSTRAP_DRAWS = 20_000
LOWER_INDEX = 499
UPPER_INDEX = 19_499


def _target(row: Mapping[str, Any]) -> list[float]:
    logits = row["logits"]
    soft = row.get("soft")
    label = row["label"]
    if not isinstance(logits, list) or len(logits) < 2:
        raise ContractError("metric row needs at least two logits")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in logits):
        raise ContractError("metric logits must be finite numbers")
    if type(label) is not int or not 0 <= label < len(logits):
        raise ContractError("metric hard label is invalid")
    if soft is None:
        return [1.0 if index == label else 0.0 for index in range(len(logits))]
    if (
        not isinstance(soft, list) or len(soft) != len(logits)
        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in soft)
        or not math.isclose(sum(soft), 1.0, rel_tol=0, abs_tol=1e-6)
    ):
        raise ContractError("metric soft target is invalid")
    return [float(value) for value in soft]


def _probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ContractError("temperature must be finite and positive")
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def per_question_metrics(row: Mapping[str, Any], temperature: float) -> dict[str, Any]:
    target = _target(row)
    probabilities = _probabilities(row["logits"], temperature)
    action = max(range(len(row["logits"])), key=lambda index: row["logits"][index])
    nll = -math.fsum(
        weight * math.log(probability)
        for weight, probability in zip(target, probabilities)
        if weight > 0
    )
    brier_histogram = math.fsum(
        (probability - weight) ** 2
        for probability, weight in zip(probabilities, target)
    )
    brier_annotation = 1.0 + math.fsum(value * value for value in probabilities) - 2 * math.fsum(
        probability * weight for probability, weight in zip(probabilities, target)
    )
    result = {
        "correct": int(action == row["label"]),
        "action": action,
        "confidence": max(probabilities),
        "nll": nll,
        "brier_histogram": brier_histogram,
        "brier_annotation": brier_annotation,
    }
    if row["primitive"] == "score":
        probability_cdf = []
        target_cdf = []
        running_probability = 0.0
        running_target = 0.0
        for probability, weight in zip(probabilities[:-1], target[:-1]):
            running_probability += probability
            running_target += weight
            probability_cdf.append(running_probability)
            target_cdf.append(running_target)
        result["rps"] = math.fsum(
            (left - right) ** 2 for left, right in zip(probability_cdf, target_cdf)
        ) / len(probability_cdf)
        expected_index = math.fsum(index * value for index, value in enumerate(probabilities))
        result["expected_index_mae"] = abs(expected_index - row["label"])
    else:
        result["rps"] = None
        result["expected_index_mae"] = None
    return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ContractError("cannot average an empty metric population")
    return math.fsum(values) / len(values)


def _macro_f1(rows: list[Mapping[str, Any]], metrics: list[Mapping[str, Any]]) -> float | None:
    ontologies = {tuple(row["ordered_option_keys"]) for row in rows}
    if len(ontologies) != 1:
        return None
    labels = next(iter(ontologies))
    scores = []
    for index in range(len(labels)):
        true_positive = sum(
            row["label"] == index and metric["action"] == index
            for row, metric in zip(rows, metrics)
        )
        false_positive = sum(
            row["label"] != index and metric["action"] == index
            for row, metric in zip(rows, metrics)
        )
        false_negative = sum(
            row["label"] == index and metric["action"] != index
            for row, metric in zip(rows, metrics)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return _mean(scores)


def aggregate_metrics(rows: Iterable[Mapping[str, Any]], temperature: float) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ContractError("metric population is empty")
    metrics = [per_question_metrics(row, temperature) for row in rows]
    result = {
        "questions": len(rows),
        "correct": sum(metric["correct"] for metric in metrics),
        "accuracy": _mean(metric["correct"] for metric in metrics),
        "nll": _mean(metric["nll"] for metric in metrics),
        "brier_histogram": _mean(metric["brier_histogram"] for metric in metrics),
        "brier_annotation": _mean(metric["brier_annotation"] for metric in metrics),
        "macro_f1": _macro_f1(rows, metrics),
    }
    score = [metric for metric in metrics if metric["rps"] is not None]
    result["rps"] = _mean(metric["rps"] for metric in score) if score else None
    result["expected_index_mae"] = (
        _mean(metric["expected_index_mae"] for metric in score) if score else None
    )
    sources: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        sources[row["task_source"]].append(index)
    per_source = {}
    for source, indices in sorted(sources.items()):
        per_source[source] = {
            "questions": len(indices),
            "accuracy": _mean(metrics[index]["correct"] for index in indices),
            "nll": _mean(metrics[index]["nll"] for index in indices),
            "brier_histogram": _mean(metrics[index]["brier_histogram"] for index in indices),
            "brier_annotation": _mean(metrics[index]["brier_annotation"] for index in indices),
            "rps": (
                _mean(metrics[index]["rps"] for index in indices)
                if all(metrics[index]["rps"] is not None for index in indices) else None
            ),
        }
    result["per_source"] = per_source
    result["source_macro"] = {
        name: _mean(value[name] for value in per_source.values())
        for name in ("accuracy", "nll", "brier_histogram", "brier_annotation")
    }
    rps_sources = [value["rps"] for value in per_source.values() if value["rps"] is not None]
    result["source_macro"]["rps"] = _mean(rps_sources) if rps_sources else None
    bins = []
    ece = 0.0
    for bin_index in range(15):
        lower, upper = bin_index / 15, (bin_index + 1) / 15
        members = [
            metric for metric in metrics
            if lower <= metric["confidence"] < upper
            or (bin_index == 14 and metric["confidence"] == 1.0)
        ]
        if members:
            confidence = _mean(metric["confidence"] for metric in members)
            accuracy = _mean(metric["correct"] for metric in members)
            ece += len(members) / len(metrics) * abs(confidence - accuracy)
            bins.append({
                "index": bin_index, "count": len(members),
                "confidence": confidence, "accuracy": accuracy,
            })
    result["ece15"] = ece
    result["ece15_bins"] = bins
    return result


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "value": 0.0 if denominator == 0 else numerator / denominator,
        "denominator_zero": denominator == 0,
    }


def offensive_diagnostics(rows: Iterable[Mapping[str, Any]], temperature: float) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ContractError("offensive population is empty")
    labels = {tuple(row["ordered_option_keys"]) for row in rows}
    if labels != {("true", "false")}:
        raise ContractError("offensive population has the wrong semantic ontology")
    pairs = []
    for row in rows:
        metric = per_question_metrics(row, temperature)
        truth = row["label"] == 0
        prediction = metric["action"] == 0
        pairs.append((truth, prediction, row["logits"][0] - row["logits"][1]))
    tp = sum(truth and prediction for truth, prediction, _ in pairs)
    fp = sum(not truth and prediction for truth, prediction, _ in pairs)
    tn = sum(not truth and not prediction for truth, prediction, _ in pairs)
    fn = sum(truth and not prediction for truth, prediction, _ in pairs)
    positives = [score for truth, _, score in pairs if truth]
    negatives = [score for truth, _, score in pairs if not truth]
    if not positives or not negatives:
        raise ContractError("offensive AUROC requires both classes")
    wins = math.fsum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives
    )
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall["value"] + specificity["value"]) / 2,
        "accuracy": (tp + tn) / len(rows),
        "predicted_positive_rate": (tp + fp) / len(rows),
        "positive_label_prevalence": (tp + fn) / len(rows),
        "auroc": wins / (len(positives) * len(negatives)),
    }


def bootstrap_parent_differences(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    rng_seed: int,
    draws: int = BOOTSTRAP_DRAWS,
    aggregation: str = "question_weighted",
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if draws != BOOTSTRAP_DRAWS:
        raise ContractError("T05 bootstrap requires exactly 20,000 draws")
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[(row["origin_source"], row["parent_id"])].append(row)
    strata: dict[tuple[str, tuple[str, ...]], list[tuple[str, str]]] = defaultdict(list)
    for parent, parent_rows in by_parent.items():
        incidence = tuple(sorted({row["task_source"] for row in parent_rows}))
        strata[(parent[0], incidence)].append(parent)
    if any(len(parents) < 2 for parents in strata.values()):
        raise ContractError("bootstrap stratum has fewer than two parents")
    rng = random.Random(rng_seed)
    samples = []
    for _ in range(draws):
        selected = []
        for key in sorted(strata):
            parents = sorted(strata[key])
            selected.extend(rng.choice(parents) for _ in parents)
        values = []
        for parent in selected:
            for row in by_parent[parent]:
                values.append((row["task_source"], row["targeted"] - row["control"]))
        if aggregation == "question_weighted":
            samples.append(_mean(value for _, value in values))
        elif aggregation in {"equal_task_source_macro", "equal_score_task_source_macro"}:
            by_source: dict[str, list[float]] = defaultdict(list)
            for source, value in values:
                by_source[source].append(value)
            samples.append(_mean(_mean(cell) for cell in by_source.values()))
        else:
            raise ContractError("unknown bootstrap aggregation")
    samples.sort()
    return {
        "draws": draws,
        "rng_seed": rng_seed,
        "parents": len(by_parent),
        "strata": len(strata),
        "lower95": samples[LOWER_INDEX],
        "upper95": samples[UPPER_INDEX],
        "one_sided_97_5_lower": samples[LOWER_INDEX],
    }


def _metric_value(metrics: Mapping[str, Any], requirement: Mapping[str, Any]) -> float:
    population = metrics[requirement["population"]]
    if "raw" in population and "fixed" in population:
        population = population["fixed" if "temperature" in requirement else "raw"]
    aggregation = requirement["aggregation"]
    metric = requirement["metric"]
    if aggregation in {"equal_task_source_macro", "equal_score_task_source_macro"}:
        return float(population["source_macro"][metric])
    return float(population[metric])


def compute_pilot_gate(
    *,
    seed: int,
    protocol_sha256: str,
    producer_code: Mapping[str, Any],
    control_metrics: Mapping[str, Any],
    targeted_metrics: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    arm_checkpoints: Mapping[str, Any],
    arm_reports: Mapping[str, Any],
) -> dict[str, Any]:
    decisions = []
    for requirement in requirements:
        control = _metric_value(control_metrics, requirement)
        targeted = _metric_value(targeted_metrics, requirement)
        delta = targeted - control
        passed = delta >= requirement["threshold"] if requirement["operator"] == ">=" else delta <= requirement["threshold"]
        if requirement["id"] == "emotion_gain":
            passed = (
                targeted_metrics["emotion_native"]["correct"]
                - control_metrics["emotion_native"]["correct"] >= 5
            )
        elif requirement["id"] == "offensive_gain":
            passed = (
                targeted_metrics["offensive_native"]["correct"]
                - control_metrics["offensive_native"]["correct"] >= 4
            )
        decisions.append({
            "id": requirement["id"], "control": control, "targeted": targeted,
            "delta": delta, "operator": requirement["operator"],
            "threshold": requirement["threshold"], "passed": passed,
        })
    result = {
        "schema_version": 1,
        "status": "passed" if all(row["passed"] for row in decisions) else "failed",
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "producer_code": dict(producer_code),
        "arm_checkpoints": dict(arm_checkpoints),
        "arm_reports": dict(arm_reports),
        "result_descriptor": {"decisions_sha256": digest_object(decisions)},
        "requirements": decisions,
        "all_requirements_passed": all(row["passed"] for row in decisions),
        "seed23_eligible": seed == 17 and all(row["passed"] for row in decisions),
    }
    result["gate_sha256"] = digest_object(result)
    return result


def verify_pilot_gate(
    gate: Mapping[str, Any],
    **arguments,
) -> dict[str, Any]:
    actual = copy.deepcopy(dict(gate))
    if actual.get("gate_sha256") != digest_object(actual, "gate_sha256"):
        raise ContractError("pilot gate self-digest mismatch")
    expected = compute_pilot_gate(**arguments)
    if actual != expected:
        raise ContractError("pilot gate differs from independent replay")
    return expected


def replication_summary(
    seed17: Mapping[str, Any],
    seed23: Mapping[str, Any],
    seed17_target_bootstrap: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    for gate in (seed17, seed23):
        if gate.get("gate_sha256") != digest_object(gate, "gate_sha256"):
            raise ContractError("replication gate self-digest mismatch")
    target_ids = {"emotion_gain", "offensive_gain"}
    seed23_targets = {
        row["id"]: row["delta"] for row in seed23["requirements"] if row["id"] in target_ids
    }
    directional = set(seed23_targets) == target_ids and all(value > 0 for value in seed23_targets.values())
    bounds = seed17_target_bootstrap or {}
    positive_bounds = (
        set(bounds) == target_ids
        and all(bounds[target]["one_sided_97_5_lower"] > 0 for target in target_ids)
    )
    result = {
        "both_point_gates_pass": bool(
            seed17["all_requirements_passed"] and seed23["all_requirements_passed"]
        ),
        "directional_replication": directional,
        "seed17_positive_one_sided_bounds": positive_bounds,
        "replicated_candidate": bool(
            seed17["all_requirements_passed"]
            and seed23["all_requirements_passed"]
            and positive_bounds
        ),
    }
    return result


def verify_result(result: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    if result.get("result_sha256") != digest_object(result, "result_sha256"):
        raise ContractError("paired result self-digest mismatch")
    if dict(result) != dict(recomputed):
        raise ContractError("paired result differs from exact replay")
