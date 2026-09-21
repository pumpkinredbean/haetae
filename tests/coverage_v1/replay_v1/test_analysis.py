from __future__ import annotations

import copy
import math

import pytest

from experiments.coverage_v1.replay_v1.analysis import (
    BOOTSTRAP_DRAWS,
    aggregate_metrics,
    bootstrap_parent_differences,
    compute_pilot_gate,
    offensive_diagnostics,
    per_question_metrics,
    replication_summary,
    verify_pilot_gate,
)
from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object


def row(
    logits,
    label=0,
    *,
    primitive="choice",
    source="task",
    parent="parent",
    soft=None,
    keys=None,
):
    keys = keys or [str(index) for index in range(len(logits))]
    return {
        "logits": list(logits), "label": label, "soft": soft,
        "primitive": primitive, "task_source": source,
        "origin_source": "origin", "parent_id": parent,
        "ordered_option_keys": keys,
    }


def test_huge_common_offset_retains_log_two_nll() -> None:
    metrics = per_question_metrics(row([1e20, 1e20]), 1.0)
    assert metrics["nll"] == pytest.approx(math.log(2), rel=0, abs=1e-15)
    assert metrics["action"] == 0


def test_soft_brier_and_score_rps_are_independent() -> None:
    metrics = per_question_metrics(
        row([0.2, -0.3, 0.1], 1, primitive="score", soft=[0.1, 0.7, 0.2]),
        1.0,
    )
    probabilities = [
        math.exp(value) / math.fsum(math.exp(item) for item in [0.2, -0.3, 0.1])
        for value in [0.2, -0.3, 0.1]
    ]
    expected_histogram = math.fsum(
        (left - right) ** 2 for left, right in zip(probabilities, [0.1, 0.7, 0.2])
    )
    expected_annotation = 1 + math.fsum(value * value for value in probabilities) - 2 * math.fsum(
        left * right for left, right in zip(probabilities, [0.1, 0.7, 0.2])
    )
    assert metrics["brier_histogram"] == pytest.approx(expected_histogram)
    assert metrics["brier_annotation"] == pytest.approx(expected_annotation)
    assert metrics["rps"] is not None


def test_macro_f1_requires_one_semantic_ontology() -> None:
    rows = [row([2, 1], keys=["a", "b"]), row([1, 2], 1, keys=["x", "y"])]
    assert aggregate_metrics(rows, 1.0)["macro_f1"] is None
    rows[1]["ordered_option_keys"] = ["a", "b"]
    assert aggregate_metrics(rows, 1.0)["macro_f1"] == 1.0


def test_offensive_counts_polarity_and_tied_auroc() -> None:
    rows = [
        row([1, 0], 0, keys=["true", "false"]),
        row([0, 1], 0, keys=["true", "false"]),
        row([1, 0], 1, keys=["true", "false"]),
        row([0, 1], 1, keys=["true", "false"]),
    ]
    result = offensive_diagnostics(rows, 1.0)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["auroc"] == 0.5
    assert result["recall"]["value"] == 0.5


def test_bootstrap_keeps_whole_parent_and_fixed_nearest_rank() -> None:
    rows = []
    for parent, difference in (("p1", 1.0), ("p2", 0.0), ("p3", -1.0), ("p4", 0.5)):
        for task in ("a", "b"):
            rows.append({
                "origin_source": "origin", "parent_id": parent,
                "task_source": task, "targeted": difference, "control": 0.0,
            })
    first = bootstrap_parent_differences(rows, metric="accuracy", rng_seed=17)
    second = bootstrap_parent_differences(rows, metric="accuracy", rng_seed=17)
    assert first == second
    assert first["draws"] == BOOTSTRAP_DRAWS
    assert first["parents"] == 4


def metric_fixture(emotion_correct: int, offensive_correct: int) -> dict:
    return {
        "emotion_native": {"correct": emotion_correct, "accuracy": emotion_correct / 92},
        "offensive_native": {"correct": offensive_correct, "accuracy": offensive_correct / 80},
        "decision_common_clean": {"source_macro": {"accuracy": 0.7}},
        "korean_common_clean": {"source_macro": {"accuracy": 0.6}},
        "other_transfer": {"source_macro": {"nll": 0.8}},
        "score_common_clean": {"source_macro": {"rps": 0.2}},
    }


def requirements() -> list[dict]:
    return [
        {"id": "emotion_gain", "population": "emotion_native", "metric": "accuracy", "aggregation": "question_weighted", "operator": ">=", "threshold": 0.05},
        {"id": "offensive_gain", "population": "offensive_native", "metric": "accuracy", "aggregation": "question_weighted", "operator": ">=", "threshold": 0.05},
        {"id": "decision_retention", "population": "decision_common_clean", "metric": "accuracy", "aggregation": "equal_task_source_macro", "operator": ">=", "threshold": -0.01},
        {"id": "korean_retention", "population": "korean_common_clean", "metric": "accuracy", "aggregation": "equal_task_source_macro", "operator": ">=", "threshold": -0.01},
        {"id": "other_transfer_retention", "population": "other_transfer", "metric": "nll", "aggregation": "equal_task_source_macro", "operator": "<=", "threshold": 0.03},
        {"id": "ordinal_retention", "population": "score_common_clean", "metric": "rps", "aggregation": "equal_score_task_source_macro", "operator": "<=", "threshold": 0.02},
    ]


def test_gate_boundaries_use_integer_target_counts_and_unrounded_guardrails() -> None:
    control = metric_fixture(40, 40)
    targeted = metric_fixture(45, 44)
    targeted["decision_common_clean"]["source_macro"]["accuracy"] = 0.691
    targeted["korean_common_clean"]["source_macro"]["accuracy"] = 0.591
    targeted["other_transfer"]["source_macro"]["nll"] = 0.82
    targeted["score_common_clean"]["source_macro"]["rps"] = 0.21
    arguments = {
        "seed": 17, "protocol_sha256": "1" * 64,
        "producer_code": {"sha256": "2" * 64},
        "control_metrics": control, "targeted_metrics": targeted,
        "requirements": requirements(), "arm_checkpoints": {}, "arm_reports": {},
    }
    gate = compute_pilot_gate(**arguments)
    assert gate["all_requirements_passed"] is True
    assert gate["seed23_eligible"] is True
    verify_pilot_gate(gate, **arguments)
    tampered = copy.deepcopy(gate)
    tampered["all_requirements_passed"] = False
    tampered["gate_sha256"] = digest_object(tampered, "gate_sha256")
    with pytest.raises(ContractError, match="independent replay"):
        verify_pilot_gate(tampered, **arguments)


def test_replication_does_not_promote_without_uncertainty() -> None:
    control = metric_fixture(40, 40)
    targeted = metric_fixture(45, 44)
    base = {
        "seed": 17, "protocol_sha256": "1" * 64,
        "producer_code": {}, "control_metrics": control,
        "targeted_metrics": targeted, "requirements": requirements(),
        "arm_checkpoints": {}, "arm_reports": {},
    }
    first = compute_pilot_gate(**base)
    base["seed"] = 23
    second = compute_pilot_gate(**base)
    result = replication_summary(first, second)
    assert result["directional_replication"] is True
    assert result["replicated_candidate"] is False
    result = replication_summary(first, second, {
        "emotion_gain": {"one_sided_97_5_lower": 0.001},
        "offensive_gain": {"one_sided_97_5_lower": 0.002},
    })
    assert result["replicated_candidate"] is True
