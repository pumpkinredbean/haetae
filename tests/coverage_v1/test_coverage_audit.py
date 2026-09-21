import copy
import math
from pathlib import Path

import pytest

from experiments.coverage_v1.audit import (
    _compare_grouped_metrics,
    _prediction_metrics,
    _validate_input_budget,
    _validate_report_identity,
    audit_exposure_weights,
    audit_role_overlap,
    audit_semantics,
    audit_supervised_coverage,
)
from experiments.coverage_v1.registry import canonical_sha256


def request(
    identity,
    *,
    origin="fixture",
    state=None,
    questions=None,
    repo="fixture/repo",
    revision="1" * 40,
    split="development",
):
    return {
        "state": state or f"state {identity}",
        "questions": questions or {
            "answer": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"left": "Left", "right": "Right"},
                "label": "left",
                "src": origin,
            }
        },
        "_meta": {
            "source": origin,
            "repo": repo,
            "revision": revision,
            "split": split,
            "id": f"{origin}/{split}/{identity}",
            "group_id": f"{origin}/{split}/{identity}",
        },
    }


def roles(*, transfer=None):
    return {
        "train": [request("train", split="train")],
        "calibration": [request("calibration", split="calibration")],
        "decision_development": [request("decision")],
        "transfer_development": transfer or [request("transfer")],
        "korean_development": [request("korean", origin="korean")],
    }


def target_requests():
    emotion = request(
        "emotion",
        origin="emotion",
        repo="dair-ai/emotion",
        revision="cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        split="test",
        questions={
            "emotion": {
                "type": "choice",
                "instructions": "Which emotion does the writer express?",
                "criteria": {
                    "sadness": None,
                    "joy": None,
                    "love": None,
                    "anger": None,
                    "fear": None,
                    "surprise": None,
                },
                "label": "joy",
                "src": "emotion",
            }
        },
    )
    offensive = request(
        "offensive",
        origin="tweet_offensive",
        repo="cardiffnlp/tweet_eval",
        revision="b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
        split="test",
        questions={
            "offensive": {
                "type": "noul",
                "instructions": "Is this post offensive?",
                "criteria": {"true": "Offensive", "false": "Not offensive"},
                "label": False,
                "src": "tweet_offensive",
            }
        },
    )
    return [emotion, offensive]


def test_semantics_preserves_choice_order_and_yes_first_noul():
    result = audit_semantics(roles(transfer=target_requests()))
    assert result["status"] == "passed"
    assert result["targets"]["emotion"]["native_ontology"] == [
        "sadness", "joy", "love", "anger", "fear", "surprise",
    ]
    assert result["targets"]["emotion"]["augmentation_labels"] == ["none_of_these"]
    assert result["targets"]["tweet_offensive"]["observed_option_orders"] == [
        ["true", "false"]
    ]
    assert result["targets"]["tweet_offensive"]["label_counts"] == {"false": 1}


def test_semantics_rejects_negative_and_boolean_score_labels():
    for invalid in (-1, True):
        broken = roles(transfer=target_requests())
        broken["train"][0]["questions"]["answer"] = {
            "type": "score",
            "instructions": "Score.",
            "criteria": ["low", "high"],
            "label": invalid,
            "src": "fixture",
        }
        with pytest.raises(ValueError):
            audit_semantics(broken)


def test_role_audit_detects_rendered_state_overlap_and_missing_role():
    values = roles()
    values["calibration"][0]["state"] = values["train"][0]["state"]
    result = audit_role_overlap(values)
    assert result["status"] == "failed"
    assert result["overlap_status"] == "overlap_detected"
    assert result["label_conflict_status"] == "passed"
    assert result["pairwise_overlap"]["train__calibration"]["rendered_state_overlap"] == 1
    missing = dict(values)
    missing.pop("calibration")
    with pytest.raises(ValueError, match="roles are incomplete"):
        audit_role_overlap(missing)


def test_role_audit_counts_exact_duplicates_and_label_conflicts():
    values = roles()
    duplicate = request("duplicate", split="train")
    duplicate["state"] = values["train"][0]["state"]
    values["train"].append(duplicate)
    result = audit_role_overlap(values)
    assert result["roles"]["train"]["duplicate_requests"] == 1

    conflict = request("conflict", split="train")
    conflict["state"] = values["train"][0]["state"]
    conflict["questions"]["answer"]["label"] = "right"
    values["train"].append(conflict)
    result = audit_role_overlap(values)
    assert result["roles"]["train"]["conflicting_rendered_question_labels"] == 1
    assert result["status"] == "failed"
    assert result["label_conflict_status"] == "conflict_detected"


def test_role_audit_detects_choice_conflict_after_option_reordering():
    values = roles()
    first = values["train"][0]
    second = copy.deepcopy(first)
    second["_meta"]["id"] = "fixture/train/reordered"
    second["_meta"]["group_id"] = "fixture/train/reordered"
    second["questions"]["answer"]["criteria"] = {
        "right": "Right",
        "left": "Left",
    }
    second["questions"]["answer"]["label"] = "right"
    values["train"].append(second)
    result = audit_role_overlap(values)
    assert result["roles"]["train"]["conflicting_rendered_question_labels"] == 1
    assert result["label_conflict_status"] == "conflict_detected"
    assert result["status"] == "failed"


def test_coverage_preserves_task_fragmentation_and_source_order():
    fragmented = request(
        "fragmented",
        origin="news",
        split="train",
        questions={
            "topic": {
                "type": "choice",
                "instructions": "Topic?",
                "criteria": {"a": None, "b": None},
                "label": "a",
                "src": "news_topic",
            },
            "check": {
                "type": "noul",
                "instructions": "Is it news?",
                "label": True,
                "src": "news_check",
            },
        },
    )
    values = roles()
    values["train"] = [fragmented]
    baseline = {
        "spec": {
            "config": {
                "sources": "emotion,ag_news",
                "per_source": 1500,
                "eval_per_source": 200,
            }
        }
    }
    first = audit_supervised_coverage(values, baseline)
    reordered = {key: list(reversed(value)) for key, value in reversed(values.items())}
    second = audit_supervised_coverage(reordered, baseline)
    assert first == second
    train = first["shared"]["train"]
    assert train["questions_by_task_source"] == {"news_check": 1, "news_topic": 1}
    assert train["questions_by_origin_and_task"] == {
        "news\0news_check": 1,
        "news\0news_topic": 1,
    }


def test_exposure_replays_request_mean_objective():
    first = request(
        "first",
        origin="origin",
        split="train",
        questions={
            "a": {
                "type": "choice",
                "instructions": "A?",
                "criteria": {"x": None, "y": None},
                "label": "x",
                "src": "task_a",
            },
            "b": {
                "type": "noul",
                "instructions": "B?",
                "label": True,
                "src": "task_b",
            },
        },
    )
    second = request("second", origin="origin", split="train")
    run = {
        "spec": {
            "config": {"steps": 1, "batch": 2, "seed": 17},
            "experiment": {
                "request_objective": "mean question loss, then mean request loss",
                "train_descriptor": {"records": 2},
            },
        }
    }
    result = audit_exposure_weights([first, second], run)
    assert result["request_draws"] == 2
    assert result["question_draws"] == 3
    assert result["coefficient_total"] == pytest.approx(1.0)
    assert result["groups"]["task_source"]["task_a"]["coefficient_sum"] == pytest.approx(0.25)
    assert result["groups"]["task_source"]["task_b"]["coefficient_sum"] == pytest.approx(0.25)


def test_stale_report_run_binding_is_rejected():
    run = {"run_id": "run", "spec_sha256": "spec"}
    manifest = {
        "current": {"generation": 3, "sha256": "checkpoint"},
    }
    plan = {"plan_sha256": "plan"}
    report = {
        "status": "evaluated",
        "locked_test_opened": False,
        "comparison_plan_sha256": "plan",
        "run": {
            "run_id": "run",
            "spec_sha256": "spec",
            "generation": 3,
            "checkpoint_sha256": "checkpoint",
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    _validate_report_identity("fixture", report, run, manifest, plan)
    stale = copy.deepcopy(report)
    stale["run"]["generation"] = 2
    unsigned = copy.deepcopy(stale)
    unsigned.pop("report_sha256")
    stale["report_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="another completed run"):
        _validate_report_identity("fixture", stale, run, manifest, plan)


def test_grouped_metric_check_covers_macro_and_source_membership():
    metrics = {
        key: 0.25 for key in (
            "accuracy", "nll", "brier_histogram", "brier_annotation",
        )
    }
    actual = {
        "all": dict(metrics),
        "source_macro": dict(metrics),
        "source": {"source": dict(metrics)},
    }
    expected = copy.deepcopy(actual)
    expected["source_macro"]["accuracy"] = 0.987654
    with pytest.raises(ValueError, match="source macro"):
        _compare_grouped_metrics(actual, expected, "fixture")
    expected = copy.deepcopy(actual)
    expected["source"]["other"] = expected["source"].pop("source")
    with pytest.raises(ValueError, match="source membership"):
        _compare_grouped_metrics(actual, expected, "fixture")


def test_input_budget_requires_positive_integer_and_shared_consistency():
    base = {
        "request_id": "request",
        "state_sha256": "a" * 64,
        "packed_request_tokens": 17,
    }
    result = _validate_input_budget("shared", {"role": [base]}, 32)
    assert result["maximum_observed"] == 17
    for invalid in (-1, 0, True, 1.5):
        row = {**base, "packed_request_tokens": invalid}
        with pytest.raises(ValueError, match="positive integer"):
            _validate_input_budget("shared", {"role": [row]}, 32)
    other = {**base, "packed_request_tokens": 18}
    with pytest.raises(ValueError, match="differs across questions"):
        _validate_input_budget("shared", {"role": [base, other]}, 32)


def test_prediction_nll_uses_centered_log_probabilities():
    row = {
        "logits": [1000.0, 0.0],
        "options": ["first", "second"],
        "label": 1,
        "soft": None,
    }
    assert _prediction_metrics(row, 1.0)["nll"] == pytest.approx(1000.0)


def test_equal_large_logits_preserve_uniform_nll():
    row = {
        "logits": [1e16, 1e16],
        "options": ["left", "right"],
        "label": 1,
        "soft": None,
    }
    assert _prediction_metrics(row, 1.0)["nll"] == pytest.approx(math.log(2.0))


def test_coverage_modules_do_not_import_model_runtimes():
    root = Path(__file__).resolve().parents[2] / "experiments" / "coverage_v1"
    sources = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "import torch" not in sources
    assert "from transformers" not in sources
