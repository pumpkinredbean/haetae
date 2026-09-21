import math
import json

import numpy as np
import pytest

import experiments.coverage_v1.diagnostic_v1 as diagnostic
from experiments.coverage_v1.diagnostic_v1 import (
    _variant_row,
    exclusion_index,
    historical_accuracy_baselines,
    select_support,
    semantic_variants,
)
from experiments.coverage_v1.run_diagnostic_v1 import (
    _read_jsonl,
    diagnostic_decisions,
    fit_positive_logistic,
    fit_temperature,
    paired_parent_bootstrap,
    require_reviewed_identity,
    semantic_results,
)


def raw_request(source, parent, state):
    return {
        "state": state,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"left": None, "right": None},
                "label": "left",
                "src": "fixture",
            }
        },
        "_meta": {"source": source, "id": parent, "group_id": parent},
    }


def test_exclusion_index_uses_rendered_state_and_source_parent():
    result = exclusion_index({
        "train": [raw_request("one", "p", "same rendered state")],
        "development": [raw_request("two", "q", "same rendered state")],
    })
    assert len(result["state_sha256"]) == 1
    assert result["state_sha256"][0]["roles"] == ["development", "train"]
    assert len(result["source_parents"]) == 2


def test_support_selection_removes_conflicts_duplicates_and_exclusions(monkeypatch):
    monkeypatch.setattr(diagnostic, "FIT_PER_CLASS", 1)
    monkeypatch.setattr(diagnostic, "CALIBRATION_PER_CLASS", 1)
    rows = []
    for label in diagnostic.NATIVE_LABELS["tweet_offensive"]:
        for index in range(4):
            state = f"{label}-{index}"
            rows.append({
                "source": "tweet_offensive",
                "split": "train",
                "row": len(rows),
                "parent_id": f"tweet_offensive/train/{len(rows)}",
                "state": state,
                "state_sha256": diagnostic.canonical_sha256(state),
                "label": label,
            })
    duplicate = dict(rows[0], row=100, parent_id="tweet_offensive/train/100")
    conflict = dict(rows[1], row=101, parent_id="tweet_offensive/train/101", label="true")
    rows.extend([duplicate, conflict])
    excluded = rows[2]["state_sha256"]
    fit, calibration, removals, summary = select_support(rows, {excluded}, set())
    assert len(fit) == 2
    assert len(calibration) == 2
    assert summary["same_label_duplicate_rows"] == 1
    assert summary["conflicting_states"] == 1
    assert {row["reason"] for row in removals} >= {
        "same_label_duplicate", "conflicting_labels", "rendered_state_exclusion",
    }


def development_row(question_type="choice"):
    if question_type == "choice":
        question = {
            "id": "emotion",
            "type": "choice",
            "instructions": "Emotion?",
            "options": ["sadness", "joy", "love", "anger", "fear", "surprise"],
            "option_keys": ["sadness", "joy", "love", "anger", "fear", "surprise"],
            "label": 1,
        }
        target = "emotion"
        population = "native_six_class"
        semantic_label = "joy"
    else:
        question = {
            "id": "offensive",
            "type": "noul",
            "instructions": "Offensive?",
            "options": ["yes: offensive", "no: not offensive"],
            "option_keys": ["true", "false"],
            "label": 1,
        }
        target = "tweet_offensive"
        population = "native_binary"
        semantic_label = "false"
    return {
        "development_id": f"{target}/test/1",
        "source": target,
        "parent_id": f"{target}/test/1",
        "target": target,
        "population": population,
        "state": "text",
        "state_sha256": diagnostic.canonical_sha256("text"),
        "question": question,
        "semantic_label": semantic_label,
    }


def test_semantic_variants_remap_choice_and_preserve_noul_polarity():
    choice, noul = development_row(), development_row("noul")
    rows = semantic_variants([choice, noul])
    choice_rows = [row for row in rows if row["target"] == "emotion"]
    noul_rows = [row for row in rows if row["target"] == "tweet_offensive"]
    assert len(choice_rows) == 8
    assert len(noul_rows) == 2
    rotated = next(row for row in choice_rows if row["family"] == "cyclic_rotation_1")
    assert rotated["question"]["option_keys"][-1] == "sadness"
    assert rotated["question"]["label"] == 0
    assert {row["family"] for row in noul_rows} == {
        "original", "native_taxonomy_lexicon",
    }
    assert all(row["question"]["option_keys"] == ["true", "false"] for row in noul_rows)


def test_variant_rejects_a_missing_semantic_label():
    row = development_row()
    with pytest.raises(ValueError, match="semantic label is absent"):
        _variant_row(row, "bad", ["sadness"], ["sadness"])


def historical_fixture():
    development = []
    report_rows = {"baseline": [], "shared": []}
    specifications = {
        "emotion": ("native_six_class", 92, 62, 33),
        "tweet_offensive": ("native_binary", 80, 50, 39),
    }
    for target, (population, questions, baseline_correct, shared_correct) in (
        specifications.items()
    ):
        for index in range(questions):
            request_id = f"{target}/development/{index}"
            question_id = f"{target}-{index}"
            development.append({
                "development_id": request_id,
                "source": target,
                "parent_id": f"{target}/parent/{index % 80}",
                "target": target,
                "population": population,
                "question": {"id": question_id},
            })
            for model, correct in (
                ("baseline", baseline_correct),
                ("shared", shared_correct),
            ):
                report_rows[model].append({
                    "request_id": request_id,
                    "question_id": question_id,
                    "logits": [1.0, 0.0] if index < correct else [0.0, 1.0],
                    "label": 0,
                })

    class Store:
        def read_json(self, evidence_id):
            model = "baseline" if evidence_id.startswith("baseline") else "shared"
            return {"predictions": {"transfer_development": report_rows[model]}}

        def binding(self, evidence_id):
            return {"evidence_id": evidence_id}

    return Store(), development, report_rows


def test_historical_accuracy_baselines_use_exact_matching_populations():
    store, development, _ = historical_fixture()
    result = historical_accuracy_baselines(store, development)
    assert result["targets"]["emotion"]["baseline"] == {
        "correct": 62,
        "questions": 92,
        "accuracy": pytest.approx(62 / 92),
    }
    assert result["targets"]["emotion"]["shared"] == {
        "correct": 33,
        "questions": 92,
        "accuracy": pytest.approx(33 / 92),
    }
    assert result["targets"]["tweet_offensive"][
        "baseline_minus_shared_accuracy"
    ] == pytest.approx(11 / 80)


def test_historical_accuracy_baselines_reject_duplicate_membership_substitution():
    store, development, report_rows = historical_fixture()
    report_rows["shared"][91] = dict(report_rows["shared"][0])
    with pytest.raises(ValueError, match="historical target membership differs"):
        historical_accuracy_baselines(store, development)


def test_runner_jsonl_reader_preserves_unicode_line_separator(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(
        json.dumps({"state": "left\u2028right"}, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    assert _read_jsonl(path) == [{"state": "left\u2028right"}]


def test_semantic_results_omit_slot_based_macro_f1():
    variants = [
        {
            "variant_id": "first",
            "development_id": "one",
            "target": "emotion",
            "population": "native_six_class",
            "family": "original",
            "question": {"option_keys": ["sadness", "joy"], "label": 1},
        },
        {
            "variant_id": "second",
            "development_id": "two",
            "target": "emotion",
            "population": "native_six_class",
            "family": "original",
            "question": {"option_keys": ["joy", "sadness"], "label": 0},
        },
    ]
    result = semantic_results(variants, [
        {"variant_id": "first", "logits": [0.0, 1.0]},
        {"variant_id": "second", "logits": [0.0, 1.0]},
    ])["emotion"]["native_six_class"]["original"]
    assert result["accuracy"] == 0.5
    assert "macro_f1" not in result


def test_reviewed_identity_is_required_before_inference():
    manifest = {"manifest_sha256": "a" * 64}
    protocol = {"protocol_sha256": "b" * 64}
    require_reviewed_identity(manifest, protocol, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="reviewed identity"):
        require_reviewed_identity(manifest, protocol, "c" * 64, "b" * 64)


def test_calibrators_are_positive_and_improve_separable_fixture():
    multiclass_logits = np.array([
        [6.0, 0.0], [5.0, 0.0], [0.0, 6.0], [0.0, 5.0],
    ])
    labels = np.array([0, 0, 1, 1])
    temperature = fit_temperature(multiclass_logits, labels)
    assert math.isfinite(temperature) and temperature > 0

    scores = np.array([-3.0, -2.0, 2.0, 3.0])
    slope, intercept = fit_positive_logistic(scores, labels)
    assert math.isfinite(slope) and slope > 0
    assert math.isfinite(intercept)


def test_parent_bootstrap_keeps_duplicate_questions_clustered():
    rows = [
        {"source": "emotion", "parent_id": "a"},
        {"source": "emotion", "parent_id": "a"},
        {"source": "emotion", "parent_id": "b"},
    ]
    result = paired_parent_bootstrap(
        rows,
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        seed=7,
    )
    assert result["parents"] == 2
    assert result["draws"] == diagnostic.BOOTSTRAP_DRAWS
    assert result["difference"] == pytest.approx(1 / 3)


def test_diagnostic_decisions_apply_predeclared_half_gap():
    metric = {
        "accuracy": 0.6,
        "nll": 0.4,
        "multiclass_brier": 0.3,
        "macro_f1": 0.5,
        "n": 10,
    }
    probes = {}
    semantics = {}
    for target, population in (
        ("emotion", "native_six_class"),
        ("tweet_offensive", "native_binary"),
    ):
        raw = dict(metric, nll=0.5, multiclass_brier=0.4)
        calibrated = dict(metric)
        if target == "tweet_offensive":
            raw["auroc"] = calibrated["auroc"] = 0.7
        probes[target] = {
            "shared": {"raw": raw, "calibrated": calibrated},
            "accessible_feature_regression": False,
        }
        semantics[target] = {
            population: {
                "original": {"accuracy": 0.4},
                "native_taxonomy_lexicon": {"accuracy": 0.6},
            }
        }
    protocol = {"metrics": {
        "historical_accuracy_gaps": {
            "emotion": 0.28,
            "tweet_offensive": 0.14,
        },
        "historical_accuracy_baselines": {"targets": {
            "emotion": {"shared": {"accuracy": 0.32}},
            "tweet_offensive": {"shared": {"accuracy": 0.49}},
        }},
    }}
    result = diagnostic_decisions(probes, semantics, protocol)
    assert result["emotion"]["semantic_family_recovers_half_gap"][
        "native_taxonomy_lexicon"
    ] is True
    assert result["tweet_offensive"]["calibration_only_pattern"] is True


def test_native_emotion_recovery_does_not_use_all_emotion_baseline():
    probes = {
        "emotion": {
            "shared": {
                "raw": {
                    "accuracy": 45 / 92,
                    "nll": 1.0,
                    "multiclass_brier": 1.0,
                },
                "calibrated": {
                    "accuracy": 45 / 92,
                    "nll": 1.0,
                    "multiclass_brier": 1.0,
                },
            },
            "accessible_feature_regression": False,
        },
        "tweet_offensive": {
            "shared": {
                "raw": {
                    "accuracy": 39 / 80,
                    "nll": 1.0,
                    "multiclass_brier": 1.0,
                    "auroc": 0.5,
                },
                "calibrated": {
                    "accuracy": 39 / 80,
                    "nll": 1.0,
                    "multiclass_brier": 1.0,
                    "auroc": 0.5,
                },
            },
            "accessible_feature_regression": False,
        },
    }
    semantics = {
        "emotion": {"native_six_class": {"original": {"accuracy": 0.4}}},
        "tweet_offensive": {"native_binary": {"original": {"accuracy": 0.4}}},
    }
    protocol = {"metrics": {
        "historical_accuracy_gaps": {
            "emotion": 29 / 92,
            "tweet_offensive": 11 / 80,
        },
        "historical_accuracy_baselines": {"targets": {
            "emotion": {"shared": {"accuracy": 33 / 92}},
            "tweet_offensive": {"shared": {"accuracy": 39 / 80}},
        }},
    }}
    result = diagnostic_decisions(probes, semantics, protocol)
    assert result["emotion"]["shared_probe_accuracy_recovery"] == pytest.approx(
        12 / 92
    )
    assert result["emotion"]["shared_probe_recovers_half_gap"] is False
