import math

import numpy as np
import pytest

import experiments.coverage_v1.diagnostic_v1 as diagnostic
from experiments.coverage_v1.diagnostic_v1 import (
    _variant_row,
    exclusion_index,
    select_support,
    semantic_variants,
)
from experiments.coverage_v1.run_diagnostic_v1 import (
    diagnostic_decisions,
    fit_positive_logistic,
    fit_temperature,
    paired_parent_bootstrap,
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
    protocol = {"metrics": {"historical_accuracy_gaps": {
        "emotion": 0.28,
        "tweet_offensive": 0.14,
    }}}
    result = diagnostic_decisions(probes, semantics, protocol)
    assert result["emotion"]["semantic_family_recovers_half_gap"][
        "native_taxonomy_lexicon"
    ] is True
    assert result["tweet_offensive"]["calibration_only_pattern"] is True
