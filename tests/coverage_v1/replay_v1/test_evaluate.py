from __future__ import annotations

import copy

import pytest
import torch

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.evaluate import (
    cpu_first_logits,
    run_initial_parity,
    validate_predictions,
)


def parity_rows() -> tuple[list[dict], list[dict]]:
    observations, references = [], []
    for index in range(196):
        key = ["transfer", f"request-{index}", "answer", "original"]
        base = {
            "key": key,
            "state_sha256": digest_object(["state", index]),
            "question_sha256": digest_object(["question", index]),
            "logits": [0.25, -0.5],
        }
        observations.append(copy.deepcopy(base))
        references.append(copy.deepcopy(base))
    return observations, references


def test_cpu_first_output_transfer_requires_float32() -> None:
    result = cpu_first_logits(torch.tensor([0.25, -0.5], dtype=torch.float32))
    assert result == {"cpu_float32": [0.25, -0.5], "logits": [0.25, -0.5]}
    with pytest.raises(ContractError, match="float32"):
        cpu_first_logits(torch.tensor([0.25, -0.5], dtype=torch.float64))
    with pytest.raises(ContractError, match="nonfinite"):
        cpu_first_logits(torch.tensor([0.25, float("nan")], dtype=torch.float32))


def test_step_zero_parity_exact_and_zeroed_outputs_fail() -> None:
    observations, references = parity_rows()
    assert run_initial_parity(observations, references)["passed"] is True
    zeroed = copy.deepcopy(observations)
    for row in zeroed:
        row["logits"] = [0.0, 0.0]
    result = run_initial_parity(zeroed, references)
    assert result["passed"] is False
    assert result["material_action_changes"] == 0
    assert result["raw_components_passed"] is False


def prediction_fixture() -> tuple[dict, dict]:
    inventory = {
        "key": ["decision", "request", "answer", "original"],
        "request_id": "request", "question_id": "answer",
        "origin_source": "origin", "parent_id": "parent",
        "task_source": "task", "primitive": "choice",
        "population_ids": ["decision_common_clean"],
        "state_sha256": "1" * 64, "question_sha256": "2" * 64,
        "ordered_option_keys": ["a", "b"],
        "ordered_option_descriptions": ["a", "b"],
        "label": 0, "soft": None, "encoding_sha256": "3" * 64,
        "packed_tokens": 10,
    }
    row = {"schema_version": 1, **inventory, "cpu_float32": [0.25, -0.5], "logits": [0.25, -0.5]}
    row["row_sha256"] = digest_object(row)
    return row, inventory


def test_prediction_inventory_rejects_reorder_and_inexact_widening() -> None:
    row, inventory = prediction_fixture()
    assert validate_predictions([row], [inventory]) == [row]
    bad = copy.deepcopy(row)
    bad["cpu_float32"][0] = 0.1
    bad["logits"][0] = 0.1
    bad["row_sha256"] = digest_object(bad, "row_sha256")
    with pytest.raises(ContractError, match="float32"):
        validate_predictions([bad], [inventory])
    with pytest.raises(ContractError, match="differs"):
        validate_predictions([row], [{**inventory, "request_id": "foreign"}])
