from __future__ import annotations

import copy
import math

import pytest
import torch

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.train import (
    apply_tape_slot,
    initialize_arm,
    microbatch_coefficients,
    run_update,
)
from experiments.train_shared import question_loss, request_batch_loss


class TinyDecisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)

    def forward(self, encodings):
        outputs = []
        for encoding in encodings:
            value = self.head(self.backbone(encoding))
            outputs.append([value])
        return outputs


class SpyOptimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.steps = 0
        self.zeroes = 0

    def zero_grad(self, set_to_none=True):
        self.zeroes += 1
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def step(self):
        self.steps += 1


class SpyScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def question(kind: str = "choice", label: int = 0, soft=None) -> dict:
    return {
        "id": "answer", "type": kind, "instructions": "Choose.",
        "options": ["a", "b"], "option_keys": ["a", "b"],
        "label": label, "soft": soft, "source": "fixture",
    }


def requests(count: int = 8) -> list[dict]:
    return [{"questions": [question(label=index % 2)]} for index in range(count)]


def test_independent_loss_formula_matches_hard_soft_and_score() -> None:
    logits = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.25, 0.75], dtype=torch.float64)
    probability = torch.softmax(logits, dim=-1)
    expected = -(target * torch.log_softmax(logits, dim=-1)).sum()
    expected += 0.5 * ((probability - target) ** 2).sum()
    expected += 0.25 * ((probability.cumsum(0)[:-1] - target.cumsum(0)[:-1]) ** 2).mean()
    actual = question_loss(logits, question("score", 1, [0.25, 0.75]), 0.5, 0.25)
    assert torch.allclose(actual, expected, rtol=0, atol=1e-14)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_request_mean_differs_from_global_question_mean() -> None:
    outputs = [
        [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 3.0])],
        [torch.tensor([0.0, 3.0])],
    ]
    batch = [
        {"questions": [question(label=0), question(label=0)]},
        {"questions": [question(label=0)]},
    ]
    loss = request_batch_loss(outputs, batch, 0.5, 0.25)
    first = torch.stack([
        question_loss(outputs[0][0], batch[0]["questions"][0], 0.5, 0.25),
        question_loss(outputs[0][1], batch[0]["questions"][1], 0.5, 0.25),
    ]).mean()
    second = question_loss(outputs[1][0], batch[1]["questions"][0], 0.5, 0.25)
    assert torch.allclose(loss, (first + second) / 2)


def test_uneven_microbatch_coefficients() -> None:
    assert microbatch_coefficients(5, [2, 2, 1]) == [0.4, 0.4, 0.2]
    with pytest.raises(ContractError, match="sum"):
        microbatch_coefficients(5, [2, 2])


def test_stored_permutation_moves_label_and_soft_target_together() -> None:
    request = {
        "request_uid": "r",
        "questions": [{
            "id": "answer", "type": "choice", "options": ["a", "b", "c"],
            "option_keys": ["a", "b", "c"], "label": 1,
            "soft": [0.1, 0.7, 0.2],
        }],
    }
    augmented = copy.deepcopy(request)
    permutation = [2, 0, 1]
    question_value = augmented["questions"][0]
    question_value["options"] = ["c", "a", "b"]
    question_value["option_keys"] = ["c", "a", "b"]
    question_value["soft"] = [0.2, 0.1, 0.7]
    question_value["label"] = 2
    slot = {
        "request_uid": "r",
        "question_permutations": [{"question_id": "answer", "new_to_old": permutation}],
        "augmented_request_sha256": digest_object(augmented),
    }
    assert apply_tape_slot(request, slot) == augmented


def test_scheduler_all_513_entries_match_independent_formula() -> None:
    model = TinyDecisionModel()
    optimizer, scheduler, _ = initialize_arm(
        copy.deepcopy(model.state_dict()), model, seed=17, arm="control",
    )
    factors = [scheduler.lr_lambdas[0](step) for step in range(513)]
    expected = []
    for step in range(513):
        if step < 30:
            expected.append(step / 30)
        else:
            progress = (step - 30) / (512 - 30)
            expected.append(0.5 * (1 + math.cos(math.pi * progress)))
    assert factors == pytest.approx(expected, rel=0, abs=1e-15)
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_weight_only_initialization_rejects_alias_and_wrong_shape() -> None:
    source = TinyDecisionModel()
    model = TinyDecisionModel()
    source_state = copy.deepcopy(source.state_dict())
    optimizer, scheduler, receipt = initialize_arm(
        source_state, model, seed=17, arm="targeted",
    )
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    assert receipt["reset_state"]["source_optimizer_inherited"] is False
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert not torch.equal(next(model.parameters()), next(source.parameters()))
    bad = copy.deepcopy(source_state)
    first = next(iter(bad))
    bad[first] = bad[first].reshape(-1)
    with pytest.raises(ContractError, match="shape or dtype"):
        initialize_arm(bad, TinyDecisionModel(), seed=17, arm="control")


def test_update_engine_uses_four_forwards_and_one_boundary_step() -> None:
    model = TinyDecisionModel()
    optimizer = SpyOptimizer(model.parameters())
    scheduler = SpyScheduler()
    clip_calls = []

    def clip(parameters, maximum):
        clip_calls.append((len(list(parameters)), maximum))
        return torch.tensor(1.0)

    result = run_update(
        model=model, optimizer=optimizer, scheduler=scheduler,
        requests=requests(), encodings=[torch.tensor([1.0, -1.0])] * 8,
        model_rng_seed=123, clip_grad=clip,
    )
    assert result["model_forward_calls"] == 4
    assert optimizer.steps == 1
    assert scheduler.steps == 1
    assert len(clip_calls) == 1
    assert optimizer.zeroes == 2


def test_full_batch_and_microbatch_gradients_match_without_optimizer_step() -> None:
    model_full = TinyDecisionModel().double()
    model_micro = copy.deepcopy(model_full)
    batch = requests()
    values = [torch.tensor([index / 10, -index / 10], dtype=torch.float64) for index in range(8)]
    request_batch_loss(model_full(values), batch, 0.5, 0.25).backward()
    for start in range(0, 8, 2):
        loss = request_batch_loss(
            model_micro(values[start:start + 2]), batch[start:start + 2], 0.5, 0.25,
        )
        (loss * 0.25).backward()
    for full, micro in zip(model_full.parameters(), model_micro.parameters()):
        assert torch.allclose(full.grad, micro.grad, rtol=1e-12, atol=1e-12)
