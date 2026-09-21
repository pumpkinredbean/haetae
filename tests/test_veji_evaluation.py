import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.comparison_protocol import canonical_sha256
from experiments.evaluate_veji import (
    infer_requests,
    load_protocol,
    stage_digest,
    veji_question,
)


class FakeVEJI:
    def __init__(self):
        self.cfg = SimpleNamespace(max_context_chars=1000)
        self.compiled = []
        self.questions = []

    def compile_state(self, state):
        self.compiled.append(state)
        return SimpleNamespace(state=state, chunks=[state])

    def forward_question(self, compiled, question):
        self.questions.append((compiled.state, question))
        return {"logits": torch.tensor([0.25, -0.5])}


def request():
    return {
        "state": "shared state",
        "meta": {"id": "request-1", "group_id": "parent-1"},
        "questions": [
            {
                "id": "q1",
                "type": "choice",
                "instructions": "choose",
                "options": ["a", "b"],
                "option_keys": ["a", "b"],
                "label": 0,
                "soft": None,
                "source": "source-1",
            },
            {
                "id": "q2",
                "type": "noul",
                "instructions": "verify",
                "options": ["yes", "no"],
                "option_keys": ["true", "false"],
                "label": 1,
                "soft": None,
                "source": "source-1",
            },
        ],
    }


def test_request_compiles_state_once_and_omits_task_hint():
    model = FakeVEJI()
    predictions = infer_requests(model, [request()], progress_every=0)
    assert model.compiled == ["shared state"]
    assert len(predictions) == 2
    assert all("semantic_type" not in question for _, question in model.questions)
    assert model.questions[0][1] == {
        "id": "q1",
        "type": "choice",
        "instruction": "choose",
        "options": ["a", "b"],
    }
    assert predictions[1]["group_id"] == "parent-1"
    assert predictions[1]["logits"] == [0.25, -0.5]


def test_veji_question_does_not_mutate_normalized_question():
    original = request()["questions"][0]
    converted = veji_question(original)
    converted["options"][0] = "changed"
    assert original["options"][0] == "a"


def test_protocol_digest_is_checked(tmp_path: Path):
    value = {"version": 1, "status": "frozen", "locked_test_opened": False}
    value["protocol_sha256"] = canonical_sha256(value)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value))
    assert load_protocol(path)["status"] == "frozen"
    value["status"] = "changed"
    path.write_text(json.dumps(value))
    try:
        load_protocol(path)
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("changed protocol was accepted")


def test_stage_digest_excludes_only_its_signature():
    stage = {"status": "complete", "predictions": [{"logits": [1.0, 2.0]}]}
    digest = stage_digest(stage)
    stage["stage_sha256"] = digest
    assert stage_digest(stage) == digest
    stage["predictions"][0]["logits"][0] = 3.0
    assert stage_digest(stage) != digest
