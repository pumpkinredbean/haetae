from types import SimpleNamespace

import pytest
import torch

from haetae.runtime import (
    HaetaeRuntime,
    HaetaeRuntimeError,
    distribution_confidence,
    resolve_device,
    widen_logits_cpu_first,
)
from haetae.shared_v1 import DELIMITER_TOKENS


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, value, add_special_tokens=False):
        return {"input_ids": [10 + (ord(character) % 20) for character in value]}


class Model:
    def __init__(self, outputs=None):
        self.outputs = outputs
        self.calls = 0

    def __call__(self, encodings):
        self.calls += 1
        if self.outputs is not None:
            return self.outputs
        return [
            [
                torch.arange(len(indexes), dtype=torch.float32)
                for indexes in encoding.option_indices
            ]
            for encoding in encodings
        ]


def runtime(maximum_length=128, outputs=None):
    manifest = {
        "model": {
            "maximum_length": maximum_length,
            "run_id": "run-1",
            "generation": 79,
        }
    }
    return HaetaeRuntime(
        bundle=SimpleNamespace(manifest=manifest),
        model=Model(outputs),
        tokenizer=Tokenizer(),
        delimiters={token: 100 + index for index, token in enumerate(DELIMITER_TOKENS)},
        device=torch.device("cpu"),
    )


def question(**changes):
    value = {
        "id": "q1",
        "type": "choice",
        "instructions": "choose",
        "options": ["first", "second"],
    }
    value.update(changes)
    return value


def test_runtime_returns_raw_probabilities_without_calibration_claim():
    loaded = runtime()
    result = loaded.decide("state", [question()])
    assert loaded.model.calls == 1
    assert result[0]["choice"] == 1
    assert result[0]["calibrated"] is False
    probabilities = torch.tensor(result[0]["probabilities"], dtype=torch.float64)
    assert sum(result[0]["probabilities"]) == pytest.approx(1.0)
    assert result[0]["confidence"] == pytest.approx(
        distribution_confidence(probabilities)
    )


def test_runtime_accepts_described_noul_options():
    loaded = runtime()
    result = loaded.decide(
        "state",
        [
            question(
                type="noul",
                options=["yes: requires action", "no: no action"],
            )
        ],
    )
    assert result[0]["type"] == "noul"
    assert result[0]["choice"] == 1


def test_runtime_moves_logits_to_cpu_before_widening():
    class DeviceGuard:
        def __init__(self):
            self.device = "mps"
            self.events = []

        def detach(self):
            self.events.append("detach")
            return self

        def cpu(self):
            self.events.append("cpu")
            self.device = "cpu"
            return self

        def to(self, *, dtype):
            if self.device != "cpu":
                raise AssertionError("float64 conversion happened before CPU transfer")
            self.events.append(("to", dtype))
            return self

    guarded = DeviceGuard()
    assert widen_logits_cpu_first(guarded) is guarded
    assert guarded.events == ["detach", "cpu", ("to", torch.float64)]


@pytest.mark.parametrize(
    "bad_question,message",
    [
        (question(type="noul"), "Noul options"),
        (question(type="noul", options=["yes: ", "no"]), "Noul options"),
        (question(options=["same", "same"]), "unique"),
        (question(extra=True), "fields differ"),
    ],
)
def test_runtime_rejects_invalid_question_schema(bad_question, message):
    with pytest.raises(HaetaeRuntimeError, match=message):
        runtime().decide("state", [bad_question])


def test_runtime_rejects_state_truncation_before_model_call():
    loaded = runtime(maximum_length=30)
    with pytest.raises(HaetaeRuntimeError, match="state content exceeds"):
        loaded.decide("a" * 100, [question(instructions="x", options=["a", "b"])])
    assert loaded.model.calls == 0


def test_runtime_rejects_nonfinite_state_and_logits():
    loaded = runtime(outputs=[[torch.tensor([float("nan"), 0.0])]])
    with pytest.raises(HaetaeRuntimeError, match="finite JSON"):
        loaded.decide({"value": float("nan")}, [question()])
    with pytest.raises(HaetaeRuntimeError, match="finite vector"):
        loaded.decide("state", [question()])


def test_device_resolution_has_no_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(HaetaeRuntimeError, match="MPS was requested"):
        resolve_device("mps")
    with pytest.raises(HaetaeRuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
    assert resolve_device("cpu") == torch.device("cpu")
