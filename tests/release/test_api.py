from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from haetae import api
from haetae.runtime import HaetaeRuntimeError


class Runtime:
    device = SimpleNamespace(type="cpu")
    bundle = SimpleNamespace(
        manifest={
            "model": {"run_id": "run-1", "generation": 79},
        }
    )
    last_call = None

    def decide(self, state, questions):
        self.last_call = (state, questions)
        if state == "reject":
            raise HaetaeRuntimeError("rejected by runtime")
        return [
            {
                "id": questions[0]["id"],
                "type": questions[0]["type"],
                "options": questions[0]["options"],
                "probabilities": [0.25, 0.75],
                "choice": 1,
                "confidence": 0.75,
                "calibrated": False,
            }
        ]


@pytest.fixture
def client(monkeypatch):
    loaded = Runtime()
    monkeypatch.setattr(api, "load_runtime", lambda bundle, device: loaded)
    client = TestClient(api.create_app("local-bundle", device="cpu"))
    client.runtime = loaded
    return client


def request(state="state"):
    return {
        "state": state,
        "questions": [
            {
                "id": "q1",
                "type": "choice",
                "instructions": "choose",
                "options": ["first", "second"],
            }
        ],
    }


def test_factory_exposes_bound_health_and_decision(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "run_id": "run-1",
        "generation": 79,
        "device": "cpu",
    }
    response = client.post("/v1/decide", json=request())
    assert response.status_code == 200
    assert response.json()["decisions"][0]["choice"] == 1
    assert response.json()["calibrated"] is False

    empty = request()
    empty["questions"][0]["instructions"] = ""
    assert client.post("/v1/decide", json=empty).status_code == 200


def test_api_rejects_extra_fields_and_runtime_errors(client):
    extra = request()
    extra["bundle"] = "/untrusted/client/path"
    assert client.post("/v1/decide", json=extra).status_code == 422

    response = client.post("/v1/decide", json=request(state="reject"))
    assert response.status_code == 422
    assert response.json()["detail"] == "rejected by runtime"


def test_systemone_compatibility_endpoint(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": "state",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "choose",
                    "criteria": {"billing": "billing issue", "other": None},
                },
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"]["route"] == {
        "choice": "other",
        "probabilities": {"billing": 0.25, "other": 0.75},
        "confidence": 0.75,
    }
    assert body["answers"] == body["choices"]
    assert body["calibrated"] is False
    assert body["model"] == "haetae-shared-v1"
    assert body["run_id"] == "run-1"
    assert body["generation"] == 79


def test_systemone_rejects_an_unavailable_model_selector(client):
    response = client.post(
        "/v1/systemone",
        json={
            "model": "not-the-loaded-model",
            "state": "state",
            "questions": {
                "route": {
                    "type": "choice",
                    "criteria": {"first": None, "second": None},
                },
            },
        },
    )
    assert response.status_code == 422


def test_systemone_rejects_ambiguous_noul_criteria(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": "state",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "criteria": {"maybe": "uncertain", "false": "no"},
                },
            },
        },
    )
    assert response.status_code == 422
    assert "exactly true and false" in response.json()["detail"]


def test_systemone_preserves_noul_labels_and_renders_structured_state(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": {"ticket": {"priority": "high"}, "flags": ["manual"]},
            "questions": {
                "urgent": {
                    "type": "noul",
                    "criteria": {
                        "true": "requires immediate handling",
                        "false": "can wait",
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    state, questions = client.runtime.last_call
    assert state == "ticket:\n  priority: high\nflags:\n  - manual"
    assert questions[0] == {
        "id": "urgent",
        "type": "noul",
        "instructions": "",
        "options": ["yes: requires immediate handling", "no: can wait"],
    }
    assert response.json()["nouls"]["urgent"] == {"noul": 0.25}


def test_systemone_normalization_matches_training_adapter():
    question = api.SystemOneQuestion(
        type="noul",
        criteria={"true": "requires action", "false": "no action"},
    )
    assert api._systemone_options(question) == (
        ["yes: requires action", "no: no action"],
        ["yes", "no"],
    )
    identical = api.SystemOneQuestion(
        type="noul",
        criteria={"true": "same description", "false": "same description"},
    )
    options, names = api._systemone_options(identical)
    assert options == ["yes: same description", "no: same description"]
    assert names == ["yes", "no"]
    assert (
        api._render_systemone_value(
            {"ticket": {"priority": "high"}, "flags": ["manual"]}
        )
        == "ticket:\n  priority: high\nflags:\n  - manual"
    )
