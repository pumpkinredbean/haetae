from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from haetae import api
from haetae.runtime import HaetaeRuntimeError


class Runtime:
    device = SimpleNamespace(type="cpu")
    bundle = SimpleNamespace(manifest={
        "model": {"run_id": "run-1", "generation": 79},
    })

    def decide(self, state, questions):
        if state == "reject":
            raise HaetaeRuntimeError("rejected by runtime")
        return [{
            "id": questions[0]["id"],
            "type": questions[0]["type"],
            "options": questions[0]["options"],
            "probabilities": [0.25, 0.75],
            "choice": 1,
            "confidence": 0.75,
            "calibrated": False,
        }]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_runtime", lambda bundle, device: Runtime())
    return TestClient(api.create_app("local-bundle", device="cpu"))


def request(state="state"):
    return {
        "state": state,
        "questions": [{
            "id": "q1",
            "type": "choice",
            "instructions": "choose",
            "options": ["first", "second"],
        }],
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


def test_api_rejects_extra_fields_and_runtime_errors(client):
    extra = request()
    extra["bundle"] = "/untrusted/client/path"
    assert client.post("/v1/decide", json=extra).status_code == 422

    response = client.post("/v1/decide", json=request(state="reject"))
    assert response.status_code == 422
    assert response.json()["detail"] == "rejected by runtime"
