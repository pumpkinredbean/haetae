"""FastAPI factory for a verified local shared-v1 bundle."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .runtime import HaetaeRuntimeError, load_runtime


class RuntimeQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    type: str
    instructions: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=255)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: str | dict[str, Any] | list[Any]
    questions: list[RuntimeQuestion] = Field(min_length=1)


def create_app(bundle: str, device: str = "cpu") -> FastAPI:
    runtime = load_runtime(bundle, device=device)
    app = FastAPI(title="Haetae shared-v1 runtime", version="1")

    @app.get("/health")
    def health() -> dict:
        model = runtime.bundle.manifest["model"]
        return {
            "status": "ok",
            "run_id": model["run_id"],
            "generation": model["generation"],
            "device": runtime.device.type,
        }

    @app.post("/v1/decide")
    def decide(request: DecisionRequest) -> dict:
        try:
            decisions = runtime.decide(
                request.state,
                [question.model_dump() for question in request.questions],
            )
        except HaetaeRuntimeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        model = runtime.bundle.manifest["model"]
        return {
            "model": "haetae-shared-v1",
            "run_id": model["run_id"],
            "generation": model["generation"],
            "decisions": decisions,
            "calibrated": False,
        }

    return app
