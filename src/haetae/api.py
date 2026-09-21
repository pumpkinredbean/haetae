"""FastAPI factory for a verified local shared-v1 bundle."""

from __future__ import annotations

import math
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .runtime import HaetaeRuntimeError, load_runtime


class RuntimeQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    type: str
    instructions: str = ""
    options: list[str] = Field(min_length=2, max_length=255)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: str | dict[str, Any] | list[Any]
    questions: list[RuntimeQuestion] = Field(min_length=1)


class SystemOneQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["choice", "noul", "score"]
    instructions: str = ""
    criteria: dict[str, Any] | list[Any] | None = None


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, SystemOneQuestion]
    model: Literal["haetae-shared-v1"] = "haetae-shared-v1"


def _render_systemone_value(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise HaetaeRuntimeError("SystemOne values must contain finite numbers")
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(
            f"{pad}- {_render_systemone_value(item, indent + 1).lstrip()}"
            for item in value
        )
    if isinstance(value, dict):
        return "\n".join(
            (
                f"{pad}{key}:\n{_render_systemone_value(item, indent + 1)}"
                if isinstance(item, (dict, list))
                else f"{pad}{key}: {_render_systemone_value(item)}"
            )
            for key, item in value.items()
        )
    raise HaetaeRuntimeError("SystemOne values must contain JSON-compatible data")


def _systemone_option(name: str, description: Any) -> str:
    rendered = _render_systemone_value(description)
    return name if not rendered else f"{name}: {rendered}"


def _systemone_options(question: SystemOneQuestion) -> tuple[list[str], list[str]]:
    criteria = question.criteria
    if question.type == "noul":
        if criteria is None:
            return ["yes", "no"], ["yes", "no"]
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise HaetaeRuntimeError(
                "Noul criteria must contain exactly true and false"
            )
        return [
            _systemone_option("yes", criteria["true"]),
            _systemone_option("no", criteria["false"]),
        ], ["yes", "no"]
    if isinstance(criteria, dict):
        if len(criteria) < 2:
            raise HaetaeRuntimeError("Choice and Score need at least two criteria")
        names = list(criteria)
        options = [
            _systemone_option(name, description)
            for name, description in criteria.items()
        ]
        return options, names
    if isinstance(criteria, list) and len(criteria) >= 2:
        rendered = [_render_systemone_value(item) for item in criteria]
        if any(not item for item in rendered):
            raise HaetaeRuntimeError("criteria list must contain nonempty values")
        return rendered, rendered
    raise HaetaeRuntimeError("Choice and Score require at least two criteria")


def _systemone_decide(runtime, request: SystemOneRequest) -> dict:
    if not request.questions:
        raise HaetaeRuntimeError("a request needs at least one question")
    prepared = []
    metadata = []
    for identifier, question in request.questions.items():
        if not identifier:
            raise HaetaeRuntimeError("question IDs must be nonempty")
        options, names = _systemone_options(question)
        prepared.append(
            {
                "id": identifier,
                "type": question.type,
                "instructions": question.instructions,
                "options": options,
            }
        )
        metadata.append((identifier, question.type, names))
    decisions = runtime.decide(_render_systemone_value(request.state), prepared)
    nouls, choices, scores = {}, {}, {}
    for decision, (identifier, primitive, names) in zip(decisions, metadata):
        probabilities = decision["probabilities"]
        named = {
            name: round(probability, 4)
            for name, probability in zip(names, probabilities)
        }
        if primitive == "noul":
            nouls[identifier] = {"noul": round(probabilities[0], 4)}
        elif primitive == "score":
            expected = sum(index * value for index, value in enumerate(probabilities))
            scores[identifier] = {
                "score": round(expected, 3),
                "probabilities": named,
                "confidence": round(decision["confidence"], 4),
                "legend": names,
            }
        else:
            choices[identifier] = {
                "choice": names[decision["choice"]],
                "probabilities": named,
                "confidence": round(decision["confidence"], 4),
            }
    answers = {**nouls, **choices, **scores}
    model = runtime.bundle.manifest["model"]
    return {
        "model": "haetae-shared-v1",
        "run_id": model["run_id"],
        "generation": model["generation"],
        "nouls": nouls,
        "choices": choices,
        "scores": scores,
        "answers": answers,
        "calibrated": False,
    }


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

    @app.post("/v1/systemone")
    def systemone(request: SystemOneRequest) -> dict:
        try:
            return _systemone_decide(runtime, request)
        except HaetaeRuntimeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
