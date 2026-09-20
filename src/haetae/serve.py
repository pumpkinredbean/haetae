"""/v1/systemone-compatible serving (Jev wire shape)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import math
import numbers
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from .checkpoint import (
    CheckpointError,
    load_completed_checkpoint,
    model_config_fingerprint,
    tokenizer_fingerprint,
)
from .model import PACKING_POLICY, HaetaeModel, collate, confidence_from_probs


@asynccontextmanager
async def lifespan(app):
    checkpoint = os.environ.get("HAETAE_CHECKPOINT")
    if checkpoint:
        load(checkpoint)
    yield


app = FastAPI(lifespan=lifespan)
_model = _tok = None
_device = "mps" if torch.backends.mps.is_available() else "cpu"
_temperature = 1.0
_max_len = 8192
_calibrated = False


class Question(BaseModel):
    type: str  # "choice" | "noul" | "score"
    instructions: str = ""
    criteria: dict | list | None = None


class Req(BaseModel):
    state: str | dict | list
    questions: dict[str, Question]
    model: str = "haetae-latest"


def load(checkpoint="runs/v1"):
    global _model, _tok, _temperature, _max_len, _calibrated
    ck, run, manifest = load_completed_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer_fingerprint(tokenizer) != run["spec"]["tokenizer_fingerprint"]:
        raise CheckpointError("serving tokenizer does not match the completed run")
    model = HaetaeModel(ck["config"]["backbone"])
    if model_config_fingerprint(model) != run["spec"]["model_config_fingerprint"]:
        raise CheckpointError("serving model configuration does not match the run")
    model.backbone.load_state_dict(ck["backbone"])
    model.head.load_state_dict(ck["head"])
    model.to(_device).eval()
    temperature, calibrated = _load_calibration(checkpoint, run, manifest)

    _tok = tokenizer
    _model = model
    _max_len = int(ck["config"]["max_len"])
    _temperature = temperature
    _calibrated = calibrated


def _load_calibration(checkpoint, run, manifest):
    calibration_path = os.path.join(checkpoint, "calibration.json")
    if not os.path.exists(calibration_path):
        return 1.0, False
    with open(calibration_path, encoding="utf-8") as handle:
        calibration = json.load(handle)
    required = {
        "version", "run_id", "checkpoint_generation", "checkpoint_sha256",
        "tokenizer_fingerprint", "inference_policy", "calibration_data",
        "temperature",
    }
    if not isinstance(calibration, dict) or required - set(calibration):
        raise CheckpointError("calibration artifact is incomplete")
    current = manifest["current"]
    if calibration["version"] != 1:
        raise CheckpointError("calibration artifact version is unsupported")
    if calibration["run_id"] != run["run_id"]:
        raise CheckpointError("calibration artifact belongs to another run")
    if calibration["checkpoint_generation"] != current["generation"]:
        raise CheckpointError("calibration artifact generation mismatch")
    if calibration["checkpoint_sha256"] != current["sha256"]:
        raise CheckpointError("calibration artifact checkpoint digest mismatch")
    if calibration["tokenizer_fingerprint"] != run["spec"]["tokenizer_fingerprint"]:
        raise CheckpointError("calibration artifact tokenizer mismatch")
    expected_policy = {
        "packing": PACKING_POLICY,
        "max_len": run["spec"]["config"]["max_len"],
        "temperature_scaling": "scalar-v1",
    }
    if calibration["inference_policy"] != expected_policy:
        raise CheckpointError("calibration artifact inference policy mismatch")
    calibration_data = calibration["calibration_data"]
    if (not isinstance(calibration_data, dict)
            or not isinstance(calibration_data.get("dataset"), str)
            or not calibration_data["dataset"]
            or not isinstance(calibration_data.get("split"), str)
            or not calibration_data["split"]
            or not isinstance(calibration_data.get("fingerprint"), str)
            or len(calibration_data["fingerprint"]) != 64
            or any(character not in "0123456789abcdef"
                   for character in calibration_data["fingerprint"])
            or isinstance(calibration_data.get("n"), bool)
            or not isinstance(calibration_data.get("n"), int)
            or calibration_data["n"] <= 0):
        raise CheckpointError("calibration data identity is invalid")
    raw_temperature = calibration["temperature"]
    if (isinstance(raw_temperature, bool)
            or not isinstance(raw_temperature, numbers.Real)):
        raise CheckpointError("calibration temperature is not numeric")
    temperature = float(raw_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise CheckpointError("calibration temperature must be finite and positive")
    return temperature, True


def _options(q: Question):
    """Always return (option_descriptions, option_names)."""
    if q.type == "noul":
        if isinstance(q.criteria, dict):
            return ([q.criteria.get("true", "yes"), q.criteria.get("false", "no")],
                    ["yes", "no"])
        return ["yes", "no"], ["yes", "no"]
    if isinstance(q.criteria, dict):
        return [f"{k}: {v}" for k, v in q.criteria.items()], list(q.criteria)
    return list(q.criteria or []), list(q.criteria or [])


@app.post("/v1/systemone")
def systemone(req: Req):
    if _model is None or _tok is None:
        raise HTTPException(503, "model is not loaded; set HAETAE_CHECKPOINT")
    state = (req.state if isinstance(req.state, str)
             else json.dumps(req.state, ensure_ascii=False, sort_keys=True))
    if not req.questions:
        raise HTTPException(422, "empty questions")
    nouls, choices, scores = {}, {}, {}
    # All questions in one batch — one forward pass for the request.
    items = [(name, q, *_options(q)) for name, q in req.questions.items()]
    recs = [{"state": state, "instructions": q.instructions, "options": opts}
            for name, q, opts, names in items]
    b = collate(_tok, recs, _max_len)
    if any(b["dropped_options"]):
        bad = [name for (name, *_), d in zip(items, b["dropped_options"]) if d]
        raise HTTPException(422, {"error": "question does not fit context",
                                  "questions": bad})
    with torch.no_grad():
        logits_list = _model(b["input_ids"].to(_device), b["attention_mask"].to(_device),
                             b["option_pos"].to(_device), b["group_ptr"].to(_device))
    for (name, q, opts, names), logits in zip(items, logits_list):
        p = torch.softmax(logits / _temperature, -1)
        probs = {names[i]: round(float(p[i]), 4) for i in range(len(names))}
        conf = confidence_from_probs(p)
        if q.type == "noul":
            nouls[name] = {"noul": round(float(p[0]), 4)}
        elif q.type == "score":
            ev = sum(i * float(p[i]) for i in range(len(names)))
            scores[name] = {"score": round(ev, 3), "probabilities": probs,
                            "confidence": round(conf, 4), "legend": names}
        else:
            choices[name] = {"choice": names[int(p.argmax())],
                             "probabilities": probs, "confidence": round(conf, 4)}
    answers = {**nouls, **choices, **scores}
    return {"model": req.model, "nouls": nouls, "choices": choices,
            "scores": scores, "answers": answers,
            "calibrated": _calibrated}
