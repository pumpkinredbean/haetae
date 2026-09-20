"""/v1/systemone-compatible serving (Jev wire shape)."""

from __future__ import annotations

import json
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from .model import HaetaeModel, collate, confidence_from_probs

app = FastAPI()
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
    _tok = AutoTokenizer.from_pretrained(checkpoint)
    ck = torch.load(f"{checkpoint}/model.pt", map_location="cpu", weights_only=False)
    _model = HaetaeModel(ck["config"]["backbone"])
    _model.backbone.load_state_dict(ck["backbone"])
    _model.head.load_state_dict(ck["head"])
    _model.to(_device).eval()
    _max_len = int(ck["config"].get("max_len", 8192))
    calibration_path = os.path.join(checkpoint, "calibration.json")
    if os.path.exists(calibration_path):
        with open(calibration_path, encoding="utf-8") as f:
            calibration = json.load(f)
        temperature = float(calibration["temperature"])
        if not temperature > 0:
            raise ValueError("calibration temperature must be positive")
        _temperature = temperature
        _calibrated = True
    else:
        _temperature = 1.0
        _calibrated = False


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
    state = req.state if isinstance(req.state, str) else str(req.state)
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
