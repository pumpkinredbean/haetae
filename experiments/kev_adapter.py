"""Read a frozen kev suite without depending on kev's runtime package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FROZEN_SPLITS = ("train", "calibration", "development", "test")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(value, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(
            f"{pad}- {render(item, indent + 1).lstrip()}" for item in value
        )
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def option_text(name: str, description) -> str:
    rendered = render(description)
    return name if not rendered else f"{name}: {rendered}"


def normalize_question(identifier: str, question: dict) -> dict:
    question_type = question["type"]
    criteria = question.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"Choice {identifier} has no criteria")
        keys = list(criteria)
        if question["label"] not in criteria:
            raise ValueError(f"Choice {identifier} label is absent")
        options = [option_text(key, criteria[key]) for key in keys]
        label = keys.index(question["label"])
    elif question_type == "noul":
        criteria = criteria or {}
        if not isinstance(question["label"], bool):
            raise ValueError(f"Noul {identifier} label is not Boolean")
        options = [
            option_text("yes", criteria.get("true")),
            option_text("no", criteria.get("false")),
        ]
        keys = ["true", "false"]
        label = 0 if question["label"] else 1
    elif question_type == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError(f"Score {identifier} has invalid criteria")
        options = [render(value) for value in criteria]
        keys = [str(index) for index in range(len(options))]
        label = question["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"Score {identifier} label is not an integer")
    else:
        raise ValueError(f"unsupported question type: {question_type}")
    if not 0 <= label < len(options):
        raise ValueError(f"question {identifier} label is out of range")
    return {
        "id": identifier,
        "type": question_type,
        "instructions": render(question["instructions"]),
        "options": options,
        "option_keys": keys,
        "label": label,
        "soft": None,
        "source": question.get("src"),
    }


def normalize_request(request: dict) -> dict:
    questions = [
        normalize_question(identifier, question)
        for identifier, question in request["questions"].items()
    ]
    return {
        "state": render(request["state"]),
        "questions": questions,
        "meta": request.get("_meta", {}),
    }


def load_frozen_split(
    suite: str | Path,
    split: str,
    *,
    allow_test: bool = False,
) -> tuple[list[dict], dict]:
    if split not in FROZEN_SPLITS:
        raise ValueError(f"unknown frozen split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit allow_test=True")
    suite = Path(suite).resolve()
    manifest = json.loads((suite / "manifest.json").read_text())
    filename = f"{split}.jsonl"
    descriptor = manifest.get("files", {}).get(filename)
    if descriptor is None:
        raise ValueError(f"suite does not declare {filename}")
    path = suite / filename
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"suite digest mismatch: {filename}")
    with path.open(encoding="utf-8") as handle:
        requests = [
            normalize_request(json.loads(line))
            for line in handle if line.strip()
        ]
    if len(requests) != descriptor["records"]:
        raise ValueError(f"suite record count mismatch: {filename}")
    questions = sum(len(request["questions"]) for request in requests)
    if questions != descriptor["questions"]:
        raise ValueError(f"suite question count mismatch: {filename}")
    return requests, manifest


def flatten_for_baseline(requests: list[dict]) -> list[dict]:
    records = []
    for request in requests:
        parent = request["meta"].get("group_id") or request["meta"].get("id")
        for question in request["questions"]:
            records.append({
                "state": request["state"],
                "type": question["type"],
                "instructions": question["instructions"],
                "options": question["options"],
                "label": question["label"],
                "soft": None,
                "source": question["source"] or request["meta"].get("source"),
                "parent": parent,
                "external_id": (
                    f"{request['meta'].get('id', '')}:{question['id']}"
                ),
            })
    return records
