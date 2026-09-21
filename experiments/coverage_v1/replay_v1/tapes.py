"""Deterministic paired draw tapes for T05."""

from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.inputs import EMOTION_LABELS, OFFENSIVE_LABELS

UPDATES = 512
SLOTS = 8
ZERO_SHA256 = "0" * 64
MODEL_SEED_MODULUS = 2**63 - 1


def hash_order(
    items: Iterable[Mapping[str, Any]],
    domain: str,
    seed: int,
    identity: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    decorated = []
    for item in items:
        item = dict(item)
        tie_break = identity(item)
        decorated.append((digest_object([domain, seed, tie_break]), tie_break, item))
    decorated.sort(key=lambda entry: (entry[0], entry[1]))
    return [item for _, _, item in decorated]


def make_original_stream(
    original_pool: Iterable[Mapping[str, Any]], seed: int,
) -> list[dict[str, Any]]:
    pool = list(original_pool)
    if len(pool) != 15572:
        raise ContractError("original replay pool must contain exactly 15,572 requests")
    if len({row["request_uid"] for row in pool}) != len(pool):
        raise ContractError("original replay request UIDs are not unique")
    ordered = hash_order(pool, "t05-original-v1", seed, lambda row: row["request_uid"])
    return ordered[: UPDATES * SLOTS]


def _semantic_label(request: Mapping[str, Any]) -> str:
    questions = request["questions"]
    if len(questions) != 1:
        raise ContractError("target support request must have exactly one question")
    question = questions[0]
    return question["option_keys"][question["label"]]


def make_support_stream(
    target_pool: Iterable[Mapping[str, Any]], seed: int, target: str,
) -> list[dict[str, Any]]:
    if target not in {"emotion", "tweet_offensive"}:
        raise ContractError("unknown target support stream")
    labels = EMOTION_LABELS if target == "emotion" else OFFENSIVE_LABELS
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in target_pool:
        if request["origin_source"] == target:
            groups[_semantic_label(request)].append(dict(request))
    if set(groups) != set(labels) or any(len(groups[label]) != 256 for label in labels):
        raise ContractError("target support classes must each contain 256 rows")
    class_order = sorted(
        labels,
        key=lambda label: (
            digest_object(["t05-class-order-v1", seed, target, label]), label,
        ),
    )
    ordered_groups = {}
    for label in labels:
        ordered_groups[label] = sorted(
            groups[label],
            key=lambda row: (
                digest_object([
                    "t05-support-order-v1", seed, target, label,
                    row["parent_id"], row["state_sha256"],
                ]),
                row["parent_id"],
                row["request_uid"],
            ),
        )
    stream = []
    for draw in range(UPDATES):
        label = class_order[draw % len(class_order)]
        stream.append(ordered_groups[label][draw // len(class_order)])
    if len({row["parent_id"] for row in stream}) != UPDATES:
        raise ContractError("support stream repeats a parent")
    return stream


def make_choice_permutations(
    request: Mapping[str, Any], seed: int, update_index: int, slot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    augmented = copy.deepcopy(dict(request))
    rng = random.Random(int(digest_object([
        "t05-choice-v1", seed, update_index, slot, request["request_uid"],
    ]), 16))
    stored = []
    for question in augmented["questions"]:
        size = len(question["options"])
        permutation = list(range(size))
        if question["type"] == "choice":
            rng.shuffle(permutation)
        original_label = question["label"]
        question["options"] = [question["options"][index] for index in permutation]
        question["option_keys"] = [question["option_keys"][index] for index in permutation]
        if question["soft"] is not None:
            question["soft"] = [question["soft"][index] for index in permutation]
        question["label"] = permutation.index(original_label)
        stored.append({
            "question_id": question["id"],
            "new_to_old": permutation,
        })
    return augmented, stored


def _encoding_identity(encoded: Mapping[str, Any]) -> tuple[str, int]:
    required = {
        "input_ids", "segments", "position_ids", "decision_indices",
        "option_indices", "labels", "state_tokens", "retained_state_tokens",
    }
    if set(encoded) != required:
        raise ContractError("packed encoding has the wrong identity fields")
    if encoded["state_tokens"] != encoded["retained_state_tokens"]:
        raise ContractError("T05 training occurrence would truncate its state")
    tokens = len(encoded["input_ids"])
    if tokens <= 0 or tokens > 2048:
        raise ContractError("T05 packed token count is outside the frozen budget")
    if len(encoded["segments"]) != tokens or len(encoded["position_ids"]) != tokens:
        raise ContractError("packed encoding arrays have inconsistent lengths")
    return digest_object(dict(encoded)), tokens


def _make_slot(
    request: Mapping[str, Any],
    *,
    seed: int,
    update_index: int,
    slot: int,
    role: str,
    encoder: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    augmented, permutations = make_choice_permutations(
        request, seed, update_index, slot,
    )
    encoding_sha256, tokens = _encoding_identity(encoder(augmented))
    return {
        "slot": slot,
        "role": role,
        "request_uid": request["request_uid"],
        "origin_source": request["origin_source"],
        "parent_id": request["parent_id"],
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "question_permutations": permutations,
        "augmented_request_sha256": digest_object(augmented),
        "encoding_sha256": encoding_sha256,
        "tokens": tokens,
    }


def replay_update(
    *,
    seed: int,
    arm: str,
    update_index: int,
    original_stream: list[Mapping[str, Any]],
    emotion_stream: list[Mapping[str, Any]],
    offensive_stream: list[Mapping[str, Any]],
    encoder: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    previous_row_sha256: str,
) -> dict[str, Any]:
    if arm not in {"control", "targeted"}:
        raise ContractError("unknown paired-replay arm")
    if not 0 <= update_index < UPDATES:
        raise ContractError("update index is outside the frozen tape")
    start = update_index * SLOTS
    slots = []
    for slot in range(6):
        slots.append(_make_slot(
            original_stream[start + slot], seed=seed, update_index=update_index,
            slot=slot, role="common_original", encoder=encoder,
        ))
    if arm == "control":
        for slot in (6, 7):
            slots.append(_make_slot(
                original_stream[start + slot], seed=seed, update_index=update_index,
                slot=slot, role="control_original", encoder=encoder,
            ))
    else:
        slots.append(_make_slot(
            emotion_stream[update_index], seed=seed, update_index=update_index,
            slot=6, role="target_emotion", encoder=encoder,
        ))
        slots.append(_make_slot(
            offensive_stream[update_index], seed=seed, update_index=update_index,
            slot=7, role="target_offensive", encoder=encoder,
        ))
    model_digest = digest_object(["t05-model-rng-v1", seed, update_index])
    row = {
        "schema_version": 1,
        "seed": seed,
        "arm": arm,
        "update_index": update_index,
        "model_rng_seed": int(model_digest[:16], 16) % MODEL_SEED_MODULUS,
        "slots": slots,
        "previous_row_sha256": previous_row_sha256,
    }
    row["row_sha256"] = digest_object(row)
    return row


def build_pair_tapes(
    original_pool: Iterable[Mapping[str, Any]],
    target_pool: Iterable[Mapping[str, Any]],
    seed: int,
    encoder: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if seed not in {17, 23}:
        raise ContractError("experiment seed must be 17 or 23")
    original_stream = make_original_stream(original_pool, seed)
    targets = list(target_pool)
    emotion_stream = make_support_stream(targets, seed, "emotion")
    offensive_stream = make_support_stream(targets, seed, "tweet_offensive")
    tapes = {}
    for arm in ("control", "targeted"):
        rows = []
        previous = ZERO_SHA256
        for update_index in range(UPDATES):
            row = replay_update(
                seed=seed, arm=arm, update_index=update_index,
                original_stream=original_stream, emotion_stream=emotion_stream,
                offensive_stream=offensive_stream, encoder=encoder,
                previous_row_sha256=previous,
            )
            rows.append(row)
            previous = row["row_sha256"]
        tapes[arm] = rows
    validate_pair_alignment(tapes["control"], tapes["targeted"], seed)
    return tapes


def _validate_row(row: Mapping[str, Any], seed: int, arm: str, index: int, previous: str) -> None:
    required = {
        "schema_version", "seed", "arm", "update_index", "model_rng_seed",
        "slots", "previous_row_sha256", "row_sha256",
    }
    if set(row) != required:
        raise ContractError("tape row has the wrong fields")
    if row["schema_version"] != 1 or row["seed"] != seed or row["arm"] != arm:
        raise ContractError("tape row identity mismatch")
    if type(row["update_index"]) is not int or row["update_index"] != index:
        raise ContractError("tape update index mismatch")
    if row["previous_row_sha256"] != previous:
        raise ContractError("tape hash chain is broken")
    if row["row_sha256"] != digest_object(row, "row_sha256"):
        raise ContractError("tape row digest mismatch")
    if len(row["slots"]) != SLOTS or [slot["slot"] for slot in row["slots"]] != list(range(SLOTS)):
        raise ContractError("tape slots are incomplete or reordered")


def validate_pair_alignment(
    control: list[Mapping[str, Any]],
    targeted: list[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    if len(control) != UPDATES or len(targeted) != UPDATES:
        raise ContractError("each paired tape must contain 512 rows")
    previous = {"control": ZERO_SHA256, "targeted": ZERO_SHA256}
    target_counts = Counter()
    target_parents = set()
    for index, (control_row, targeted_row) in enumerate(zip(control, targeted)):
        _validate_row(control_row, seed, "control", index, previous["control"])
        _validate_row(targeted_row, seed, "targeted", index, previous["targeted"])
        previous["control"] = control_row["row_sha256"]
        previous["targeted"] = targeted_row["row_sha256"]
        if control_row["model_rng_seed"] != targeted_row["model_rng_seed"]:
            raise ContractError("paired model RNG seeds differ")
        for slot in range(6):
            left = dict(control_row["slots"][slot])
            right = dict(targeted_row["slots"][slot])
            if left != right:
                raise ContractError("paired common slots differ")
        for slot in targeted_row["slots"][6:]:
            target_counts[(slot["origin_source"],)] += 1
            parent = (slot["origin_source"], slot["parent_id"])
            if parent in target_parents:
                raise ContractError("targeted tape repeats a support parent")
            target_parents.add(parent)
    if target_counts != Counter({("emotion",): 512, ("tweet_offensive",): 512}):
        raise ContractError("targeted tape has the wrong target counts")
    return {
        "seed": seed,
        "updates": UPDATES,
        "common_original_slots": UPDATES * 6,
        "control_only_original_slots": UPDATES * 2,
        "target_emotion_slots": UPDATES,
        "target_offensive_slots": UPDATES,
        "control_terminal_row_sha256": control[-1]["row_sha256"],
        "targeted_terminal_row_sha256": targeted[-1]["row_sha256"],
    }


def exposure_report(
    tapes: Mapping[str, list[Mapping[str, Any]]],
    requests_by_uid: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"arms": {}}
    for arm, rows in sorted(tapes.items()):
        groups: dict[tuple[str, str, str, str], dict[str, float | int]] = {}
        labels = Counter()
        for row in rows:
            for slot in row["slots"]:
                request = requests_by_uid[slot["request_uid"]]
                question_count = len(request["questions"])
                for question in request["questions"]:
                    rubric = digest_object({
                        "instructions": question["instructions"],
                        "option_keys": question["option_keys"],
                        "options": question["options"],
                    })
                    key = (
                        request["origin_source"], question["source"],
                        question["type"], rubric,
                    )
                    cell = groups.setdefault(key, {"question_draws": 0, "loss_weight": 0.0})
                    cell["question_draws"] += 1
                    cell["loss_weight"] += 1.0 / (8 * question_count)
                    labels[(request["origin_source"], question["option_keys"][question["label"]])] += 1
        report["arms"][arm] = {
            "groups": [
                {
                    "origin_source": key[0], "task_source": key[1],
                    "primitive": key[2], "rubric_sha256": key[3], **value,
                }
                for key, value in sorted(groups.items())
            ],
            "semantic_labels": [
                {"origin_source": key[0], "label": key[1], "draws": value}
                for key, value in sorted(labels.items())
            ],
        }
    report["report_sha256"] = digest_object(report)
    return report
