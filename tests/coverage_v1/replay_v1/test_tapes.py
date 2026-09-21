from __future__ import annotations

import copy
from collections import Counter

import pytest

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.inputs import EMOTION_LABELS, OFFENSIVE_LABELS
from experiments.coverage_v1.replay_v1.tapes import (
    build_pair_tapes,
    make_choice_permutations,
    make_original_stream,
    validate_pair_alignment,
)


def normalized_request(index: int, *, source: str = "original", label: str = "a") -> dict:
    question = {
        "id": "answer",
        "type": "choice",
        "instructions": "Choose one.",
        "options": ["a: first", "b: second", "c: third"],
        "option_keys": ["a", "b", "c"],
        "label": ["a", "b", "c"].index(label),
        "soft": None,
        "source": f"{source}_task",
    }
    core = {
        "origin_source": source,
        "parent_id": f"{source}/parent/{index}",
        "request_id": f"{source}/request/{index}",
        "state": f"state {source} {index}",
        "questions": [question],
        "original_record_sha256": digest_object([source, index]),
    }
    request_sha256 = digest_object(core)
    return {
        "request_uid": digest_object([
            source, core["parent_id"], core["request_id"], request_sha256,
        ]),
        **core,
        "state_sha256": digest_object(core["state"]),
        "request_sha256": request_sha256,
    }


def target_request(index: int, source: str, label: str) -> dict:
    request = normalized_request(index, source=source)
    question = request["questions"][0]
    labels = EMOTION_LABELS if source == "emotion" else OFFENSIVE_LABELS
    question["type"] = "choice" if source == "emotion" else "noul"
    question["options"] = list(labels)
    question["option_keys"] = list(labels)
    question["label"] = labels.index(label)
    question["source"] = source
    core = {key: request[key] for key in (
        "origin_source", "parent_id", "request_id", "state", "questions",
        "original_record_sha256",
    )}
    request["request_sha256"] = digest_object(core)
    request["request_uid"] = digest_object([
        source, request["parent_id"], request["request_id"], request["request_sha256"],
    ])
    return request


def encoder(request: dict) -> dict:
    input_ids = [1]
    segments = [0]
    position_ids = [0]
    decision_indices = []
    option_indices = []
    labels = []
    for segment, question in enumerate(request["questions"], start=1):
        indices = []
        for option in question["options"]:
            input_ids.append(int(digest_object(option)[:8], 16))
            segments.append(segment)
            position_ids.append(len(position_ids))
            indices.append(len(input_ids) - 1)
        decision_indices.append(indices[-1])
        option_indices.append(indices)
        labels.append(question["label"])
    return {
        "input_ids": input_ids,
        "segments": segments,
        "position_ids": position_ids,
        "decision_indices": decision_indices,
        "option_indices": option_indices,
        "labels": labels,
        "state_tokens": 3,
        "retained_state_tokens": 3,
    }


@pytest.fixture(scope="module")
def pools() -> tuple[list[dict], list[dict]]:
    original = [normalized_request(index) for index in range(15572)]
    target = []
    for label_index, label in enumerate(EMOTION_LABELS):
        target += [target_request(label_index * 256 + i, "emotion", label) for i in range(256)]
    for label_index, label in enumerate(OFFENSIVE_LABELS):
        target += [target_request(label_index * 256 + i, "tweet_offensive", label) for i in range(256)]
    return original, target


@pytest.fixture(scope="module")
def seed17_pair(pools) -> dict[str, list[dict]]:
    return build_pair_tapes(*pools, 17, encoder)


def test_original_stream_is_source_order_independent(pools) -> None:
    original, _ = pools
    forward = make_original_stream(original, 17)
    reverse = make_original_stream(reversed(original), 17)
    assert [row["request_uid"] for row in forward] == [row["request_uid"] for row in reverse]


def test_pair_alignment_and_chain(seed17_pair) -> None:
    summary = validate_pair_alignment(
        seed17_pair["control"], seed17_pair["targeted"], 17,
    )
    assert summary["common_original_slots"] == 3072
    assert summary["control_only_original_slots"] == 1024
    assert summary["target_emotion_slots"] == 512
    assert summary["target_offensive_slots"] == 512


def test_first_six_are_fully_identical_and_rng_is_update_scoped(seed17_pair) -> None:
    for control, targeted in zip(seed17_pair["control"], seed17_pair["targeted"]):
        assert control["model_rng_seed"] == targeted["model_rng_seed"]
        assert control["slots"][:6] == targeted["slots"][:6]
    assert len({row["model_rng_seed"] for row in seed17_pair["control"]}) == 512


def test_tape_regeneration_rejects_rehashed_substitution(seed17_pair) -> None:
    control = copy.deepcopy(seed17_pair["control"])
    control[10]["slots"][0]["parent_id"] = "foreign"
    control[10]["row_sha256"] = digest_object(control[10], "row_sha256")
    control[11]["previous_row_sha256"] = control[10]["row_sha256"]
    control[11]["row_sha256"] = digest_object(control[11], "row_sha256")
    with pytest.raises(ContractError, match="common slots differ"):
        validate_pair_alignment(control, seed17_pair["targeted"], 17)


def test_target_balance_and_no_parent_reuse(seed17_pair, pools) -> None:
    _, target = pools
    by_uid = {row["request_uid"]: row for row in target}
    counts = Counter()
    parents = set()
    for row in seed17_pair["targeted"]:
        for slot in row["slots"][6:]:
            request = by_uid[slot["request_uid"]]
            question = request["questions"][0]
            counts[(request["origin_source"], question["option_keys"][question["label"]])] += 1
            parents.add((request["origin_source"], request["parent_id"]))
    assert sorted(count for (source, _), count in counts.items() if source == "emotion") == [85, 85, 85, 85, 86, 86]
    assert sorted(count for (source, _), count in counts.items() if source == "tweet_offensive") == [256, 256]
    assert len(parents) == 1024


def test_choice_permutation_preserves_semantic_label_and_nonchoice_order() -> None:
    request = normalized_request(1, label="b")
    before = request["questions"][0]["option_keys"][request["questions"][0]["label"]]
    augmented, stored = make_choice_permutations(request, 17, 0, 0)
    after = augmented["questions"][0]["option_keys"][augmented["questions"][0]["label"]]
    assert after == before
    assert sorted(stored[0]["new_to_old"]) == [0, 1, 2]
    request["questions"][0]["type"] = "score"
    _, stored = make_choice_permutations(request, 17, 0, 0)
    assert stored[0]["new_to_old"] == [0, 1, 2]
