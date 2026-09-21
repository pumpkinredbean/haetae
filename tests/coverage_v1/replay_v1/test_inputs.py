from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.coverage_v1.replay_v1.contracts import (
    ContractError,
    capture_allowed_file,
    digest_object,
    strict_json_loads,
    validate_descriptor,
    validate_schema,
    write_exclusive_json,
)
from experiments.coverage_v1.replay_v1.inputs import (
    EMOTION_LABELS,
    TARGET_RENDERING,
    audit_exclusions,
    normalize_original_pool,
    validate_semantic_identity,
    validate_target_support,
)


def original_request(identifier: str = "row/1", label: str = "a") -> dict:
    return {
        "_meta": {
            "id": identifier,
            "group_id": identifier,
            "source": "fixture",
        },
        "state": f"state {identifier}",
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"a": "first", "b": "second"},
                "label": label,
                "src": "fixture_task",
            }
        },
    }


def support_row(source: str, label, index: int) -> dict:
    state = f"{source} state {label} {index}"
    return {
        "label": label,
        "parent_id": f"{source}/train/{label}/{index}",
        "row": index,
        "selection_sha256": hashlib.sha256(f"selection {source} {label} {index}".encode()).hexdigest(),
        "source": source,
        "split": "train",
        "state": state,
        "state_sha256": digest_object(state),
    }


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ContractError, match="duplicate"):
        strict_json_loads('{"x":1,"x":2}')
    with pytest.raises(ContractError, match="non-finite"):
        strict_json_loads('{"x":NaN}')
    with pytest.raises(ContractError, match="non-finite"):
        strict_json_loads('{"x":1e9999}')


def test_descriptor_rejects_boolean_integer_and_unknown_fields() -> None:
    descriptor = {"artifact_id": "x", "sha256": "0" * 64, "size_bytes": True}
    with pytest.raises(ContractError, match="plain integer"):
        validate_descriptor(descriptor)
    descriptor["size_bytes"] = 0
    descriptor["extra"] = 1
    with pytest.raises(ContractError, match="unknown"):
        validate_descriptor(descriptor)


def test_schema_rejects_extra_field_and_boolean_integer() -> None:
    schema = {
        "type": "object",
        "required": ["count"],
        "properties": {"count": {"type": "integer", "minimum": 0}},
        "additionalProperties": False,
    }
    validate_schema({"count": 1}, schema)
    with pytest.raises(ContractError, match="wrong type"):
        validate_schema({"count": True}, schema)
    with pytest.raises(ContractError, match="unknown"):
        validate_schema({"count": 1, "other": 2}, schema)


def test_capture_hashes_and_parses_one_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text('{"value":1}', encoding="utf-8")
    data = path.read_bytes()
    captured = capture_allowed_file(
        tmp_path,
        "value.json",
        {
            "artifact_id": "value",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        },
    )
    path.write_text('{"value":2}', encoding="utf-8")
    assert captured.json() == {"value": 1}
    assert captured.sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("relative", ["../sentinel", "/tmp/sentinel", "a/../../sentinel"])
def test_capture_rejects_path_escape(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ContractError, match="path"):
        capture_allowed_file(tmp_path, relative)


def test_capture_rejects_leaf_and_directory_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("secret")
    (tmp_path / "leaf").symlink_to(outside / "sentinel")
    (tmp_path / "directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError, match="symbolic"):
        capture_allowed_file(tmp_path, "leaf")
    with pytest.raises(ContractError, match="symbolic"):
        capture_allowed_file(tmp_path, "directory/sentinel")


def test_exclusive_json_never_clobbers(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    descriptor = write_exclusive_json(path, {"value": 1})
    assert descriptor["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_exclusive_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_original_pool_is_source_order_independent() -> None:
    first = original_request("row/1")
    second = original_request("row/2", "b")
    forward = normalize_original_pool([first, second])
    reverse = normalize_original_pool([second, first])
    assert forward == reverse
    assert len(forward) == 2
    assert all(validate_semantic_identity(row) == row for row in forward)


def test_question_target_validation_rejects_boolean_and_invalid_soft() -> None:
    request = normalize_original_pool([original_request()])[0]
    request["questions"][0]["label"] = True
    core = {key: request[key] for key in (
        "origin_source", "parent_id", "request_id", "state", "questions",
        "original_record_sha256",
    )}
    request["request_sha256"] = digest_object(core)
    request["request_uid"] = digest_object([
        request["origin_source"], request["parent_id"], request["request_id"],
        request["request_sha256"],
    ])
    with pytest.raises(ContractError, match="plain integer"):
        validate_semantic_identity(request)

    request = normalize_original_pool([original_request()])[0]
    request["questions"][0]["soft"] = [0.25, 0.25]
    core["questions"] = request["questions"]
    request["request_sha256"] = digest_object(core)
    request["request_uid"] = digest_object([
        request["origin_source"], request["parent_id"], request["request_id"],
        request["request_sha256"],
    ])
    with pytest.raises(ContractError, match="sum to one"):
        validate_semantic_identity(request)


def test_native_target_templates_are_literal() -> None:
    assert TARGET_RENDERING["emotion"]["instructions"] == "Which emotion does the writer express?"
    assert TARGET_RENDERING["emotion"]["option_keys"] == list(EMOTION_LABELS)
    assert TARGET_RENDERING["tweet_offensive"]["option_keys"] == ["true", "false"]
    assert TARGET_RENDERING["tweet_offensive"]["options"][0].startswith("yes:")


def test_support_native_ontology_and_offensive_orientation() -> None:
    rows = [support_row("emotion", label, index) for label in EMOTION_LABELS for index in range(2)]
    rows += [support_row("tweet_offensive", label, index) for label in ("false", "true") for index in range(2)]
    rendered = validate_target_support(rows, expected_counts=False)
    offensive = [row for row in rendered if row["origin_source"] == "tweet_offensive"]
    assert {row["questions"][0]["label"] for row in offensive} == {0, 1}
    true_row = next(row for row in offensive if row["request_id"].endswith("/0") and "true" in row["state"])
    assert true_row["questions"][0]["label"] == 0


def test_support_rejects_none_of_these_nontrain_and_overlap() -> None:
    invalid = support_row("emotion", "none_of_these", 1)
    with pytest.raises(ContractError, match="emotion label"):
        validate_target_support([invalid], expected_counts=False)
    invalid = support_row("emotion", "joy", 1)
    invalid["split"] = "test"
    with pytest.raises(ContractError, match="training sources"):
        validate_target_support([invalid], expected_counts=False)
    row = support_row("emotion", "joy", 1)
    with pytest.raises(ContractError, match="excluded"):
        validate_target_support(
            [row], forbidden_state_sha256={row["state_sha256"]},
            expected_counts=False,
        )


def test_support_rejects_duplicate_parent_without_reselection() -> None:
    first = support_row("emotion", "joy", 1)
    second = support_row("emotion", "joy", 2)
    second["parent_id"] = first["parent_id"]
    with pytest.raises(ContractError, match="duplicate"):
        validate_target_support([first, second], expected_counts=False)


def test_exclusion_audit_uses_source_namespaced_parent() -> None:
    target = validate_target_support(
        [support_row("emotion", "joy", 1)], expected_counts=False,
    )[0]
    comparison = dict(target)
    comparison["origin_source"] = "other"
    comparison["parent_id"] = target["parent_id"]
    comparison["state"] = "different"
    comparison["state_sha256"] = digest_object("different")
    core = {key: comparison[key] for key in (
        "origin_source", "parent_id", "request_id", "state", "questions",
        "original_record_sha256",
    )}
    comparison["request_sha256"] = digest_object(core)
    comparison["request_uid"] = digest_object([
        comparison["origin_source"], comparison["parent_id"], comparison["request_id"],
        comparison["request_sha256"],
    ])
    audit = audit_exclusions([target], [comparison])
    assert audit["parent_overlap"] == 0
