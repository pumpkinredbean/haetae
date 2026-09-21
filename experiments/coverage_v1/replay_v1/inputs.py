"""Immutable public-input normalization for T05 paired replay."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from experiments.coverage_v1.replay_v1.contracts import (
    CapturedFile,
    ContractError,
    capture_allowed_file,
    digest_object,
    strict_json_loads,
)
from experiments.kev_adapter import normalize_request

EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
OFFENSIVE_LABELS = ("true", "false")
TARGET_RENDERING = {
    "emotion": {
        "id": "emotion",
        "type": "choice",
        "instructions": "Which emotion does the writer express?",
        "options": list(EMOTION_LABELS),
        "option_keys": list(EMOTION_LABELS),
        "soft": None,
        "source": "emotion",
    },
    "tweet_offensive": {
        "id": "tweet_offensive",
        "type": "noul",
        "instructions": "Is this post offensive?",
        "options": [
            "yes: Contains insults, threats, profanity directed at someone, or hateful content",
            "no: Not offensive",
        ],
        "option_keys": list(OFFENSIVE_LABELS),
        "soft": None,
        "source": "tweet_offensive",
    },
}


def _strict_jsonl(captured: CapturedFile) -> list[dict[str, Any]]:
    rows = []
    for index, raw in enumerate(captured.data.splitlines(), start=1):
        if not raw.strip():
            continue
        value = strict_json_loads(raw)
        if not isinstance(value, dict):
            raise ContractError(f"JSONL row {index} is not an object")
        rows.append(value)
    return rows


def _resolve_artifact(paths: Mapping[str, Any], artifact_id: str) -> tuple[Path, str]:
    if set(paths) != {"schema_version", "roots", "artifacts", "output_root", "mps_owner_lock"}:
        raise ContractError("private path map has the wrong top-level fields")
    if paths["schema_version"] != 1:
        raise ContractError("unsupported private path-map schema")
    try:
        location = paths["artifacts"][artifact_id]
        root = paths["roots"][location["root"]]
        relative = location["relative_path"]
    except (KeyError, TypeError) as error:
        raise ContractError(f"missing path mapping for {artifact_id}") from error
    if set(location) != {"root", "relative_path"} or set(root) != {"path", "allow_prefixes"}:
        raise ContractError("path mapping has unknown fields")
    allowed = root["allow_prefixes"]
    if not isinstance(allowed, list) or not allowed:
        raise ContractError("artifact root needs explicit allow prefixes")
    if not any(relative == prefix or relative.startswith(prefix.rstrip("/") + "/") for prefix in allowed):
        raise ContractError(f"artifact path is outside its allowlist: {artifact_id}")
    return Path(root["path"]), relative


def load_bound_sources(
    path_map_file: str | Path,
    registry_file: str | Path,
    required_artifacts: Iterable[str],
) -> dict[str, CapturedFile]:
    """Capture registered inputs without invoking dataset loaders or downloads."""

    path_map_path = Path(path_map_file)
    registry_path = Path(registry_file)
    path_map = strict_json_loads(path_map_path.read_bytes())
    registry = strict_json_loads(registry_path.read_bytes())
    if not isinstance(path_map, dict) or not isinstance(registry, dict):
        raise ContractError("path map and registry must be JSON objects")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("registry has no artifact table")
    captured: dict[str, CapturedFile] = {}
    for artifact_id in sorted(set(required_artifacts)):
        descriptor = artifacts.get(artifact_id)
        if not isinstance(descriptor, dict) or descriptor.get("contains_locked_data") is not False:
            raise ContractError(f"artifact is absent or not public: {artifact_id}")
        root, relative = _resolve_artifact(path_map, artifact_id)
        captured[artifact_id] = capture_allowed_file(
            root,
            relative,
            {
                "artifact_id": artifact_id,
                "sha256": descriptor["sha256"],
                "size_bytes": descriptor["size_bytes"],
            },
        )
    return captured


def _validate_question(question: Mapping[str, Any]) -> dict[str, Any]:
    required = {"id", "type", "instructions", "options", "option_keys", "label", "soft", "source"}
    if not isinstance(question, Mapping) or set(question) != required:
        raise ContractError("normalized question has the wrong fields")
    result = copy.deepcopy(dict(question))
    for field in ("id", "type", "instructions", "source"):
        if not isinstance(result[field], str) or not result[field]:
            raise ContractError(f"question {field} must be nonempty text")
    if result["type"] not in {"choice", "noul", "score"}:
        raise ContractError("unsupported normalized question type")
    options, keys = result["options"], result["option_keys"]
    if (
        not isinstance(options, list) or not isinstance(keys, list)
        or len(options) < 2 or len(options) != len(keys)
        or any(not isinstance(item, str) or not item for item in options + keys)
        or len(set(keys)) != len(keys)
    ):
        raise ContractError("question options or keys are invalid")
    label = result["label"]
    if type(label) is not int or not 0 <= label < len(options):
        raise ContractError("question label must be an in-range plain integer")
    soft = result["soft"]
    if soft is not None:
        if not isinstance(soft, list) or len(soft) != len(options):
            raise ContractError("soft target shape differs from options")
        if any(type(item) not in (int, float) or not math.isfinite(item) or item < 0 for item in soft):
            raise ContractError("soft targets must be finite and nonnegative")
        if not math.isclose(sum(soft), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ContractError("soft targets must sum to one")
    if result["type"] == "noul" and keys != ["true", "false"]:
        raise ContractError("Noul option order must be true then false")
    if result["type"] == "score" and keys != [str(index) for index in range(len(keys))]:
        raise ContractError("Score order must remain ordinal")
    return result


def validate_semantic_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "request_uid", "origin_source", "parent_id", "request_id", "state",
        "state_sha256", "questions", "original_record_sha256", "request_sha256",
    }
    if not isinstance(request, Mapping) or set(request) != required:
        raise ContractError("normalized request has the wrong fields")
    result = copy.deepcopy(dict(request))
    for field in ("request_uid", "origin_source", "parent_id", "request_id", "state"):
        if not isinstance(result[field], str) or not result[field]:
            raise ContractError(f"request {field} must be nonempty text")
    for field in ("request_uid", "state_sha256", "original_record_sha256", "request_sha256"):
        if not isinstance(result[field], str) or len(result[field]) != 64:
            raise ContractError(f"request {field} is not a SHA-256")
    if digest_object(result["state"]) != result["state_sha256"]:
        raise ContractError("request state digest mismatch")
    questions = result["questions"]
    if not isinstance(questions, list) or not questions:
        raise ContractError("request must contain questions")
    result["questions"] = [_validate_question(question) for question in questions]
    identifiers = [question["id"] for question in result["questions"]]
    if len(identifiers) != len(set(identifiers)):
        raise ContractError("question IDs must be unique within a request")
    core = {
        "origin_source": result["origin_source"],
        "parent_id": result["parent_id"],
        "request_id": result["request_id"],
        "state": result["state"],
        "questions": result["questions"],
        "original_record_sha256": result["original_record_sha256"],
    }
    if digest_object(core) != result["request_sha256"]:
        raise ContractError("normalized request digest mismatch")
    expected_uid = digest_object([
        result["origin_source"], result["parent_id"], result["request_id"],
        result["request_sha256"],
    ])
    if result["request_uid"] != expected_uid:
        raise ContractError("normalized request UID mismatch")
    return result


def normalize_original_pool(raw_requests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pool = []
    seen_uids: set[str] = set()
    semantic_targets: dict[tuple[str, str, str, str], tuple[Any, ...]] = {}
    for raw in raw_requests:
        if not isinstance(raw, Mapping):
            raise ContractError("original training row is not an object")
        normalized = normalize_request(dict(raw))
        meta = normalized["meta"]
        origin = meta.get("source")
        request_id = meta.get("id")
        parent = meta.get("group_id") or request_id
        if any(not isinstance(value, str) or not value for value in (origin, request_id, parent)):
            raise ContractError("original request identity is incomplete")
        questions = [_validate_question(question) for question in normalized["questions"]]
        original_digest = digest_object(raw)
        core = {
            "origin_source": origin,
            "parent_id": parent,
            "request_id": request_id,
            "state": normalized["state"],
            "questions": questions,
            "original_record_sha256": original_digest,
        }
        request_digest = digest_object(core)
        request_uid = digest_object([origin, parent, request_id, request_digest])
        request = {
            "request_uid": request_uid,
            **core,
            "state_sha256": digest_object(normalized["state"]),
            "request_sha256": request_digest,
        }
        if request_uid in seen_uids:
            raise ContractError("original request UID collision")
        seen_uids.add(request_uid)
        for question in questions:
            associations = tuple(sorted(zip(question["option_keys"], question["options"]))) if question["type"] == "choice" else tuple(zip(question["option_keys"], question["options"]))
            target = (associations, question["option_keys"][question["label"]], question["soft"])
            key = (origin, request["state_sha256"], question["id"], question["instructions"])
            old = semantic_targets.setdefault(key, target)
            if old != target:
                raise ContractError("contradictory original semantic target")
        pool.append(validate_semantic_identity(request))
    return sorted(pool, key=lambda item: item["request_uid"])


def freeze_target_rendering(support_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for support in support_rows:
        source = support["source"]
        template = copy.deepcopy(TARGET_RENDERING[source])
        label = support["label"]
        if source == "emotion":
            if label not in EMOTION_LABELS:
                raise ContractError("support emotion label is outside the native ontology")
            template["label"] = EMOTION_LABELS.index(label)
        else:
            if label not in OFFENSIVE_LABELS:
                raise ContractError("offensive support label is outside the native ontology")
            template["label"] = 0 if label == "true" else 1
        core = {
            "origin_source": source,
            "parent_id": support["parent_id"],
            "request_id": f"t05/support/{source}/{support['row']}",
            "state": support["state"],
            "questions": [_validate_question(template)],
            "original_record_sha256": support["selection_sha256"],
        }
        request_sha256 = digest_object(core)
        request = {
            "request_uid": digest_object([
                source, core["parent_id"], core["request_id"], request_sha256,
            ]),
            **core,
            "state_sha256": digest_object(core["state"]),
            "request_sha256": request_sha256,
        }
        requests.append(validate_semantic_identity(request))
    return requests


def validate_target_support(
    support_rows: Iterable[Mapping[str, Any]],
    *,
    forbidden_state_sha256: set[str] | None = None,
    forbidden_parents: set[tuple[str, str]] | None = None,
    expected_counts: bool = True,
) -> list[dict[str, Any]]:
    rows = [copy.deepcopy(dict(row)) for row in support_rows]
    forbidden_state_sha256 = forbidden_state_sha256 or set()
    forbidden_parents = forbidden_parents or set()
    seen_states: set[str] = set()
    seen_parents: set[tuple[str, str]] = set()
    counts: dict[tuple[str, Any], int] = {}
    required = {"label", "parent_id", "row", "selection_sha256", "source", "split", "state", "state_sha256"}
    for row in rows:
        if set(row) != required:
            raise ContractError("support row has the wrong fields")
        if row["source"] not in TARGET_RENDERING or row["split"] != "train":
            raise ContractError("support row is outside the frozen training sources")
        if type(row["row"]) is not int or row["row"] < 0:
            raise ContractError("support source row must be a nonnegative plain integer")
        if not isinstance(row["state"], str) or not row["state"]:
            raise ContractError("support state must be nonempty text")
        if digest_object(row["state"]) != row["state_sha256"]:
            raise ContractError("support state digest mismatch")
        if any(not isinstance(row[field], str) or not row[field] for field in ("parent_id", "selection_sha256")):
            raise ContractError("support identity is incomplete")
        if row["source"] == "emotion":
            if row["label"] not in EMOTION_LABELS:
                raise ContractError("augmented emotion label is forbidden")
        elif row["label"] not in OFFENSIVE_LABELS:
            raise ContractError("offensive semantic label is outside the native ontology")
        parent_key = (row["source"], row["parent_id"])
        if row["state_sha256"] in seen_states or parent_key in seen_parents:
            raise ContractError("support contains a duplicate state or parent")
        if row["state_sha256"] in forbidden_state_sha256 or parent_key in forbidden_parents:
            raise ContractError("support intersects an excluded population")
        seen_states.add(row["state_sha256"])
        seen_parents.add(parent_key)
        counts[(row["source"], row["label"])] = counts.get((row["source"], row["label"]), 0) + 1
    if expected_counts:
        expected = {("emotion", label): 256 for label in EMOTION_LABELS}
        expected.update({("tweet_offensive", label): 256 for label in OFFENSIVE_LABELS})
        if counts != expected or len(rows) != 2048:
            raise ContractError("support class counts differ from the accepted T04 fit pool")
    return freeze_target_rendering(rows)


def audit_exclusions(
    target_requests: Iterable[Mapping[str, Any]],
    comparison_requests: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    targets = list(target_requests)
    comparisons = list(comparison_requests)
    target_states = {request["state_sha256"] for request in targets}
    target_parents = {(request["origin_source"], request["parent_id"]) for request in targets}
    comparison_states = {request["state_sha256"] for request in comparisons}
    comparison_parents = {(request["origin_source"], request["parent_id"]) for request in comparisons}
    state_overlap = sorted(target_states & comparison_states)
    parent_overlap = sorted(target_parents & comparison_parents)
    if state_overlap or parent_overlap:
        raise ContractError("target support intersects an excluded population")
    return {
        "target_requests": len(targets),
        "comparison_requests": len(comparisons),
        "state_overlap": 0,
        "parent_overlap": 0,
        "target_state_sha256": digest_object(sorted(target_states)),
        "target_parent_sha256": digest_object(sorted(target_parents)),
    }


def build_evaluation_inventory(populations: Mapping[str, Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    inventory = []
    seen: set[tuple[str, str, str, str]] = set()
    for population, requests in sorted(populations.items()):
        for request in requests:
            normalized = validate_semantic_identity(request)
            for question in normalized["questions"]:
                key = (population, normalized["request_id"], question["id"], "original")
                if key in seen:
                    raise ContractError("evaluation inventory contains duplicate keys")
                seen.add(key)
                inventory.append({
                    "key": list(key),
                    "population": population,
                    "request_uid": normalized["request_uid"],
                    "request_id": normalized["request_id"],
                    "question_id": question["id"],
                    "origin_source": normalized["origin_source"],
                    "parent_id": normalized["parent_id"],
                    "state_sha256": normalized["state_sha256"],
                    "question_sha256": digest_object(question),
                    "request_sha256": normalized["request_sha256"],
                })
    return inventory


def build_zero_reference(
    inventory: Iterable[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {tuple(row["key"]): row for row in inventory}
    result = []
    for observation in observations:
        key = tuple(observation.get("key", ()))
        if key not in expected:
            raise ContractError("zero-reference row is outside the evaluation inventory")
        logits = observation.get("logits")
        if not isinstance(logits, list) or len(logits) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
            raise ContractError("zero-reference logits are malformed")
        result.append({
            "key": list(key),
            "state_sha256": expected[key]["state_sha256"],
            "question_sha256": expected[key]["question_sha256"],
            "logits": [float(value) for value in logits],
        })
    if {tuple(row["key"]) for row in result} != set(expected):
        raise ContractError("zero-reference membership differs from the inventory")
    return sorted(result, key=lambda row: tuple(row["key"]))
