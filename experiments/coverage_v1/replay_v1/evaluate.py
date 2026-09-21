"""Terminal-only evaluation and generation-79 parity for T05."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch

from experiments.coverage_v1.replay_v1.analysis import (
    FIXED_TEMPERATURE,
    aggregate_metrics,
    offensive_diagnostics,
)
from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object


def validate_evaluation_request(
    request: Mapping[str, Any], inventory_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not request.get("questions") or len(request["questions"]) != len(inventory_rows):
        raise ContractError("evaluation request question count differs from inventory")
    for question, inventory in zip(request["questions"], inventory_rows):
        if question["id"] != inventory["question_id"]:
            raise ContractError("evaluation question order differs from inventory")
        if request["state_sha256"] != inventory["state_sha256"]:
            raise ContractError("evaluation state identity differs from inventory")
        if digest_object(question) != inventory["question_sha256"]:
            raise ContractError("evaluation question identity differs from inventory")


def cpu_first_logits(value: torch.Tensor) -> dict[str, list[float]]:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() < 2:
        raise ContractError("model output is not a candidate vector")
    detached = value.detach()
    cpu_float32 = detached.cpu()
    if cpu_float32.device.type != "cpu" or cpu_float32.dtype != torch.float32:
        raise ContractError("model output must be copied to CPU as float32")
    if not torch.isfinite(cpu_float32).all():
        raise ContractError("model output contains a nonfinite value")
    widened = cpu_float32.to(torch.float64)
    return {
        "cpu_float32": [float(item) for item in cpu_float32.tolist()],
        "logits": [float(item) for item in widened.tolist()],
    }


def _probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    values = [math.exp(value - maximum) for value in scaled]
    total = math.fsum(values)
    return [value / total for value in values]


def _log_probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    normalizer = maximum + math.log(math.fsum(math.exp(value - maximum) for value in scaled))
    return [value - normalizer for value in scaled]


def run_initial_parity(
    observations: Iterable[Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    observations = list(observations)
    references = list(references)
    if len(observations) != 196 or len(references) != 196:
        raise ContractError("step-zero parity requires exactly 196 controls")
    reference_by_key = {tuple(row["key"]): row for row in references}
    if len(reference_by_key) != 196:
        raise ContractError("step-zero reference contains duplicate keys")
    maximum_log_error = 0.0
    material_changes = 0
    near_tie_changes = []
    temperature_summaries = {}
    raw_components_passed = True
    for observation in observations:
        key = tuple(observation["key"])
        if key not in reference_by_key:
            raise ContractError("step-zero observation membership differs")
        reference = reference_by_key[key]
        if observation["state_sha256"] != reference["state_sha256"] or observation["question_sha256"] != reference["question_sha256"]:
            raise ContractError("step-zero semantic identity differs")
        if len(observation["logits"]) != len(reference["logits"]):
            raise ContractError("step-zero candidate count differs")
        for actual, expected in zip(observation["logits"], reference["logits"]):
            if abs(actual - expected) > 1e-4 + 1e-5 * abs(expected):
                raw_components_passed = False
        actual_action = max(range(len(observation["logits"])), key=observation["logits"].__getitem__)
        expected_action = max(range(len(reference["logits"])), key=reference["logits"].__getitem__)
        reference_probabilities = _probabilities(reference["logits"], 1.0)
        ordered = sorted(reference_probabilities, reverse=True)
        margin = ordered[0] - ordered[1]
        if actual_action != expected_action:
            if margin > 2e-4:
                material_changes += 1
            else:
                near_tie_changes.append(list(key))
        for temperature in (1.0, FIXED_TEMPERATURE):
            actual_log = _log_probabilities(observation["logits"], temperature)
            expected_log = _log_probabilities(reference["logits"], temperature)
            maximum_log_error = max(
                maximum_log_error,
                max(abs(left - right) for left, right in zip(actual_log, expected_log)),
            )
    for temperature in (1.0, FIXED_TEMPERATURE):
        televisions = []
        for observation in observations:
            reference = reference_by_key[tuple(observation["key"])]
            actual = _probabilities(observation["logits"], temperature)
            expected = _probabilities(reference["logits"], temperature)
            televisions.append(0.5 * math.fsum(abs(left - right) for left, right in zip(actual, expected)))
        ordered = sorted(televisions)
        temperature_summaries[str(temperature)] = {
            "maximum_total_variation": ordered[-1],
            "p99_total_variation": ordered[math.ceil(0.99 * len(ordered)) - 1],
        }
    passed = (
        raw_components_passed and maximum_log_error <= 1e-3
        and material_changes == 0
        and all(value["maximum_total_variation"] <= 1e-4 for value in temperature_summaries.values())
        and all(value["p99_total_variation"] <= 1e-5 for value in temperature_summaries.values())
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "original_rows": 196,
        "raw_components_passed": raw_components_passed,
        "maximum_absolute_log_probability_error": maximum_log_error,
        "material_action_changes": material_changes,
        "near_tie_action_changes": near_tie_changes,
        "temperature_summaries": temperature_summaries,
    }


def _float32_exact(value: float) -> bool:
    return float(struct.unpack("!f", struct.pack("!f", value))[0]) == value


def validate_predictions(
    rows: Iterable[Mapping[str, Any]], inventory: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    inventory = [dict(row) for row in inventory]
    if len(rows) != len(inventory):
        raise ContractError("prediction row count differs from inventory")
    required = {
        "schema_version", "key", "request_id", "question_id", "origin_source",
        "parent_id", "task_source", "primitive", "population_ids",
        "state_sha256", "question_sha256", "ordered_option_keys",
        "ordered_option_descriptions", "label", "soft", "encoding_sha256",
        "packed_tokens", "cpu_float32", "logits", "row_sha256",
    }
    validated = []
    for row, expected in zip(rows, inventory):
        if set(row) != required:
            raise ContractError("prediction row has unexpected or missing fields")
        if row["row_sha256"] != digest_object(row, "row_sha256"):
            raise ContractError("prediction row self-digest mismatch")
        for field in (
            "key", "request_id", "question_id", "origin_source", "parent_id",
            "task_source", "primitive", "population_ids", "state_sha256",
            "question_sha256", "ordered_option_keys", "ordered_option_descriptions",
            "label", "soft", "encoding_sha256", "packed_tokens",
        ):
            if row[field] != expected[field]:
                raise ContractError(f"prediction identity differs: {field}")
        if (
            len(row["cpu_float32"]) != len(row["ordered_option_keys"])
            or row["logits"] != row["cpu_float32"]
            or any(type(value) not in (int, float) or not math.isfinite(value) or not _float32_exact(value) for value in row["cpu_float32"])
        ):
            raise ContractError("prediction float32 evidence is inconsistent")
        validated.append(row)
    return validated


def evaluate_endpoint(
    *,
    model,
    requests_by_uid: Mapping[str, Mapping[str, Any]],
    inventory_by_uid: Mapping[str, Sequence[Mapping[str, Any]]],
    schedule: Sequence[Mapping[str, Any]],
    encode: Callable[[Mapping[str, Any]], tuple[Any, str, int]],
) -> list[dict[str, Any]]:
    observations = []
    with torch.inference_mode():
        for batch in schedule:
            requests = [requests_by_uid[uid] for uid in batch["request_uids"]]
            encodings = []
            encoding_meta = []
            for request in requests:
                validate_evaluation_request(request, inventory_by_uid[request["request_uid"]])
                packed, encoding_sha256, tokens = encode(request)
                encodings.append(packed)
                encoding_meta.append((encoding_sha256, tokens))
            outputs = model(encodings)
            if len(outputs) != len(requests):
                raise ContractError("endpoint model output count differs from requests")
            for request, question_outputs, metadata in zip(requests, outputs, encoding_meta):
                rows = inventory_by_uid[request["request_uid"]]
                if len(question_outputs) != len(rows):
                    raise ContractError("endpoint question output count differs")
                for expected, logits in zip(rows, question_outputs):
                    values = cpu_first_logits(logits)
                    row = {
                        **{key: expected[key] for key in (
                            "key", "request_id", "question_id", "origin_source",
                            "parent_id", "task_source", "primitive", "population_ids",
                            "state_sha256", "question_sha256", "ordered_option_keys",
                            "ordered_option_descriptions", "label", "soft",
                        )},
                        "schema_version": 1,
                        "encoding_sha256": metadata[0],
                        "packed_tokens": metadata[1],
                        **values,
                    }
                    row["row_sha256"] = digest_object(row)
                    observations.append(row)
    expected = [row for batch in schedule for uid in batch["request_uids"] for row in inventory_by_uid[uid]]
    return validate_predictions(observations, expected)


def replay_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    by_population: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for population in row["population_ids"]:
            by_population.setdefault(population, []).append(row)
    result = {}
    for population, population_rows in sorted(by_population.items()):
        result[population] = {
            "raw": aggregate_metrics(population_rows, 1.0),
            "fixed": aggregate_metrics(population_rows, FIXED_TEMPERATURE),
        }
        if population == "offensive_native":
            result[population]["offensive_raw"] = offensive_diagnostics(
                population_rows, 1.0,
            )
            result[population]["offensive_fixed"] = offensive_diagnostics(
                population_rows, FIXED_TEMPERATURE,
            )
    return result
