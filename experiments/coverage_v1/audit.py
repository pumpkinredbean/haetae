"""Audit frozen coverage and cached predictions without constructing a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from experiments.coverage_v1.registry import canonical_sha256
from experiments.kev_adapter import normalize_request


ROLE_ORDER = (
    "train",
    "calibration",
    "decision_development",
    "transfer_development",
    "korean_development",
)
TARGET_ONTOLOGIES = {
    "emotion": ("sadness", "joy", "love", "anger", "fear", "surprise", "none_of_these"),
    "tweet_offensive": ("false", "true"),
}
TARGET_NATIVE_ONTOLOGIES = {
    "emotion": ("sadness", "joy", "love", "anger", "fear", "surprise"),
    "tweet_offensive": ("false", "true"),
}
TARGET_METADATA = {
    "emotion": {
        "origin_source": "emotion",
        "repo": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "split": "test",
        "type": "choice",
    },
    "tweet_offensive": {
        "origin_source": "tweet_offensive",
        "repo": "cardiffnlp/tweet_eval",
        "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
        "split": "test",
        "type": "noul",
    },
}
PRIMARY_METRICS = ("accuracy", "nll", "brier_histogram", "brier_annotation")


def _semantic_label(question: dict) -> str:
    return str(question["option_keys"][question["label"]])


def _rubric_descriptor(question: dict) -> dict:
    if question["type"] == "choice":
        options = sorted(zip(question["option_keys"], question["options"]))
    else:
        options = list(zip(question["option_keys"], question["options"]))
    return {
        "source": question["source"],
        "type": question["type"],
        "instructions": question["instructions"],
        "semantic_options": options,
    }


def _request_identity(raw: dict, request: dict) -> tuple[str, str, str]:
    meta = raw.get("_meta", {})
    origin = meta.get("source")
    request_id = meta.get("id")
    parent = meta.get("group_id") or request_id
    if not all(isinstance(value, str) and value for value in (origin, request_id, parent)):
        raise ValueError("request metadata is incomplete")
    return origin, request_id, parent


def _state_sha256(request: dict) -> str:
    return canonical_sha256(request["state"])


def _normalized_partitions(partitions: dict[str, list[dict]]) -> dict[str, list[tuple[dict, dict]]]:
    if set(partitions) != set(ROLE_ORDER):
        raise ValueError("coverage audit roles are incomplete")
    return {
        role: [(raw, normalize_request(raw)) for raw in partitions[role]]
        for role in ROLE_ORDER
    }


def audit_role_overlap(partitions: dict[str, list[dict]]) -> dict:
    """Inventory roles and detect parent, state, duplicate, and label overlap."""
    normalized = _normalized_partitions(partitions)
    inventories = {}
    role_sets = {}
    for role in ROLE_ORDER:
        parents = set()
        states = set()
        questions = set()
        exact_requests = Counter()
        semantic_labels = defaultdict(set)
        for raw, request in normalized[role]:
            origin, request_id, parent = _request_identity(raw, request)
            state = _state_sha256(request)
            parents.add((origin, parent))
            states.add(state)
            exact_requests[canonical_sha256({
                "state": request["state"],
                "questions": request["questions"],
            })] += 1
            for question in request["questions"]:
                source = question.get("source")
                if not isinstance(source, str) or not source:
                    raise ValueError("question source is missing")
                question_key = (source, request_id, question["id"], state)
                if question_key in questions:
                    raise ValueError(f"duplicate question identity in {role}")
                questions.add(question_key)
                semantic_key = (
                    state,
                    source,
                    question["type"],
                    question["instructions"],
                    tuple(question["option_keys"]),
                    tuple(question["options"]),
                )
                semantic_labels[semantic_key].add(_semantic_label(question))
        conflicts = sum(len(labels) > 1 for labels in semantic_labels.values())
        inventories[role] = {
            "requests": len(normalized[role]),
            "questions": len(questions),
            "unique_source_parents": len(parents),
            "unique_rendered_states": len(states),
            "duplicate_requests": sum(count - 1 for count in exact_requests.values()),
            "conflicting_rendered_question_labels": conflicts,
        }
        role_sets[role] = {"parents": parents, "states": states}
    overlaps = {}
    for left, right in combinations(ROLE_ORDER, 2):
        overlaps[f"{left}__{right}"] = {
            "source_parent_overlap": len(
                role_sets[left]["parents"] & role_sets[right]["parents"]
            ),
            "rendered_state_overlap": len(
                role_sets[left]["states"] & role_sets[right]["states"]
            ),
        }
    return {
        "status": "passed" if not any(
            value for pair in overlaps.values() for value in pair.values()
        ) else "overlap_detected",
        "roles": inventories,
        "pairwise_overlap": overlaps,
    }


def audit_semantics(partitions: dict[str, list[dict]]) -> dict:
    """Validate label ontology, option order, and Noul polarity."""
    normalized = _normalized_partitions(partitions)
    type_counts = Counter()
    target_rows = defaultdict(list)
    option_signatures = defaultdict(set)
    for role in ROLE_ORDER:
        for raw, request in normalized[role]:
            origin, _, _ = _request_identity(raw, request)
            raw_questions = raw.get("questions")
            if not isinstance(raw_questions, dict):
                raise ValueError("raw request questions must be an object")
            normalized_by_id = {question["id"]: question for question in request["questions"]}
            if set(raw_questions) != set(normalized_by_id):
                raise ValueError("normalized question membership changed")
            for identifier, raw_question in raw_questions.items():
                question = normalized_by_id[identifier]
                question_type = raw_question.get("type")
                type_counts[question_type] += 1
                if question_type == "choice":
                    criteria = raw_question.get("criteria")
                    label = raw_question.get("label")
                    if not isinstance(criteria, dict) or not criteria or label not in criteria:
                        raise ValueError("Choice label is absent from its criteria")
                    keys = list(criteria)
                    if question["option_keys"] != keys or question["label"] != keys.index(label):
                        raise ValueError("Choice option order or label mapping changed")
                elif question_type == "noul":
                    label = raw_question.get("label")
                    if not isinstance(label, bool):
                        raise ValueError("Noul label must be Boolean")
                    if question["option_keys"] != ["true", "false"]:
                        raise ValueError("Noul option order is not yes-first")
                    expected = 0 if label else 1
                    if question["label"] != expected:
                        raise ValueError("Noul polarity is inverted")
                elif question_type == "score":
                    label = raw_question.get("label")
                    criteria = raw_question.get("criteria")
                    if (
                        type(label) is not int
                        or not isinstance(criteria, list)
                        or label < 0
                        or label >= len(criteria)
                    ):
                        raise ValueError("Score label is outside its ordered criteria")
                else:
                    raise ValueError(f"unsupported primitive {question_type!r}")
                source = question.get("source")
                signature = (
                    question["type"],
                    question["instructions"],
                    tuple(question["option_keys"]),
                    tuple(question["options"]),
                )
                option_signatures[source].add(signature)
                if source in TARGET_ONTOLOGIES:
                    target_rows[source].append((raw, question, origin))
    targets = {}
    for source, ontology in TARGET_ONTOLOGIES.items():
        rows = target_rows[source]
        if not rows:
            raise ValueError(f"target source {source!r} is absent")
        expected = TARGET_METADATA[source]
        label_counts = Counter()
        option_orders = set()
        parents = set()
        for raw, question, origin in rows:
            meta = raw["_meta"]
            if any(
                meta.get(key) != value
                for key, value in expected.items()
                if key not in {"origin_source", "type"}
            ):
                raise ValueError(f"target source {source!r} metadata changed")
            if origin != expected["origin_source"] or question["type"] != expected["type"]:
                raise ValueError(f"target source {source!r} identity changed")
            keys = tuple(question["option_keys"])
            if not set(keys).issubset(set(ontology)):
                raise ValueError(f"target source {source!r} contains an unknown label")
            label = _semantic_label(question)
            if label not in ontology:
                raise ValueError(f"target source {source!r} label changed")
            label_counts[label] += 1
            option_orders.add(keys)
            parents.add(meta.get("group_id") or meta["id"])
        targets[source] = {
            "questions": len(rows),
            "parents": len(parents),
            "native_ontology": list(TARGET_NATIVE_ONTOLOGIES[source]),
            "evaluation_ontology": list(ontology),
            "augmentation_labels": sorted(
                set(ontology) - set(TARGET_NATIVE_ONTOLOGIES[source])
            ),
            "label_counts": dict(sorted(label_counts.items())),
            "observed_option_orders": [list(value) for value in sorted(option_orders)],
            "metadata": expected,
        }
    return {
        "status": "passed",
        "primitive_counts": dict(sorted(type_counts.items())),
        "task_rubric_counts": {
            source: len(signatures)
            for source, signatures in sorted(option_signatures.items())
        },
        "targets": targets,
    }


def _coverage_counts(rows: list[tuple[dict, dict]]) -> dict:
    origins = Counter()
    tasks = Counter()
    primitives = Counter()
    labels = Counter()
    origin_tasks = Counter()
    rubrics = {}
    for raw, request in rows:
        origin, _, _ = _request_identity(raw, request)
        origins[origin] += 1
        for question in request["questions"]:
            source = question["source"]
            tasks[source] += 1
            primitives[question["type"]] += 1
            labels[f"{source}\0{_semantic_label(question)}"] += 1
            origin_tasks[f"{origin}\0{source}"] += 1
            rubric = _rubric_descriptor(question)
            rubrics[canonical_sha256(rubric)] = {
                "source": source,
                "type": question["type"],
                "instructions_sha256": canonical_sha256(question["instructions"]),
                "semantic_options_sha256": canonical_sha256(
                    rubric["semantic_options"]
                ),
                "option_count": len(question["options"]),
            }
    return {
        "requests_by_origin_source": dict(sorted(origins.items())),
        "questions_by_task_source": dict(sorted(tasks.items())),
        "questions_by_primitive": dict(sorted(primitives.items())),
        "questions_by_task_label": dict(sorted(labels.items())),
        "questions_by_origin_and_task": dict(sorted(origin_tasks.items())),
        "rubrics": [
            {"rubric_sha256": identity, **rubrics[identity]}
            for identity in sorted(rubrics)
        ],
    }


def audit_supervised_coverage(
    partitions: dict[str, list[dict]],
    baseline_run: dict,
) -> dict:
    """Count supervised questions separately from words present in state text."""
    normalized = _normalized_partitions(partitions)
    shared = {
        role: _coverage_counts(normalized[role])
        for role in ROLE_ORDER
    }
    config = baseline_run["spec"]["config"]
    baseline_sources = sorted(config["sources"].split(","))
    return {
        "shared": shared,
        "baseline": {
            "status": "configured_sources_only",
            "reason": (
                "The registered baseline run binds aggregate fingerprints but does not "
                "contain an itemized consumed-question inventory."
            ),
            "configured_sources": baseline_sources,
            "configured_per_source": config["per_source"],
            "configured_internal_validation_per_source": config["eval_per_source"],
            "target_configuration": {
                "emotion": "configured for baseline supervision",
                "tweet_offensive": "not configured for baseline supervision",
            },
        },
        "interpretation": (
            "Counts use question source metadata. State-text occurrences are not counted "
            "as supervision."
        ),
    }


def _consume_choice_randomness(request: dict, rng: random.Random) -> None:
    for question in request["questions"]:
        if question["type"] == "choice":
            order = list(range(len(question["options"])))
            rng.shuffle(order)


def _add_weight(group: dict[str, dict], key: str, coefficient: float) -> None:
    value = group.setdefault(key, {"draws": 0, "coefficient_sum": 0.0})
    value["draws"] += 1
    value["coefficient_sum"] += coefficient


def audit_exposure_weights(train: list[dict], shared_run: dict) -> dict:
    """Replay request sampling and account for the request-mean objective."""
    requests = [normalize_request(raw) for raw in train]
    experiment = shared_run["spec"]["experiment"]
    config = shared_run["spec"]["config"]
    if experiment.get("request_objective") != "mean question loss, then mean request loss":
        raise ValueError("shared request objective changed")
    if experiment.get("train_descriptor", {}).get("records") != len(requests):
        raise ValueError("shared run train count differs from frozen input")
    steps = int(config["steps"])
    batch_size = int(config["batch"])
    rng = random.Random(int(config["seed"]))
    order = list(range(len(requests)))
    rng.shuffle(order)
    cursor = 0
    epoch = 0
    groups = {
        "origin_source": {},
        "task_source": {},
        "origin_and_task": {},
        "primitive": {},
        "task_label": {},
        "rubric": {},
    }
    short_batches = []
    total_request_draws = 0
    total_question_draws = 0
    for step in range(steps):
        if cursor >= len(order):
            epoch += 1
            order = list(range(len(requests)))
            rng.shuffle(order)
            cursor = 0
        indexes = order[cursor:cursor + batch_size]
        cursor += len(indexes)
        if len(indexes) != batch_size:
            short_batches.append({"step": step + 1, "size": len(indexes)})
        for index in indexes:
            total_request_draws += 1
            raw = train[index]
            request = requests[index]
            _consume_choice_randomness(request, rng)
            origin = raw["_meta"]["source"]
            question_weight = 1.0 / (len(indexes) * len(request["questions"]))
            for question in request["questions"]:
                total_question_draws += 1
                source = question["source"]
                rubric = canonical_sha256(_rubric_descriptor(question))
                _add_weight(groups["origin_source"], origin, question_weight)
                _add_weight(groups["task_source"], source, question_weight)
                _add_weight(groups["origin_and_task"], f"{origin}\0{source}", question_weight)
                _add_weight(groups["primitive"], question["type"], question_weight)
                _add_weight(
                    groups["task_label"],
                    f"{source}\0{_semantic_label(question)}",
                    question_weight,
                )
                _add_weight(groups["rubric"], rubric, question_weight)
    for values in groups.values():
        for value in values.values():
            value["coefficient_share"] = value["coefficient_sum"] / steps
    total = sum(item["coefficient_sum"] for item in groups["task_source"].values())
    if not math.isclose(total, steps, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("objective coefficients do not sum to one per update")
    return {
        "steps": steps,
        "configured_batch": batch_size,
        "request_draws": total_request_draws,
        "question_draws": total_question_draws,
        "completed_epochs": epoch + int(cursor >= len(order)),
        "short_batches": short_batches,
        "coefficient_total": total,
        "groups": {
            name: dict(sorted(values.items()))
            for name, values in groups.items()
        },
        "objective": "mean question loss within request, then mean request loss within update",
    }


def _softmax(logits: list[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = [float(value) / temperature for value in logits]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("logits must be a nonempty finite list")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _target(row: dict) -> list[float]:
    count = len(row["options"])
    label = row.get("label")
    if type(label) is not int or label < 0 or label >= count:
        raise ValueError("cached prediction has an invalid hard label")
    soft = row.get("soft")
    if soft is None:
        return [float(index == label) for index in range(count)]
    if (
        not isinstance(soft, list)
        or len(soft) != count
        or not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in soft)
        or not math.isclose(sum(soft), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("cached prediction has an invalid soft target")
    return [float(value) for value in soft]


def _prediction_metrics(row: dict, temperature: float) -> dict[str, float]:
    if len(row.get("logits", [])) != len(row.get("options", [])):
        raise ValueError("cached logits do not match options")
    probabilities = _softmax(row["logits"], temperature)
    target = _target(row)
    predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "accuracy": float(predicted == row["label"]),
        "nll": -sum(
            target_value * math.log(max(probability, 1e-300))
            for target_value, probability in zip(target, probabilities)
        ),
        "brier_histogram": sum(
            (probability - target_value) ** 2
            for probability, target_value in zip(probabilities, target)
        ),
        "brier_annotation": (
            1.0
            + sum(probability * probability for probability in probabilities)
            - 2.0 * sum(
                probability * target_value
                for probability, target_value in zip(probabilities, target)
            )
        ),
    }


def _mean_metrics(rows: list[dict], temperature: float) -> dict:
    if not rows:
        raise ValueError("metric reproduction requires cached rows")
    values = [_prediction_metrics(row, temperature) for row in rows]
    return {
        metric: sum(value[metric] for value in values) / len(values)
        for metric in PRIMARY_METRICS
    }


def _reproduce_grouped(rows: list[dict], temperature: float) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    source = {
        key: _mean_metrics(values, temperature)
        for key, values in sorted(grouped.items())
    }
    return {
        "all": _mean_metrics(rows, temperature),
        "source": source,
        "source_macro": {
            metric: sum(value[metric] for value in source.values()) / len(source)
            for metric in PRIMARY_METRICS
        },
    }


def _prediction_key(row: dict) -> tuple:
    return (
        row["source"],
        row["request_id"],
        row["question_id"],
        row["state_sha256"],
    )


def _inventory(raw_rows: list[dict]) -> dict:
    result = {}
    for raw in raw_rows:
        request = normalize_request(raw)
        origin, request_id, _ = _request_identity(raw, request)
        state_sha256 = _state_sha256(request)
        for question in request["questions"]:
            expected = {
                "request_id": request_id,
                "group_id": request["meta"].get("group_id"),
                "state_sha256": state_sha256,
                "question_id": question["id"],
                "source": question["source"],
                "type": question["type"],
                "options": question["options"],
                "option_keys": question["option_keys"],
                "label": question["label"],
                "soft": question["soft"],
                "origin_source": origin,
            }
            key = _prediction_key(expected)
            if key in result:
                raise ValueError("frozen population contains a duplicate question")
            result[key] = expected
    return result


def _validate_population(rows: list[dict], inventory: dict, label: str) -> None:
    indexed = {}
    fields = (
        "request_id", "group_id", "state_sha256", "question_id", "source",
        "type", "options", "option_keys", "label", "soft",
    )
    for row in rows:
        key = _prediction_key(row)
        if key in indexed:
            raise ValueError(f"{label} contains a duplicate prediction")
        indexed[key] = row
        _prediction_metrics(row, 1.0)
    if set(indexed) != set(inventory):
        raise ValueError(f"{label} differs from its frozen population")
    for key, row in indexed.items():
        if any(row.get(field) != inventory[key].get(field) for field in fields):
            raise ValueError(f"{label} metadata differs from its frozen population")


def _clean_rows(rows: list[dict], exclusions: list[dict]) -> list[dict]:
    expected = {item["state_sha256"]: int(item["questions"]) for item in exclusions}
    observed = {key: 0 for key in expected}
    clean = []
    for row in rows:
        state = row["state_sha256"]
        if state in observed:
            observed[state] += 1
        else:
            clean.append(row)
    if observed != expected:
        raise ValueError("cached predictions differ from frozen exclusions")
    return clean


def _compare_metrics(actual: dict, expected: dict, label: str) -> float:
    maximum = 0.0
    for metric in PRIMARY_METRICS:
        delta = abs(float(actual[metric]) - float(expected[metric]))
        maximum = max(maximum, delta)
        if delta > 1e-10:
            raise ValueError(f"{label} cached {metric} is stale")
    return maximum


def _validate_report_identity(
    name: str,
    report: dict,
    run: dict,
    manifest: dict,
    comparison_plan: dict,
) -> None:
    expected_digest = report.get("report_sha256")
    unsigned = copy.deepcopy(report)
    unsigned.pop("report_sha256", None)
    if expected_digest != canonical_sha256(unsigned):
        raise ValueError(f"{name} report self-digest differs")
    if report.get("status") != "evaluated" or report.get("locked_test_opened") is not False:
        raise ValueError(f"{name} report status is invalid")
    if report.get("comparison_plan_sha256") != comparison_plan.get("plan_sha256"):
        raise ValueError(f"{name} report uses another comparison plan")
    expected_run = {
        "run_id": run["run_id"],
        "spec_sha256": run["spec_sha256"],
        "generation": manifest["current"]["generation"],
        "checkpoint_sha256": manifest["current"]["sha256"],
    }
    if report.get("run") != expected_run:
        raise ValueError(f"{name} report uses another completed run")


def _semantic_target_metrics(rows: list[dict], source: str, temperature: float) -> dict:
    ontology = TARGET_ONTOLOGIES[source]
    confusion = {truth: {predicted: 0 for predicted in ontology} for truth in ontology}
    class_counts = Counter()
    predicted_counts = Counter()
    nll = brier = accuracy = top_margin = true_margin = 0.0
    offensive_scores = []
    for row in rows:
        probabilities = _softmax(row["logits"], temperature)
        keys = [str(value) for value in row["option_keys"]]
        truth = keys[row["label"]]
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        predicted = keys[predicted_index]
        if truth not in ontology or predicted not in ontology:
            raise ValueError(f"{source} prediction contains an unknown semantic label")
        confusion[truth][predicted] += 1
        class_counts[truth] += 1
        predicted_counts[predicted] += 1
        values = _prediction_metrics(row, temperature)
        nll += values["nll"]
        brier += values["brier_histogram"]
        accuracy += values["accuracy"]
        ordered = sorted(probabilities, reverse=True)
        top_margin += ordered[0] - ordered[1]
        true_probability = probabilities[row["label"]]
        true_margin += true_probability - max(
            value for index, value in enumerate(probabilities) if index != row["label"]
        )
        if source == "tweet_offensive":
            positive = probabilities[keys.index("true")]
            offensive_scores.append((positive, truth == "true"))
    f1_values = []
    for label in ontology:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[truth][label] for truth in ontology if truth != label)
        false_negative = sum(confusion[label][predicted] for predicted in ontology if predicted != label)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    result = {
        "n": len(rows),
        "temperature": temperature,
        "accuracy": accuracy / len(rows),
        "fixed_ontology_macro_f1": sum(f1_values) / len(f1_values),
        "nll": nll / len(rows),
        "brier": brier / len(rows),
        "mean_top_margin": top_margin / len(rows),
        "mean_true_margin": true_margin / len(rows),
        "class_frequency": {label: class_counts[label] for label in ontology},
        "predicted_class_frequency": {label: predicted_counts[label] for label in ontology},
        "confusion": confusion,
    }
    if offensive_scores:
        positives = [score for score, label in offensive_scores if label]
        negatives = [score for score, label in offensive_scores if not label]
        if not positives or not negatives:
            raise ValueError("offensive AUROC requires both labels")
        wins = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives for negative in negatives
        )
        result["auroc"] = wins / (len(positives) * len(negatives))
    return result


def _paired_bootstrap(
    baseline: list[dict],
    shared: list[dict],
    samples: int,
    seed: int,
) -> dict:
    baseline_index = {_prediction_key(row): row for row in baseline}
    shared_index = {_prediction_key(row): row for row in shared}
    if set(baseline_index) != set(shared_index):
        raise ValueError("target cached prediction membership differs")
    parent_values = defaultdict(lambda: defaultdict(list))
    for key in sorted(baseline_index):
        baseline_row = baseline_index[key]
        shared_row = shared_index[key]
        if any(
            baseline_row[field] != shared_row[field]
            for field in ("group_id", "label", "option_keys", "options", "type")
        ):
            raise ValueError("paired target prediction metadata differs")
        parent = baseline_row.get("group_id") or baseline_row["request_id"]
        baseline_values = _prediction_metrics(baseline_row, 1.0)
        shared_values = _prediction_metrics(shared_row, 1.0)
        for metric in ("accuracy", "nll", "brier_histogram"):
            parent_values[parent][metric].append(shared_values[metric] - baseline_values[metric])
    parents = sorted(parent_values)
    if len(parents) < 2:
        raise ValueError("paired bootstrap needs at least two parents")
    result = {}
    for offset, metric in enumerate(("accuracy", "nll", "brier_histogram")):
        generator = random.Random(seed + offset)
        estimates = []
        for _ in range(samples):
            selected = generator.choices(parents, k=len(parents))
            values = [
                value
                for parent in selected
                for value in parent_values[parent][metric]
            ]
            estimates.append(sum(values) / len(values))
        estimates.sort()
        observed = [value for parent in parents for value in parent_values[parent][metric]]
        result[metric] = {
            "shared_minus_baseline": sum(observed) / len(observed),
            "interval_95": {
                "lower": estimates[math.ceil(0.025 * samples) - 1],
                "upper": estimates[math.ceil(0.975 * samples) - 1],
            },
            "draws": samples,
            "unit": "source-namespaced request parent",
        }
    return result


def verify_cached_reports(
    reports: dict[str, dict],
    partitions: dict[str, list[dict]],
    suite_manifests: dict[str, dict],
    runs: dict[str, dict],
    run_manifests: dict[str, dict],
    comparison_plan: dict,
    *,
    bootstrap_draws: int = 20000,
    bootstrap_seed: int = 20260921,
) -> dict:
    """Reproduce cached primary metrics and target diagnostics from stored logits."""
    if set(reports) != {"baseline", "shared"}:
        raise ValueError("baseline and shared cached reports are required")
    population_map = {
        "calibration": "calibration",
        "decision_development": "decision_development",
        "transfer_development": "transfer_development",
        "korean_development": "korean_development",
    }
    inventories = {
        prediction_role: _inventory(partitions[partition_role])
        for prediction_role, partition_role in population_map.items()
    }
    reproduced = {}
    maximum_delta = 0.0
    for model_name, report in reports.items():
        _validate_report_identity(
            model_name,
            report,
            runs[model_name],
            run_manifests[model_name],
            comparison_plan,
        )
        temperature = float(report["temperature"]["value"])
        model_result = {"temperature": temperature, "partitions": {}}
        for prediction_role, partition_role in population_map.items():
            rows = report["predictions"].get(prediction_role)
            if not isinstance(rows, list):
                raise ValueError(f"{model_name} report is missing {prediction_role}")
            _validate_population(rows, inventories[prediction_role], f"{model_name} {prediction_role}")
            if prediction_role == "calibration":
                continue
            suite_key = prediction_role.removesuffix("_development")
            suite_key = "decision" if suite_key == "decision" else suite_key
            manifest_key = "decision" if prediction_role == "decision_development" else suite_key
            section = report[prediction_role]
            filename = "development.jsonl"
            if section.get("split_sha256") != suite_manifests[manifest_key]["files"][filename]["sha256"]:
                raise ValueError(f"{model_name} {prediction_role} split binding is stale")
            partition_result = {}
            for scale_name, scale in (("raw", 1.0), ("scaled", temperature)):
                actual = _reproduce_grouped(rows, scale)
                expected = section[scale_name]
                delta = _compare_metrics(actual["all"], expected["all"], f"{model_name} {prediction_role} {scale_name} all")
                maximum_delta = max(maximum_delta, delta)
                delta = _compare_metrics(actual["source_macro"], expected["source_macro"], f"{model_name} {prediction_role} {scale_name} source macro")
                maximum_delta = max(maximum_delta, delta)
                for source, metrics in actual["source"].items():
                    delta = _compare_metrics(metrics, expected["source"][source], f"{model_name} {prediction_role} {scale_name} {source}")
                    maximum_delta = max(maximum_delta, delta)
                partition_result[scale_name] = {
                    "all": actual["all"],
                    "source_macro": actual["source_macro"],
                }
            exclusions = comparison_plan["excluded_development_requests"][manifest_key]
            clean = _clean_rows(rows, exclusions)
            for scale_name, scale in (("raw", 1.0), ("scaled", temperature)):
                actual = _reproduce_grouped(clean, scale)
                expected = section["common_clean"][scale_name]
                delta = _compare_metrics(actual["all"], expected["all"], f"{model_name} common-clean {prediction_role} {scale_name}")
                maximum_delta = max(maximum_delta, delta)
            partition_result["questions"] = len(rows)
            model_result["partitions"][prediction_role] = partition_result
        max_length = int(runs[model_name]["spec"]["config"]["max_len"])
        token_field = "packed_request_tokens" if model_name == "shared" else "packed_question_tokens"
        token_values = [
            int(row[token_field])
            for rows in report["predictions"].values()
            for row in rows
        ]
        if max(token_values) > max_length:
            raise ValueError(f"{model_name} cached input exceeds its bound context")
        model_result["input_budget"] = {
            "field": token_field,
            "maximum_observed": max(token_values),
            "maximum_allowed": max_length,
            "overflow": 0,
        }
        reproduced[model_name] = model_result
    targets = {}
    for source in TARGET_ONTOLOGIES:
        baseline_rows = [
            row for row in reports["baseline"]["predictions"]["transfer_development"]
            if row["source"] == source
        ]
        shared_rows = [
            row for row in reports["shared"]["predictions"]["transfer_development"]
            if row["source"] == source
        ]
        if not baseline_rows or not shared_rows:
            raise ValueError(f"cached target source {source!r} is absent")
        targets[source] = {
            "baseline": {
                "raw": _semantic_target_metrics(baseline_rows, source, 1.0),
                "scaled": _semantic_target_metrics(
                    baseline_rows,
                    source,
                    float(reports["baseline"]["temperature"]["value"]),
                ),
            },
            "shared": {
                "raw": _semantic_target_metrics(shared_rows, source, 1.0),
                "scaled": _semantic_target_metrics(
                    shared_rows,
                    source,
                    float(reports["shared"]["temperature"]["value"]),
                ),
            },
            "paired_raw": _paired_bootstrap(
                baseline_rows,
                shared_rows,
                bootstrap_draws,
                bootstrap_seed,
            ),
        }
    return {
        "status": "passed",
        "metric_tolerance": 1e-10,
        "maximum_absolute_reproduction_delta": maximum_delta,
        "reports": reproduced,
        "targets": targets,
        "model_forwards": 0,
        "optimizer_updates": 0,
    }


def audit_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_manifest_sha256 = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected_manifest_sha256 != canonical_sha256(unsigned):
        raise ValueError("coverage manifest self-digest differs")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("coverage manifest is incomplete")
    if manifest.get("locked_test_opened") is not False:
        raise ValueError("coverage manifest did not preserve the test lock")
    if manifest.get("model_forwards") != 0 or manifest.get("optimizer_updates") != 0:
        raise ValueError("coverage manifest exceeded the C0 compute boundary")
    for filename, descriptor in manifest.get("outputs", {}).items():
        output = path.parent / filename
        data = output.read_bytes()
        if len(data) != descriptor["size_bytes"] or hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
            raise ValueError(f"coverage output changed: {filename}")
    return {
        "status": "verified",
        "manifest_sha256": expected_manifest_sha256,
        "manifest_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "outputs": sorted(manifest["outputs"]),
        "model_forwards": 0,
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_manifest(args.plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
