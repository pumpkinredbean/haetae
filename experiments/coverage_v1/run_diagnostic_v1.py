"""Run the reviewed T04 fixed-weight diagnostic without updating model weights."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile

import numpy as np
from safetensors.torch import save_file
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer

from experiments.coverage_v1.diagnostic_v1 import (
    BOOTSTRAP_DRAWS,
    NATIVE_LABELS,
    ORIGINAL_MODEL_FILES,
    SEED,
    SHARED_TOKENIZER_FILES,
    _load_local_paths,
    _verify_descriptor,
    diagnostic_code_identity,
    verify_frozen,
)
from experiments.coverage_v1.registry import (
    canonical_json_bytes,
    canonical_sha256,
    load_json_bytes,
)
from experiments.shared_state import (
    PackedRequest,
    branch_attention_masks,
    collate_requests,
    encode_request,
    load_shared_state_model,
)
from haetae.checkpoint import (
    load_completed_checkpoint,
    model_config_fingerprint,
    tokenizer_fingerprint,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        load_json_bytes(line, f"{path.name} line {number}")
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if line
    ]


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise ValueError(f"requested device is unavailable: {name}")


def _runtime_identity(device: torch.device) -> dict:
    def sysctl(name: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "device": device.type,
        "parameter_dtype": "float32",
        "attention_implementation": "eager",
        "hardware": {
            "model_identifier": sysctl("hw.model"),
            "chip": sysctl("machdep.cpu.brand_string"),
            "memory_gib": int(sysctl("hw.memsize")) // (1024 ** 3),
        },
        "software": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
            "numpy": importlib.metadata.version("numpy"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
    }


def _load_models(local: dict, device: torch.device, protocol: dict):
    original_dir = Path(local["original_model_directory"]).resolve(strict=True)
    shared_dir = Path(local["shared_run_directory"]).resolve(strict=True)
    for name, descriptor in ORIGINAL_MODEL_FILES.items():
        _verify_descriptor(original_dir / name, descriptor, f"original model {name}")
    for name, descriptor in SHARED_TOKENIZER_FILES.items():
        _verify_descriptor(shared_dir / name, descriptor, f"shared tokenizer {name}")
    original_tokenizer = AutoTokenizer.from_pretrained(
        original_dir, local_files_only=True, trust_remote_code=False,
    )
    original_binding = protocol["models"]["original_pretrained"]
    original = AutoModel.from_pretrained(
        original_dir,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(device).eval()
    original.config._name_or_path = original_binding["model"]
    if (
        tokenizer_fingerprint(original_tokenizer)
        != original_binding["tokenizer_fingerprint"]
        or hashlib.sha256(canonical_json_bytes(original.config.to_dict())).hexdigest()
        != original_binding["model_config_fingerprint"]
    ):
        raise ValueError("original model runtime identity differs")

    payload, run, manifest = load_completed_checkpoint(shared_dir)
    shared_tokenizer = AutoTokenizer.from_pretrained(
        shared_dir, local_files_only=True, trust_remote_code=False,
    )
    config = run["spec"]["config"]
    experiment = run["spec"]["experiment"]
    shared, delimiters = load_shared_state_model(
        config["backbone"],
        shared_tokenizer,
        str(device),
        revision=experiment["backbone_revision"],
        pointer_size=experiment["pointer_size"],
        attention_implementation=experiment["attention_implementation"],
        local_files_only=True,
    )
    if (
        tokenizer_fingerprint(shared_tokenizer) != run["spec"]["tokenizer_fingerprint"]
        or model_config_fingerprint(shared) != run["spec"]["model_config_fingerprint"]
        or manifest["current"]["generation"] != 79
        or manifest["current"]["sha256"]
        != protocol["models"]["shared_generation_79"]["checkpoint"]["sha256"]
    ):
        raise ValueError("shared model runtime identity differs")
    shared.backbone.load_state_dict(payload["backbone"], strict=True)
    shared.head.load_state_dict(payload["head"], strict=True)
    shared.eval()
    return original, original_tokenizer, shared, shared_tokenizer, delimiters


def _state_encoding(tokenizer, state: str, maximum_length: int) -> PackedRequest:
    token_ids = tokenizer(state, add_special_tokens=False)["input_ids"]
    input_ids = [tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id]
    if len(input_ids) > maximum_length:
        raise ValueError("diagnostic state exceeds the fixed feature budget")
    return PackedRequest(
        input_ids=input_ids,
        segments=[0] * len(input_ids),
        position_ids=list(range(len(input_ids))),
        decision_indices=[],
        option_indices=[],
        labels=[],
        state_tokens=len(token_ids),
        retained_state_tokens=len(token_ids),
    )


def extract_features(
    backbone,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    microbatch: int,
    maximum_length: int,
) -> torch.Tensor:
    features = []
    with torch.inference_mode():
        for start in range(0, len(rows), microbatch):
            batch_rows = rows[start:start + microbatch]
            encodings = [
                _state_encoding(tokenizer, row["state"], maximum_length)
                for row in batch_rows
            ]
            input_ids, position_ids = collate_requests(
                encodings, tokenizer.pad_token_id, device,
            )
            dtype = next(backbone.parameters()).dtype
            masks = branch_attention_masks(
                encodings,
                device,
                dtype,
                int(backbone.config.sliding_window),
            )
            hidden = backbone(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=masks,
            ).last_hidden_state
            for index, encoding in enumerate(encodings):
                if encoding.state_tokens <= 0:
                    raise ValueError("diagnostic state has no content tokens")
                features.append(
                    hidden[index, 1:1 + encoding.state_tokens]
                    .mean(dim=0).detach().to(device="cpu", dtype=torch.float32)
                )
    return torch.stack(features)


def infer_semantic_logits(
    model,
    tokenizer,
    delimiters: dict[str, int],
    rows: list[dict],
    maximum_length: int,
    microbatch: int,
) -> list[dict]:
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), microbatch):
            batch_rows = rows[start:start + microbatch]
            encodings = [
                encode_request(
                    tokenizer,
                    row["state"],
                    [row["question"]],
                    maximum_length,
                    delimiters,
                )
                for row in batch_rows
            ]
            if any(value is None for value in encodings):
                raise ValueError("semantic variant exceeds the fixed input budget")
            if any(value.retained_state_tokens != value.state_tokens for value in encodings):
                raise ValueError("semantic variant would truncate state content")
            logits = model(encodings)
            for row, request_logits in zip(batch_rows, logits):
                values = request_logits[0].detach().to(device="cpu", dtype=torch.float64)
                if values.ndim != 1 or len(values) != len(row["question"]["options"]):
                    raise ValueError("semantic logits differ from the option inventory")
                output.append({
                    "variant_id": row["variant_id"],
                    "logits": values.tolist(),
                })
    return output


def _softmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probabilities = shifted - log_normalizer
    return np.exp(log_probabilities), log_probabilities


def _metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    log_probabilities: np.ndarray,
) -> dict:
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("metric arrays differ")
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    result = {
        "n": len(labels),
        "accuracy": float(np.mean(predicted == labels)),
        "macro_f1": float(f1_score(
            labels,
            predicted,
            labels=list(range(probabilities.shape[1])),
            average="macro",
            zero_division=0,
        )),
        "nll": float(-np.mean(log_probabilities[np.arange(len(labels)), labels])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }
    if probabilities.shape[1] == 2:
        result["auroc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    return result


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature):
        _, log_probabilities = _softmax(
            logits / math.exp(float(log_temperature))
        )
        return -float(np.mean(
            log_probabilities[np.arange(len(labels)), labels]
        ))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-5.0, 5.0),
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("emotion temperature fitting failed")
    return math.exp(float(result.x))


def fit_positive_logistic(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.float64)

    def objective(parameters):
        slope = math.exp(float(parameters[0]))
        values = slope * scores + float(parameters[1])
        return float(np.mean(np.logaddexp(0.0, values) - labels * values))

    result = minimize(
        objective,
        np.array([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=((-5.0, 5.0), (-20.0, 20.0)),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("offensive logistic calibration failed")
    return math.exp(float(result.x[0])), float(result.x[1])


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def paired_parent_bootstrap(
    rows: list[dict],
    first_correct: np.ndarray,
    second_correct: np.ndarray,
    seed: int,
) -> dict:
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, first, second in zip(rows, first_correct, second_correct):
        clusters[(row["source"], row["parent_id"])].append(float(first - second))
    identities = sorted(clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [identities[rng.randrange(len(identities))] for _ in identities]
        values = [value for identity in sampled for value in clusters[identity]]
        draws.append(sum(values) / len(values))
    point = float(np.mean(first_correct - second_correct))
    return {
        "difference": point,
        "parents": len(identities),
        "draws": BOOTSTRAP_DRAWS,
        "one_sided_97_5_lower": _nearest_rank(draws, 0.025),
        "ci95": [_nearest_rank(draws, 0.025), _nearest_rank(draws, 0.975)],
    }


def probe_results(
    feature_rows: list[dict],
    features: dict[str, np.ndarray],
    support_fit: list[dict],
    support_calibration: list[dict],
    development: list[dict],
) -> dict:
    indexes = {row["state_sha256"]: index for index, row in enumerate(feature_rows)}
    results = {}
    correctness = {}
    for target, development_population in (
        ("emotion", "native_six_class"),
        ("tweet_offensive", "native_binary"),
    ):
        labels = list(NATIVE_LABELS[target])
        label_index = {label: index for index, label in enumerate(labels)}
        fit_rows = [row for row in support_fit if row["source"] == target]
        calibration_rows = [row for row in support_calibration if row["source"] == target]
        development_rows = [
            row for row in development
            if row["target"] == target and row["population"] == development_population
        ]
        target_results = {}
        for backbone in ("original", "shared"):
            matrix = features[backbone]
            fit_x = np.stack([matrix[indexes[row["state_sha256"]]] for row in fit_rows])
            fit_y = np.array([label_index[row["label"]] for row in fit_rows])
            calibration_x = np.stack([
                matrix[indexes[row["state_sha256"]]] for row in calibration_rows
            ])
            calibration_y = np.array([
                label_index[row["label"]] for row in calibration_rows
            ])
            development_x = np.stack([
                matrix[indexes[row["state_sha256"]]] for row in development_rows
            ])
            development_y = np.array([
                label_index[row["semantic_label"]] for row in development_rows
            ])
            standardizer = StandardScaler(with_mean=True, with_std=True)
            fit_x = standardizer.fit_transform(fit_x)
            calibration_x = standardizer.transform(calibration_x)
            development_x = standardizer.transform(development_x)
            classifier = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                fit_intercept=True,
                class_weight=None,
                random_state=SEED,
            ).fit(fit_x, fit_y)
            if list(classifier.classes_) != list(range(len(labels))):
                raise ValueError("probe classifier class order differs")
            if target == "emotion":
                calibration_logits = classifier.decision_function(calibration_x)
                development_logits = classifier.decision_function(development_x)
                raw, raw_log = _softmax(development_logits)
                temperature = fit_temperature(calibration_logits, calibration_y)
                calibrated, calibrated_log = _softmax(
                    development_logits / temperature
                )
                calibration = {"temperature": temperature}
            else:
                calibration_scores = classifier.decision_function(calibration_x)
                development_scores = classifier.decision_function(development_x)
                raw_positive = np.exp(-np.logaddexp(0.0, -development_scores))
                raw = np.column_stack((1.0 - raw_positive, raw_positive))
                raw_log = np.column_stack((
                    -np.logaddexp(0.0, development_scores),
                    -np.logaddexp(0.0, -development_scores),
                ))
                slope, intercept = fit_positive_logistic(
                    calibration_scores, calibration_y,
                )
                values = slope * development_scores + intercept
                calibrated_positive = np.exp(-np.logaddexp(0.0, -values))
                calibrated = np.column_stack((
                    1.0 - calibrated_positive, calibrated_positive,
                ))
                calibrated_log = np.column_stack((
                    -np.logaddexp(0.0, values),
                    -np.logaddexp(0.0, -values),
                ))
                calibration = {"slope": slope, "intercept": intercept}
            target_results[backbone] = {
                "fit_rows": len(fit_rows),
                "calibration_rows": len(calibration_rows),
                "development_rows": len(development_rows),
                "calibration": calibration,
                "raw": _metrics(raw, development_y, raw_log),
                "calibrated": _metrics(
                    calibrated, development_y, calibrated_log,
                ),
            }
            correctness[(target, backbone)] = (
                calibrated.argmax(axis=1) == development_y
            ).astype(np.float64)
        uncertainty = paired_parent_bootstrap(
            development_rows,
            correctness[(target, "original")],
            correctness[(target, "shared")],
            SEED + len(results),
        )
        target_results["original_minus_shared_calibrated_accuracy"] = uncertainty
        target_results["accessible_feature_regression"] = bool(
            uncertainty["difference"] >= 0.05
            and uncertainty["one_sided_97_5_lower"] > 0.0
        )
        results[target] = target_results
    return results


def semantic_results(variants: list[dict], logits_rows: list[dict]) -> dict:
    logits_by_id = {row["variant_id"]: np.asarray(row["logits"], dtype=np.float64) for row in logits_rows}
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in variants:
        grouped[(row["target"], row["population"], row["family"])].append(row)
    original_actions = {}
    original_probabilities = {}
    for row in variants:
        if row["family"] != "original":
            continue
        probabilities = _softmax(
            logits_by_id[row["variant_id"]][None, :]
        )[0][0]
        keys = row["question"]["option_keys"]
        original_actions[row["development_id"]] = keys[int(probabilities.argmax())]
        original_probabilities[row["development_id"]] = dict(zip(keys, probabilities))

    result = {}
    for (target, population, family), rows in sorted(grouped.items()):
        probabilities = []
        log_probabilities = []
        labels = []
        stable = []
        total_variations = []
        for row in rows:
            values, log_values = _softmax(
                logits_by_id[row["variant_id"]][None, :]
            )
            values = values[0]
            log_values = log_values[0]
            keys = row["question"]["option_keys"]
            probabilities.append(values)
            log_probabilities.append(log_values)
            labels.append(int(row["question"]["label"]))
            action = keys[int(values.argmax())]
            stable.append(action == original_actions[row["development_id"]])
            baseline = original_probabilities[row["development_id"]]
            current = dict(zip(keys, values))
            semantic_keys = sorted(set(baseline) | set(current))
            total_variations.append(0.5 * sum(
                abs(float(baseline.get(key, 0.0)) - float(current.get(key, 0.0)))
                for key in semantic_keys
            ))
        metrics = _metrics(
            np.stack(probabilities),
            np.asarray(labels),
            np.stack(log_probabilities),
        )
        metrics.pop("macro_f1")
        metrics["semantic_action_stability"] = float(np.mean(stable))
        metrics["mean_total_variation"] = float(np.mean(total_variations))
        result.setdefault(target, {}).setdefault(population, {})[family] = metrics
    return result


def diagnostic_decisions(probes: dict, semantics: dict, protocol: dict) -> dict:
    populations = {
        "emotion": "native_six_class",
        "tweet_offensive": "native_binary",
    }
    decisions = {}
    for target, population in populations.items():
        gap = protocol["metrics"]["historical_accuracy_gaps"][target]
        historical = protocol["metrics"]["historical_accuracy_baselines"][
            "targets"
        ][target]
        shared_probe = probes[target]["shared"]
        probe_recovery = (
            shared_probe["calibrated"]["accuracy"]
            - historical["shared"]["accuracy"]
        )
        original_semantic = semantics[target][population]["original"]["accuracy"]
        semantic_recoveries = {
            family: values["accuracy"] - original_semantic
            for family, values in semantics[target][population].items()
            if family != "original"
        }
        raw = shared_probe["raw"]
        calibrated = shared_probe["calibrated"]
        calibration_improved = (
            calibrated["nll"] < raw["nll"]
            or calibrated["multiclass_brier"] < raw["multiclass_brier"]
        )
        ranking_unchanged = (
            "auroc" not in raw
            or math.isclose(
                raw["auroc"], calibrated["auroc"], rel_tol=0.0, abs_tol=1e-12,
            )
        )
        decisions[target] = {
            "historical_accuracy_gap": gap,
            "half_gap": gap / 2.0,
            "shared_probe_accuracy_recovery": probe_recovery,
            "shared_probe_recovers_half_gap": probe_recovery >= gap / 2.0,
            "semantic_accuracy_recovery_by_family": semantic_recoveries,
            "semantic_family_recovers_half_gap": {
                family: recovery >= gap / 2.0
                for family, recovery in semantic_recoveries.items()
            },
            "accessible_feature_regression": probes[target][
                "accessible_feature_regression"
            ],
            "calibration_only_pattern": calibration_improved and ranking_unchanged,
        }
    return decisions


def _write_json(path: Path, value: dict) -> dict:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    data = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "rows": len(rows)}


def require_reviewed_identity(
    manifest: dict,
    protocol: dict,
    reviewed_manifest_sha256: str,
    reviewed_protocol_sha256: str,
) -> None:
    if (
        manifest.get("manifest_sha256") != reviewed_manifest_sha256
        or protocol.get("protocol_sha256") != reviewed_protocol_sha256
    ):
        raise ValueError("diagnostic plan differs from the reviewed identity")


def run(
    plan: Path,
    evidence_paths: Path,
    local_paths: Path,
    output: Path,
    device_name: str,
    registry_path: Path,
    reviewed_manifest_sha256: str,
    reviewed_protocol_sha256: str,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = verify_frozen(
        plan,
        evidence_paths=evidence_paths,
        local_paths=local_paths,
        registry_path=registry_path,
    )
    protocol = load_json_bytes(
        (plan / "protocol.json").read_bytes(), "diagnostic protocol",
    )
    require_reviewed_identity(
        manifest,
        protocol,
        reviewed_manifest_sha256,
        reviewed_protocol_sha256,
    )
    repository = Path(__file__).resolve().parents[2]
    if protocol["code"] != diagnostic_code_identity(repository):
        raise ValueError("diagnostic code differs from the frozen protocol")
    local = _load_local_paths(local_paths)
    device = _resolve_device(device_name)
    if _runtime_identity(device) != protocol.get("runtime"):
        raise ValueError("diagnostic runtime differs from the frozen identity")
    support_fit = _read_jsonl(plan / "support-fit.jsonl")
    support_calibration = _read_jsonl(plan / "support-calibration.jsonl")
    development = _read_jsonl(plan / "development.jsonl")
    variants = _read_jsonl(plan / "semantic-variants.jsonl")
    unique_feature_rows = {}
    for row in [*support_fit, *support_calibration, *development]:
        unique_feature_rows.setdefault(row["state_sha256"], {
            "state_sha256": row["state_sha256"], "state": row["state"],
        })
    feature_rows = [unique_feature_rows[key] for key in sorted(unique_feature_rows)]
    expected = protocol["compute"]
    if (
        len(feature_rows) != expected["original_backbone_feature_sequences"]
        or len(feature_rows) != expected["shared_backbone_feature_sequences"]
        or len(variants) != expected["shared_pointer_semantic_sequences"]
    ):
        raise ValueError("runtime sequence inventory differs from the frozen budget")

    original, original_tokenizer, shared, shared_tokenizer, delimiters = _load_models(
        local, device, protocol,
    )
    original_features = extract_features(
        original,
        original_tokenizer,
        feature_rows,
        device,
        expected["microbatch"],
        protocol["feature_rendering"]["maximum_length"],
    )
    shared_features = extract_features(
        shared.backbone,
        original_tokenizer,
        feature_rows,
        device,
        expected["microbatch"],
        protocol["feature_rendering"]["maximum_length"],
    )
    semantic_logits = infer_semantic_logits(
        shared,
        shared_tokenizer,
        delimiters,
        variants,
        protocol["models"]["shared_generation_79"]["maximum_length"],
        expected["microbatch"],
    )
    features = {
        "original": original_features.numpy().astype(np.float64),
        "shared": shared_features.numpy().astype(np.float64),
    }
    probes = probe_results(
        feature_rows,
        features,
        support_fit,
        support_calibration,
        development,
    )
    semantics = semantic_results(variants, semantic_logits)
    decisions = diagnostic_decisions(probes, semantics, protocol)

    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "plan_manifest_sha256": manifest["manifest_sha256"],
        "device": device.type,
        "sequence_counts": {
            "original_backbone_features": len(feature_rows),
            "shared_backbone_features": len(feature_rows),
            "shared_pointer_semantics": len(variants),
            "aggregate": 2 * len(feature_rows) + len(variants),
        },
        "probe_results": probes,
        "semantic_results": semantics,
        "diagnostic_decisions": decisions,
        "encoded_sequences": 2 * len(feature_rows) + len(variants),
        "model_forward_calls": (
            math.ceil(len(feature_rows) / expected["microbatch"]) * 2
            + math.ceil(len(variants) / expected["microbatch"])
        ),
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }
    result["result_sha256"] = canonical_sha256(result)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        save_file(
            {"original": original_features.contiguous(), "shared": shared_features.contiguous()},
            temporary / "features.safetensors",
            metadata={"protocol_sha256": protocol["protocol_sha256"]},
        )
        files = {
            "feature-rows.jsonl": _write_jsonl(temporary / "feature-rows.jsonl", feature_rows),
            "features.safetensors": {
                "sha256": hashlib.sha256((temporary / "features.safetensors").read_bytes()).hexdigest(),
                "size_bytes": (temporary / "features.safetensors").stat().st_size,
            },
            "semantic-logits.jsonl": _write_jsonl(temporary / "semantic-logits.jsonl", semantic_logits),
            "result.json": _write_json(temporary / "result.json", result),
        }
        output_manifest = {
            "schema_version": 1,
            "status": "complete",
            "protocol_sha256": protocol["protocol_sha256"],
            "plan_manifest_sha256": manifest["manifest_sha256"],
            "files": files,
            "encoded_sequences": result["encoded_sequences"],
            "model_forward_calls": result["model_forward_calls"],
            "optimizer_updates": 0,
            "locked_test_opened": False,
        }
        output_manifest["manifest_sha256"] = canonical_sha256(output_manifest)
        _write_json(temporary / "manifest.json", output_manifest)
        output.mkdir()
        try:
            for name in files:
                os.rename(temporary / name, output / name)
            os.rename(temporary / "manifest.json", output / "manifest.json")
        except BaseException:
            shutil.rmtree(output, ignore_errors=True)
            raise
        temporary.rmdir()
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--registry", default=Path("research/evidence/index.json"), type=Path)
    parser.add_argument("--reviewed-manifest-sha256", required=True)
    parser.add_argument("--reviewed-protocol-sha256", required=True)
    parser.add_argument("--allow-reviewed-inference", action="store_true")
    args = parser.parse_args()
    if not args.allow_reviewed_inference:
        parser.error("M3-reviewed inference requires --allow-reviewed-inference")
    print(json.dumps(run(
        args.plan,
        args.paths,
        args.local_paths,
        args.out,
        args.device,
        args.registry,
        args.reviewed_manifest_sha256,
        args.reviewed_protocol_sha256,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
