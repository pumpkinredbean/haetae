"""Repair and verify only the held T04 semantic-output evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

from experiments.coverage_v1.diagnostic_v1 import (
    OUTPUT_FILES as ORIGINAL_PLAN_FILES,
    SHARED_TOKENIZER_FILES,
    _load_local_paths,
    _verify_descriptor,
    diagnostic_code_identity,
    semantic_variants,
)
from experiments.coverage_v1.registry import (
    ArtifactStore,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_json_bytes,
)
from experiments.coverage_v1.run_diagnostic_v1 import (
    _resolve_device,
    _runtime_identity,
    diagnostic_decisions,
    probe_results,
    semantic_results,
)
from experiments.shared_state import encode_request, load_shared_state_model
from haetae.checkpoint import (
    load_completed_checkpoint,
    model_config_fingerprint,
    tokenizer_fingerprint,
)


AMENDMENT_ID = "t04-semantic-output-adapter-v1"
ORIGINAL_PLAN_MANIFEST_SHA256 = (
    "ea41f5f7a3ba799dc14c24ce3368c38c97684d4c4bf9bafff993dcc6e8a1a663"
)
ORIGINAL_PROTOCOL_SHA256 = (
    "e376a60ebbe52d7a8c0129a78c50d2ca4b3950a225089146c4d1ed6dc83b7201"
)
HELD_OUTPUT_MANIFEST_SHA256 = (
    "ed76876fd9ed4aa6ff9824a1c73186ebb9f2113a5cc282cc4598c3abdfd62d23"
)
HELD_RESULT_SHA256 = (
    "861b63a15427e56719e23d9917712de819cad0b721a0685f3c31d44476e0bc23"
)
HELD_REVIEW_ARCHIVE_SHA256 = (
    "4199afc3538898a0b1b71a6f801557b24ab3b04d584794970cc926b8f0bd71fb"
)
HELD_PRODUCER_COMMIT = "399509c3b02ec55859c37377881dec4595006575"
COPY_PRODUCER_COMMIT = "e50e0b209782368f7028372abb44c9fc50290aee"
COPY_RESULT_DESCRIPTOR = {
    "sha256": "caa7028ef112402c252f7896d3053c83ff64b92fa6e883cd42bcf8324e344b8a",
    "size_bytes": 15_401,
}
COPY_RESULT_SELF_SHA256 = (
    "5fac32c20a2a5c737786bfecf2929b3cbb77df1261011372c64bcb8d90f56c3d"
)
CACHED_REPORT_DESCRIPTOR = {
    "evidence_id": "shared_development_report",
    "sha256": "960349119103216be9b0945565e578ca1a5303afd70b0338ef2c64b92e4afd23",
    "size_bytes": 11_043_015,
}
CACHED_REPORT_SELF_SHA256 = (
    "8dbe775cd09539b9d366047aa886717a9344dd2a271aa65dd030663b0f8a1d97"
)
CHECKPOINT_SHA256 = (
    "e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c"
)
SHARED_TEMPERATURE = 1.5197255188671874
GENERATION = 79
EXPECTED_VARIANTS = 1_100
EXPECTED_ORIGINALS = 196
EXPECTED_FEATURE_ROWS = 2_720
EXPECTED_FEATURE_SHAPE = (2_720, 384)
MICROBATCH = 2
MAXIMUM_FORWARD_CALLS = 550
MAXIMUM_LENGTH = 2_048
RAW_ATOL = 1e-4
RAW_RTOL = 1e-5
MAX_LOG_PROBABILITY_ERROR = 1e-3
MAX_TOTAL_VARIATION = 1e-4
P99_TOTAL_VARIATION = 1e-5
MATERIAL_MARGIN = 2e-4
PROBE_SCORE_ATOL = 1e-7
PROBE_CALIBRATOR_ATOL = 1e-6
OUTPUT_FILES = frozenset({
    "reuse.json",
    "semantic-logits.jsonl",
    "original-parity.json",
    "result.json",
    "execution-receipt.json",
    "manifest.json",
})
HELD_OUTPUT_FILES = frozenset({
    "feature-rows.jsonl",
    "features.safetensors",
    "semantic-logits.jsonl",
    "result.json",
    "manifest.json",
})
CODE_FILES = (
    "experiments/coverage_v1/semantic_output_amendment_v1.py",
    "experiments/coverage_v1/diagnostic_v1.py",
    "experiments/coverage_v1/registry.py",
    "experiments/coverage_v1/run_diagnostic_v1.py",
    "experiments/shared_state.py",
    "src/haetae/checkpoint.py",
)


def _descriptor(path: Path) -> dict:
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _descriptor_with_rows(path: Path) -> dict:
    result = _descriptor(path)
    result["rows"] = sum(bool(line) for line in path.read_bytes().splitlines())
    return result


def _load_self_digested(path: Path, label: str, field: str) -> dict:
    value = load_json_bytes(path.read_bytes(), label)
    expected = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if expected != canonical_sha256(unsigned):
        raise ValueError(f"{label} self-digest differs")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if line:
            rows.append(load_json_bytes(line, f"{label} line {number}"))
    return rows


def _write_json(path: Path, value: dict) -> dict:
    data = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    data = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(data)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "rows": len(rows),
    }


def _append_jsonl(handle, row: dict) -> None:
    handle.write(canonical_json_bytes(row) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def _json_safe_values(values: torch.Tensor) -> list[float | str]:
    encoded: list[float | str] = []
    for value in values.tolist():
        number = float(value)
        if math.isnan(number):
            encoded.append("NaN")
        elif number == float("inf"):
            encoded.append("+Infinity")
        elif number == float("-inf"):
            encoded.append("-Infinity")
        else:
            encoded.append(number)
    return encoded


def amendment_code_identity(repository: Path) -> dict:
    files = {name: file_sha256(repository / name) for name in CODE_FILES}
    digest = hashlib.sha256()
    for name, sha256 in files.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": files}


def _safe_file(directory: Path, name: str, allowed: set[str] | frozenset[str]) -> Path:
    if name not in allowed:
        raise ValueError(f"file is outside the explicit allowlist: {name}")
    if directory.is_symlink():
        raise ValueError("artifact directory cannot be a symlink")
    root = directory.resolve(strict=True)
    path = root / name
    if path.is_symlink():
        raise ValueError(f"artifact file cannot be a symlink: {name}")
    resolved = path.resolve(strict=True)
    if resolved.parent != root or not resolved.is_file():
        raise ValueError(f"artifact file escapes its directory: {name}")
    return resolved


def _verify_files(directory: Path, descriptors: dict[str, dict], allowed) -> None:
    for name, expected in descriptors.items():
        path = _safe_file(directory, name, allowed)
        actual = _descriptor_with_rows(path) if name.endswith(".jsonl") else _descriptor(path)
        if actual != expected:
            raise ValueError(f"artifact descriptor differs: {name}")


def validate_original_plan(plan: Path) -> dict:
    root = plan.resolve(strict=True)
    expected_names = set(ORIGINAL_PLAN_FILES) | {"manifest.json"}
    if root.is_symlink() or {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("original plan inventory differs")
    manifest = _load_self_digested(
        _safe_file(root, "manifest.json", expected_names),
        "original plan manifest",
        "manifest_sha256",
    )
    if manifest.get("manifest_sha256") != ORIGINAL_PLAN_MANIFEST_SHA256:
        raise ValueError("original plan identity differs")
    if set(manifest.get("files", {})) != set(ORIGINAL_PLAN_FILES):
        raise ValueError("original plan file allowlist differs")
    _verify_files(root, manifest["files"], set(ORIGINAL_PLAN_FILES))
    protocol = _load_self_digested(
        _safe_file(root, "protocol.json", set(ORIGINAL_PLAN_FILES)),
        "original protocol",
        "protocol_sha256",
    )
    repository = Path(__file__).resolve().parents[2]
    if (
        protocol.get("protocol_sha256") != ORIGINAL_PROTOCOL_SHA256
        or protocol.get("protocol_sha256") != manifest.get("protocol_sha256")
        or protocol.get("code") != diagnostic_code_identity(repository)
        or protocol.get("compute", {}).get("shared_pointer_semantic_sequences")
        != EXPECTED_VARIANTS
        or protocol.get("compute", {}).get("microbatch") != MICROBATCH
    ):
        raise ValueError("original protocol boundary differs")
    support_fit = _read_jsonl(root / "support-fit.jsonl", "support fit")
    support_calibration = _read_jsonl(
        root / "support-calibration.jsonl", "support calibration",
    )
    development = _read_jsonl(root / "development.jsonl", "development")
    variants = _read_jsonl(root / "semantic-variants.jsonl", "semantic variants")
    if variants != semantic_variants(development):
        raise ValueError("semantic variants do not reproduce from development")
    if len(development) != EXPECTED_ORIGINALS or len(variants) != EXPECTED_VARIANTS:
        raise ValueError("original plan row counts differ")
    unique_rows: dict[str, dict] = {}
    for row in [*support_fit, *support_calibration, *development]:
        unique_rows.setdefault(row["state_sha256"], {
            "state_sha256": row["state_sha256"],
            "state": row["state"],
        })
    feature_rows = [unique_rows[key] for key in sorted(unique_rows)]
    if len(feature_rows) != EXPECTED_FEATURE_ROWS:
        raise ValueError("feature-row inventory differs")
    return {
        "manifest": manifest,
        "protocol": protocol,
        "support_fit": support_fit,
        "support_calibration": support_calibration,
        "development": development,
        "variants": variants,
        "feature_rows": feature_rows,
    }


def validate_held_output(held_output: Path, expected_feature_rows: list[dict]) -> dict:
    root = held_output.resolve(strict=True)
    if root.is_symlink() or {path.name for path in root.iterdir()} != set(HELD_OUTPUT_FILES):
        raise ValueError("held output inventory differs")
    manifest = _load_self_digested(
        _safe_file(root, "manifest.json", HELD_OUTPUT_FILES),
        "held output manifest",
        "manifest_sha256",
    )
    if (
        manifest.get("manifest_sha256") != HELD_OUTPUT_MANIFEST_SHA256
        or manifest.get("plan_manifest_sha256") != ORIGINAL_PLAN_MANIFEST_SHA256
        or manifest.get("protocol_sha256") != ORIGINAL_PROTOCOL_SHA256
        or manifest.get("status") != "complete"
        or set(manifest.get("files", {})) != HELD_OUTPUT_FILES - {"manifest.json"}
    ):
        raise ValueError("held output boundary differs")
    _verify_files(root, manifest["files"], HELD_OUTPUT_FILES)
    result = _load_self_digested(
        root / "result.json", "held result", "result_sha256",
    )
    if result.get("result_sha256") != HELD_RESULT_SHA256:
        raise ValueError("held result identity differs")
    feature_rows = _read_jsonl(root / "feature-rows.jsonl", "held feature rows")
    if feature_rows != expected_feature_rows:
        raise ValueError("held feature-row order differs")
    with safe_open(root / "features.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"original", "shared"}:
            raise ValueError("held feature tensor keys differ")
        if handle.metadata() != {"protocol_sha256": ORIGINAL_PROTOCOL_SHA256}:
            raise ValueError("held feature metadata differs")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    for tensor in tensors.values():
        if (
            tuple(tensor.shape) != EXPECTED_FEATURE_SHAPE
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("held feature tensor contract differs")
    return {
        "manifest": manifest,
        "result": result,
        "feature_rows": feature_rows,
        "features": tensors,
    }


def validate_copy_result(path: Path) -> dict:
    if path.is_symlink() or _descriptor(path) != COPY_RESULT_DESCRIPTOR:
        raise ValueError("copy diagnostic file differs")
    result = _load_self_digested(path, "copy diagnostic", "result_sha256")
    cases = result.get("cases")
    expected_inventory = {
        (context, length, offset)
        for context in ("no_grad", "inference_mode")
        for length in (2, 6, 7)
        for offset in (0, 1, 256)
    }
    if (
        result.get("result_sha256") != COPY_RESULT_SELF_SHA256
        or result.get("status") != "failed"
        or result.get("all_passed") is not False
        or not isinstance(cases, list)
        or len(cases) != 18
        or {(row.get("context"), row.get("length"), row.get("offset")) for row in cases}
        != expected_inventory
        or result.get("checkpoint_accessed") is not False
        or result.get("dataset_accessed") is not False
        or result.get("locked_test_opened") is not False
        or result.get("model_forwards") != 0
        or result.get("optimizer_updates") != 0
    ):
        raise ValueError("copy diagnostic boundary differs")
    for row in cases:
        if (
            row.get("fused") != [0.0] * row["length"]
            or row.get("fused_exact") is not False
            or row.get("staged_exact") is not True
            or row.get("source_unchanged") is not True
            or row.get("storage_offset_matches") is not True
            or row.get("staged") != row.get("reference")
            or row.get("source_after") != row.get("reference")
            or row.get("passed") is not False
        ):
            raise ValueError("copy diagnostic observation differs")
    return result


def _validate_number_vector(value: Any, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} candidate count differs")
    if any(type(item) not in (int, float) for item in value):
        raise ValueError(f"{label} contains a non-numeric or Boolean value")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{label} is not finite")
    return result


def validate_cached_report(
    paths: Path,
    registry: Path,
    development: list[dict],
    variants: list[dict],
) -> dict:
    store = ArtifactStore.from_files(registry, paths)
    if store.binding("shared_development_report") != CACHED_REPORT_DESCRIPTOR:
        raise ValueError("cached report registry binding differs")
    report = store.read_json("shared_development_report")
    expected = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if (
        expected != CACHED_REPORT_SELF_SHA256
        or canonical_sha256(unsigned) != expected
        or report.get("run", {}).get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report.get("run", {}).get("generation") != GENERATION
        or report.get("inference") != {"batch": 2, "device": "mps"}
        or report.get("temperature", {}).get("value") != SHARED_TEMPERATURE
        or report.get("locked_test_opened") is not False
    ):
        raise ValueError("cached report identity differs")
    predictions = report.get("predictions", {}).get("transfer_development")
    if not isinstance(predictions, list):
        raise ValueError("cached transfer predictions are missing")
    report_index = {}
    for row in predictions:
        key = (row.get("request_id"), row.get("question_id"))
        if key in report_index:
            raise ValueError("cached report contains duplicate request/question keys")
        report_index[key] = row
    development_index = {}
    for row in development:
        key = (row["development_id"], row["question"]["id"])
        if key in development_index:
            raise ValueError("development contains duplicate request/question keys")
        development_index[key] = row
    originals = [row for row in variants if row.get("family") == "original"]
    if len(originals) != EXPECTED_ORIGINALS:
        raise ValueError("original-family inventory differs")
    references = {}
    for variant in originals:
        key = (variant["development_id"], variant["question"]["id"])
        development_row = development_index.get(key)
        reference = report_index.get(key)
        if development_row is None or reference is None or variant["variant_id"] in references:
            raise ValueError("original-family membership differs")
        question = development_row["question"]
        if (
            variant["state"] != development_row["state"]
            or variant["state_sha256"] != development_row["state_sha256"]
            or variant["parent_id"] != development_row["parent_id"]
            or variant["question"]["instructions"] != question["instructions"]
            or variant["question"]["options"] != question["options"]
            or variant["question"]["option_keys"] != question["option_keys"]
            or variant["question"]["type"] != question["type"]
            or variant["question"]["label"] != question["label"]
            or reference.get("request_id") != development_row["development_id"]
            or reference.get("group_id") != development_row["parent_id"]
            or reference.get("state_sha256") != development_row["state_sha256"]
            or reference.get("question_id") != question["id"]
            or reference.get("source") != question["source"]
            or reference.get("type") != question["type"]
            or reference.get("options") != question["options"]
            or reference.get("option_keys") != question["option_keys"]
            or reference.get("label") != question["label"]
            or reference.get("soft") != question.get("soft")
            or reference.get("questions_in_request") != 1
        ):
            raise ValueError("original-family semantic or input identity differs")
        logits = _validate_number_vector(
            reference.get("logits"), len(question["options"]), "cached logits",
        )
        references[variant["variant_id"]] = {
            "key": list(key),
            "reference_logits": logits.tolist(),
        }
    if len(references) != EXPECTED_ORIGINALS:
        raise ValueError("original-family reference count differs")
    return {"report": report, "references": references}


def _input_bindings(
    plan: Path,
    held_output: Path,
    copy_result: Path,
    paths: Path,
    local_paths: Path,
    registry: Path,
    plan_values: dict,
    held_values: dict,
    report_values: dict,
) -> dict:
    return {
        "original_plan": {
            "manifest_sha256": ORIGINAL_PLAN_MANIFEST_SHA256,
            "protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
            "files": plan_values["manifest"]["files"],
        },
        "held_output": {
            "manifest_sha256": HELD_OUTPUT_MANIFEST_SHA256,
            "result_sha256": HELD_RESULT_SHA256,
            "files": held_values["manifest"]["files"],
        },
        "copy_result": {
            **COPY_RESULT_DESCRIPTOR,
            "result_sha256": COPY_RESULT_SELF_SHA256,
            "producer_commit": COPY_PRODUCER_COMMIT,
        },
        "cached_report": {
            **CACHED_REPORT_DESCRIPTOR,
            "report_sha256": CACHED_REPORT_SELF_SHA256,
        },
        "path_maps": {
            "evidence_paths": _descriptor(paths),
            "local_paths": _descriptor(local_paths),
            "registry": _descriptor(registry),
        },
        "models": plan_values["protocol"]["models"],
        "runtime": plan_values["protocol"]["runtime"],
        "historical_code": plan_values["protocol"]["code"],
        "held_producer_commit": HELD_PRODUCER_COMMIT,
        "held_review_archive_sha256": HELD_REVIEW_ARCHIVE_SHA256,
        "captured_paths": {
            "plan": str(plan.resolve()),
            "held_output": str(held_output.resolve()),
            "copy_result": str(copy_result.resolve()),
        },
        "validated_reference_rows": len(report_values["references"]),
    }


def collect_inputs(
    plan: Path,
    held_output: Path,
    copy_result: Path,
    paths: Path,
    local_paths: Path,
    registry: Path,
) -> dict:
    plan_values = validate_original_plan(plan)
    held_values = validate_held_output(held_output, plan_values["feature_rows"])
    validate_copy_result(copy_result)
    report_values = validate_cached_report(
        paths,
        registry,
        plan_values["development"],
        plan_values["variants"],
    )
    return {
        **plan_values,
        "held": held_values,
        "cached_report": report_values["report"],
        "references": report_values["references"],
        "bindings": _input_bindings(
            plan,
            held_output,
            copy_result,
            paths,
            local_paths,
            registry,
            plan_values,
            held_values,
            report_values,
        ),
    }


def protocol_value(bindings: dict) -> dict:
    repository = Path(__file__).resolve().parents[2]
    value = {
        "schema_version": 1,
        "status": "frozen_before_inference",
        "amendment_id": AMENDMENT_ID,
        "purpose": (
            "Regenerate only the held semantic logits with CPU-first conversion; "
            "reuse validated fixed-feature evidence; require original-family parity."
        ),
        "inputs": bindings,
        "code": amendment_code_identity(repository),
        "conversion": {
            "cpu_float32": "logits.detach().cpu()",
            "cpu_float64": "cpu_float32.to(dtype=torch.float64)",
            "fused_transfer_forbidden": True,
            "store_both": True,
            "exact_widening": True,
        },
        "compute": {
            "new_semantic_sequences": EXPECTED_VARIANTS,
            "new_model_forward_calls_max": MAXIMUM_FORWARD_CALLS,
            "microbatch": MICROBATCH,
            "accelerator_processes": 1,
            "feature_extraction_forwards": 0,
            "original_pretrained_model_loads": 0,
            "optimizer_updates": 0,
            "automatic_retry": False,
            "automatic_resume": False,
            "device": "mps",
            "context": "torch.inference_mode",
            "maximum_length": MAXIMUM_LENGTH,
        },
        "parity": {
            "original_rows": EXPECTED_ORIGINALS,
            "raw_atol": RAW_ATOL,
            "raw_rtol": RAW_RTOL,
            "temperatures": [1.0, SHARED_TEMPERATURE],
            "max_absolute_log_probability_error": MAX_LOG_PROBABILITY_ERROR,
            "maximum_total_variation": MAX_TOTAL_VARIATION,
            "p99_total_variation": P99_TOTAL_VARIATION,
            "p99_nearest_rank_index": 194,
            "material_reference_probability_margin": MATERIAL_MARGIN,
            "allowed_material_action_changes": 0,
            "allow_common_offset_adjustment": False,
        },
        "feature_reuse": {
            "keys": ["original", "shared"],
            "shape": list(EXPECTED_FEATURE_SHAPE),
            "dtype": "float32",
            "rows": EXPECTED_FEATURE_ROWS,
            "protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
            "logical_sequences": 5_440,
            "probe_score_atol": PROBE_SCORE_ATOL,
            "probe_calibrator_atol": PROBE_CALIBRATOR_ATOL,
        },
        "required_outputs": sorted(OUTPUT_FILES),
        "locked_test_opened": False,
    }
    value["protocol_sha256"] = canonical_sha256(value)
    return value


def freeze(
    plan: Path,
    held_output: Path,
    copy_result: Path,
    paths: Path,
    local_paths: Path,
    registry: Path,
    output: Path,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    inputs = collect_inputs(
        plan, held_output, copy_result, paths, local_paths, registry,
    )
    protocol = protocol_value(inputs["bindings"])
    output.mkdir(parents=True)
    try:
        protocol_descriptor = _write_json(output / "protocol.json", protocol)
        manifest = {
            "schema_version": 1,
            "status": "frozen_before_inference",
            "amendment_id": AMENDMENT_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "files": {"protocol.json": protocol_descriptor},
            "model_forwards": 0,
            "optimizer_updates": 0,
            "locked_test_opened": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(output / "manifest.json", manifest)
    except BaseException:
        for path in output.iterdir():
            path.unlink()
        output.rmdir()
        raise
    return manifest


def read_freeze(amendment: Path) -> tuple[dict, dict]:
    root = amendment.resolve(strict=True)
    if root.is_symlink() or {path.name for path in root.iterdir()} != {
        "protocol.json", "manifest.json",
    }:
        raise ValueError("amendment freeze inventory differs")
    manifest = _load_self_digested(
        root / "manifest.json", "amendment manifest", "manifest_sha256",
    )
    protocol = _load_self_digested(
        root / "protocol.json", "amendment protocol", "protocol_sha256",
    )
    if (
        manifest.get("status") != "frozen_before_inference"
        or manifest.get("amendment_id") != AMENDMENT_ID
        or manifest.get("protocol_sha256") != protocol.get("protocol_sha256")
        or manifest.get("files") != {
            "protocol.json": _descriptor(root / "protocol.json")
        }
        or protocol.get("status") != "frozen_before_inference"
        or protocol.get("amendment_id") != AMENDMENT_ID
    ):
        raise ValueError("amendment freeze boundary differs")
    return manifest, protocol


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
        raise ValueError("amendment differs from the reviewed identity")


def verify_freeze(
    amendment: Path,
    plan: Path,
    held_output: Path,
    copy_result: Path,
    paths: Path,
    local_paths: Path,
    registry: Path,
    reviewed_manifest_sha256: str | None = None,
    reviewed_protocol_sha256: str | None = None,
) -> tuple[dict, dict, dict]:
    manifest, protocol = read_freeze(amendment)
    if reviewed_manifest_sha256 is not None or reviewed_protocol_sha256 is not None:
        require_reviewed_identity(
            manifest,
            protocol,
            reviewed_manifest_sha256 or "",
            reviewed_protocol_sha256 or "",
        )
    repository = Path(__file__).resolve().parents[2]
    if protocol.get("code") != amendment_code_identity(repository):
        raise ValueError("amendment code identity differs")
    inputs = collect_inputs(
        plan, held_output, copy_result, paths, local_paths, registry,
    )
    if protocol != protocol_value(inputs["bindings"]):
        raise ValueError("amendment protocol does not reproduce")
    return manifest, protocol, inputs


def _centered_log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    centered = logits / temperature
    centered = centered - centered.max()
    return centered - math.log(float(np.exp(centered).sum()))


def parity_result(
    variants: list[dict],
    observations: list[dict],
    references: dict[str, dict],
) -> dict:
    variant_index = {row["variant_id"]: row for row in variants}
    if len(variant_index) != len(variants):
        raise ValueError("semantic variants contain duplicate IDs")
    observation_index = {}
    for row in observations:
        variant_id = row.get("variant_id")
        if variant_id in observation_index:
            raise ValueError("semantic observations contain duplicate IDs")
        observation_index[variant_id] = row
    if set(observation_index) != set(variant_index):
        raise ValueError("semantic observation inventory differs")
    details = []
    raw_passed = True
    material_changes = 0
    near_tie_changes = []
    maximum_log_probability_error = 0.0
    televisions_by_temperature: dict[str, list[float]] = {
        str(value): [] for value in (1.0, SHARED_TEMPERATURE)
    }
    for variant in variants:
        if variant["family"] != "original":
            continue
        observation = observation_index[variant["variant_id"]]
        expected_count = len(variant["question"]["options"])
        cpu32 = _validate_number_vector(
            observation.get("cpu_float32"), expected_count, "stored CPU float32",
        )
        logits = _validate_number_vector(
            observation.get("logits"), expected_count, "stored float64 logits",
        )
        widened = cpu32.astype(np.float32).astype(np.float64)
        if not np.array_equal(widened, logits):
            raise ValueError("stored float32 and float64 logits do not widen exactly")
        reference = np.asarray(
            references[variant["variant_id"]]["reference_logits"],
            dtype=np.float64,
        )
        errors = np.abs(logits - reference)
        allowed = RAW_ATOL + RAW_RTOL * np.abs(reference)
        row_raw_passed = bool(np.all(errors <= allowed))
        raw_passed = raw_passed and row_raw_passed
        temperature_results = {}
        action_change = int(logits.argmax()) != int(reference.argmax())
        reference_raw_log = _centered_log_softmax(reference, 1.0)
        reference_raw_probabilities = np.exp(reference_raw_log)
        ordered = np.sort(reference_raw_probabilities)
        reference_margin = float(ordered[-1] - ordered[-2])
        if action_change and reference_margin > MATERIAL_MARGIN:
            material_changes += 1
        elif action_change:
            near_tie_changes.append(variant["variant_id"])
        for temperature in (1.0, SHARED_TEMPERATURE):
            candidate_log = _centered_log_softmax(logits, temperature)
            reference_log = _centered_log_softmax(reference, temperature)
            log_error = float(np.max(np.abs(candidate_log - reference_log)))
            television = float(
                0.5 * np.abs(np.exp(candidate_log) - np.exp(reference_log)).sum()
            )
            maximum_log_probability_error = max(
                maximum_log_probability_error, log_error,
            )
            televisions_by_temperature[str(temperature)].append(television)
            temperature_results[str(temperature)] = {
                "max_absolute_log_probability_error": log_error,
                "total_variation": television,
            }
        details.append({
            "variant_id": variant["variant_id"],
            "key": references[variant["variant_id"]]["key"],
            "reference_logits": reference.tolist(),
            "candidate_logits": logits.tolist(),
            "maximum_absolute_raw_error": float(errors.max()),
            "raw_component_passed": row_raw_passed,
            "reference_probability_margin": reference_margin,
            "action_changed": action_change,
            "temperatures": temperature_results,
        })
    if len(details) != EXPECTED_ORIGINALS:
        raise ValueError("serialized original-family count differs")
    summaries = {}
    maximum_tv = 0.0
    maximum_p99 = 0.0
    for temperature, values in televisions_by_temperature.items():
        ordered = sorted(values)
        p99 = float(ordered[math.ceil(0.99 * len(ordered)) - 1])
        maximum = float(ordered[-1])
        maximum_tv = max(maximum_tv, maximum)
        maximum_p99 = max(maximum_p99, p99)
        summaries[temperature] = {
            "maximum_total_variation": maximum,
            "p99_total_variation": p99,
        }
    passed = bool(
        raw_passed
        and maximum_log_probability_error <= MAX_LOG_PROBABILITY_ERROR
        and maximum_tv <= MAX_TOTAL_VARIATION
        and maximum_p99 <= P99_TOTAL_VARIATION
        and material_changes == 0
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "original_rows": len(details),
        "raw_components_passed": raw_passed,
        "maximum_absolute_log_probability_error": maximum_log_probability_error,
        "temperature_summaries": summaries,
        "material_action_changes": material_changes,
        "near_tie_action_changes": near_tie_changes,
        "thresholds": {
            "raw_atol": RAW_ATOL,
            "raw_rtol": RAW_RTOL,
            "maximum_absolute_log_probability_error": MAX_LOG_PROBABILITY_ERROR,
            "maximum_total_variation": MAX_TOTAL_VARIATION,
            "p99_total_variation": P99_TOTAL_VARIATION,
            "material_reference_probability_margin": MATERIAL_MARGIN,
            "allowed_material_action_changes": 0,
        },
        "details": details,
        "passed": passed,
    }


def staged_observation(variant_id: str, logits: torch.Tensor, expected: int) -> dict:
    cpu32 = logits.detach().cpu()
    base = {"variant_id": variant_id}
    if (
        cpu32.device.type != "cpu"
        or cpu32.dtype != torch.float32
        or cpu32.ndim != 1
        or cpu32.numel() != expected
        or not torch.isfinite(cpu32).all()
    ):
        return {
            **base,
            "status": "invalid_cpu_float32",
            "cpu_float32": _json_safe_values(cpu32),
            "logits": None,
        }
    cpu64 = cpu32.to(dtype=torch.float64)
    if not torch.equal(cpu32.to(dtype=torch.float64), cpu64):
        raise RuntimeError("float32-to-float64 widening differs")
    return {
        **base,
        "status": "complete",
        "cpu_float32": cpu32.tolist(),
        "logits": cpu64.tolist(),
    }


def _original_row_gate(
    variant: dict,
    observation: dict,
    reference_value: dict,
) -> tuple[bool, dict]:
    expected_count = len(variant["question"]["options"])
    candidate = _validate_number_vector(
        observation.get("logits"), expected_count, "candidate original logits",
    )
    reference = _validate_number_vector(
        reference_value.get("reference_logits"),
        expected_count,
        "reference original logits",
    )
    errors = np.abs(candidate - reference)
    raw_passed = bool(np.all(errors <= RAW_ATOL + RAW_RTOL * np.abs(reference)))
    raw_reference_probabilities = np.exp(_centered_log_softmax(reference, 1.0))
    ordered = np.sort(raw_reference_probabilities)
    margin = float(ordered[-1] - ordered[-2])
    action_changed = int(candidate.argmax()) != int(reference.argmax())
    material_change = bool(action_changed and margin > MATERIAL_MARGIN)
    temperatures = {}
    probability_passed = True
    for temperature in (1.0, SHARED_TEMPERATURE):
        candidate_log = _centered_log_softmax(candidate, temperature)
        reference_log = _centered_log_softmax(reference, temperature)
        log_error = float(np.max(np.abs(candidate_log - reference_log)))
        television = float(
            0.5 * np.abs(np.exp(candidate_log) - np.exp(reference_log)).sum()
        )
        probability_passed = bool(
            probability_passed
            and log_error <= MAX_LOG_PROBABILITY_ERROR
            and television <= MAX_TOTAL_VARIATION
        )
        temperatures[str(temperature)] = {
            "max_absolute_log_probability_error": log_error,
            "total_variation": television,
        }
    passed = raw_passed and probability_passed and not material_change
    return passed, {
        "variant_id": variant["variant_id"],
        "key": reference_value["key"],
        "reference_logits": reference.tolist(),
        "candidate_logits": candidate.tolist(),
        "maximum_absolute_raw_error": float(errors.max()),
        "raw_component_passed": raw_passed,
        "reference_probability_margin": margin,
        "action_changed": action_changed,
        "material_action_change": material_change,
        "temperatures": temperatures,
    }


def infer_semantic_schedule(
    model,
    tokenizer,
    delimiters: dict[str, int],
    variants: list[dict],
    output_path: Path,
    references: dict[str, dict],
    counters: dict,
    parity_failure_path: Path | None = None,
) -> list[dict]:
    if len(variants) != EXPECTED_VARIANTS:
        raise ValueError("semantic schedule count differs")
    if counters != {
        "attempted_model_forward_calls": 0,
        "completed_model_forward_calls": 0,
        "completed_semantic_sequences": 0,
    }:
        raise ValueError("semantic counters must start at zero")
    observations = []
    original_details = []
    with output_path.open("xb") as handle, torch.inference_mode():
        for start in range(0, len(variants), MICROBATCH):
            batch_rows = variants[start:start + MICROBATCH]
            encodings = [
                encode_request(
                    tokenizer,
                    row["state"],
                    [row["question"]],
                    MAXIMUM_LENGTH,
                    delimiters,
                )
                for row in batch_rows
            ]
            if any(value is None for value in encodings):
                raise ValueError("semantic variant exceeds the fixed input budget")
            if any(value.retained_state_tokens != value.state_tokens for value in encodings):
                raise ValueError("semantic variant would truncate state content")
            if counters["attempted_model_forward_calls"] >= MAXIMUM_FORWARD_CALLS:
                raise RuntimeError("semantic forward-call cap would be exceeded")
            counters["attempted_model_forward_calls"] += 1
            outputs = model(encodings)
            counters["completed_model_forward_calls"] += 1
            if len(outputs) != len(batch_rows):
                raise ValueError("semantic model batch count differs")
            for row, request_logits in zip(batch_rows, outputs):
                if len(request_logits) != 1:
                    raise ValueError("semantic model question count differs")
                observation = staged_observation(
                    row["variant_id"],
                    request_logits[0],
                    len(row["question"]["options"]),
                )
                _append_jsonl(handle, observation)
                observations.append(observation)
                if observation["status"] != "complete":
                    raise RuntimeError("semantic output failed CPU float32 validation")
                counters["completed_semantic_sequences"] += 1
                if row["family"] == "original":
                    passed, detail = _original_row_gate(
                        row,
                        observation,
                        references[row["variant_id"]],
                    )
                    original_details.append(detail)
                    if not passed:
                        failure = {
                            "schema_version": 1,
                            "status": "failed_during_execution",
                            "completed_original_rows": len(original_details),
                            "failed_variant_id": row["variant_id"],
                            "details": original_details,
                            "passed": False,
                        }
                        if parity_failure_path is not None:
                            _write_json(parity_failure_path, failure)
                        raise RuntimeError("original-family parity failed during execution")
    if counters != {
        "attempted_model_forward_calls": MAXIMUM_FORWARD_CALLS,
        "completed_model_forward_calls": MAXIMUM_FORWARD_CALLS,
        "completed_semantic_sequences": EXPECTED_VARIANTS,
    }:
        raise RuntimeError("semantic execution counters differ")
    return observations


def _probe_replay(held: dict, recomputed: dict) -> dict:
    score_deltas = []
    calibrator_deltas = []

    def compare(expected: Any, actual: Any, path: tuple[str, ...]) -> None:
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(expected) != set(actual):
                raise ValueError(f"probe replay fields differ at {'.'.join(path)}")
            for key in expected:
                compare(expected[key], actual[key], (*path, key))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                raise ValueError(f"probe replay list differs at {'.'.join(path)}")
            for index, (left, right) in enumerate(zip(expected, actual)):
                compare(left, right, (*path, str(index)))
            return
        if type(expected) is float:
            if type(actual) not in (int, float) or not math.isfinite(float(actual)):
                raise ValueError(f"probe replay numeric value differs at {'.'.join(path)}")
            delta = abs(float(actual) - expected)
            if "calibration" in path and path[-1] in {"temperature", "slope", "intercept"}:
                calibrator_deltas.append(delta)
                if delta > PROBE_CALIBRATOR_ATOL:
                    raise ValueError(f"probe calibrator replay differs at {'.'.join(path)}")
            elif path[-1] in {"nll", "multiclass_brier"}:
                score_deltas.append(delta)
                if delta > PROBE_SCORE_ATOL:
                    raise ValueError(f"probe score replay differs at {'.'.join(path)}")
            elif actual != expected:
                raise ValueError(f"probe exact replay differs at {'.'.join(path)}")
            return
        if actual != expected:
            raise ValueError(f"probe replay differs at {'.'.join(path)}")

    compare(held, recomputed, ("probe_results",))
    return {
        "maximum_nll_or_brier_delta": max(score_deltas, default=0.0),
        "maximum_calibrator_parameter_delta": max(calibrator_deltas, default=0.0),
        "exact_other_fields": True,
    }


def _load_shared_only(local: dict, device: torch.device, protocol: dict):
    shared_dir = Path(local["shared_run_directory"]).resolve(strict=True)
    for name, descriptor in SHARED_TOKENIZER_FILES.items():
        _verify_descriptor(shared_dir / name, descriptor, f"shared tokenizer {name}")
    payload, run, manifest = load_completed_checkpoint(shared_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        shared_dir, local_files_only=True, trust_remote_code=False,
    )
    config = run["spec"]["config"]
    experiment = run["spec"]["experiment"]
    model, delimiters = load_shared_state_model(
        config["backbone"],
        tokenizer,
        str(device),
        revision=experiment["backbone_revision"],
        pointer_size=experiment["pointer_size"],
        attention_implementation=experiment["attention_implementation"],
        local_files_only=True,
    )
    binding = protocol["inputs"]["models"]["shared_generation_79"]
    if (
        tokenizer_fingerprint(tokenizer) != run["spec"]["tokenizer_fingerprint"]
        or model_config_fingerprint(model) != run["spec"]["model_config_fingerprint"]
        or manifest["current"]["generation"] != GENERATION
        or manifest["current"]["sha256"] != CHECKPOINT_SHA256
        or binding["checkpoint"]["sha256"] != CHECKPOINT_SHA256
    ):
        raise ValueError("shared generation-79 runtime identity differs")
    model.backbone.load_state_dict(payload["backbone"], strict=True)
    model.head.load_state_dict(payload["head"], strict=True)
    model.eval()
    return model, tokenizer, delimiters


def _receipt(
    protocol: dict,
    status: str,
    counters: dict,
    failure: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "amendment_id": AMENDMENT_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "new_execution": {
            **counters,
            "feature_extraction_forwards": 0,
            "original_pretrained_model_loads": 0,
            "optimizer_updates": 0,
            "accelerator_processes": 1,
        },
        "reused_feature_sequences": 5_440,
        "logical_combined_sequences": 6_540,
        "old_plus_amendment_executed_sequences": 7_640,
        "old_plus_amendment_model_forward_calls": 3_820,
        "automatic_retry": False,
        "automatic_resume": False,
        "locked_test_opened": False,
        "failure": failure,
    }


def _runtime_process_identity() -> dict:
    return {
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(os.sys.argv),
    }


def run(
    amendment: Path,
    plan: Path,
    held_output: Path,
    copy_result: Path,
    paths: Path,
    local_paths: Path,
    registry: Path,
    output: Path,
    device_name: str,
    reviewed_manifest_sha256: str,
    reviewed_protocol_sha256: str,
) -> dict:
    manifest, protocol = read_freeze(amendment)
    require_reviewed_identity(
        manifest,
        protocol,
        reviewed_manifest_sha256,
        reviewed_protocol_sha256,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    _, protocol, inputs = verify_freeze(
        amendment,
        plan,
        held_output,
        copy_result,
        paths,
        local_paths,
        registry,
        reviewed_manifest_sha256,
        reviewed_protocol_sha256,
    )
    device = _resolve_device(device_name)
    if device.type != "mps" or _runtime_identity(device) != protocol["inputs"]["runtime"]:
        raise ValueError("amendment runtime differs from the frozen identity")
    output.mkdir(parents=True)
    counters = {
        "attempted_model_forward_calls": 0,
        "completed_model_forward_calls": 0,
        "completed_semantic_sequences": 0,
    }
    _write_json(
        output / "execution-receipt.json",
        {
            **_receipt(protocol, "running", counters),
            "process": _runtime_process_identity(),
        },
    )
    try:
        local = _load_local_paths(local_paths)
        model, tokenizer, delimiters = _load_shared_only(local, device, protocol)
        observations = infer_semantic_schedule(
            model,
            tokenizer,
            delimiters,
            inputs["variants"],
            output / "semantic-logits.jsonl",
            inputs["references"],
            counters,
            output / "original-parity.json",
        )
        parity = parity_result(inputs["variants"], observations, inputs["references"])
        _write_json(output / "original-parity.json", parity)
        serialized = _read_jsonl(
            output / "semantic-logits.jsonl", "serialized semantic logits",
        )
        replayed_parity = parity_result(
            inputs["variants"], serialized, inputs["references"],
        )
        if replayed_parity != parity or not parity["passed"]:
            raise RuntimeError("serialized original-family parity failed")
        feature_arrays = {
            key: tensor.numpy().astype(np.float64)
            for key, tensor in inputs["held"]["features"].items()
        }
        probes = probe_results(
            inputs["feature_rows"],
            feature_arrays,
            inputs["support_fit"],
            inputs["support_calibration"],
            inputs["development"],
        )
        probe_replay = _probe_replay(
            inputs["held"]["result"]["probe_results"], probes,
        )
        semantic_logits = [
            {"variant_id": row["variant_id"], "logits": row["logits"]}
            for row in serialized
        ]
        semantics = semantic_results(inputs["variants"], semantic_logits)
        decisions = diagnostic_decisions(probes, semantics, inputs["protocol"])
        reuse = {
            "schema_version": 1,
            "status": "validated_reuse",
            "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
            "held_output_manifest_sha256": HELD_OUTPUT_MANIFEST_SHA256,
            "held_result_sha256": HELD_RESULT_SHA256,
            "feature_rows": inputs["held"]["manifest"]["files"]["feature-rows.jsonl"],
            "features": inputs["held"]["manifest"]["files"]["features.safetensors"],
            "feature_tensor_contract": {
                "keys": ["original", "shared"],
                "shape": list(EXPECTED_FEATURE_SHAPE),
                "dtype": "float32",
                "finite": True,
                "metadata": {"protocol_sha256": ORIGINAL_PROTOCOL_SHA256},
            },
            "probe_replay": probe_replay,
        }
        _write_json(output / "reuse.json", reuse)
        result = {
            "schema_version": 2,
            "status": "complete",
            "amendment_id": AMENDMENT_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "original_plan_manifest_sha256": ORIGINAL_PLAN_MANIFEST_SHA256,
            "held_output_manifest_sha256": HELD_OUTPUT_MANIFEST_SHA256,
            "copy_result_sha256": COPY_RESULT_SELF_SHA256,
            "original_parity": {
                key: value for key, value in parity.items() if key != "details"
            },
            "probe_results": probes,
            "semantic_results": semantics,
            "diagnostic_decisions": decisions,
            "provenance": {
                "reused_feature_sequences": 5_440,
                "new_semantic_sequences": EXPECTED_VARIANTS,
                "new_model_forward_calls": MAXIMUM_FORWARD_CALLS,
                "logical_combined_sequences": 6_540,
                "old_plus_amendment_executed_sequences": 7_640,
                "old_plus_amendment_model_forward_calls": 3_820,
            },
            "optimizer_updates": 0,
            "locked_test_opened": False,
        }
        result["result_sha256"] = canonical_sha256(result)
        _write_json(output / "result.json", result)
        receipt = {
            **_receipt(protocol, "complete", counters),
            "process": _runtime_process_identity(),
        }
        _write_json(output / "execution-receipt.json", receipt)
        files = {
            name: (
                _descriptor_with_rows(output / name)
                if name.endswith(".jsonl") else _descriptor(output / name)
            )
            for name in sorted(OUTPUT_FILES - {"manifest.json"})
        }
        output_manifest = {
            "schema_version": 1,
            "status": "complete",
            "amendment_id": AMENDMENT_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "files": files,
            "new_semantic_sequences": EXPECTED_VARIANTS,
            "new_model_forward_calls": MAXIMUM_FORWARD_CALLS,
            "feature_extraction_forwards": 0,
            "original_pretrained_model_loads": 0,
            "optimizer_updates": 0,
            "locked_test_opened": False,
        }
        output_manifest["manifest_sha256"] = canonical_sha256(output_manifest)
        _write_json(output / "manifest.json", output_manifest)
        verify_output(
            amendment,
            plan,
            held_output,
            copy_result,
            paths,
            local_paths,
            registry,
            output,
        )
        return output_manifest
    except BaseException as error:
        failure_receipt = {
            **_receipt(protocol, "failed", counters, str(error)),
            "process": _runtime_process_identity(),
        }
        _write_json(output / "execution-receipt.json", failure_receipt)
        raise


def verify_output(
    amendment: Path,
    plan: Path,
    held_output: Path,
    copy_result: Path,
    paths: Path,
    local_paths: Path,
    registry: Path,
    output: Path,
) -> dict:
    _, protocol, inputs = verify_freeze(
        amendment,
        plan,
        held_output,
        copy_result,
        paths,
        local_paths,
        registry,
    )
    root = output.resolve(strict=True)
    if root.is_symlink() or {path.name for path in root.iterdir()} != set(OUTPUT_FILES):
        raise ValueError("amendment output inventory differs")
    manifest = _load_self_digested(
        root / "manifest.json", "amendment output manifest", "manifest_sha256",
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("amendment_id") != AMENDMENT_ID
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or set(manifest.get("files", {})) != OUTPUT_FILES - {"manifest.json"}
        or manifest.get("new_semantic_sequences") != EXPECTED_VARIANTS
        or manifest.get("new_model_forward_calls") != MAXIMUM_FORWARD_CALLS
        or manifest.get("feature_extraction_forwards") != 0
        or manifest.get("original_pretrained_model_loads") != 0
        or manifest.get("optimizer_updates") != 0
        or manifest.get("locked_test_opened") is not False
    ):
        raise ValueError("amendment output boundary differs")
    _verify_files(root, manifest["files"], OUTPUT_FILES)
    observations = _read_jsonl(root / "semantic-logits.jsonl", "semantic logits")
    parity = load_json_bytes((root / "original-parity.json").read_bytes(), "original parity")
    replayed_parity = parity_result(
        inputs["variants"], observations, inputs["references"],
    )
    if parity != replayed_parity or not replayed_parity["passed"]:
        raise ValueError("original parity does not replay")
    feature_arrays = {
        key: tensor.numpy().astype(np.float64)
        for key, tensor in inputs["held"]["features"].items()
    }
    probes = probe_results(
        inputs["feature_rows"],
        feature_arrays,
        inputs["support_fit"],
        inputs["support_calibration"],
        inputs["development"],
    )
    replay = _probe_replay(inputs["held"]["result"]["probe_results"], probes)
    reuse = load_json_bytes((root / "reuse.json").read_bytes(), "reuse")
    if reuse.get("probe_replay") != replay:
        raise ValueError("feature probe replay evidence differs")
    logits = [
        {"variant_id": row["variant_id"], "logits": row["logits"]}
        for row in observations
    ]
    semantics = semantic_results(inputs["variants"], logits)
    decisions = diagnostic_decisions(probes, semantics, inputs["protocol"])
    result = _load_self_digested(root / "result.json", "amendment result", "result_sha256")
    if (
        result.get("probe_results") != probes
        or result.get("semantic_results") != semantics
        or result.get("diagnostic_decisions") != decisions
        or result.get("original_parity")
        != {key: value for key, value in parity.items() if key != "details"}
    ):
        raise ValueError("amendment scientific result does not replay")
    receipt = load_json_bytes(
        (root / "execution-receipt.json").read_bytes(), "execution receipt",
    )
    if (
        receipt.get("status") != "complete"
        or receipt.get("new_execution", {}).get("attempted_model_forward_calls")
        != MAXIMUM_FORWARD_CALLS
        or receipt.get("new_execution", {}).get("completed_model_forward_calls")
        != MAXIMUM_FORWARD_CALLS
        or receipt.get("new_execution", {}).get("completed_semantic_sequences")
        != EXPECTED_VARIANTS
        or receipt.get("new_execution", {}).get("feature_extraction_forwards") != 0
        or receipt.get("new_execution", {}).get("original_pretrained_model_loads") != 0
        or receipt.get("new_execution", {}).get("optimizer_updates") != 0
        or receipt.get("locked_test_opened") is not False
    ):
        raise ValueError("amendment execution receipt differs")
    return manifest


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--held-output", required=True, type=Path)
    parser.add_argument("--copy-result", required=True, type=Path)
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--registry", default=Path("research/evidence/index.json"), type=Path)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--plan", required=True, type=Path)
    freeze_parser.add_argument("--held-output", required=True, type=Path)
    freeze_parser.add_argument("--copy-result", required=True, type=Path)
    freeze_parser.add_argument("--paths", required=True, type=Path)
    freeze_parser.add_argument("--local-paths", required=True, type=Path)
    freeze_parser.add_argument("--registry", default=Path("research/evidence/index.json"), type=Path)
    freeze_parser.add_argument("--out", required=True, type=Path)
    run_parser = commands.add_parser("run")
    _common_arguments(run_parser)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--device", choices=("mps",), required=True)
    run_parser.add_argument("--reviewed-manifest-sha256", required=True)
    run_parser.add_argument("--reviewed-protocol-sha256", required=True)
    run_parser.add_argument("--allow-reviewed-inference", action="store_true")
    verify_parser = commands.add_parser("verify")
    _common_arguments(verify_parser)
    verify_parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze(
            args.plan,
            args.held_output,
            args.copy_result,
            args.paths,
            args.local_paths,
            args.registry,
            args.out,
        )
    elif args.command == "run":
        if not args.allow_reviewed_inference:
            parser.error("reviewed amendment inference requires --allow-reviewed-inference")
        value = run(
            args.amendment,
            args.plan,
            args.held_output,
            args.copy_result,
            args.paths,
            args.local_paths,
            args.registry,
            args.out,
            args.device,
            args.reviewed_manifest_sha256,
            args.reviewed_protocol_sha256,
        )
    else:
        value = verify_output(
            args.amendment,
            args.plan,
            args.held_output,
            args.copy_result,
            args.paths,
            args.local_paths,
            args.registry,
            args.out,
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
