"""Frozen bindings and common-clean filtering for development comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.kev_adapter import file_sha256


PLAN_VERSION = 1
SUITE_KEYS = ("decision", "transfer", "korean")


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_state_sha256(request: dict) -> str:
    return canonical_sha256(request["state"])


def comparison_code_identity(repository: Path) -> dict:
    files = [
        repository / "experiments" / "audit_shared_suite.py",
        repository / "experiments" / "comparison_protocol.py",
        repository / "experiments" / "evaluate_baseline.py",
        repository / "experiments" / "evaluate_shared.py",
        repository / "experiments" / "freeze_comparison.py",
        repository / "experiments" / "freeze_shared_suite.py",
        repository / "experiments" / "kev_adapter.py",
        repository / "src" / "haetae" / "calibrate.py",
        repository / "src" / "haetae" / "checkpoint.py",
        repository / "src" / "haetae" / "certify.py",
        repository / "src" / "haetae" / "model.py",
    ]
    digest = hashlib.sha256()
    values = {}
    for path in files:
        relative = str(path.relative_to(repository))
        sha256 = file_sha256(path)
        values[relative] = sha256
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": values}


def load_comparison_plan(path: str | Path, *, enforce_code: bool = True) -> dict:
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    expected = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise ValueError("comparison plan digest mismatch")
    if plan.get("version") != PLAN_VERSION or plan.get("status") != "frozen":
        raise ValueError("unsupported comparison plan")
    if set(plan.get("suites", {})) != set(SUITE_KEYS):
        raise ValueError("comparison plan suite bindings are incomplete")
    if set(plan.get("excluded_development_requests", {})) != set(SUITE_KEYS):
        raise ValueError("comparison plan exclusions are incomplete")
    if not isinstance(plan.get("excluded_calibration_requests"), list):
        raise ValueError("comparison calibration exclusions are incomplete")
    if not isinstance(plan.get("rendered_state_audit"), dict):
        raise ValueError("comparison rendered-state audit is missing")
    if enforce_code:
        repository = Path(__file__).resolve().parents[1]
        if comparison_code_identity(repository) != plan.get("code"):
            raise ValueError("comparison code differs from the frozen plan")
    return plan


def validate_suite_binding(plan: dict, key: str, suite: str | Path) -> dict:
    if key not in SUITE_KEYS:
        raise ValueError(f"unknown comparison suite: {key}")
    suite = Path(suite).resolve()
    manifest_path = suite / "manifest.json"
    binding = plan["suites"][key]
    if file_sha256(manifest_path) != binding["manifest_sha256"]:
        raise ValueError(f"{key} suite manifest differs from the comparison plan")
    if key == "decision":
        audit_path = suite / "rendered-state-audit.json"
        if file_sha256(audit_path) != plan["rendered_state_audit"][
                "file_sha256"]:
            raise ValueError(
                "rendered-state audit differs from the comparison plan"
            )
        audit = json.loads(audit_path.read_text())
        validate_external_audit_bindings(plan["suites"], audit)
    manifest = json.loads(manifest_path.read_text())
    for filename, descriptor in binding["files"].items():
        if manifest.get("files", {}).get(filename) != descriptor:
            raise ValueError(f"{key} suite split differs from the comparison plan")
    return manifest


def validate_external_audit_bindings(
    suite_bindings: dict,
    rendered_state_audit: dict,
) -> None:
    for key in ("transfer", "korean"):
        if rendered_state_audit.get(key) != suite_bindings.get(key):
            raise ValueError(
                f"{key} suite differs from the rendered-state audit"
            )


def _clean_predictions(
    predictions: list[dict],
    exclusions: list[dict],
    label: str,
) -> tuple[list[dict], dict]:
    expected = {
        item["state_sha256"]: int(item["questions"])
        for item in exclusions
    }
    if len(expected) != len(exclusions):
        raise ValueError(f"{label} comparison exclusions contain duplicate states")
    observed = {identity: 0 for identity in expected}
    clean = []
    for prediction in predictions:
        identity = prediction.get("state_sha256")
        if identity in observed:
            observed[identity] += 1
        else:
            clean.append(prediction)
    if observed != expected:
        raise ValueError(
            f"{label} predictions do not match frozen comparison exclusions"
        )
    return clean, {
        "excluded_requests": len(exclusions),
        "excluded_questions": sum(expected.values()),
        "retained_questions": len(clean),
    }


def common_clean_predictions(
    predictions: list[dict],
    plan: dict,
    suite_key: str,
) -> tuple[list[dict], dict]:
    return _clean_predictions(
        predictions,
        plan["excluded_development_requests"][suite_key],
        f"{suite_key} development",
    )


def training_clean_calibration_predictions(
    predictions: list[dict],
    plan: dict,
) -> tuple[list[dict], dict]:
    return _clean_predictions(
        predictions,
        plan["excluded_calibration_requests"],
        "decision calibration",
    )
