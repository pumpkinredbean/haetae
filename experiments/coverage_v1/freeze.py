"""Freeze the metadata-only coverage audit and the next diagnostic protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from experiments.coverage_v1.audit import (
    audit_exposure_weights,
    audit_role_overlap,
    audit_semantics,
    audit_supervised_coverage,
    verify_cached_reports,
)
from experiments.coverage_v1.registry import (
    ArtifactStore,
    canonical_sha256,
    load_json_bytes,
)


PARTITION_EVIDENCE = {
    "train": "shared_suite_train",
    "calibration": "shared_suite_calibration",
    "decision_development": "shared_suite_development",
    "transfer_development": "transfer_suite_development",
    "korean_development": "korean_suite_development",
}
JSON_EVIDENCE = (
    "baseline_development_report",
    "baseline_run_manifest",
    "baseline_run_spec",
    "comparison_plan",
    "korean_suite_manifest",
    "shared_development_report",
    "shared_run_manifest",
    "shared_run_spec",
    "shared_suite_manifest",
    "shared_suite_rendered_state_audit",
    "transfer_suite_manifest",
)
OUTPUT_NAMES = (
    "coverage.json",
    "semantic_checks.json",
    "cached_metrics.json",
    "diagnostic_protocol.json",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_identity(repository: Path) -> dict:
    names = (
        "experiments/coverage_v1/__init__.py",
        "experiments/coverage_v1/registry.py",
        "experiments/coverage_v1/freeze.py",
        "experiments/coverage_v1/audit.py",
        "experiments/coverage_v1/protocol.template.json",
    )
    files = {name: _file_sha256(repository / name) for name in names}
    digest = hashlib.sha256()
    for name, sha256 in files.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": files}


def _partition_summary(rows: list[dict]) -> dict:
    origins = set()
    parents = set()
    task_sources = set()
    questions = 0
    for row in rows:
        meta = row.get("_meta", {})
        origin = meta.get("source")
        request_id = meta.get("id")
        parent = meta.get("group_id") or request_id
        if not isinstance(origin, str) or not isinstance(parent, str):
            raise ValueError("partition contains incomplete source-parent metadata")
        origins.add(origin)
        parents.add((origin, parent))
        raw_questions = row.get("questions")
        if not isinstance(raw_questions, dict):
            raise ValueError("partition questions must be an object")
        questions += len(raw_questions)
        for question in raw_questions.values():
            source = question.get("src")
            if not isinstance(source, str) or not source:
                raise ValueError("partition question source is missing")
            task_sources.add(source)
    return {
        "requests": len(rows),
        "questions": questions,
        "unique_source_parents": len(parents),
        "origin_sources": sorted(origins),
        "task_sources": sorted(task_sources),
    }


def _validate_input_bindings(
    store: ArtifactStore,
    partitions: dict[str, list[dict]],
    values: dict[str, dict],
) -> None:
    shared_manifest = values["shared_suite_manifest"]
    transfer_manifest = values["transfer_suite_manifest"]
    korean_manifest = values["korean_suite_manifest"]
    manifest_map = {
        "train": (shared_manifest, "train.jsonl"),
        "calibration": (shared_manifest, "calibration.jsonl"),
        "decision_development": (shared_manifest, "development.jsonl"),
        "transfer_development": (transfer_manifest, "development.jsonl"),
        "korean_development": (korean_manifest, "development.jsonl"),
    }
    for role, (manifest, filename) in manifest_map.items():
        evidence_id = PARTITION_EVIDENCE[role]
        descriptor = manifest.get("files", {}).get(filename)
        binding = store.binding(evidence_id)
        if not isinstance(descriptor, dict):
            raise ValueError(f"{role} manifest descriptor is missing")
        if (
            descriptor.get("sha256") != binding["sha256"]
            or (
                "bytes" in descriptor
                and descriptor["bytes"] != binding["size_bytes"]
            )
            or descriptor.get("records") != len(partitions[role])
            or descriptor.get("questions")
            != sum(len(row["questions"]) for row in partitions[role])
        ):
            raise ValueError(f"{role} differs from its frozen suite manifest")
    comparison = values["comparison_plan"]
    expected = comparison.get("plan_sha256")
    unsigned = dict(comparison)
    unsigned.pop("plan_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise ValueError("comparison plan self-digest differs")
    if comparison.get("locked_test_opened") is not False:
        raise ValueError("comparison plan did not preserve the test lock")
    rendered = values["shared_suite_rendered_state_audit"]
    if rendered.get("locked_test_opened") is not False:
        raise ValueError("rendered-state audit did not preserve the test lock")
    for model in ("baseline", "shared"):
        run = values[f"{model}_run_spec"]
        manifest = values[f"{model}_run_manifest"]
        if (
            manifest.get("status") != "completed"
            or manifest.get("run_id") != run.get("run_id")
            or manifest.get("spec_sha256") != run.get("spec_sha256")
            or manifest.get("current", {}).get("status") != "completed"
        ):
            raise ValueError(f"{model} completed-run binding differs")


def build_manifest(
    store: ArtifactStore,
    partitions: dict[str, list[dict]],
    outputs: dict[str, dict],
    code: dict,
) -> dict:
    """Build a path-free manifest for the completed cached audit."""
    input_ids = sorted(set(PARTITION_EVIDENCE.values()) | set(JSON_EVIDENCE))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "base_commit": store.registry["base_commit"],
        "evidence_registry_sha256": canonical_sha256(store.registry),
        "inputs": {evidence_id: store.binding(evidence_id) for evidence_id in input_ids},
        "partitions": {
            role: {
                "evidence_id": PARTITION_EVIDENCE[role],
                **_partition_summary(partitions[role]),
            }
            for role in PARTITION_EVIDENCE
        },
        "outputs": outputs,
        "code": code,
        "compute_class": "C0 cached CPU",
        "bootstrap_draws": 20000,
        "model_forwards": 0,
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _encoded_json(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: dict) -> dict:
    data = _encoded_json(value)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def freeze(paths_path: Path, output: Path, registry_path: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output.name}")
    store = ArtifactStore.from_files(registry_path, paths_path)
    partitions = {
        role: store.read_jsonl(evidence_id)
        for role, evidence_id in PARTITION_EVIDENCE.items()
    }
    values = {evidence_id: store.read_json(evidence_id) for evidence_id in JSON_EVIDENCE}
    _validate_input_bindings(store, partitions, values)

    role_overlap = audit_role_overlap(partitions)
    semantic_checks = audit_semantics(partitions)
    coverage = {
        "schema_version": 1,
        "status": "complete",
        "role_overlap": role_overlap,
        "supervised_coverage": audit_supervised_coverage(
            partitions,
            values["baseline_run_spec"],
        ),
        "exposure_weights": audit_exposure_weights(
            partitions["train"],
            values["shared_run_spec"],
        ),
        "model_forwards": 0,
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }
    cached_metrics = verify_cached_reports(
        {
            "baseline": values["baseline_development_report"],
            "shared": values["shared_development_report"],
        },
        partitions,
        {
            "decision": values["shared_suite_manifest"],
            "transfer": values["transfer_suite_manifest"],
            "korean": values["korean_suite_manifest"],
        },
        {
            "baseline": values["baseline_run_spec"],
            "shared": values["shared_run_spec"],
        },
        {
            "baseline": values["baseline_run_manifest"],
            "shared": values["shared_run_manifest"],
        },
        values["comparison_plan"],
    )
    template_path = Path(__file__).with_name("protocol.template.json")
    protocol_data = template_path.read_bytes()
    protocol = load_json_bytes(protocol_data, "diagnostic protocol template")
    protocol["template_sha256"] = hashlib.sha256(protocol_data).hexdigest()
    protocol["input_evidence"] = {
        evidence_id: store.binding(evidence_id)
        for evidence_id in (
            "shared_suite_train",
            "shared_suite_calibration",
            "shared_suite_development",
            "transfer_suite_development",
            "korean_suite_development",
            "baseline_development_report",
            "shared_development_report",
        )
    }
    repository = Path(__file__).resolve().parents[2]
    code = _code_identity(repository)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        output_values = {
            "coverage.json": coverage,
            "semantic_checks.json": semantic_checks,
            "cached_metrics.json": cached_metrics,
            "diagnostic_protocol.json": protocol,
        }
        descriptors = {
            filename: _write_new(temporary / filename, value)
            for filename, value in output_values.items()
        }
        manifest = build_manifest(store, partitions, descriptors, code)
        _write_new(temporary / "manifest.json", manifest)
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/evidence/index.json"),
    )
    args = parser.parse_args()
    manifest = freeze(args.paths, args.out, args.registry)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "partitions": {
            role: {
                "requests": value["requests"],
                "questions": value["questions"],
            }
            for role, value in manifest["partitions"].items()
        },
        "model_forwards": 0,
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
