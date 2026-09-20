"""Audit frozen public partitions by the state text visible to the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from haetae.checkpoint import atomic_json_save
from haetae.measure import digest_value
from experiments.freeze_shared_suite import assert_disjoint, read_raw_split
from experiments.kev_adapter import file_sha256


ARTIFACT_NAME = "rendered-state-audit.json"


def code_identity(repository: Path) -> dict:
    files = [
        repository / "experiments" / "audit_shared_suite.py",
        repository / "experiments" / "freeze_shared_suite.py",
        repository / "experiments" / "kev_adapter.py",
        repository / "src" / "haetae" / "checkpoint.py",
        repository / "src" / "haetae" / "measure.py",
    ]
    values = {}
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(repository))
        sha256 = file_sha256(path)
        values[relative] = sha256
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": values}


def manifest_binding(path: Path, filenames: tuple[str, ...]) -> tuple[dict, dict]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return manifest, {
        "manifest_sha256": file_sha256(manifest_path),
        "files": {name: manifest["files"][name] for name in filenames},
    }


def audit(args) -> dict:
    suite = Path(args.suite).resolve()
    output = suite / ARTIFACT_NAME
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    transfer = Path(args.transfer_suite).resolve()
    korean = Path(args.korean_suite).resolve()
    suite_manifest, suite_binding = manifest_binding(
        suite,
        ("train.jsonl", "calibration.jsonl", "development.jsonl"),
    )
    transfer_manifest, transfer_binding = manifest_binding(
        transfer, ("development.jsonl",),
    )
    korean_manifest, korean_binding = manifest_binding(
        korean, ("development.jsonl",),
    )
    partitions = {
        split: read_raw_split(suite, suite_manifest, split)
        for split in ("train", "calibration", "development")
    }
    partitions["transfer_development"] = read_raw_split(
        transfer, transfer_manifest, "development",
    )
    partitions["korean_development"] = read_raw_split(
        korean, korean_manifest, "development",
    )
    result = assert_disjoint(partitions)
    repository = Path(__file__).resolve().parents[1]
    artifact = {
        "version": 1,
        "status": "complete",
        "method": (
            "normalize every request through the runtime adapter, hash the "
            "rendered state string, and compare source-namespaced parents"
        ),
        "suite": suite_binding,
        "transfer": transfer_binding,
        "korean": korean_binding,
        "partition_counts": {
            key: {
                "requests": len(records),
                "questions": sum(len(record["questions"]) for record in records),
            }
            for key, records in partitions.items()
        },
        "audit": result,
        "locked_test_opened": False,
        "code": code_identity(repository),
    }
    artifact["artifact_sha256"] = digest_value(artifact)
    atomic_json_save(artifact, output)
    return artifact


def load_rendered_state_audit(suite: str | Path) -> dict:
    suite = Path(suite).resolve()
    path = suite / ARTIFACT_NAME
    artifact = json.loads(path.read_text())
    expected = artifact.get("artifact_sha256")
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    if expected != digest_value(unsigned):
        raise ValueError("rendered-state audit digest mismatch")
    if artifact.get("version") != 1 or artifact.get("status") != "complete":
        raise ValueError("rendered-state audit is incomplete")
    if artifact.get("locked_test_opened") is not False:
        raise ValueError("rendered-state audit does not preserve the test lock")
    result = artifact.get("audit", {})
    if result.get("state_overlap") != 0 or result.get("source_parent_overlap") != 0:
        raise ValueError("rendered-state audit contains partition overlap")
    manifest_path = suite / "manifest.json"
    if file_sha256(manifest_path) != artifact.get("suite", {}).get(
            "manifest_sha256"):
        raise ValueError("rendered-state audit belongs to another suite")
    manifest = json.loads(manifest_path.read_text())
    if any(
        manifest.get("files", {}).get(name) != descriptor
        for name, descriptor in artifact["suite"]["files"].items()
    ):
        raise ValueError("rendered-state audit suite files changed")
    repository = Path(__file__).resolve().parents[1]
    if artifact.get("code") != code_identity(repository):
        raise ValueError("rendered-state audit code changed")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--korean-suite", required=True)
    args = parser.parse_args()
    artifact = audit(args)
    print(json.dumps({
        "artifact_sha256": artifact["artifact_sha256"],
        "partition_counts": artifact["partition_counts"],
        "audit": artifact["audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
