import hashlib
import json
from pathlib import Path

import pytest

from experiments.coverage_v1.audit import (
    COVERAGE_OUTPUT_NAMES,
    audit_manifest,
)
from experiments.coverage_v1.freeze import build_manifest
from experiments.coverage_v1.registry import (
    CODE_IDENTITY_FILES,
    PARTITION_EVIDENCE,
    ArtifactStore,
    canonical_sha256,
    coverage_code_identity,
    load_json_bytes,
)


def _encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()


def _rewrite_manifest(path, manifest):
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(unsigned)
    path.write_bytes(_encoded(manifest))


def _request(role):
    return {
        "state": f"state {role}",
        "questions": {
            "answer": {
                "src": f"task_{role}",
            }
        },
        "_meta": {
            "source": f"origin_{role}",
            "id": f"{role}/request",
            "group_id": f"{role}/parent",
        },
    }


def _valid_archive(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    registry = load_json_bytes(
        (repository / "research/evidence/index.json").read_bytes(),
        "registry",
    )
    store = ArtifactStore(registry, {})
    descriptors = {}
    for filename in COVERAGE_OUTPUT_NAMES:
        data = _encoded({"name": filename})
        (tmp_path / filename).write_bytes(data)
        descriptors[filename] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    partitions = {
        role: [_request(role)] for role in PARTITION_EVIDENCE
    }
    manifest = build_manifest(
        store,
        partitions,
        descriptors,
        coverage_code_identity(repository),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(_encoded(manifest))
    return manifest_path, manifest


def test_manifest_verifier_requires_complete_bound_archive(tmp_path):
    manifest_path, manifest = _valid_archive(tmp_path)
    assert audit_manifest(manifest_path)["status"] == "verified"

    manifest["outputs"] = {}
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="output inventory"):
        audit_manifest(manifest_path)


def test_manifest_verifier_rejects_output_symlink_escape(tmp_path):
    manifest_path, _ = _valid_archive(tmp_path)
    output = tmp_path / COVERAGE_OUTPUT_NAMES[0]
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_bytes(output.read_bytes())
    output.unlink()
    output.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        audit_manifest(manifest_path)


def test_manifest_verifier_rejects_stale_code_and_inputs(tmp_path):
    manifest_path, manifest = _valid_archive(tmp_path)
    manifest["code"]["sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="producer identity"):
        audit_manifest(manifest_path)

    manifest_path, manifest = _valid_archive(tmp_path / "second")
    manifest["inputs"].pop(next(iter(manifest["inputs"])))
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="input bindings"):
        audit_manifest(manifest_path)


def test_code_identity_covers_normalization_and_resolution_helpers():
    assert "experiments/kev_adapter.py" in CODE_IDENTITY_FILES
    assert "tools/evidence_registry.py" in CODE_IDENTITY_FILES
