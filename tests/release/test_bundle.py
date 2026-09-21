import copy
import json
from pathlib import Path

import pytest

import haetae.bundle as bundle_module
from haetae.bundle import (
    BUNDLE_FILES,
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    SHARED_V1_IDENTITY,
    BundleError,
    canonical_sha256,
    file_sha256,
    verify_bundle,
)


def write_bundle(root: Path) -> dict:
    content = {
        "config.json": b"{}",
        "model.safetensors": b"data",
        "special_tokens_map.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for name, data in content.items():
        (root / name).write_bytes(data)
    tensor_index = {"head.key.bias": {"dtype": "float32", "shape": [2]}}
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "format": BUNDLE_FORMAT,
        "model": dict(SHARED_V1_IDENTITY),
        "architecture_source": {
            "path": "experiments/shared_state.py",
            "sha256": "6" * 64,
            "base_commit": "7" * 40,
            "runtime_sha256": "8" * 64,
        },
        "files": {
            name: {
                "sha256": file_sha256(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in BUNDLE_FILES
        },
        "tensor_index": tensor_index,
        "tensor_index_sha256": canonical_sha256(tensor_index),
        "model_forwards": 0,
        "optimizer_state_included": False,
        "random_state_included": False,
        "locked_test_opened": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    return manifest


def rewrite_manifest(root: Path, manifest: dict) -> None:
    value = copy.deepcopy(manifest)
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = canonical_sha256(value)
    (root / "manifest.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    )


def trust_fixture(monkeypatch, manifest):
    monkeypatch.setattr(
        bundle_module,
        "EXPECTED_BUNDLE_MANIFEST_SHA256",
        manifest["manifest_sha256"],
    )


def test_verified_bundle_accepts_exact_inventory(tmp_path, monkeypatch):
    manifest = write_bundle(tmp_path)
    trust_fixture(monkeypatch, manifest)
    bundle = verify_bundle(tmp_path)
    assert bundle.manifest == manifest
    assert set(bundle.files) == set(BUNDLE_FILES)


def test_bundle_rejects_changed_file(tmp_path, monkeypatch):
    trust_fixture(monkeypatch, write_bundle(tmp_path))
    (tmp_path / "config.json").write_text('{"changed":true}')
    with pytest.raises(BundleError, match="file differs"):
        verify_bundle(tmp_path)


def test_bundle_rejects_extra_file_and_symlink(tmp_path, monkeypatch):
    trust_fixture(monkeypatch, write_bundle(tmp_path))
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(BundleError, match="inventory differs"):
        verify_bundle(tmp_path)
    (tmp_path / "extra").unlink()
    (tmp_path / "config.json").unlink()
    (tmp_path / "config.json").symlink_to(tmp_path / "tokenizer.json")
    with pytest.raises(BundleError, match="symbolic link"):
        verify_bundle(tmp_path)


def test_bundle_rejects_false_export_boundary_claim(tmp_path, monkeypatch):
    manifest = write_bundle(tmp_path)
    manifest["optimizer_state_included"] = True
    rewrite_manifest(tmp_path, manifest)
    trust_fixture(
        monkeypatch,
        json.loads((tmp_path / "manifest.json").read_text()),
    )
    with pytest.raises(BundleError, match="export boundary"):
        verify_bundle(tmp_path)


def test_bundle_rejects_unreviewed_self_consistent_manifest(tmp_path):
    manifest = write_bundle(tmp_path)
    assert manifest["manifest_sha256"] != bundle_module.EXPECTED_BUNDLE_MANIFEST_SHA256
    with pytest.raises(BundleError, match="reviewed release candidate"):
        verify_bundle(tmp_path)
