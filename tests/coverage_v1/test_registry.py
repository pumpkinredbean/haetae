import hashlib

import pytest

from experiments.coverage_v1.registry import ArtifactStore, load_json_bytes


def test_store_rejects_unregistered_and_locked_inputs(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    data = b'{"state":"safe","questions":{},"_meta":{}}\n'
    (root / "train.jsonl").write_bytes(data)
    registry = {
        "artifacts": {
            "shared_suite_train": {
                "kind": "original",
                "contains_locked_data": False,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            },
            "shared_suite_calibration": {
                "kind": "original",
                "contains_locked_data": True,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            },
        }
    }
    paths = {
        "schema_version": 1,
        "roots": {
            "fixture": {
                "path": str(root),
                "allow_prefixes": ["train.jsonl"],
            }
        },
        "artifacts": {
            "shared_suite_train": {
                "root": "fixture",
                "relative_path": "train.jsonl",
            },
            "shared_suite_calibration": {
                "root": "fixture",
                "relative_path": "train.jsonl",
            },
        },
    }
    store = ArtifactStore(registry, paths)
    assert store.read_bytes("shared_suite_train") == data
    with pytest.raises(PermissionError, match="cannot read"):
        store.read_bytes("shared_checkpoint_g79")
    with pytest.raises(PermissionError, match="locked data"):
        store.read_bytes("shared_suite_calibration")


def test_json_reader_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate key"):
        load_json_bytes(b'{"value":1,"value":2}', "fixture")


def test_store_rejects_changed_registered_bytes(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    expected = b"{}\n"
    (root / "manifest.json").write_bytes(b'{"changed":true}\n')
    registry = {
        "artifacts": {
            "shared_suite_manifest": {
                "kind": "original",
                "contains_locked_data": False,
                "sha256": hashlib.sha256(expected).hexdigest(),
                "size_bytes": len(expected),
            }
        }
    }
    paths = {
        "schema_version": 1,
        "roots": {
            "fixture": {
                "path": str(root),
                "allow_prefixes": ["manifest.json"],
            }
        },
        "artifacts": {
            "shared_suite_manifest": {
                "root": "fixture",
                "relative_path": "manifest.json",
            }
        },
    }
    with pytest.raises(ValueError, match="differs from its registry"):
        ArtifactStore(registry, paths).read_json("shared_suite_manifest")
