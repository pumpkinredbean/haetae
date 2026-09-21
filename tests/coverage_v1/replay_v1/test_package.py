from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from experiments.coverage_v1.replay_v1.contracts import ContractError
from experiments.coverage_v1.replay_v1.freeze import FROZEN_FILES
from experiments.coverage_v1.replay_v1.package import build_m4_package, verify_package


def test_review_package_is_indexed_and_excludes_checkpoint_payload(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    for name in FROZEN_FILES:
        (plan / name).write_text("{}\n")
    (plan / "protocol.json").write_text(json.dumps({
        "producer_commit": "1" * 40, "protocol_sha256": "2" * 64,
    }))
    (plan / "manifest.json").write_text("{}\n")
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.package.verify_plan",
        lambda value: {
            "manifest_sha256": "3" * 64,
            "pre_m4_counters": {
                "checkpoint_deserializations": 0, "model_forwards": 0,
                "mps_operations": 0, "optimizer_updates": 0,
                "locked_test_accesses": 0,
            },
        },
    )
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.package._run_check",
        lambda repository, command: {
            "argv": command, "returncode": 0, "stdout": "passed", "stderr": "",
        },
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"public":true}\n')
    metadata = tmp_path / "tokenizer.json"
    metadata.write_text("{}\n")
    output = tmp_path / "review.zip"
    result = build_m4_package(
        repository=Path.cwd(), plan=plan, out=output,
        evidence_files={"shared/evidence.json": evidence},
        model_metadata_files={"tokenizer.json": metadata},
        archive_map={
            "checkpoint": {
                "included": False,
                "descriptor": {"sha256": "4" * 64, "size_bytes": 100},
            },
        },
    )
    assert result["indexed_payloads"] > 0
    with zipfile.ZipFile(output) as archive:
        assert not any(name.endswith(".pt") for name in archive.namelist())
        assert "model-metadata/g79-checkpoint-descriptor.json" in archive.namelist()


def test_package_verifier_rejects_traversal(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside", b"bad")
        archive.writestr("review-index.json", b"{}")
    with pytest.raises(ContractError, match="unsafe"):
        verify_package(path)
