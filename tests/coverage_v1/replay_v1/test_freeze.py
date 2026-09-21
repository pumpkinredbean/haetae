from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.freeze import freeze, verify_plan
from experiments.coverage_v1.replay_v1.inputs import EMOTION_LABELS, OFFENSIVE_LABELS
from tests.coverage_v1.replay_v1.test_tapes import (
    encoder,
    normalized_request,
    target_request,
)


@pytest.fixture(scope="module")
def freeze_pools() -> tuple[list[dict], list[dict]]:
    original = [normalized_request(index) for index in range(15572)]
    target = []
    for label_index, label in enumerate(EMOTION_LABELS):
        target += [
            target_request(label_index * 256 + index, "emotion", label)
            for index in range(256)
        ]
    for label_index, label in enumerate(OFFENSIVE_LABELS):
        target += [
            target_request(label_index * 256 + index, "tweet_offensive", label)
            for index in range(256)
        ]
    return original, target


def evaluation_inventory() -> list[dict]:
    counts = {"decision": 1199, "transfer": 764, "korean": 2000, "secondary": 904}
    rows = []
    for track, count in counts.items():
        for index in range(count):
            rows.append({
                "track": track,
                "request_uid": digest_object([track, index]),
                "question_id": "answer",
                "row_sha256": digest_object([track, index, "answer"]),
            })
    return rows


def test_freeze_is_repeatable_and_complete(tmp_path: Path, monkeypatch, freeze_pools) -> None:
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.freeze.freeze_source_identity",
        lambda repository: {"files": [], "sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.freeze._head", lambda repository: "2" * 40,
    )
    spec = json.loads(Path("plans/next-v1/t05-v1/implementation-spec.json").read_text())
    inventory = evaluation_inventory()
    arguments = {
        "repository": Path.cwd(),
        "spec": spec,
        "runtime": {"device": "mps", "parameter_dtype": "float32"},
        "input_bindings": {"fixture": {"sha256": "3" * 64, "size_bytes": 1}},
        "original_pool": freeze_pools[0],
        "target_pool": freeze_pools[1],
        "exclusion_audit": {"state_overlap": 0, "parent_overlap": 0},
        "evaluation_inventory": inventory,
        "zero_reference": [{"key": row["row_sha256"]} for row in inventory],
        "zero_reference_metrics": {"rows": len(inventory)},
        "bootstrap_strata": {"populations": {}},
        "encoding_budget": {"maximum_length": 2048},
        "historical_source_digests": {"files": {}},
        "encoder": encoder,
    }
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = freeze(out=first, **arguments)
    freeze(out=second, **arguments)
    assert manifest["pre_m4_counters"]["optimizer_updates"] == 0
    assert [path.name for path in sorted(first.iterdir())] == [path.name for path in sorted(second.iterdir())]
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert verify_plan(first)["manifest_sha256"] == manifest["manifest_sha256"]


def test_verify_rejects_tape_substitution(tmp_path: Path, monkeypatch, freeze_pools) -> None:
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.freeze.freeze_source_identity",
        lambda repository: {"files": [], "sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.freeze._head", lambda repository: "2" * 40,
    )
    spec = json.loads(Path("plans/next-v1/t05-v1/implementation-spec.json").read_text())
    inventory = evaluation_inventory()
    plan = tmp_path / "plan"
    freeze(
        repository=Path.cwd(), out=plan, spec=spec, runtime={}, input_bindings={},
        original_pool=freeze_pools[0], target_pool=freeze_pools[1], exclusion_audit={},
        evaluation_inventory=inventory,
        zero_reference=[{"key": row["row_sha256"]} for row in inventory],
        zero_reference_metrics={}, bootstrap_strata={}, encoding_budget={},
        historical_source_digests={}, encoder=encoder,
    )
    tape = plan / "tape-s17-control.jsonl"
    tape.write_bytes(tape.read_bytes().replace(b'"seed":17', b'"seed":23', 1))
    with pytest.raises(ContractError, match="descriptor mismatch"):
        verify_plan(plan)
