from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.coverage_v1.replay_v1.contracts import ContractError, digest_object
from experiments.coverage_v1.replay_v1.state import (
    MpsOwnerLock,
    tensor_state_identity,
    validate_t05_checkpoint,
    verify_initial_pair,
)
from haetae.checkpoint import CheckpointError


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.LayerNorm(3))
        self.head = torch.nn.Linear(3, 2)


def initialization(arm: str, identity: dict) -> dict:
    return {
        "protocol_sha256": "1" * 64,
        "seed": 17,
        "arm": arm,
        "run_id": f"run-{arm}",
        "tensor_state_identity": identity,
        "reset_state": {
            "optimizer_moments": 0,
            "scheduler_completed_updates": 0,
            "source_optimizer_inherited": False,
            "source_rng_inherited": False,
        },
        "parity_report": {"passed": True},
    }


def test_tensor_identity_covers_every_named_tensor() -> None:
    model = TinyModel()
    identity = tensor_state_identity(model)
    assert [row["name"] for row in identity["tensors"]] == list(model.state_dict())
    changed = copy.deepcopy(model)
    with torch.no_grad():
        next(changed.parameters()).view(-1)[0] += 1
    assert tensor_state_identity(changed)["sha256"] != identity["sha256"]


def test_initial_pair_requires_tensor_reset_and_parity() -> None:
    identity = tensor_state_identity(TinyModel())
    result = verify_initial_pair(
        initialization("control", identity), initialization("targeted", identity),
        protocol_sha256="1" * 64, seed=17,
    )
    assert result["all_tensors_equal"] is True
    assert result["pair_initialization_sha256"] == digest_object(
        result, "pair_initialization_sha256",
    )
    bad = initialization("targeted", identity)
    bad["reset_state"]["source_rng_inherited"] = True
    with pytest.raises(ContractError, match="not reset"):
        verify_initial_pair(
            initialization("control", identity), bad,
            protocol_sha256="1" * 64, seed=17,
        )


def test_owner_lock_is_nonblocking(tmp_path: Path) -> None:
    path = tmp_path / "mps.lock"
    with (
        MpsOwnerLock(path, "1" * 64),
        pytest.raises(CheckpointError, match="owns"),
        MpsOwnerLock(path, "1" * 64),
    ):
        pass


def test_replay_checkpoint_relations_are_exact(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.coverage_v1.replay_v1.state.validate_checkpoint",
        lambda *args, **kwargs: None,
    )
    step = 3
    prefixes = [digest_object(list(range(index))) for index in range(513)]
    payload = {
        "version": 3, "run_id": "run", "spec_sha256": "2" * 64,
        "producer": {}, "status": "running", "head": {}, "backbone": {},
        "optimizer": {}, "scheduler": {}, "step": step, "skipped": 0,
        "config": {}, "rng": {},
        "data_state": {"epoch": 0, "order": list(range(512)), "cursor": step},
        "replay_state": {
            "protocol_sha256": "1" * 64, "tape_sha256": "3" * 64,
            "seed": 17, "arm": "control", "next_update": step,
            "committed_tape_prefix_sha256": prefixes[step],
            "committed_model_forward_calls": step * 4,
            "committed_optimizer_updates": step,
        },
    }
    validate_t05_checkpoint(
        payload, {}, SimpleNamespace(), None, None, None, device="cpu",
        protocol_sha256="1" * 64, tape_sha256="3" * 64, seed=17,
        arm="control", tape_prefixes=prefixes,
    )
    payload["replay_state"]["next_update"] = 3894
    with pytest.raises(CheckpointError, match="committed tape"):
        validate_t05_checkpoint(
            payload, {}, SimpleNamespace(), None, None, None, device="cpu",
            protocol_sha256="1" * 64, tape_sha256="3" * 64, seed=17,
            arm="control", tape_prefixes=prefixes,
        )
