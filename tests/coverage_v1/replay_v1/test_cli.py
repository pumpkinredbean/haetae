from __future__ import annotations

import pytest

from experiments.coverage_v1.replay_v1.__main__ import parser


def test_freeze_has_no_recipe_overrides() -> None:
    arguments = parser().parse_args([
        "freeze", "--paths", "paths.json", "--registry", "registry.json",
        "--out", "freeze",
    ])
    assert vars(arguments) == {
        "command": "freeze", "paths": "paths.json", "registry": "registry.json",
        "out": "freeze",
    }
    with pytest.raises(SystemExit):
        parser().parse_args([
            "freeze", "--paths", "paths.json", "--registry", "registry.json",
            "--out", "freeze", "--steps", "513",
        ])


def test_reviewed_training_requires_all_exact_identities() -> None:
    base = [
        "train", "--plan", "plan", "--paths", "paths", "--seed", "17",
        "--arm", "control", "--run", "run", "--initial-pair", "pair",
    ]
    with pytest.raises(SystemExit):
        parser().parse_args(base)
    arguments = parser().parse_args(base + [
        "--reviewed-commit", "1" * 40,
        "--reviewed-protocol-sha256", "2" * 64,
        "--reviewed-manifest-sha256", "3" * 64,
        "--allow-reviewed-training",
    ])
    assert arguments.resume_current is False


def test_seed_and_arm_are_closed_enumerations() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args([
            "initialize", "--plan", "plan", "--paths", "paths", "--seed", "99",
            "--arm", "control", "--run", "run", "--reviewed-commit", "1" * 40,
            "--reviewed-protocol-sha256", "2" * 64,
            "--reviewed-manifest-sha256", "3" * 64,
            "--allow-reviewed-inference",
        ])
