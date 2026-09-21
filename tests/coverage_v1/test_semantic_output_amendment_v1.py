from types import SimpleNamespace

import pytest
import torch

import experiments.coverage_v1.semantic_output_amendment_v1 as amendment
from experiments.coverage_v1.semantic_output_amendment_v1 import (
    _probe_replay,
    freeze,
    infer_semantic_schedule,
    parity_result,
    require_reviewed_identity,
    staged_observation,
)


def semantic_fixture(reference=(3.0, 1.0)):
    variants = []
    references = {}
    for index in range(amendment.EXPECTED_VARIANTS):
        family = "original" if index < amendment.EXPECTED_ORIGINALS else "alias_prefix"
        variant_id = f"variant-{index:04d}"
        variants.append({
            "variant_id": variant_id,
            "development_id": f"request-{index:04d}",
            "family": family,
            "state": f"state {index}",
            "question": {
                "id": "q",
                "options": ["left", "right"],
                "option_keys": ["left", "right"],
                "label": 0,
            },
        })
        if family == "original":
            references[variant_id] = {
                "key": [f"request-{index:04d}", "q"],
                "reference_logits": list(reference),
            }
    observations = [
        {
            "variant_id": row["variant_id"],
            "status": "complete",
            "cpu_float32": list(reference),
            "logits": list(reference),
        }
        for row in variants
    ]
    return variants, references, observations


def test_staged_observation_copies_before_widening(monkeypatch):
    trace = []

    class Widened:
        def tolist(self):
            return [2.0, -1.0]

    class CpuValues:
        device = SimpleNamespace(type="cpu")
        dtype = torch.float32
        ndim = 1

        def numel(self):
            return 2

        def to(self, *, dtype):
            assert dtype == torch.float64
            trace.append("to-float64")
            return Widened()

        def tolist(self):
            return [2.0, -1.0]

    class Source:
        def detach(self):
            trace.append("detach")
            return self

        def cpu(self):
            trace.append("cpu")
            return CpuValues()

    monkeypatch.setattr(
        amendment.torch,
        "isfinite",
        lambda value: SimpleNamespace(all=lambda: True),
    )
    monkeypatch.setattr(amendment.torch, "equal", lambda left, right: True)
    result = staged_observation("v", Source(), 2)
    assert trace[:2] == ["detach", "cpu"]
    assert trace[2:] == ["to-float64", "to-float64"]
    assert result == {
        "variant_id": "v",
        "status": "complete",
        "cpu_float32": [2.0, -1.0],
        "logits": [2.0, -1.0],
    }


def test_staged_observation_preserves_nonfinite_failure():
    result = staged_observation(
        "v",
        torch.tensor([float("nan"), float("inf")], dtype=torch.float32),
        2,
    )
    assert result == {
        "variant_id": "v",
        "status": "invalid_cpu_float32",
        "cpu_float32": ["NaN", "+Infinity"],
        "logits": None,
    }


def test_parity_accepts_honest_serialized_evidence_and_rejects_all_zero():
    variants, references, observations = semantic_fixture()
    result = parity_result(variants, observations, references)
    assert result["passed"] is True
    assert result["original_rows"] == amendment.EXPECTED_ORIGINALS
    assert result["temperature_summaries"]["1.0"]["maximum_total_variation"] == 0

    corrupted = [dict(row) for row in observations]
    for row in corrupted[:amendment.EXPECTED_ORIGINALS]:
        row["cpu_float32"] = [0.0, 0.0]
        row["logits"] = [0.0, 0.0]
    failed = parity_result(variants, corrupted, references)
    assert failed["passed"] is False
    assert failed["raw_components_passed"] is False


def test_parity_rejects_inventory_substitution_and_threshold_violation():
    variants, references, observations = semantic_fixture()
    with pytest.raises(ValueError, match="inventory differs"):
        parity_result(variants, observations[:-1], references)
    reordered = list(observations)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="order differs"):
        parity_result(variants, reordered, references)

    changed = [dict(row) for row in observations]
    changed[0]["cpu_float32"] = [4.0, 1.0]
    changed[0]["logits"] = [4.0, 1.0]
    assert parity_result(variants, changed, references)["passed"] is False


@pytest.mark.parametrize(
    ("cpu_float32", "logits"),
    [
        ([3.0, 1.0], [100.0, -100.0]),
        ([["NaN"], 1.0], [3.0, 1.0]),
        ([0.1, 1.0], [float(torch.tensor(0.1, dtype=torch.float32)), 1.0]),
    ],
)
def test_parity_rejects_invalid_non_original_numerical_evidence(
    cpu_float32,
    logits,
):
    variants, references, observations = semantic_fixture()
    row = observations[amendment.EXPECTED_ORIGINALS]
    row["cpu_float32"] = cpu_float32
    row["logits"] = logits
    with pytest.raises(ValueError):
        parity_result(variants, observations, references)


def test_parity_reports_near_tie_action_change_without_dropping_it():
    variants, references, observations = semantic_fixture(reference=(0.0, 0.0))
    small_float32 = 2 ** -17
    observations[0]["cpu_float32"] = [0.0, small_float32]
    observations[0]["logits"] = [0.0, small_float32]
    result = parity_result(variants, observations, references)
    assert result["passed"] is True
    assert result["material_action_changes"] == 0
    assert result["near_tie_action_changes"] == ["variant-0000"]


def test_semantic_schedule_uses_exactly_550_model_calls(monkeypatch, tmp_path):
    variants, references, _ = semantic_fixture()
    monkeypatch.setattr(
        amendment,
        "encode_request",
        lambda *args, **kwargs: SimpleNamespace(
            state_tokens=1,
            retained_state_tokens=1,
        ),
    )

    class Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, encodings):
            self.calls += 1
            return [
                [torch.tensor([3.0, 1.0], dtype=torch.float32)]
                for _ in encodings
            ]

    counters = {
        "attempted_model_forward_calls": 0,
        "completed_model_forward_calls": 0,
        "completed_semantic_sequences": 0,
    }
    model = Model()
    observations = infer_semantic_schedule(
        model,
        object(),
        {},
        variants,
        tmp_path / "semantic-logits.jsonl",
        references,
        counters,
        tmp_path / "original-parity.json",
    )
    assert model.calls == amendment.MAXIMUM_FORWARD_CALLS
    assert len(observations) == amendment.EXPECTED_VARIANTS
    assert counters == {
        "attempted_model_forward_calls": amendment.MAXIMUM_FORWARD_CALLS,
        "completed_model_forward_calls": amendment.MAXIMUM_FORWARD_CALLS,
        "completed_semantic_sequences": amendment.EXPECTED_VARIANTS,
    }
    with pytest.raises(FileExistsError):
        infer_semantic_schedule(
            model,
            object(),
            {},
            variants,
            tmp_path / "semantic-logits.jsonl",
            references,
            {
                "attempted_model_forward_calls": 0,
                "completed_model_forward_calls": 0,
                "completed_semantic_sequences": 0,
            },
        )


def test_semantic_schedule_aborts_and_preserves_failed_original(monkeypatch, tmp_path):
    variants, references, _ = semantic_fixture()
    monkeypatch.setattr(
        amendment,
        "encode_request",
        lambda *args, **kwargs: SimpleNamespace(
            state_tokens=1,
            retained_state_tokens=1,
        ),
    )

    class AllZeroModel:
        def __call__(self, encodings):
            return [
                [torch.tensor([0.0, 0.0], dtype=torch.float32)]
                for _ in encodings
            ]

    counters = {
        "attempted_model_forward_calls": 0,
        "completed_model_forward_calls": 0,
        "completed_semantic_sequences": 0,
    }
    parity_path = tmp_path / "original-parity.json"
    with pytest.raises(RuntimeError, match="parity failed"):
        infer_semantic_schedule(
            AllZeroModel(),
            object(),
            {},
            variants,
            tmp_path / "semantic-logits.jsonl",
            references,
            counters,
            parity_path,
        )
    assert counters == {
        "attempted_model_forward_calls": 1,
        "completed_model_forward_calls": 1,
        "completed_semantic_sequences": 1,
    }
    assert parity_path.exists()
    assert '"failed_during_execution"' in parity_path.read_text()


def test_probe_replay_uses_predeclared_numerical_tolerances():
    expected = {
        "accuracy": 0.5,
        "nll": 1.0,
        "multiclass_brier": 0.4,
        "calibration": {"temperature": 1.5},
    }
    actual = {
        "accuracy": 0.5,
        "nll": 1.0 + 5e-8,
        "multiclass_brier": 0.4 - 5e-8,
        "calibration": {"temperature": 1.5 + 5e-7},
    }
    replay = _probe_replay(expected, actual)
    assert replay["maximum_nll_or_brier_delta"] == pytest.approx(5e-8)
    assert replay["maximum_calibrator_parameter_delta"] == pytest.approx(5e-7)
    with pytest.raises(ValueError, match="probe score replay"):
        _probe_replay(expected, {**actual, "nll": 1.0 + 2e-7})


def test_reviewed_identity_rejects_before_other_work():
    manifest = {"manifest_sha256": "a" * 64}
    protocol = {"protocol_sha256": "b" * 64}
    require_reviewed_identity(manifest, protocol, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="reviewed identity"):
        require_reviewed_identity(manifest, protocol, "c" * 64, "b" * 64)


def test_freeze_refuses_an_existing_output_before_reading_inputs(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        freeze(
            tmp_path / "missing-plan",
            tmp_path / "missing-held",
            tmp_path / "missing-copy",
            tmp_path / "missing-paths",
            tmp_path / "missing-local",
            tmp_path / "missing-registry",
            output,
        )


def _self_digested(value, field):
    result = dict(value)
    result[field] = amendment.canonical_sha256(result)
    return result


@pytest.mark.parametrize(
    ("leaf", "digest_field"),
    [
        ("manifest.json", "manifest_sha256"),
        ("protocol.json", "protocol_sha256"),
    ],
)
def test_read_freeze_rejects_leaf_symlinks(tmp_path, leaf, digest_field):
    root = tmp_path / "freeze"
    root.mkdir()
    values = {
        "manifest.json": _self_digested({}, "manifest_sha256"),
        "protocol.json": _self_digested({}, "protocol_sha256"),
    }
    other = ({"manifest.json", "protocol.json"} - {leaf}).pop()
    amendment._write_json(root / other, values[other])
    target = tmp_path / f"outside-{leaf}"
    amendment._write_json(target, values[leaf])
    (root / leaf).symlink_to(target)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        amendment.read_freeze(root)


def _write_complete_output_boundary(root, observations):
    root.mkdir()
    amendment._write_jsonl(root / "semantic-logits.jsonl", observations)
    for name in amendment.OUTPUT_FILES - {"manifest.json", "semantic-logits.jsonl"}:
        amendment._write_json(root / name, {})
    files = {
        name: (
            amendment._descriptor_with_rows(root / name)
            if name.endswith(".jsonl")
            else amendment._descriptor(root / name)
        )
        for name in sorted(amendment.OUTPUT_FILES - {"manifest.json"})
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "amendment_id": amendment.AMENDMENT_ID,
        "protocol_sha256": "p",
        "files": files,
        "new_semantic_sequences": amendment.EXPECTED_VARIANTS,
        "new_model_forward_calls": amendment.MAXIMUM_FORWARD_CALLS,
        "feature_extraction_forwards": 0,
        "original_pretrained_model_loads": 0,
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }
    amendment._write_json(
        root / "manifest.json",
        _self_digested(manifest, "manifest_sha256"),
    )


@pytest.mark.parametrize(
    ("cpu_float32", "logits"),
    [
        ([3.0, 1.0], [100.0, -100.0]),
        ([["NaN"], 1.0], [3.0, 1.0]),
    ],
)
def test_final_verifier_rejects_invalid_non_original_numerical_evidence(
    monkeypatch,
    tmp_path,
    cpu_float32,
    logits,
):
    variants, references, observations = semantic_fixture()
    row = observations[amendment.EXPECTED_ORIGINALS]
    row["cpu_float32"] = cpu_float32
    row["logits"] = logits
    output = tmp_path / "output"
    _write_complete_output_boundary(output, observations)
    monkeypatch.setattr(
        amendment,
        "verify_freeze",
        lambda *args, **kwargs: (
            {},
            {"protocol_sha256": "p"},
            {"variants": variants, "references": references},
        ),
    )

    with pytest.raises(ValueError):
        amendment.verify_output(
            *(tmp_path / name for name in (
                "amendment", "plan", "held", "copy", "paths", "local", "registry",
            )),
            output,
        )


def test_final_verifier_rejects_manifest_leaf_symlink(monkeypatch, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for name in amendment.OUTPUT_FILES - {"manifest.json"}:
        (output / name).write_text("{}\n")
    target = tmp_path / "outside-manifest.json"
    target.write_text("{}\n")
    (output / "manifest.json").symlink_to(target)
    monkeypatch.setattr(
        amendment,
        "verify_freeze",
        lambda *args, **kwargs: ({}, {"protocol_sha256": "p"}, {}),
    )

    with pytest.raises(ValueError, match="cannot be a symlink"):
        amendment.verify_output(
            *(tmp_path / name for name in (
                "amendment", "plan", "held", "copy", "paths", "local", "registry",
            )),
            output,
        )
