import json

import pytest
import torch

import tools.diagnose_mps_copy_only as diagnostic
from experiments.coverage_v1.registry import canonical_sha256
from tools.diagnose_mps_copy_only import (
    execute,
    json_safe_values,
    run_cases,
    write_result,
)


def test_cpu_reference_exercises_the_complete_case_inventory():
    cases = run_cases(torch.device("cpu"))
    assert len(cases) == 18
    assert {row["length"] for row in cases} == {2, 6, 7}
    assert {row["offset"] for row in cases} == {0, 1, 256}
    assert {row["context"] for row in cases} == {
        "no_grad",
        "inference_mode",
    }
    assert all(row["passed"] for row in cases)


def test_result_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "result.json"
    descriptor = write_result(output, {"status": "fixture"})
    assert descriptor["size_bytes"] == output.stat().st_size
    with pytest.raises(FileExistsError):
        write_result(output, {"status": "replacement"})


def test_nonfinite_values_use_strict_json_tokens():
    values = json_safe_values(torch.tensor([
        float("nan"), float("inf"), float("-inf"), 1.25,
    ], dtype=torch.float64))
    assert values == ["NaN", "+Infinity", "-Infinity", 1.25]


def test_nonfinite_case_publishes_complete_failure_report(
    monkeypatch,
    tmp_path,
):
    real_transfer = diagnostic.fused_transfer
    calls = 0

    def inject_one_nan(source):
        nonlocal calls
        result = real_transfer(source)
        if calls == 0:
            result = result.clone()
            result[0] = float("nan")
        calls += 1
        return result

    monkeypatch.setattr(diagnostic, "fused_transfer", inject_one_nan)
    output = tmp_path / "failure.json"
    result, descriptor = execute(
        output,
        torch.device("cpu"),
        {"device": "cpu fixture"},
    )
    assert result["status"] == "failed"
    assert result["all_passed"] is False
    assert sum(row["passed"] for row in result["cases"]) == 17
    assert result["cases"][0]["fused"][0] == "NaN"
    assert descriptor["size_bytes"] == output.stat().st_size

    loaded = json.loads(output.read_text())
    expected = loaded.pop("result_sha256")
    assert expected == canonical_sha256(loaded)
