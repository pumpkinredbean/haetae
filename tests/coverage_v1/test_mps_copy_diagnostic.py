import pytest
import torch

from tools.diagnose_mps_copy_only import run_cases, write_result


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
