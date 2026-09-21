"""Compare fused and staged MPS-to-CPU dtype conversion without model access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from experiments.coverage_v1.registry import canonical_sha256


LENGTHS = (2, 6, 7)
OFFSETS = (0, 1, 256)
CONTEXTS = ("no_grad", "inference_mode")
EXPECTED_RUNTIME = {
    "device": "mps",
    "parameter_dtype": "float32",
    "hardware": {
        "model_identifier": "Mac15,7",
        "chip": "Apple M3 Pro",
        "memory_gib": 36,
    },
    "software": {
        "python": "3.13.2",
        "macos": "26.6.2",
        "torch": "2.14.0",
    },
}


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def runtime_identity() -> dict:
    return {
        "device": "mps",
        "parameter_dtype": "float32",
        "hardware": {
            "model_identifier": _sysctl("hw.model"),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "memory_gib": int(_sysctl("hw.memsize")) // (1024 ** 3),
        },
        "software": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "torch": importlib.metadata.version("torch"),
        },
    }


def _error(error: BaseException) -> dict:
    return {"type": type(error).__name__, "message": str(error)}


def run_case(
    device: torch.device,
    length: int,
    offset: int,
    context_name: str,
) -> dict:
    total = offset + length + 3
    cpu_base = torch.arange(1, total + 1, dtype=torch.float32) * 0.125 - 9.0
    reference_float32 = cpu_base[offset:offset + length].clone()
    reference_float64 = reference_float32.to(dtype=torch.float64)
    device_base = cpu_base.to(device)
    source = device_base[offset:offset + length]
    if device.type == "mps":
        torch.mps.synchronize()

    context = torch.no_grad if context_name == "no_grad" else torch.inference_mode
    fused = None
    staged = None
    fused_error = None
    staged_error = None
    with context():
        try:
            fused = source.detach().to(device="cpu", dtype=torch.float64)
        except BaseException as error:
            fused_error = _error(error)
        try:
            staged = source.detach().cpu().to(dtype=torch.float64)
        except BaseException as error:
            staged_error = _error(error)
    if device.type == "mps":
        torch.mps.synchronize()

    source_after = None
    source_error = None
    try:
        source_after = source.detach().cpu()
    except BaseException as error:
        source_error = _error(error)
    fused_exact = fused is not None and torch.equal(fused, reference_float64)
    staged_exact = staged is not None and torch.equal(staged, reference_float64)
    source_unchanged = (
        source_after is not None
        and torch.equal(source_after, reference_float32)
    )
    storage_offset_matches = source.storage_offset() == offset
    return {
        "case_id": f"{context_name}-length-{length}-offset-{offset}",
        "context": context_name,
        "length": length,
        "offset": offset,
        "actual_storage_offset": source.storage_offset(),
        "storage_offset_matches": storage_offset_matches,
        "reference": reference_float64.tolist(),
        "fused": None if fused is None else fused.tolist(),
        "staged": None if staged is None else staged.tolist(),
        "source_after": (
            None
            if source_after is None
            else source_after.to(dtype=torch.float64).tolist()
        ),
        "fused_error": fused_error,
        "staged_error": staged_error,
        "source_error": source_error,
        "fused_exact": fused_exact,
        "staged_exact": staged_exact,
        "source_unchanged": source_unchanged,
        "passed": bool(
            fused_exact
            and staged_exact
            and source_unchanged
            and storage_offset_matches
        ),
    }


def run_cases(device: torch.device) -> list[dict]:
    return [
        run_case(device, length, offset, context_name)
        for context_name in CONTEXTS
        for length in LENGTHS
        for offset in OFFSETS
    ]


def write_result(path: Path, result: dict) -> dict:
    data = (
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"refusing to overwrite {args.out}")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    runtime = runtime_identity()
    if runtime != EXPECTED_RUNTIME:
        parser.error("runtime differs from the recorded diagnostic environment")

    cases = run_cases(torch.device("mps"))
    result = {
        "schema_version": 1,
        "status": "complete" if all(row["passed"] for row in cases) else "failed",
        "purpose": "Test fused versus staged MPS-to-CPU float64 conversion.",
        "runtime": runtime,
        "case_inventory": {
            "lengths": list(LENGTHS),
            "offsets": list(OFFSETS),
            "contexts": list(CONTEXTS),
            "count": len(cases),
        },
        "cases": cases,
        "all_passed": all(row["passed"] for row in cases),
        "checkpoint_accessed": False,
        "dataset_accessed": False,
        "model_forwards": 0,
        "optimizer_updates": 0,
        "locked_test_opened": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    descriptor = write_result(args.out, result)
    print(json.dumps({"result": result, "file": descriptor}, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
