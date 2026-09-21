"""Private, path-safe M4 review package for T05."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.coverage_v1.replay_v1.contracts import (
    ContractError,
    canonical_bytes,
    digest_object,
    strict_json_loads,
)
from experiments.coverage_v1.replay_v1.freeze import FROZEN_FILES, verify_plan

SOURCE_FILES = (
    "experiments/coverage_v1/replay_v1/__init__.py",
    "experiments/coverage_v1/replay_v1/__main__.py",
    "experiments/coverage_v1/replay_v1/analysis.py",
    "experiments/coverage_v1/replay_v1/contracts.py",
    "experiments/coverage_v1/replay_v1/evaluate.py",
    "experiments/coverage_v1/replay_v1/freeze.py",
    "experiments/coverage_v1/replay_v1/inputs.py",
    "experiments/coverage_v1/replay_v1/package.py",
    "experiments/coverage_v1/replay_v1/state.py",
    "experiments/coverage_v1/replay_v1/tapes.py",
    "experiments/coverage_v1/replay_v1/train.py",
    "experiments/train_shared.py",
    "experiments/shared_state.py",
    "experiments/kev_adapter.py",
    "src/haetae/checkpoint.py",
    "pyproject.toml",
    "uv.lock",
)
TEST_FILES = (
    "tests/coverage_v1/replay_v1/test_inputs.py",
    "tests/coverage_v1/replay_v1/test_tapes.py",
    "tests/coverage_v1/replay_v1/test_freeze.py",
    "tests/coverage_v1/replay_v1/test_state.py",
    "tests/coverage_v1/replay_v1/test_train.py",
    "tests/coverage_v1/replay_v1/test_evaluate.py",
    "tests/coverage_v1/replay_v1/test_analysis.py",
    "tests/coverage_v1/replay_v1/test_cli.py",
    "tests/coverage_v1/replay_v1/test_package.py",
)


def _safe_member(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or any(part in ("", ".", "..") for part in value.parts):
        raise ContractError(f"unsafe archive member: {name}")
    return value


def _descriptor(data: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _git_blob(repository: Path, relative: str) -> str:
    return subprocess.run(
        ["git", "hash-object", relative], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _zip_bytes(files: Mapping[str, bytes], output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(files.items()):
                _safe_member(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_check(repository: Path, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, env={
            **os.environ,
            "PYTHONPATH": ".:src",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        check=False,
    )
    return {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_m4_package(
    *,
    repository: str | Path,
    plan: str | Path,
    out: str | Path,
    evidence_files: Mapping[str, str | Path],
    model_metadata_files: Mapping[str, str | Path],
    archive_map: Mapping[str, Any],
) -> dict[str, Any]:
    repository = Path(repository)
    plan = Path(plan)
    out = Path(out)
    manifest = verify_plan(plan)
    protocol = strict_json_loads((plan / "protocol.json").read_bytes())
    specification = strict_json_loads(
        (repository / "plans/next-v1/t05-v1/implementation-spec.json").read_bytes(),
    )
    files: dict[str, bytes] = {}
    for name in (*FROZEN_FILES, "manifest.json"):
        files[f"freeze/{name}"] = (plan / name).read_bytes()
    files["specification/implementation-spec.json"] = (
        repository / "plans/next-v1/t05-v1/implementation-spec.json"
    ).read_bytes()
    files["specification/artifact-schemas.json"] = (
        repository / "plans/next-v1/t05-v1/artifact-schemas.json"
    ).read_bytes()
    source_inventory = []
    for relative in (*SOURCE_FILES, *TEST_FILES):
        path = repository / relative
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"review source is missing or unsafe: {relative}")
        data = path.read_bytes()
        member = f"source/{relative}"
        files[member] = data
        source_inventory.append({
            "path": relative, **_descriptor(data), "git_blob": _git_blob(repository, relative),
        })
    files["source/inventory.json"] = canonical_bytes({"files": source_inventory}) + b"\n"
    for member, source in sorted(evidence_files.items()):
        member = f"evidence/{member}"
        _safe_member(member)
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ContractError(f"review evidence is missing or unsafe: {source}")
        files[member] = source.read_bytes()
    for member, source in sorted(model_metadata_files.items()):
        member = f"model-metadata/{member}"
        _safe_member(member)
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ContractError(f"model metadata is missing or unsafe: {source}")
        files[member] = source.read_bytes()
    files["model-metadata/g79-checkpoint-descriptor.json"] = canonical_bytes(
        archive_map["checkpoint"],
    ) + b"\n"
    files["maps/archive-map.json"] = canonical_bytes(dict(archive_map)) + b"\n"

    checks = {
        "pytest": _run_check(repository, [
            "uv", "run", "--frozen", "--with", "pytest", "pytest",
            "tests/coverage_v1/replay_v1", "-q",
        ]),
        "ruff": _run_check(repository, [
            "uv", "run", "--frozen", "--with", "ruff", "ruff", "check",
            "experiments/coverage_v1/replay_v1", "tests/coverage_v1/replay_v1",
        ]),
        "compile": _run_check(repository, [
            "python", "-m", "compileall", "-q",
            "experiments/coverage_v1/replay_v1", "tests/coverage_v1/replay_v1",
        ]),
    }
    if any(value["returncode"] != 0 for value in checks.values()):
        raise ContractError("a pre-M4 package check failed")
    files["checks/pre-m4-checks.json"] = canonical_bytes(checks) + b"\n"
    review = (
        "# T05 M4 review\n\n"
        f"Producer commit: `{protocol['producer_commit']}`\n\n"
        f"Specification SHA-256: `{specification['spec_sha256']}`\n\n"
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`\n\n"
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`\n\n"
        "This package authorizes no execution. It contains no checkpoint tensor payload. "
        "All pre-M4 execution counters are zero.\n"
    )
    files["M4_REVIEW.md"] = review.encode("utf-8")
    members = [{"path": name, **_descriptor(data)} for name, data in sorted(files.items())]
    index = {
        "schema_version": 1,
        "kind": "haetae-t05-paired-replay-v1-prerun",
        "producer_commit": protocol["producer_commit"],
        "spec_sha256": specification["spec_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "members": members,
        "pre_m4_counters": manifest["pre_m4_counters"],
        "privacy_scope": "private_exact_review_public_development_only",
    }
    index["index_sha256"] = digest_object(index)
    files["review-index.json"] = canonical_bytes(index) + b"\n"
    _zip_bytes(files, out)
    verification = verify_package(out)
    return {"archive": {**_descriptor(out.read_bytes()), "path": str(out)}, **verification}


def verify_package(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ContractError("review archive contains duplicate members")
        for info in infos:
            _safe_member(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ContractError("review archive contains a symbolic link")
        if archive.testzip() is not None:
            raise ContractError("review archive failed CRC verification")
        if "review-index.json" not in names:
            raise ContractError("review archive has no index")
        index = strict_json_loads(archive.read("review-index.json"))
        if index["index_sha256"] != digest_object(index, "index_sha256"):
            raise ContractError("review index self-digest mismatch")
        expected = {row["path"]: row for row in index["members"]}
        if set(expected) != set(names) - {"review-index.json"}:
            raise ContractError("review index membership differs from archive")
        for name, descriptor in expected.items():
            if _descriptor(archive.read(name)) != {
                "sha256": descriptor["sha256"], "size_bytes": descriptor["size_bytes"],
            }:
                raise ContractError(f"review archive descriptor mismatch: {name}")
    return {
        "members": len(names),
        "indexed_payloads": len(expected),
        "index_sha256": index["index_sha256"],
    }
