"""Verify immutable research evidence without exposing local artifact paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_MARKERS = ("/Users/", "file://", "chatgpt.com/c/", "\\Users\\")


class EvidenceError(ValueError):
    """Raised when evidence identity or provenance validation fails."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise EvidenceError(f"{label} is not a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise EvidenceError(f"{label} is a placeholder digest")
    return value


def _reject_private_markers(value: Any, label: str = "registry") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_private_markers(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_private_markers(nested, f"{label}[{index}]")
    elif isinstance(value, str) and any(marker in value for marker in PRIVATE_MARKERS):
        raise EvidenceError(f"{label} contains a private location or conversation link")


def _validate_registry(registry: dict) -> None:
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("registry schema version is unsupported")
    base_commit = registry.get("base_commit")
    if not isinstance(base_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", base_commit
    ):
        raise EvidenceError("base_commit is not a full lowercase Git commit")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EvidenceError("registry artifacts must be a nonempty object")
    _reject_private_markers(registry)
    for evidence_id, entry in artifacts.items():
        if not isinstance(evidence_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", evidence_id
        ):
            raise EvidenceError("registry contains an invalid evidence ID")
        if not isinstance(entry, dict):
            raise EvidenceError(f"{evidence_id} entry must be an object")
        if entry.get("location_key") != evidence_id:
            raise EvidenceError(f"{evidence_id} location key must equal its evidence ID")
        if entry.get("kind") not in {"original", "derivative"}:
            raise EvidenceError(f"{evidence_id} has an invalid evidence kind")
        if not isinstance(entry.get("required"), bool):
            raise EvidenceError(f"{evidence_id} required must be Boolean")
        if entry.get("contains_locked_data") is not False:
            raise EvidenceError(f"{evidence_id} must explicitly exclude locked data")
        _require_sha256(entry.get("sha256"), f"{evidence_id}.sha256")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise EvidenceError(f"{evidence_id} has an invalid size")
        claims = entry.get("claims")
        if not isinstance(claims, list) or not all(
            isinstance(claim, str) and claim for claim in claims
        ):
            raise EvidenceError(f"{evidence_id} claims must be nonempty strings")
        if entry["kind"] == "derivative":
            source_id = entry.get("source_evidence_id")
            source_sha256 = entry.get("source_sha256")
            if source_id not in artifacts or source_id == evidence_id:
                raise EvidenceError(f"{evidence_id} has an invalid derivative source")
            _require_sha256(source_sha256, f"{evidence_id}.source_sha256")
            if artifacts[source_id]["sha256"] != source_sha256:
                raise EvidenceError(f"{evidence_id} source digest differs from registry")
        elif "source_evidence_id" in entry or "source_sha256" in entry:
            raise EvidenceError(f"{evidence_id} original has derivative fields")
    protected = registry.get("protected_history")
    if not isinstance(protected, list) or len(protected) != len(set(protected)):
        raise EvidenceError("protected_history must contain unique evidence IDs")
    for evidence_id in protected:
        if evidence_id not in artifacts:
            raise EvidenceError("protected_history names unknown evidence")
        if artifacts[evidence_id]["kind"] != "original":
            raise EvidenceError("protected_history may contain originals only")
    reference = registry.get("reference_source_identity")
    if not isinstance(reference, dict):
        raise EvidenceError("reference_source_identity is missing")
    _require_sha256(reference.get("sha256"), "reference_source_identity.sha256")
    files = reference.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError("reference source file list is missing")
    for relative in files:
        _validate_relative_path(relative, "reference source file")


def _validate_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} is not a relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise EvidenceError(f"{label} escapes its declared root")
    return relative


def _root_and_allowlist(paths: dict, root_id: str) -> tuple[Path, list[PurePosixPath]]:
    roots = paths.get("roots")
    if not isinstance(roots, dict) or root_id not in roots:
        raise EvidenceError(f"unknown artifact root {root_id}")
    descriptor = roots[root_id]
    if not isinstance(descriptor, dict):
        raise EvidenceError(f"artifact root {root_id} is invalid")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise EvidenceError(f"artifact root {root_id} must use an absolute local path")
    allow_prefixes = descriptor.get("allow_prefixes")
    if not isinstance(allow_prefixes, list) or not allow_prefixes:
        raise EvidenceError(f"artifact root {root_id} has no allowlist")
    return Path(raw_path).resolve(), [
        _validate_relative_path(prefix, f"artifact root {root_id} allowlist")
        for prefix in allow_prefixes
    ]


def resolve_explicit_artifact(
    paths: dict,
    evidence_id: str,
) -> Path:
    """Resolve one explicitly mapped artifact within its declared allowlist."""
    if paths.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("local path map schema version is unsupported")
    mappings = paths.get("artifacts")
    if not isinstance(mappings, dict) or evidence_id not in mappings:
        raise EvidenceError(f"{evidence_id} has no explicit local path mapping")
    mapping = mappings[evidence_id]
    if not isinstance(mapping, dict):
        raise EvidenceError(f"{evidence_id} local path mapping is invalid")
    root_id = mapping.get("root")
    if not isinstance(root_id, str):
        raise EvidenceError(f"{evidence_id} local root is invalid")
    root, allowlist = _root_and_allowlist(paths, root_id)
    relative = _validate_relative_path(
        mapping.get("relative_path"), f"{evidence_id} local path"
    )
    if not any(relative == prefix or prefix in relative.parents for prefix in allowlist):
        raise EvidenceError(f"{evidence_id} local path is outside its allowlist")
    candidate = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise EvidenceError(f"{evidence_id} local path escapes its root") from error
    return candidate


def _reference_source_identity(repository: Path, files: list[str]) -> str:
    repository = repository.resolve(strict=True)
    digest = hashlib.sha256()
    for relative_text in files:
        relative = _validate_relative_path(relative_text, "reference source file")
        source = (repository / Path(*relative.parts)).resolve(strict=True)
        try:
            source.relative_to(repository)
        except ValueError as error:
            raise EvidenceError("reference source file escapes repository") from error
        if not source.is_file():
            raise EvidenceError("reference source file is missing")
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_reference(registry: dict, paths: dict) -> dict:
    descriptor = paths.get("reference_repository")
    if not isinstance(descriptor, dict):
        raise EvidenceError("local path map has no reference repository")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise EvidenceError("reference repository path is invalid")
    repository = Path(raw_path).resolve(strict=True)
    expected_commit = registry["base_commit"]
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode or process.stdout.strip() != expected_commit:
        raise EvidenceError("reference repository commit differs from registry")
    reference = registry["reference_source_identity"]
    actual = _reference_source_identity(repository, reference["files"])
    if actual != reference["sha256"]:
        raise EvidenceError("reference source identity differs from registry")
    writable = []
    for relative_text in reference["files"]:
        path = repository / relative_text
        if path.stat().st_mode & 0o222:
            writable.append(relative_text)
    if writable:
        raise EvidenceError("reference source files are not read-only")
    return {
        "status": "verified",
        "commit": expected_commit,
        "source_sha256": actual,
        "files": len(reference["files"]),
        "read_only": True,
    }


def verify_registry(registry: dict, paths: dict) -> dict:
    """Verify every mapped artifact and return a path-free report."""
    _validate_registry(registry)
    if paths.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("local path map schema version is unsupported")
    results = []
    failures = []
    for evidence_id in sorted(registry["artifacts"]):
        entry = registry["artifacts"][evidence_id]
        try:
            artifact = resolve_explicit_artifact(paths, evidence_id)
        except EvidenceError as error:
            if entry["required"]:
                failures.append(f"{evidence_id}: {error}")
                results.append({"evidence_id": evidence_id, "status": "missing_required"})
            else:
                results.append({"evidence_id": evidence_id, "status": "unavailable"})
            continue
        if not artifact.is_file():
            status = "missing_required" if entry["required"] else "unavailable"
            results.append({"evidence_id": evidence_id, "status": status})
            if entry["required"]:
                failures.append(f"{evidence_id}: required file is missing")
            continue
        actual_size = artifact.stat().st_size
        actual_sha256 = sha256_file(artifact)
        if actual_size != entry["size_bytes"] or actual_sha256 != entry["sha256"]:
            results.append({"evidence_id": evidence_id, "status": "mismatch"})
            failures.append(f"{evidence_id}: bytes differ from registry")
            continue
        results.append(
            {
                "evidence_id": evidence_id,
                "status": "verified",
                "sha256": actual_sha256,
                "size_bytes": actual_size,
                "kind": entry["kind"],
            }
        )
    reference = _verify_reference(registry, paths)
    report = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": registry["base_commit"],
        "registry_sha256": hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
        "reference": reference,
        "artifacts": results,
        "verified_count": sum(row["status"] == "verified" for row in results),
        "unavailable_optional_count": sum(
            row["status"] == "unavailable" for row in results
        ),
        "status": "verified" if not failures else "failed",
    }
    if failures:
        raise EvidenceError("; ".join(failures))
    return report


def write_public_derivative(
    registry: dict,
    paths: dict,
    source_evidence_id: str,
    payload: Any,
    output_path: Path,
) -> dict:
    """Write a new derivative that keeps its original evidence identity explicit."""
    _validate_registry(registry)
    if source_evidence_id not in registry["artifacts"]:
        raise EvidenceError("derivative source is not registered")
    source = registry["artifacts"][source_evidence_id]
    protected_paths = {
        resolve_explicit_artifact(paths, evidence_id).resolve(strict=False)
        for evidence_id in registry["protected_history"]
        if evidence_id in paths.get("artifacts", {})
    }
    resolved_output = output_path.resolve(strict=False)
    if resolved_output in protected_paths:
        raise EvidenceError("derivative output would overwrite protected evidence")
    if output_path.exists():
        raise EvidenceError("derivative output already exists")
    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "kind": "public_derivative",
        "source_evidence": {
            "evidence_id": source_evidence_id,
            "sha256": source["sha256"],
        },
        "payload": payload,
    }
    encoded = canonical_json_bytes(wrapper) + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_evidence_id": source_evidence_id,
        "source_sha256": source["sha256"],
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _self_test() -> dict:
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repository = root / "reference"
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "Evidence Test"],
            check=True,
        )
        (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "source.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reference_sha = _reference_source_identity(repository, ["source.py"])
        (repository / "source.py").chmod(0o444)
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        artifact = artifact_root / "original.json"
        artifact.write_text('{"value":1}\n', encoding="utf-8")
        entry = {
            "kind": "original",
            "location_key": "fixture",
            "required": True,
            "contains_locked_data": False,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
            "claims": ["fixture"],
        }
        registry = {
            "schema_version": SCHEMA_VERSION,
            "base_commit": commit,
            "reference_source_identity": {
                "sha256": reference_sha,
                "files": ["source.py"],
            },
            "artifacts": {"fixture": entry},
            "protected_history": ["fixture"],
        }
        paths = {
            "schema_version": SCHEMA_VERSION,
            "reference_repository": {"path": str(repository)},
            "roots": {
                "fixture": {
                    "path": str(artifact_root),
                    "allow_prefixes": ["original.json", "optional.json"],
                }
            },
            "artifacts": {
                "fixture": {"root": "fixture", "relative_path": "original.json"}
            },
        }

        def must_fail(name: str, action) -> None:
            try:
                action()
            except EvidenceError:
                checks.append(name)
            else:
                raise AssertionError(f"self-test did not reject {name}")

        assert verify_registry(registry, paths)["status"] == "verified"
        checks.append("valid registry")
        changed = copy.deepcopy(registry)
        changed["artifacts"]["fixture"]["sha256"] = "1" * 64
        must_fail("tampered digest", lambda: verify_registry(changed, paths))
        moved = artifact_root / "saved.json"
        artifact.rename(moved)
        must_fail("missing required", lambda: verify_registry(registry, paths))
        moved.rename(artifact_root / "optional.json")
        optional = copy.deepcopy(registry)
        optional_entry = optional["artifacts"].pop("fixture")
        optional_entry["location_key"] = "optional"
        optional_entry["required"] = False
        optional["artifacts"]["optional"] = optional_entry
        optional["protected_history"] = []
        optional_paths = copy.deepcopy(paths)
        optional_paths["artifacts"] = {
            "optional": {"root": "fixture", "relative_path": "missing.json"}
        }
        assert verify_registry(optional, optional_paths)["unavailable_optional_count"] == 1
        checks.append("missing optional")
        artifact = artifact_root / "original.json"
        (artifact_root / "optional.json").rename(artifact)
        unknown = copy.deepcopy(registry)
        unknown["schema_version"] = 9
        must_fail("unknown schema", lambda: verify_registry(unknown, paths))
        escaped = copy.deepcopy(paths)
        escaped["artifacts"]["fixture"]["relative_path"] = "../escape.json"
        must_fail("path escape", lambda: verify_registry(registry, escaped))
        placeholder = copy.deepcopy(registry)
        placeholder["artifacts"]["fixture"]["sha256"] = "0" * 64
        must_fail("placeholder digest", lambda: verify_registry(placeholder, paths))
        leaked = copy.deepcopy(registry)
        leaked["artifacts"]["fixture"]["claims"] = ["/Users/private/result.json"]
        must_fail("private path leakage", lambda: verify_registry(leaked, paths))
        derivative = root / "public" / "summary.json"
        identity = write_public_derivative(
            registry, paths, "fixture", {"metric": 1.0}, derivative
        )
        wrapper = json.loads(derivative.read_text(encoding="utf-8"))
        assert wrapper["source_evidence"]["sha256"] == entry["sha256"]
        assert identity["file_sha256"] != entry["sha256"]
        checks.append("derivative provenance")
        must_fail(
            "protected overwrite",
            lambda: write_public_derivative(
                registry, paths, "fixture", {"metric": 2.0}, artifact
            ),
        )
    return {"status": "passed", "checks": checks, "count": len(checks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--registry", type=Path, default=Path("research/evidence/index.json")
    )
    verify_parser.add_argument("--paths", type=Path, required=True)
    verify_parser.add_argument("--out", type=Path, required=True)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
        return
    registry = _load_json(args.registry, "registry")
    paths = _load_json(args.paths, "local path map")
    report = verify_registry(registry, paths)
    _write_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "verified_count": report["verified_count"],
        "unavailable_optional_count": report["unavailable_optional_count"],
        "registry_sha256": report["registry_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
