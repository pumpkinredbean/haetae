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
PRIVATE_MARKERS = (
    "/users/",
    "/home/",
    "/private/",
    "/var/",
    "/tmp/",
    "/volumes/",
    "file://",
    "chatgpt.com/c/",
    "chatgpt.com/share/",
    "chat.openai.com/c/",
    "chat.openai.com/share/",
    "\\users\\",
)
REGISTRY_FIELDS = {
    "schema_version",
    "base_commit",
    "reference_source_identity",
    "artifacts",
    "protected_history",
}
REFERENCE_FIELDS = {"sha256", "files"}
ORIGINAL_ARTIFACT_FIELDS = {
    "kind",
    "location_key",
    "required",
    "contains_locked_data",
    "sha256",
    "size_bytes",
    "claims",
}
DERIVATIVE_ARTIFACT_FIELDS = ORIGINAL_ARTIFACT_FIELDS | {
    "source_evidence_id",
    "source_sha256",
}


class EvidenceError(ValueError):
    """Raised when evidence identity or provenance validation fails."""


class EvidenceSourceError(EvidenceError):
    """Raised when an evidence item's registered ancestor is not verified."""


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


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _assert_stable_file(
    path: Path,
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    try:
        current = path.stat()
    except OSError as error:
        raise EvidenceError(f"{label} changed while it was being read") from error
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(current)
    ):
        raise EvidenceError(f"{label} changed while it was being read")


def _read_bytes_snapshot(path: Path, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise EvidenceError(f"{label} is not readable") from error
    _assert_stable_file(path, before, after, label)
    return data


def _sha256_file_snapshot(path: Path, label: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise EvidenceError(f"{label} is not readable") from error
    _assert_stable_file(path, before, after, label)
    return before.st_size, digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def _load_json(path: Path, label: str) -> dict:
    return _load_json_bytes(_read_bytes_snapshot(path, label), label)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise EvidenceError(f"{label} is not a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise EvidenceError(f"{label} is a placeholder digest")
    return value


def _validate_public_content(value: Any, label: str = "public content") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f"{label} contains a non-string key")
            _validate_public_content(key, f"{label} key")
            _validate_public_content(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_public_content(nested, f"{label}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("~/") or re.search(r"[a-z]:\\", lowered):
            raise EvidenceError(f"{label} contains a private filesystem path")
        if any(marker in lowered for marker in PRIVATE_MARKERS):
            raise EvidenceError(f"{label} contains a private location or conversation link")
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise EvidenceError(f"{label} contains a non-JSON value")


def _require_exact_fields(value: dict, expected: set[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise EvidenceError(f"{label} fields differ from the checked-in schema")


def _validate_registry(registry: dict) -> None:
    _require_exact_fields(registry, REGISTRY_FIELDS, "registry")
    if type(registry.get("schema_version")) is not int or registry["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("registry schema version is unsupported")
    base_commit = registry.get("base_commit")
    if not isinstance(base_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", base_commit
    ):
        raise EvidenceError("base_commit is not a full lowercase Git commit")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EvidenceError("registry artifacts must be a nonempty object")
    _validate_public_content(registry, "registry")
    for evidence_id, entry in artifacts.items():
        if not isinstance(evidence_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", evidence_id
        ):
            raise EvidenceError("registry contains an invalid evidence ID")
        if not isinstance(entry, dict):
            raise EvidenceError(f"{evidence_id} entry must be an object")
        kind = entry.get("kind")
        expected_fields = (
            DERIVATIVE_ARTIFACT_FIELDS if kind == "derivative"
            else ORIGINAL_ARTIFACT_FIELDS
        )
        _require_exact_fields(entry, expected_fields, f"{evidence_id} entry")
        if entry.get("location_key") != evidence_id:
            raise EvidenceError(f"{evidence_id} location key must equal its evidence ID")
        if kind not in {"original", "derivative"}:
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
        if not isinstance(claims, list) or not claims or not all(
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
    _require_exact_fields(reference, REFERENCE_FIELDS, "reference_source_identity")
    _require_sha256(reference.get("sha256"), "reference_source_identity.sha256")
    files = reference.get("files")
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(relative, str) for relative in files)
        or len(files) != len(set(files))
    ):
        raise EvidenceError("reference source file list is missing")
    for relative in files:
        _validate_relative_path(relative, "reference source file")
    for evidence_id in artifacts:
        visited = set()
        current = evidence_id
        while artifacts[current]["kind"] == "derivative":
            if current in visited:
                raise EvidenceError("derivative provenance contains a cycle")
            visited.add(current)
            current = artifacts[current]["source_evidence_id"]


def _validate_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value == ".":
        raise EvidenceError(f"{label} is not a relative path")
    relative = PurePosixPath(value)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
    ):
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


def _reject_symlink_components(root: Path, relative: PurePosixPath, label: str) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceError(f"{label} traverses a symbolic link")


def resolve_explicit_artifact(
    paths: dict,
    evidence_id: str,
) -> Path:
    """Resolve one explicitly mapped artifact within its declared allowlist."""
    if (
        type(paths.get("schema_version")) is not int
        or paths["schema_version"] != SCHEMA_VERSION
    ):
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
    _reject_symlink_components(root, relative, f"{evidence_id} local path")
    candidate = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise EvidenceError(f"{evidence_id} local path escapes its root") from error
    canonical_prefixes = []
    for prefix in allowlist:
        _reject_symlink_components(root, prefix, f"artifact root allowlist for {evidence_id}")
        canonical = (root / Path(*prefix.parts)).resolve(strict=False)
        try:
            canonical.relative_to(root)
        except ValueError as error:
            raise EvidenceError(f"{evidence_id} allowlist escapes its root") from error
        canonical_prefixes.append(canonical)
    if not any(
        candidate == prefix or prefix in candidate.parents
        for prefix in canonical_prefixes
    ):
        raise EvidenceError(f"{evidence_id} canonical path is outside its allowlist")
    return candidate


def _reference_source_identity(repository: Path, files: list[str]) -> str:
    repository = repository.resolve(strict=True)
    digest = hashlib.sha256()
    for relative_text in files:
        relative = _validate_relative_path(relative_text, "reference source file")
        _reject_symlink_components(repository, relative, "reference source file")
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
    branch = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if branch.returncode == 0:
        raise EvidenceError("reference repository must use detached HEAD")
    if branch.returncode != 1:
        raise EvidenceError("reference repository HEAD state is unreadable")
    reference = registry["reference_source_identity"]
    actual = _reference_source_identity(repository, reference["files"])
    if actual != reference["sha256"]:
        raise EvidenceError("reference source identity differs from registry")
    writable_files = []
    writable_directories = set()
    for relative_text in reference["files"]:
        path = repository / relative_text
        if path.stat().st_mode & 0o222:
            writable_files.append(relative_text)
        directory = path.parent
        while True:
            if directory.stat().st_mode & 0o222:
                writable_directories.add(str(directory.relative_to(repository)) or ".")
            if directory == repository:
                break
            directory = directory.parent
    if writable_files:
        raise EvidenceError("reference source files are not read-only")
    if writable_directories:
        raise EvidenceError("reference source directories are writable")
    return {
        "status": "verified",
        "commit": expected_commit,
        "source_sha256": actual,
        "files": len(reference["files"]),
        "accidental_mutation_guard": {
            "detached_head": True,
            "source_files_nonwritable": True,
            "source_directories_nonwritable": True,
        },
    }


def _validate_derivative_wrapper(
    data: bytes,
    evidence_id: str,
    entry: dict,
) -> None:
    wrapper = _load_json_bytes(data, f"{evidence_id} derivative")
    _require_exact_fields(
        wrapper,
        {"schema_version", "kind", "source_evidence", "payload"},
        f"{evidence_id} derivative",
    )
    if type(wrapper["schema_version"]) is not int or wrapper["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError(f"{evidence_id} derivative schema version is unsupported")
    if wrapper["kind"] != "public_derivative":
        raise EvidenceError(f"{evidence_id} derivative kind is invalid")
    source = wrapper["source_evidence"]
    if not isinstance(source, dict):
        raise EvidenceError(f"{evidence_id} derivative source is invalid")
    _require_exact_fields(source, {"evidence_id", "sha256"}, "derivative source")
    if (
        source["evidence_id"] != entry["source_evidence_id"]
        or source["sha256"] != entry["source_sha256"]
    ):
        raise EvidenceError(f"{evidence_id} derivative provenance differs from registry")
    _validate_public_content(wrapper, f"{evidence_id} derivative")


def _capture_registered_artifact(
    path: Path,
    evidence_id: str,
    entry: dict,
) -> dict:
    label = f"{evidence_id} evidence"
    if entry["kind"] == "derivative":
        data = _read_bytes_snapshot(path, label)
        actual_size = len(data)
        actual_sha256 = hashlib.sha256(data).hexdigest()
    else:
        actual_size, actual_sha256 = _sha256_file_snapshot(path, label)
        data = None
    if actual_size != entry["size_bytes"] or actual_sha256 != entry["sha256"]:
        raise EvidenceError(f"{evidence_id} bytes differ from registry")
    return {
        "sha256": actual_sha256,
        "size_bytes": actual_size,
        "data": data,
    }


def _validate_captured_chain(
    registry: dict,
    evidence_id: str,
    captures: dict[str, dict],
    validated: set[str],
) -> None:
    if evidence_id in validated:
        return
    entry = registry["artifacts"][evidence_id]
    if evidence_id not in captures:
        raise EvidenceSourceError(f"{evidence_id} is not verified")
    if entry["kind"] == "derivative":
        source_id = entry["source_evidence_id"]
        try:
            _validate_captured_chain(registry, source_id, captures, validated)
        except EvidenceError as error:
            raise EvidenceSourceError(
                f"{evidence_id} source chain is not verified: {error}"
            ) from error
        data = captures[evidence_id]["data"]
        if not isinstance(data, bytes):
            raise EvidenceError(f"{evidence_id} derivative snapshot is missing")
        _validate_derivative_wrapper(data, evidence_id, entry)
    validated.add(evidence_id)


def _capture_evidence_chain(
    registry: dict,
    paths: dict,
    evidence_id: str,
    captures: dict[str, dict],
) -> None:
    if evidence_id in captures:
        return
    entry = registry["artifacts"][evidence_id]
    if entry["kind"] == "derivative":
        _capture_evidence_chain(
            registry,
            paths,
            entry["source_evidence_id"],
            captures,
        )
    artifact = resolve_explicit_artifact(paths, evidence_id)
    if not artifact.is_file():
        raise EvidenceSourceError(f"{evidence_id} evidence is unavailable")
    captures[evidence_id] = _capture_registered_artifact(
        artifact,
        evidence_id,
        entry,
    )


def verify_registry(registry: dict, paths: dict) -> dict:
    """Verify every mapped artifact and return a path-free report."""
    _validate_registry(registry)
    if (
        type(paths.get("schema_version")) is not int
        or paths["schema_version"] != SCHEMA_VERSION
    ):
        raise EvidenceError("local path map schema version is unsupported")
    results_by_id = {}
    captures = {}
    failures = []
    for evidence_id in sorted(registry["artifacts"]):
        entry = registry["artifacts"][evidence_id]
        try:
            artifact = resolve_explicit_artifact(paths, evidence_id)
        except EvidenceError as error:
            failures.append(f"{evidence_id}: {error}")
            results_by_id[evidence_id] = {
                "evidence_id": evidence_id,
                "status": "invalid_mapping",
            }
            continue
        if not artifact.is_file():
            status = "missing_required" if entry["required"] else "unavailable"
            results_by_id[evidence_id] = {
                "evidence_id": evidence_id,
                "status": status,
            }
            if entry["required"]:
                failures.append(f"{evidence_id}: required file is missing")
            continue
        try:
            capture = _capture_registered_artifact(artifact, evidence_id, entry)
        except EvidenceError as error:
            results_by_id[evidence_id] = {
                "evidence_id": evidence_id,
                "status": "mismatch",
            }
            failures.append(f"{evidence_id}: {error}")
            continue
        captures[evidence_id] = capture
        results_by_id[evidence_id] = {
            "evidence_id": evidence_id,
            "status": "verified",
            "sha256": capture["sha256"],
            "size_bytes": capture["size_bytes"],
            "kind": entry["kind"],
        }
    validated_chains = set()
    for evidence_id in sorted(registry["artifacts"]):
        entry = registry["artifacts"][evidence_id]
        if entry["kind"] != "derivative":
            continue
        result = results_by_id[evidence_id]
        if result["status"] != "verified":
            continue
        try:
            _validate_captured_chain(
                registry,
                evidence_id,
                captures,
                validated_chains,
            )
        except EvidenceSourceError:
            result.clear()
            result.update({"evidence_id": evidence_id, "status": "source_unavailable"})
            if entry["required"]:
                failures.append(f"{evidence_id}: derivative source is not verified")
        except EvidenceError as error:
            result.clear()
            result.update({"evidence_id": evidence_id, "status": "invalid_derivative"})
            failures.append(f"{evidence_id}: {error}")
    results = [results_by_id[evidence_id] for evidence_id in sorted(results_by_id)]
    reference = _verify_reference(registry, paths)
    report = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": registry["base_commit"],
        "registry_sha256": hashlib.sha256(canonical_json_bytes(registry)).hexdigest(),
        "reference": reference,
        "artifacts": results,
        "verified_count": sum(row["status"] == "verified" for row in results),
        "unavailable_optional_count": sum(
            row["status"] in {"unavailable", "source_unavailable"}
            for row in results
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
    captures = {}
    _capture_evidence_chain(
        registry,
        paths,
        source_evidence_id,
        captures,
    )
    _validate_captured_chain(
        registry,
        source_evidence_id,
        captures,
        set(),
    )
    _validate_output_destination(registry, paths, output_path)
    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "kind": "public_derivative",
        "source_evidence": {
            "evidence_id": source_evidence_id,
            "sha256": source["sha256"],
        },
        "payload": payload,
    }
    _validate_public_content(wrapper, "public derivative")
    encoded = canonical_json_bytes(wrapper) + b"\n"
    _atomic_write_new(output_path, encoded)
    return {
        "source_evidence_id": source_evidence_id,
        "source_sha256": source["sha256"],
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _validate_output_destination(
    registry: dict,
    paths: dict,
    output_path: Path,
    input_paths: tuple[Path, ...] = (),
) -> None:
    resolved_output = output_path.resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise EvidenceError("output path already exists")
    for evidence_id in paths.get("artifacts", {}):
        evidence_path = resolve_explicit_artifact(paths, evidence_id).resolve(strict=False)
        if resolved_output == evidence_path:
            raise EvidenceError("output path is a registered evidence location")
    for input_path in input_paths:
        if resolved_output == input_path.resolve(strict=False):
            raise EvidenceError("output path is a verification input")
    descriptor = paths.get("reference_repository")
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise EvidenceError("local path map has no valid reference repository")
    reference = Path(descriptor["path"]).resolve(strict=True)
    try:
        resolved_output.relative_to(reference)
    except ValueError:
        pass
    else:
        raise EvidenceError("output path is inside the reference repository")


def _atomic_write_new(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise EvidenceError("output path was created by another writer") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_new(path, encoded)


def _self_test() -> dict:
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
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
        subprocess.run(
            ["git", "-C", str(repository), "checkout", "--detach", "-q"],
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
        repository.chmod(0o555)

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
                    "allow_prefixes": [
                        "original.json",
                        "optional.json",
                        "missing.json",
                        "allowed",
                        "derived.json",
                        "downstream.json",
                        "mid.json",
                        "tip.json",
                        "race.json",
                    ],
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

        def must_fail_containing(name: str, expected: str, action) -> None:
            try:
                action()
            except EvidenceError as error:
                if expected not in str(error):
                    raise AssertionError(
                        f"self-test rejected {name} for the wrong reason"
                    ) from error
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
        moved.rename(artifact)
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
        checks.append("missing allowed optional")
        invalid_optional_paths = copy.deepcopy(optional_paths)
        invalid_optional_paths["artifacts"]["optional"]["relative_path"] = "../outside.json"
        must_fail(
            "invalid optional mapping",
            lambda: verify_registry(optional, invalid_optional_paths),
        )

        for name, mutation in (
            ("unknown schema", lambda value: value.update(schema_version=9)),
            ("Boolean schema version", lambda value: value.update(schema_version=True)),
            ("unexpected registry field", lambda value: value.update(unexpected=True)),
        ):
            candidate = copy.deepcopy(registry)
            mutation(candidate)
            must_fail(name, lambda candidate=candidate: verify_registry(candidate, paths))
        boolean_paths = copy.deepcopy(paths)
        boolean_paths["schema_version"] = True
        must_fail(
            "Boolean path-map schema version",
            lambda: verify_registry(registry, boolean_paths),
        )
        empty_claims = copy.deepcopy(registry)
        empty_claims["artifacts"]["fixture"]["claims"] = []
        must_fail("empty claims", lambda: verify_registry(empty_claims, paths))
        extra_artifact = copy.deepcopy(registry)
        extra_artifact["artifacts"]["fixture"]["unexpected"] = True
        must_fail("unexpected artifact field", lambda: verify_registry(extra_artifact, paths))
        extra_reference = copy.deepcopy(registry)
        extra_reference["reference_source_identity"]["unexpected"] = True
        must_fail("unexpected reference field", lambda: verify_registry(extra_reference, paths))
        duplicate_reference = copy.deepcopy(registry)
        duplicate_reference["reference_source_identity"]["files"] = [
            "source.py", "source.py",
        ]
        must_fail("duplicate reference file", lambda: verify_registry(duplicate_reference, paths))
        duplicate_json = root / "duplicate.json"
        duplicate_json.write_text('{"schema_version":1,"schema_version":1}\n')
        must_fail("duplicate JSON key", lambda: _load_json(duplicate_json, "fixture"))

        escaped = copy.deepcopy(paths)
        escaped["artifacts"]["fixture"]["relative_path"] = "../escape.json"
        must_fail("path escape", lambda: verify_registry(registry, escaped))
        dot_allowlist = copy.deepcopy(paths)
        dot_allowlist["roots"]["fixture"]["allow_prefixes"] = ["."]
        must_fail("root-wide dot allowlist", lambda: verify_registry(registry, dot_allowlist))
        allowed = artifact_root / "allowed"
        forbidden = artifact_root / "not_allowed"
        allowed.mkdir()
        forbidden.mkdir()
        hidden = forbidden / "evidence.json"
        hidden.write_bytes(artifact.read_bytes())
        (allowed / "link.json").symlink_to(hidden)
        symlinked = copy.deepcopy(paths)
        symlinked["artifacts"]["fixture"]["relative_path"] = "allowed/link.json"
        must_fail("internal symlink escape", lambda: verify_registry(registry, symlinked))

        placeholder = copy.deepcopy(registry)
        placeholder["artifacts"]["fixture"]["sha256"] = "0" * 64
        must_fail("placeholder digest", lambda: verify_registry(placeholder, paths))
        leaked = copy.deepcopy(registry)
        leaked["artifacts"]["fixture"]["claims"] = ["/Users/private/result.json"]
        must_fail("private registry path", lambda: verify_registry(leaked, paths))

        derivative = root / "public" / "summary.json"
        identity = write_public_derivative(
            registry, paths, "fixture", {"metric": 1.0}, derivative
        )
        wrapper = _load_json(derivative, "derivative fixture")
        assert wrapper["source_evidence"]["sha256"] == entry["sha256"]
        assert identity["file_sha256"] != entry["sha256"]
        checks.append("derivative provenance")
        must_fail(
            "private derivative value",
            lambda: write_public_derivative(
                registry,
                paths,
                "fixture",
                {"source": "/home/example/private.json"},
                root / "public" / "private-value.json",
            ),
        )
        must_fail(
            "private derivative key",
            lambda: write_public_derivative(
                registry,
                paths,
                "fixture",
                {"chatgpt.com/share/private": "value"},
                root / "public" / "private-key.json",
            ),
        )
        must_fail(
            "protected overwrite",
            lambda: write_public_derivative(
                registry, paths, "fixture", {"metric": 2.0}, artifact
            ),
        )
        must_fail(
            "existing derivative output",
            lambda: write_public_derivative(
                registry, paths, "fixture", {"metric": 2.0}, derivative
            ),
        )
        original_bytes = artifact.read_bytes()
        artifact.write_text('{"value":2}\n', encoding="utf-8")
        must_fail(
            "changed derivative source",
            lambda: write_public_derivative(
                registry,
                paths,
                "fixture",
                {"metric": 2.0},
                root / "public" / "changed-source.json",
            ),
        )
        artifact.write_bytes(original_bytes)

        must_fail(
            "verification output over evidence",
            lambda: _validate_output_destination(registry, paths, artifact),
        )
        must_fail(
            "verification output over input",
            lambda: _validate_output_destination(
                registry, paths, duplicate_json, input_paths=(duplicate_json,)
            ),
        )
        must_fail(
            "verification output in reference",
            lambda: _validate_output_destination(
                registry, paths, repository / "report.json"
            ),
        )
        occupied = root / "occupied.json"
        occupied.write_text("occupied\n", encoding="utf-8")
        must_fail("atomic no-clobber", lambda: _atomic_write_new(occupied, b"changed\n"))

        wrong_wrapper = {
            "schema_version": SCHEMA_VERSION,
            "kind": "public_derivative",
            "source_evidence": {
                "evidence_id": "unrelated",
                "sha256": entry["sha256"],
            },
            "payload": {"metric": 1.0},
        }
        wrong_path = artifact_root / "derived.json"
        wrong_path.write_bytes(canonical_json_bytes(wrong_wrapper) + b"\n")
        derivative_registry = copy.deepcopy(registry)
        derivative_registry["artifacts"]["derived"] = {
            "kind": "derivative",
            "location_key": "derived",
            "required": True,
            "contains_locked_data": False,
            "sha256": sha256_file(wrong_path),
            "size_bytes": wrong_path.stat().st_size,
            "claims": ["derived fixture"],
            "source_evidence_id": "fixture",
            "source_sha256": entry["sha256"],
        }
        derivative_paths = copy.deepcopy(paths)
        derivative_paths["artifacts"]["derived"] = {
            "root": "fixture",
            "relative_path": "derived.json",
        }
        must_fail(
            "mismatched derivative wrapper",
            lambda: verify_registry(derivative_registry, derivative_paths),
        )
        downstream_wrapper = {
            "schema_version": SCHEMA_VERSION,
            "kind": "public_derivative",
            "source_evidence": {
                "evidence_id": "derived",
                "sha256": derivative_registry["artifacts"]["derived"]["sha256"],
            },
            "payload": {"metric": 2.0},
        }
        downstream_path = artifact_root / "downstream.json"
        downstream_path.write_bytes(canonical_json_bytes(downstream_wrapper) + b"\n")
        derivative_registry["artifacts"]["a_downstream"] = {
            "kind": "derivative",
            "location_key": "a_downstream",
            "required": True,
            "contains_locked_data": False,
            "sha256": sha256_file(downstream_path),
            "size_bytes": downstream_path.stat().st_size,
            "claims": ["downstream fixture"],
            "source_evidence_id": "derived",
            "source_sha256": derivative_registry["artifacts"]["derived"]["sha256"],
        }
        derivative_paths["artifacts"]["a_downstream"] = {
            "root": "fixture",
            "relative_path": "downstream.json",
        }
        must_fail_containing(
            "transitive derivative source failure",
            "a_downstream: derivative source is not verified",
            lambda: verify_registry(derivative_registry, derivative_paths),
        )
        must_fail(
            "derivative writer rejects invalid ancestor",
            lambda: write_public_derivative(
                derivative_registry,
                derivative_paths,
                "a_downstream",
                {"metric": 3.0},
                root / "public" / "invalid-ancestor.json",
            ),
        )

        mid_wrapper = {
            "schema_version": SCHEMA_VERSION,
            "kind": "public_derivative",
            "source_evidence": {
                "evidence_id": "fixture",
                "sha256": entry["sha256"],
            },
            "payload": {"metric": 1.0},
        }
        mid_path = artifact_root / "mid.json"
        mid_path.write_bytes(canonical_json_bytes(mid_wrapper) + b"\n")
        chain_registry = copy.deepcopy(registry)
        chain_registry["artifacts"]["mid"] = {
            "kind": "derivative",
            "location_key": "mid",
            "required": True,
            "contains_locked_data": False,
            "sha256": sha256_file(mid_path),
            "size_bytes": mid_path.stat().st_size,
            "claims": ["middle fixture"],
            "source_evidence_id": "fixture",
            "source_sha256": entry["sha256"],
        }
        tip_wrapper = {
            "schema_version": SCHEMA_VERSION,
            "kind": "public_derivative",
            "source_evidence": {
                "evidence_id": "mid",
                "sha256": chain_registry["artifacts"]["mid"]["sha256"],
            },
            "payload": {"metric": 2.0},
        }
        tip_path = artifact_root / "tip.json"
        tip_path.write_bytes(canonical_json_bytes(tip_wrapper) + b"\n")
        chain_registry["artifacts"]["tip"] = {
            "kind": "derivative",
            "location_key": "tip",
            "required": True,
            "contains_locked_data": False,
            "sha256": sha256_file(tip_path),
            "size_bytes": tip_path.stat().st_size,
            "claims": ["tip fixture"],
            "source_evidence_id": "mid",
            "source_sha256": chain_registry["artifacts"]["mid"]["sha256"],
        }
        chain_paths = copy.deepcopy(paths)
        chain_paths["artifacts"].update(
            {
                "mid": {"root": "fixture", "relative_path": "mid.json"},
                "tip": {"root": "fixture", "relative_path": "tip.json"},
            }
        )
        write_public_derivative(
            chain_registry,
            chain_paths,
            "tip",
            {"metric": 3.0},
            root / "public" / "valid-chain.json",
        )
        checks.append("derivative writer validates complete chain")
        original_bytes = artifact.read_bytes()
        artifact.write_text('{"value":2}\n', encoding="utf-8")
        must_fail(
            "derivative writer rejects changed ancestor",
            lambda: write_public_derivative(
                chain_registry,
                chain_paths,
                "tip",
                {"metric": 4.0},
                root / "public" / "changed-ancestor.json",
            ),
        )
        artifact.write_bytes(original_bytes)

        race_bad_wrapper = {
            "schema_version": SCHEMA_VERSION,
            "kind": "public_derivative",
            "source_evidence": {
                "evidence_id": "unrelated",
                "sha256": entry["sha256"],
            },
            "payload": {"metric": 1.0},
        }
        race_good_wrapper = copy.deepcopy(race_bad_wrapper)
        race_good_wrapper["source_evidence"]["evidence_id"] = "fixture"
        race_path = artifact_root / "race.json"
        race_bad_bytes = canonical_json_bytes(race_bad_wrapper) + b"\n"
        race_good_bytes = canonical_json_bytes(race_good_wrapper) + b"\n"
        race_path.write_bytes(race_bad_bytes)
        race_registry = copy.deepcopy(registry)
        race_registry["artifacts"]["race"] = {
            "kind": "derivative",
            "location_key": "race",
            "required": True,
            "contains_locked_data": False,
            "sha256": hashlib.sha256(race_bad_bytes).hexdigest(),
            "size_bytes": len(race_bad_bytes),
            "claims": ["race fixture"],
            "source_evidence_id": "fixture",
            "source_sha256": entry["sha256"],
        }
        race_paths = copy.deepcopy(paths)
        race_paths["artifacts"]["race"] = {
            "root": "fixture",
            "relative_path": "race.json",
        }
        original_reader = _read_bytes_snapshot

        def replace_after_read(path: Path, label: str) -> bytes:
            data = original_reader(path, label)
            if path == race_path:
                replacement = artifact_root / "race-replacement.tmp"
                replacement.write_bytes(race_good_bytes)
                os.replace(replacement, race_path)
            return data

        globals()["_read_bytes_snapshot"] = replace_after_read
        try:
            must_fail(
                "verification parses hashed derivative snapshot",
                lambda: verify_registry(race_registry, race_paths),
            )
            race_path.write_bytes(race_bad_bytes)
            must_fail(
                "writer parses hashed derivative snapshot",
                lambda: write_public_derivative(
                    race_registry,
                    race_paths,
                    "race",
                    {"metric": 2.0},
                    root / "public" / "raced-source.json",
                ),
            )
        finally:
            globals()["_read_bytes_snapshot"] = original_reader

        cyclic = copy.deepcopy(registry)
        cyclic["artifacts"] = {
            "first": {
                "kind": "derivative",
                "location_key": "first",
                "required": True,
                "contains_locked_data": False,
                "sha256": "1" * 64,
                "size_bytes": 1,
                "claims": ["first"],
                "source_evidence_id": "second",
                "source_sha256": "2" * 64,
            },
            "second": {
                "kind": "derivative",
                "location_key": "second",
                "required": True,
                "contains_locked_data": False,
                "sha256": "2" * 64,
                "size_bytes": 1,
                "claims": ["second"],
                "source_evidence_id": "first",
                "source_sha256": "1" * 64,
            },
        }
        cyclic["protected_history"] = []
        must_fail("cyclic derivative provenance", lambda: _validate_registry(cyclic))

        repository.chmod(0o755)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "switch",
                "-q",
                "-c",
                "attached-test",
            ],
            check=True,
        )
        must_fail("attached reference HEAD", lambda: verify_registry(registry, paths))
        subprocess.run(
            ["git", "-C", str(repository), "checkout", "--detach", "-q"],
            check=True,
        )
        must_fail("writable reference directory", lambda: verify_registry(registry, paths))
        repository.chmod(0o555)
        assert verify_registry(registry, paths)["status"] == "verified"
        checks.append("restored reference guard")
        repository.chmod(0o755)
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
    _validate_output_destination(
        registry,
        paths,
        args.out,
        input_paths=(args.registry, args.paths),
    )
    report = verify_registry(registry, paths)
    _write_json_new(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "verified_count": report["verified_count"],
        "unavailable_optional_count": report["unavailable_optional_count"],
        "registry_sha256": report["registry_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
