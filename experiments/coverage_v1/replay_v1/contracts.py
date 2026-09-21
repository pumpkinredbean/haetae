"""Strict, model-free contracts for the T05 paired-replay experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a frozen artifact violates its declared contract."""


@dataclass(frozen=True)
class CapturedFile:
    """One immutable userspace snapshot used for both hashing and parsing."""

    path: Path
    data: bytes
    sha256: str
    size_bytes: int

    def json(self) -> Any:
        return strict_json_loads(self.data)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_object(value: Any, self_field: str | None = None) -> str:
    if self_field is not None:
        if not isinstance(value, Mapping):
            raise ContractError("a self-digested value must be an object")
        value = {key: item for key, item in value.items() if key != self_field}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number: {value}")


def _validate_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite number at {location}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{location}.{key}")


def strict_json_loads(data: bytes | str) -> Any:
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ContractError("JSON is not strict UTF-8") from error
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("JSON input must be bytes or text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise ContractError("invalid strict JSON") from error
    _validate_finite(value)
    return value


def _schema_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in (int, float) and math.isfinite(value)
    if expected == "boolean":
        return type(value) is bool
    if expected == "null":
        return value is None
    raise ContractError(f"unsupported schema type: {expected}")


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the strict Draft 2020-12 subset used by T05 artifacts."""

    if "$ref" in schema:
        raise ContractError("schema references must be resolved by the caller")
    if "const" in schema and value != schema["const"]:
        raise ContractError(f"{path} differs from the required constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"{path} is outside the allowed enumeration")
    if "oneOf" in schema:
        passes = 0
        for child in schema["oneOf"]:
            try:
                validate_schema(value, child, path)
            except ContractError:
                continue
            passes += 1
        if passes != 1:
            raise ContractError(f"{path} does not match exactly one schema")
    if "anyOf" in schema:
        for child in schema["anyOf"]:
            try:
                validate_schema(value, child, path)
                break
            except ContractError:
                continue
        else:
            raise ContractError(f"{path} does not match an allowed schema")
    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else expected
        if not any(_schema_type(value, item) for item in types):
            raise ContractError(f"{path} has the wrong type")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ContractError(f"{path} is missing fields: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ContractError(f"{path} has unknown fields: {extra}")
        for key, child in properties.items():
            if key in value:
                validate_schema(value[key], child, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ContractError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ContractError(f"{path} has too many items")
        if schema.get("uniqueItems") and len({canonical_bytes(x) for x in value}) != len(value):
            raise ContractError(f"{path} has duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_schema(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ContractError(f"{path} is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ContractError(f"{path} does not match its pattern")
    if type(value) in (int, float):
        if "minimum" in schema and value < schema["minimum"]:
            raise ContractError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ContractError(f"{path} is above its maximum")


def validate_descriptor(descriptor: Mapping[str, Any]) -> None:
    allowed = {
        "artifact_id", "sha256", "size_bytes", "schema_id", "self_sha256",
        "rows", "questions", "archive_member",
    }
    if not isinstance(descriptor, Mapping):
        raise ContractError("artifact descriptor must be an object")
    extra = set(descriptor) - allowed
    if extra:
        raise ContractError(f"artifact descriptor has unknown fields: {sorted(extra)}")
    for field in ("artifact_id", "sha256", "size_bytes"):
        if field not in descriptor:
            raise ContractError(f"artifact descriptor is missing {field}")
    if not isinstance(descriptor["artifact_id"], str) or not descriptor["artifact_id"]:
        raise ContractError("artifact_id must be nonempty text")
    for field in ("sha256", "self_sha256"):
        if field in descriptor and (
            not isinstance(descriptor[field], str)
            or SHA256_PATTERN.fullmatch(descriptor[field]) is None
        ):
            raise ContractError(f"{field} is not a lowercase SHA-256")
    for field in ("size_bytes", "rows", "questions"):
        if field in descriptor and (type(descriptor[field]) is not int or descriptor[field] < 0):
            raise ContractError(f"{field} must be a nonnegative plain integer")
    if "archive_member" in descriptor:
        _validate_relative_path(descriptor["archive_member"])


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    if not isinstance(relative_path, str) or not relative_path:
        raise ContractError("relative path must be nonempty text")
    candidate = PurePosixPath(relative_path)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise ContractError("path traversal or absolute path is forbidden")
    return candidate


def _reject_symlink_chain(root: Path, relative: PurePosixPath) -> Path:
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ContractError("artifact root must be a real directory")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ContractError(f"artifact file is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ContractError(f"symbolic links are forbidden: {relative}")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise ContractError(f"artifact is not a regular file: {relative}")
    return current


def capture_allowed_file(
    root: str | Path,
    relative_path: str,
    descriptor: Mapping[str, Any] | None = None,
) -> CapturedFile:
    """Capture one allowed leaf once, then validate the captured bytes."""

    relative = _validate_relative_path(relative_path)
    path = _reject_symlink_chain(Path(root), relative)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"cannot open artifact safely: {relative}") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("captured artifact is not a regular file")
        pieces: list[bytes] = []
        while True:
            piece = os.read(fd, 1024 * 1024)
            if not piece:
                break
            pieces.append(piece)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ContractError("artifact changed while it was captured")
    data = b"".join(pieces)
    captured = CapturedFile(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    if descriptor is not None:
        validate_descriptor(descriptor)
        if captured.sha256 != descriptor["sha256"] or captured.size_bytes != descriptor["size_bytes"]:
            raise ContractError(f"artifact descriptor mismatch: {relative}")
    return captured


def write_exclusive_json(path: str | Path, value: Any) -> dict[str, Any]:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ContractError("output parent may not be a symbolic link")
    data = canonical_bytes(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def producer_identity(repository: str | Path, files: Iterable[str]) -> dict[str, Any]:
    repository = Path(repository)
    descriptors = []
    for relative in sorted(set(files)):
        captured = capture_allowed_file(repository, relative)
        descriptors.append({
            "path": relative,
            "sha256": captured.sha256,
            "size_bytes": captured.size_bytes,
        })
    return {"files": descriptors, "sha256": digest_object(descriptors)}


def assert_reviewed_identity(
    repository: str | Path,
    *,
    reviewed_commit: str,
    actual_protocol_sha256: str,
    reviewed_protocol_sha256: str,
    actual_manifest_sha256: str,
    reviewed_manifest_sha256: str,
    allow_reviewed_execution: bool,
) -> None:
    if not allow_reviewed_execution:
        raise ContractError("reviewed execution requires an explicit allow flag")
    if re.fullmatch(r"[0-9a-f]{40}", reviewed_commit) is None:
        raise ContractError("reviewed commit is not a full Git SHA")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != reviewed_commit:
        raise ContractError("working commit differs from the reviewed commit")
    if actual_protocol_sha256 != reviewed_protocol_sha256:
        raise ContractError("protocol differs from the reviewed protocol")
    if actual_manifest_sha256 != reviewed_manifest_sha256:
        raise ContractError("manifest differs from the reviewed manifest")
