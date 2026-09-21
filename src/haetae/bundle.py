"""Strict verification for portable shared-v1 model bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FORMAT = "haetae-shared-v1-safetensors"
BUNDLE_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "format",
    "model",
    "architecture_source",
    "files",
    "tensor_index",
    "tensor_index_sha256",
    "model_forwards",
    "optimizer_state_included",
    "random_state_included",
    "locked_test_opened",
    "manifest_sha256",
}
MODEL_FIELDS = {
    "run_id",
    "spec_sha256",
    "generation",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "backbone",
    "backbone_revision",
    "model_config_fingerprint",
    "tokenizer_fingerprint",
    "pointer_size",
    "maximum_length",
    "attention_implementation",
    "parameter_dtype",
}
ARCHITECTURE_FIELDS = {"path", "sha256", "base_commit", "runtime_sha256"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHARED_V1_IDENTITY = {
    "run_id": "71d06d48-a3cf-4ddc-bfe2-2315dbe66da8",
    "spec_sha256": "2797fc5a985c315f035658790c6ba582d634fe2ceeaa2d8fc5d35e78a100e9d5",
    "generation": 79,
    "checkpoint_sha256": (
        "e9c782407912242c34d4da88557bded76e92e1222090d7e25f974dafab588d5c"
    ),
    "checkpoint_size_bytes": 1_687_348_267,
    "backbone": "jhu-clsp/mmBERT-small",
    "backbone_revision": "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
    "model_config_fingerprint": (
        "c7de60b92d29058f37e1272cb548ef212841fee04b365ea46924df069edb6c51"
    ),
    "tokenizer_fingerprint": (
        "eedf10dda29c5fda59481b04cab55da64c03a545b8b437a33ea73562b88a5ad2"
    ),
    "pointer_size": 128,
    "maximum_length": 2048,
    "attention_implementation": "eager",
    "parameter_dtype": "float32",
}
EXPECTED_BUNDLE_MANIFEST_SHA256 = (
    "b47d19dc98b06719934174c1cc51e359039cdc539077a273352c8254184ef8e0"
)


class BundleError(ValueError):
    """Raised when a portable bundle is incomplete or untrusted."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def expected_safetensors_metadata() -> dict[str, str]:
    identity = {
        "format": BUNDLE_FORMAT,
        "generation": SHARED_V1_IDENTITY["generation"],
        "run_id": SHARED_V1_IDENTITY["run_id"],
    }
    return {"haetae": canonical_json_bytes(identity).decode("utf-8")}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_json_bytes(data: bytes, label: str) -> dict:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except BundleError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise BundleError(f"{label} is not a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise BundleError(f"{label} is a placeholder SHA-256 digest")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BundleError(f"{label} must be a positive integer")
    return value


def _resolved_regular_file(root: Path, filename: str) -> Path:
    if filename not in BUNDLE_FILES and filename != "manifest.json":
        raise BundleError(f"bundle filename is not allowed: {filename}")
    candidate = root / filename
    if candidate.is_symlink():
        raise BundleError(f"bundle file cannot be a symbolic link: {filename}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise BundleError(f"bundle file is unavailable: {filename}") from error
    if resolved.parent != root or not resolved.is_file():
        raise BundleError(f"bundle file escapes its directory: {filename}")
    return resolved


def _validate_tensor_index(value: Any) -> None:
    if not isinstance(value, dict) or not value:
        raise BundleError("tensor index must be a nonempty object")
    for name, descriptor in value.items():
        if not isinstance(name, str) or not name:
            raise BundleError("tensor index contains an invalid name")
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape"}:
            raise BundleError(f"tensor index descriptor differs: {name}")
        if not isinstance(descriptor["dtype"], str) or not descriptor["dtype"]:
            raise BundleError(f"tensor dtype is invalid: {name}")
        shape = descriptor["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise BundleError(f"tensor shape is invalid: {name}")


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest: dict
    files: dict[str, Path]


def verify_bundle(directory: str | Path) -> VerifiedBundle:
    supplied = Path(directory)
    if supplied.is_symlink():
        raise BundleError("bundle directory cannot be a symbolic link")
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise BundleError("bundle directory is unavailable") from error
    if not root.is_dir():
        raise BundleError("bundle path is not a directory")
    names = {path.name for path in root.iterdir()}
    expected_names = set(BUNDLE_FILES) | {"manifest.json"}
    if names != expected_names:
        raise BundleError("bundle file inventory differs")

    manifest_path = _resolved_regular_file(root, "manifest.json")
    manifest_data = manifest_path.read_bytes()
    manifest = load_json_bytes(manifest_data, "bundle manifest")
    if set(manifest) != MANIFEST_FIELDS:
        raise BundleError("bundle manifest fields differ")
    unsigned = dict(manifest)
    expected_manifest_sha256 = unsigned.pop("manifest_sha256", None)
    _require_sha256(expected_manifest_sha256, "manifest_sha256")
    if expected_manifest_sha256 != canonical_sha256(unsigned):
        raise BundleError("bundle manifest self-digest differs")
    if expected_manifest_sha256 != EXPECTED_BUNDLE_MANIFEST_SHA256:
        raise BundleError("bundle manifest is not the reviewed release candidate")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("format") != BUNDLE_FORMAT
    ):
        raise BundleError("bundle manifest identity differs")
    if (
        manifest.get("model_forwards") != 0
        or manifest.get("optimizer_state_included") is not False
        or manifest.get("random_state_included") is not False
        or manifest.get("locked_test_opened") is not False
    ):
        raise BundleError("bundle exceeds the export boundary")

    model = manifest.get("model")
    if not isinstance(model, dict) or set(model) != MODEL_FIELDS:
        raise BundleError("bundle model identity fields differ")
    for field in (
        "run_id", "spec_sha256", "checkpoint_sha256",
        "model_config_fingerprint", "tokenizer_fingerprint",
    ):
        if field == "run_id":
            if not isinstance(model[field], str) or not model[field]:
                raise BundleError("bundle run ID is invalid")
        else:
            _require_sha256(model[field], f"model.{field}")
    if not isinstance(model.get("backbone_revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", model["backbone_revision"]
    ):
        raise BundleError("model.backbone_revision is not a Git commit digest")
    for field in (
        "generation", "checkpoint_size_bytes", "pointer_size", "maximum_length",
    ):
        _require_positive_integer(model.get(field), f"model.{field}")
    for field in ("backbone", "attention_implementation", "parameter_dtype"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise BundleError(f"model.{field} is invalid")
    if model != SHARED_V1_IDENTITY:
        differences = sorted(
            key for key in MODEL_FIELDS
            if model.get(key) != SHARED_V1_IDENTITY.get(key)
        )
        raise BundleError(f"bundle shared-v1 identity differs: {differences}")

    architecture = manifest.get("architecture_source")
    if not isinstance(architecture, dict) or set(architecture) != ARCHITECTURE_FIELDS:
        raise BundleError("bundle architecture identity fields differ")
    if architecture.get("path") != "experiments/shared_state.py":
        raise BundleError("bundle architecture source path differs")
    for field in ("sha256", "runtime_sha256"):
        _require_sha256(architecture.get(field), f"architecture_source.{field}")
    if not isinstance(architecture.get("base_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", architecture["base_commit"]
    ):
        raise BundleError("bundle architecture base commit is invalid")

    _validate_tensor_index(manifest.get("tensor_index"))
    if manifest.get("tensor_index_sha256") != canonical_sha256(
        manifest["tensor_index"]
    ):
        raise BundleError("bundle tensor index digest differs")

    descriptors = manifest.get("files")
    if not isinstance(descriptors, dict) or set(descriptors) != set(BUNDLE_FILES):
        raise BundleError("bundle manifest file inventory differs")
    files = {}
    for filename in BUNDLE_FILES:
        descriptor = descriptors[filename]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "sha256", "size_bytes",
        }:
            raise BundleError(f"bundle file descriptor differs: {filename}")
        _require_sha256(descriptor.get("sha256"), f"files.{filename}.sha256")
        _require_positive_integer(
            descriptor.get("size_bytes"),
            f"files.{filename}.size_bytes",
        )
        file_path = _resolved_regular_file(root, filename)
        if (
            file_path.stat().st_size != descriptor["size_bytes"]
            or file_sha256(file_path) != descriptor["sha256"]
        ):
            raise BundleError(f"bundle file differs: {filename}")
        files[filename] = file_path
    return VerifiedBundle(root=root, manifest=manifest, files=files)
