"""Read only explicitly registered public-development evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.evidence_registry import resolve_explicit_artifact


PUBLIC_EVIDENCE_IDS = frozenset({
    "baseline_development_report",
    "baseline_run_manifest",
    "baseline_run_spec",
    "comparison_plan",
    "korean_suite_development",
    "korean_suite_manifest",
    "shared_development_report",
    "shared_run_manifest",
    "shared_run_spec",
    "shared_suite_calibration",
    "shared_suite_development",
    "shared_suite_manifest",
    "shared_suite_rendered_state_audit",
    "shared_suite_train",
    "transfer_suite_development",
    "transfer_suite_manifest",
})


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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_json_bytes(data: bytes, label: str) -> dict:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


class ArtifactStore:
    """Digest-check and read the fixed T01 public evidence allowlist."""

    def __init__(self, registry: dict, paths: dict):
        self.registry = registry
        self.paths = paths

    @classmethod
    def from_files(cls, registry_path: Path, paths_path: Path) -> "ArtifactStore":
        registry = load_json_bytes(registry_path.read_bytes(), "evidence registry")
        paths = load_json_bytes(paths_path.read_bytes(), "local path map")
        return cls(registry, paths)

    def binding(self, evidence_id: str) -> dict:
        if evidence_id not in PUBLIC_EVIDENCE_IDS:
            raise PermissionError(f"T01 cannot read evidence {evidence_id!r}")
        entry = self.registry.get("artifacts", {}).get(evidence_id)
        if not isinstance(entry, dict):
            raise ValueError(f"evidence {evidence_id!r} is not registered")
        if entry.get("contains_locked_data") is not False:
            raise PermissionError(f"evidence {evidence_id!r} may contain locked data")
        if entry.get("kind") != "original":
            raise ValueError(f"evidence {evidence_id!r} is not an original artifact")
        return {
            "evidence_id": evidence_id,
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }

    def read_bytes(self, evidence_id: str) -> bytes:
        binding = self.binding(evidence_id)
        path = resolve_explicit_artifact(self.paths, evidence_id)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(f"evidence {evidence_id!r} is unavailable") from error
        if (
            len(data) != binding["size_bytes"]
            or hashlib.sha256(data).hexdigest() != binding["sha256"]
        ):
            raise ValueError(f"evidence {evidence_id!r} differs from its registry")
        return data

    def read_json(self, evidence_id: str) -> dict:
        return load_json_bytes(self.read_bytes(evidence_id), evidence_id)

    def read_jsonl(self, evidence_id: str) -> list[dict]:
        data = self.read_bytes(evidence_id)
        rows = []
        for number, line in enumerate(data.splitlines(), start=1):
            if not line.strip():
                continue
            rows.append(load_json_bytes(line, f"{evidence_id} line {number}"))
        return rows

