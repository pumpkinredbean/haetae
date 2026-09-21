"""Export the completed shared-v1 generation 79 checkpoint as safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from safetensors.torch import save_file
import torch
from transformers import AutoConfig, AutoTokenizer

from haetae.bundle import (
    BUNDLE_FILES,
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    SHARED_V1_IDENTITY,
    canonical_sha256,
    expected_safetensors_metadata,
    file_sha256,
)
from haetae.checkpoint import load_completed_checkpoint
from haetae.runtime import (
    configuration_fingerprint,
    runtime_source_sha256,
    tokenizer_fingerprint,
)
from haetae.shared_v1 import (
    SOURCE_BASE_COMMIT,
    SOURCE_IMPLEMENTATION_PATH,
    SOURCE_IMPLEMENTATION_SHA256,
)


TRAINING_SOURCE_FILES = (
    "experiments/audit_shared_suite.py",
    "experiments/comparison_protocol.py",
    "experiments/evaluate_baseline.py",
    "experiments/evaluate_shared.py",
    "experiments/freeze_comparison.py",
    "experiments/freeze_shared_suite.py",
    "experiments/kev_adapter.py",
    "experiments/shared_state.py",
    "experiments/train_shared.py",
    "src/haetae/checkpoint.py",
    "pyproject.toml",
)


def _write_json(value: dict, path: Path) -> None:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(data)


def _tensor_index(state: dict[str, torch.Tensor]) -> dict:
    return {
        name: {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(state.items())
    }


def _training_source_identity(repository: Path) -> dict:
    digest = hashlib.sha256()
    for relative in TRAINING_SOURCE_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((repository / relative).read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": list(TRAINING_SOURCE_FILES)}


def _release_state(payload: dict) -> dict[str, torch.Tensor]:
    result = {
        **{f"backbone.{name}": value for name, value in payload["backbone"].items()},
        **{f"head.{name}": value for name, value in payload["head"].items()},
    }
    if not result or any(
        not torch.is_tensor(value)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        for value in result.values()
    ):
        raise ValueError("release state must contain contiguous CPU float32 tensors")
    return result


def _validate_fixed_identity(run: dict, manifest: dict) -> dict:
    current = manifest.get("current")
    if (
        run.get("run_id") != SHARED_V1_IDENTITY["run_id"]
        or run.get("spec_sha256") != SHARED_V1_IDENTITY["spec_sha256"]
        or not isinstance(current, dict)
        or current.get("generation") != SHARED_V1_IDENTITY["generation"]
        or current.get("status") != "completed"
        or current.get("sha256") != SHARED_V1_IDENTITY["checkpoint_sha256"]
        or current.get("size") != SHARED_V1_IDENTITY["checkpoint_size_bytes"]
    ):
        raise ValueError("run is not the fixed shared-v1 generation 79 checkpoint")
    return current


def export_bundle(run_dir: Path, output: Path, repository: Path) -> dict:
    repository = repository.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    output_parent = output.parent.resolve(strict=True)

    original_source = repository / SOURCE_IMPLEMENTATION_PATH
    if (
        not original_source.is_file()
        or file_sha256(original_source) != SOURCE_IMPLEMENTATION_SHA256
    ):
        raise ValueError("shared-state research implementation identity differs")

    payload, run, checkpoint_manifest = load_completed_checkpoint(run_dir)
    current = _validate_fixed_identity(run, checkpoint_manifest)
    if _training_source_identity(repository) != run["spec"]["training_code"]:
        raise ValueError("repository does not match the completed training producer")

    experiment = run["spec"]["experiment"]
    config_values = run["spec"]["config"]
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer_fingerprint(tokenizer) != run["spec"]["tokenizer_fingerprint"]:
        raise ValueError("saved tokenizer differs from the completed run")

    config = AutoConfig.from_pretrained(
        config_values["backbone"],
        revision=experiment["backbone_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(config, "auto_map", None):
        raise ValueError("backbone configuration requests custom code")
    config.vocab_size = len(tokenizer)
    config._name_or_path = config_values["backbone"]
    if configuration_fingerprint(config) != run["spec"]["model_config_fingerprint"]:
        raise ValueError("reconstructed model configuration differs from training")

    state = _release_state(payload)
    tensor_index = _tensor_index(state)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        config.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        _write_json(tokenizer.special_tokens_map, temporary / "special_tokens_map.json")
        save_file(
            state,
            temporary / "model.safetensors",
            metadata=expected_safetensors_metadata(),
        )
        for name in BUNDLE_FILES:
            (temporary / name).chmod(0o644)
        names = {path.name for path in temporary.iterdir()}
        if names != set(BUNDLE_FILES):
            raise ValueError(f"exported file inventory differs: {sorted(names)}")

        files = {
            name: {
                "sha256": file_sha256(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in BUNDLE_FILES
        }
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "status": "complete",
            "format": BUNDLE_FORMAT,
            "model": {
                "run_id": run["run_id"],
                "spec_sha256": run["spec_sha256"],
                "generation": current["generation"],
                "checkpoint_sha256": current["sha256"],
                "checkpoint_size_bytes": current["size"],
                "backbone": config_values["backbone"],
                "backbone_revision": experiment["backbone_revision"],
                "model_config_fingerprint": run["spec"]["model_config_fingerprint"],
                "tokenizer_fingerprint": run["spec"]["tokenizer_fingerprint"],
                "pointer_size": experiment["pointer_size"],
                "maximum_length": config_values["max_len"],
                "attention_implementation": experiment["attention_implementation"],
                "parameter_dtype": "float32",
            },
            "architecture_source": {
                "path": SOURCE_IMPLEMENTATION_PATH,
                "sha256": SOURCE_IMPLEMENTATION_SHA256,
                "base_commit": SOURCE_BASE_COMMIT,
                "runtime_sha256": runtime_source_sha256(),
            },
            "files": files,
            "tensor_index": tensor_index,
            "tensor_index_sha256": canonical_sha256(tensor_index),
            "model_forwards": 0,
            "optimizer_state_included": False,
            "random_state_included": False,
            "locked_test_opened": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(manifest, temporary / "manifest.json")
        published = output_parent / output.name
        published.mkdir(mode=0o755)
        try:
            for name in BUNDLE_FILES:
                os.rename(temporary / name, published / name)
            os.rename(temporary / "manifest.json", published / "manifest.json")
        except BaseException:
            shutil.rmtree(published, ignore_errors=True)
            raise
        temporary.rmdir()
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    manifest = export_bundle(args.run_dir, args.output, args.repository)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "model_forwards": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
