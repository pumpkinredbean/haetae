"""Offline runtime for a verified shared-v1 bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from .bundle import (
    BundleError,
    VerifiedBundle,
    canonical_json_bytes,
    expected_safetensors_metadata,
    verify_bundle,
)
from .shared_v1 import (
    SOURCE_BASE_COMMIT,
    SOURCE_IMPLEMENTATION_PATH,
    SOURCE_IMPLEMENTATION_SHA256,
    build_shared_state_model,
    encode_request,
)


class HaetaeRuntimeError(ValueError):
    """Raised for invalid runtime configuration or requests."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        tokenization_definition = json.loads(backend.to_str())
    else:
        tokenization_definition = {"vocab": tokenizer.get_vocab()}
    payload = {
        "class": tokenizer.__class__.__name__,
        "tokenization_definition": tokenization_definition,
        "special_tokens_map": tokenizer.special_tokens_map,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    return _sha256_bytes(canonical_json_bytes(payload))


def configuration_fingerprint(config) -> str:
    return _sha256_bytes(canonical_json_bytes(config.to_dict()))


def model_config_fingerprint(model) -> str:
    return configuration_fingerprint(model.backbone.config)


def runtime_source_sha256() -> str:
    return _sha256_bytes(
        Path(__file__).with_name("shared_v1.py").read_bytes()
    )


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise HaetaeRuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise HaetaeRuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    raise HaetaeRuntimeError("device must be one of: cpu, mps, cuda")


def _tensor_index(state: dict[str, torch.Tensor]) -> dict:
    return {
        name: {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(state.items())
    }


def _load_local_model(bundle: VerifiedBundle, device: torch.device):
    manifest = bundle.manifest
    identity = manifest["model"]
    architecture = manifest["architecture_source"]
    if (
        architecture["path"] != SOURCE_IMPLEMENTATION_PATH
        or architecture["sha256"] != SOURCE_IMPLEMENTATION_SHA256
        or architecture["base_commit"] != SOURCE_BASE_COMMIT
        or architecture["runtime_sha256"] != runtime_source_sha256()
    ):
        raise BundleError("runtime architecture identity differs from the bundle")

    tokenizer = AutoTokenizer.from_pretrained(
        bundle.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer_fingerprint(tokenizer) != identity["tokenizer_fingerprint"]:
        raise BundleError("bundle tokenizer fingerprint differs")

    config = AutoConfig.from_pretrained(
        bundle.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(config, "auto_map", None):
        raise BundleError("bundle config requests remote custom code")
    config._name_or_path = identity["backbone"]
    model, delimiters = build_shared_state_model(
        config,
        tokenizer,
        str(device),
        pointer_size=identity["pointer_size"],
        attention_implementation=identity["attention_implementation"],
    )
    if model_config_fingerprint(model) != identity["model_config_fingerprint"]:
        raise BundleError("bundle model configuration fingerprint differs")

    model_file = bundle.files["model.safetensors"]
    with safe_open(model_file, framework="pt", device="cpu") as handle:
        if handle.metadata() != expected_safetensors_metadata():
            raise BundleError("safetensors metadata differs")
    state = load_file(str(model_file), device="cpu")
    if _tensor_index(state) != manifest["tensor_index"]:
        raise BundleError("safetensors contents differ from the tensor index")
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise BundleError("bundle tensors are not float32")
    expected_keys = set(model.state_dict())
    if set(state) != expected_keys:
        raise BundleError("bundle tensor names differ from the model")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, delimiters


def _validate_runtime_question(question: dict) -> None:
    expected = {"id", "type", "instructions", "options"}
    if not isinstance(question, dict) or set(question) != expected:
        raise HaetaeRuntimeError("question fields differ from the runtime schema")
    if not isinstance(question["id"], str) or not question["id"]:
        raise HaetaeRuntimeError("question ID must be a nonempty string")
    if question["type"] not in {"choice", "noul", "score"}:
        raise HaetaeRuntimeError("question type is unsupported")
    if not isinstance(question["instructions"], str) or not question["instructions"]:
        raise HaetaeRuntimeError("question instructions must be a nonempty string")
    options = question["options"]
    if (
        not isinstance(options, list)
        or len(options) < 2
        or len(options) > 255
        or any(not isinstance(option, str) or not option for option in options)
    ):
        raise HaetaeRuntimeError("question options must contain 2 to 255 strings")
    if len(set(options)) != len(options):
        raise HaetaeRuntimeError("question options must be unique")
    if question["type"] == "noul" and options != ["yes", "no"]:
        raise HaetaeRuntimeError("Noul options must be yes then no")


def _validate_state(state: Any) -> None:
    if not isinstance(state, (str, dict, list)):
        raise HaetaeRuntimeError("state must be a string, object, or array")
    try:
        canonical_json_bytes(state)
    except (TypeError, ValueError) as error:
        raise HaetaeRuntimeError("state must contain finite JSON values") from error


@dataclass
class HaetaeRuntime:
    bundle: VerifiedBundle
    model: Any
    tokenizer: Any
    delimiters: dict[str, int]
    device: torch.device

    @property
    def maximum_length(self) -> int:
        return int(self.bundle.manifest["model"]["maximum_length"])

    def decide(self, state: Any, questions: list[dict]) -> list[dict]:
        _validate_state(state)
        if not isinstance(questions, list) or not questions:
            raise HaetaeRuntimeError("a request needs at least one question")
        identifiers = []
        for question in questions:
            _validate_runtime_question(question)
            identifiers.append(question["id"])
        if len(set(identifiers)) != len(identifiers):
            raise HaetaeRuntimeError("question IDs must be unique within a request")
        encoded = encode_request(
            self.tokenizer,
            state,
            questions,
            self.maximum_length,
            self.delimiters,
        )
        if encoded is None:
            raise HaetaeRuntimeError(
                "question and option content exceeds the input budget"
            )
        if encoded.retained_state_tokens != encoded.state_tokens:
            raise HaetaeRuntimeError(
                "state content exceeds the remaining input budget"
            )
        with torch.inference_mode():
            output = self.model([encoded])[0]
        if len(output) != len(questions):
            raise HaetaeRuntimeError("model output count differs from the request")
        results = []
        for question, logits in zip(questions, output):
            if (
                not torch.is_tensor(logits)
                or logits.ndim != 1
                or logits.numel() != len(question["options"])
                or not torch.isfinite(logits).all()
            ):
                raise HaetaeRuntimeError(
                    "model output is not a finite vector over the options"
                )
            probabilities = torch.softmax(
                logits.detach().to(dtype=torch.float64, device="cpu"),
                dim=-1,
            )
            values = probabilities.tolist()
            results.append({
                "id": question["id"],
                "type": question["type"],
                "options": list(question["options"]),
                "probabilities": values,
                "choice": int(probabilities.argmax()),
                "confidence": float(probabilities.max()),
                "calibrated": False,
            })
        return results


def load_runtime(bundle: str | Path, device: str = "cpu") -> HaetaeRuntime:
    verified = verify_bundle(bundle)
    resolved_device = resolve_device(device)
    model, tokenizer, delimiters = _load_local_model(
        verified,
        resolved_device,
    )
    return HaetaeRuntime(
        bundle=verified,
        model=model,
        tokenizer=tokenizer,
        delimiters=delimiters,
        device=resolved_device,
    )
