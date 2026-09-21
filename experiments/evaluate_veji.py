"""Evaluate the fixed VEJI-V2 release on Haetae's public development suites."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch

from experiments.compare_development import (
    inventory_from_requests,
    load_report,
    paired_summary,
    prediction_key,
    validate_population,
    validate_report_plan,
)
from experiments.comparison_protocol import (
    canonical_sha256,
    common_clean_predictions,
    load_comparison_plan,
    request_state_sha256,
    training_clean_calibration_predictions,
    validate_suite_binding,
)
from experiments.evaluate_shared import (
    canonical_report_sha256,
    development_report,
    fit_source_balanced_temperature,
)
from experiments.kev_adapter import file_sha256, load_frozen_split
from haetae.checkpoint import atomic_json_save

TRACKS = {
    "calibration": ("decision", "calibration"),
    "decision_development": ("decision", "development"),
    "transfer_development": ("transfer", "development"),
    "korean_development": ("korean", "development"),
}
DEVELOPMENT_TRACKS = {
    "decision": "decision_development",
    "transfer": "transfer_development",
    "korean": "korean_development",
}


def load_protocol(path: str | Path) -> dict:
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    expected = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise ValueError("VEJI evaluation protocol digest mismatch")
    if protocol.get("version") != 1 or protocol.get("status") != "frozen":
        raise ValueError("unsupported VEJI evaluation protocol")
    if protocol.get("locked_test_opened") is not False:
        raise ValueError("VEJI protocol does not attest a closed locked test")
    return protocol


def validate_file(path: str | Path, descriptor: dict, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.stat().st_size != descriptor["bytes"]:
        raise ValueError(f"{label} byte size differs from the frozen protocol")
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"{label} digest differs from the frozen protocol")
    return path


def validate_artifact_files(root: str | Path, files: dict, label: str) -> dict:
    root = Path(root).resolve()
    validated = {}
    for relative, descriptor in sorted(files.items()):
        path = validate_file(root / relative, descriptor, f"{label} {relative}")
        validated[relative] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": descriptor["sha256"],
        }
    return validated


def validate_implementation(protocol: dict) -> None:
    repository = Path(__file__).resolve().parents[1]
    for relative, expected in protocol["implementation_files"].items():
        path = repository / relative
        if file_sha256(path) != expected:
            raise ValueError(f"evaluation implementation changed after freeze: {relative}")


def import_veji(source: Path):
    spec = importlib.util.spec_from_file_location("frozen_veji_v2", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen VEJI source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_veji_model(
    model_directory: str | Path,
    encoder_directory: str | Path,
    protocol: dict,
    device: str,
):
    model_directory = Path(model_directory).resolve()
    encoder_directory = Path(encoder_directory).resolve()
    model_files = validate_artifact_files(
        model_directory, protocol["model"]["files"], "VEJI model",
    )
    encoder_files = validate_artifact_files(
        encoder_directory, protocol["encoder"]["files"], "VEJI encoder",
    )
    module = import_veji(model_directory / "veji.py")
    config = json.loads((model_directory / "config.json").read_text())
    if config != protocol["model"]["config"]:
        raise ValueError("VEJI configuration differs from the frozen protocol")
    config["encoder_model"] = str(encoder_directory)
    model = module.VEJIV2(
        module.VEJIConfig(**config),
        device=device,
        allow_encoder_fallback=False,
    )
    if getattr(model.encoder, "backend", None) != "sentence-transformers":
        raise RuntimeError("VEJI did not load the required sentence-transformers encoder")
    head = torch.load(
        model_directory / "veji_head.pt",
        map_location=model.device,
        weights_only=True,
    )
    model.load_state_dict(head, strict=True)
    model.eval()
    if any(parameter.requires_grad for parameter in model.encoder._model.parameters()):
        raise RuntimeError("VEJI encoder is not frozen")
    head_parameters = sum(parameter.numel() for parameter in model.parameters())
    encoder_parameters = sum(
        parameter.numel() for parameter in model.encoder._model.parameters()
    )
    runtime = {
        "device": str(model.device),
        "encoder_backend": model.encoder.backend,
        "head_parameters": head_parameters,
        "encoder_parameters": encoder_parameters,
        "total_parameters": head_parameters + encoder_parameters,
        "model_files": model_files,
        "encoder_files": encoder_files,
    }
    return model, runtime


def veji_question(question: dict) -> dict:
    return {
        "id": question["id"],
        "type": question["type"],
        "instruction": question["instructions"],
        "options": list(question["options"]),
    }


def logits_to_list(logits: torch.Tensor) -> list[float]:
    values = logits.detach().cpu().to(dtype=torch.float64).tolist()
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("VEJI returned invalid logits")
    return values


def infer_requests(model, requests: list[dict], *, progress_every: int = 100) -> list[dict]:
    predictions = []
    started = time.monotonic()
    with torch.no_grad():
        for request_index, request in enumerate(requests, start=1):
            state = request["state"]
            if len(state) > model.cfg.max_context_chars:
                raise ValueError("VEJI would truncate a public-development state")
            compiled = model.compile_state(state)
            if compiled.state != state:
                raise ValueError("VEJI compiled state differs from the normalized state")
            for question in request["questions"]:
                external_question = veji_question(question)
                if "semantic_type" in external_question:
                    raise AssertionError("external task hints must not enter VEJI")
                output = model.forward_question(compiled, external_question)
                logits = logits_to_list(output["logits"])
                if len(logits) != len(question["options"]):
                    raise ValueError("VEJI logit count differs from the option count")
                predictions.append({
                    "request_id": request["meta"].get("id"),
                    "group_id": request["meta"].get("group_id"),
                    "state_sha256": request_state_sha256(request),
                    "question_id": question["id"],
                    "source": question["source"],
                    "type": question["type"],
                    "options": question["options"],
                    "option_keys": question["option_keys"],
                    "label": question["label"],
                    "soft": question["soft"],
                    "state_chars": len(state),
                    "compiled_chunks": len(compiled.chunks),
                    "questions_in_request": len(request["questions"]),
                    "logits": logits,
                })
            if progress_every and request_index % progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"requests={request_index}/{len(requests)} "
                    f"questions={len(predictions)} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
    return predictions


def stage_digest(stage: dict) -> str:
    unsigned = dict(stage)
    unsigned.pop("stage_sha256", None)
    return canonical_sha256(unsigned)


def write_stage(
    path: Path,
    name: str,
    predictions: list[dict],
    protocol: dict,
    split_sha256: str,
    runtime: dict,
) -> dict:
    stage = {
        "version": 1,
        "status": "complete",
        "stage": name,
        "protocol_sha256": protocol["protocol_sha256"],
        "split_sha256": split_sha256,
        "runtime": runtime,
        "locked_test_opened": False,
        "predictions": predictions,
    }
    stage["stage_sha256"] = stage_digest(stage)
    atomic_json_save(stage, path)
    return stage


def load_stage(
    path: Path,
    name: str,
    protocol: dict,
    split_sha256: str,
    inventory: dict,
) -> dict:
    stage = json.loads(path.read_text())
    if stage_digest(stage) != stage.get("stage_sha256"):
        raise ValueError(f"{name} stage digest mismatch")
    expected = {
        "version": 1,
        "status": "complete",
        "stage": name,
        "protocol_sha256": protocol["protocol_sha256"],
        "split_sha256": split_sha256,
        "locked_test_opened": False,
    }
    if any(stage.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{name} stage binding mismatch")
    runtime = stage.get("runtime", {})
    if (
        runtime.get("device") != protocol["execution"]["required_device"]
        or runtime.get("encoder_backend") != "sentence-transformers"
        or runtime.get("head_parameters") != protocol["model"][
            "reported_trainable_parameters"
        ]
        or runtime.get("encoder_parameters") != protocol["encoder"][
            "parameters"
        ]
    ):
        raise ValueError(f"{name} stage runtime identity mismatch")
    validate_population(stage.get("predictions"), inventory, f"VEJI {name}")
    return stage


def training_overlap_audit(
    training_path: str | Path,
    descriptor: dict,
    development_requests: dict[str, list[dict]],
) -> dict:
    training_path = validate_file(training_path, descriptor, "VEJI training pool")
    states = {}
    for track, requests in development_requests.items():
        for request in requests:
            states.setdefault(request["state"], []).append({
                "track": track,
                "request_id": request["meta"].get("id"),
            })
    overlaps = []
    records = 0
    with training_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            if record.get("state") in states:
                overlaps.append({
                    "veji_record_id": record.get("id"),
                    "matches": states[record["state"]],
                })
    if records != descriptor["records"]:
        raise ValueError("VEJI training-pool record count differs from protocol")
    return {
        "scope": "exact rendered-state equality against the full public training pool",
        "training_pool_records": records,
        "public_development_unique_states": len(states),
        "exact_state_overlaps": len(overlaps),
        "overlaps": overlaps,
        "limitation": (
            "the exact 30,000-record subset used for the checkpoint is unpublished; "
            "zero overlap with the full 190,000-record pool also implies zero overlap "
            "with any subset of that pool"
        ),
    }


def parent_bindings(requests: list[dict], predictions: list[dict]) -> dict:
    inventory = inventory_from_requests(requests)
    keys = {prediction_key(prediction) for prediction in predictions}
    return {
        key: {
            "parent_namespace": inventory[key]["parent_namespace"],
            "parent": inventory[key]["parent"],
        }
        for key in keys
    }


def analyzer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "path": str(Path(__file__).resolve().relative_to(repository)),
        "sha256": file_sha256(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def evaluate(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    work_directory = Path(args.work_dir).resolve()
    work_directory.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.protocol)
    validate_implementation(protocol)
    comparison_plan_path = validate_file(
        args.comparison_plan,
        protocol["inputs"]["comparison_plan"]["file"],
        "comparison plan",
    )
    comparison_plan = load_comparison_plan(
        comparison_plan_path, enforce_code=False,
    )
    if comparison_plan["plan_sha256"] != protocol["inputs"][
            "comparison_plan"]["plan_sha256"]:
        raise ValueError("comparison plan identity differs from VEJI protocol")
    suite_paths = {
        "decision": Path(args.decision_suite).resolve(),
        "transfer": Path(args.transfer_suite).resolve(),
        "korean": Path(args.korean_suite).resolve(),
    }
    for track, suite_path in suite_paths.items():
        validate_suite_binding(comparison_plan, track, suite_path)
    populations = {}
    manifests = {}
    inventories = {}
    for name, (track, split) in TRACKS.items():
        requests, manifest = load_frozen_split(suite_paths[track], split)
        descriptor = manifest["files"][f"{split}.jsonl"]
        expected = protocol["inputs"]["populations"][name]
        for field in ("sha256", "records", "questions"):
            if descriptor[field] != expected[field]:
                raise ValueError(f"{name} population differs from VEJI protocol")
        populations[name] = requests
        manifests[name] = manifest
        inventories[name] = inventory_from_requests(requests)
    reference_path = validate_file(
        args.haetae_report,
        protocol["inputs"]["haetae_report"]["file"],
        "Haetae reference report",
    )
    reference = load_report(reference_path)
    validate_report_plan(reference, comparison_plan, "Haetae")
    if reference["report_sha256"] != protocol["inputs"][
            "haetae_report"]["report_sha256"]:
        raise ValueError("Haetae report identity differs from VEJI protocol")
    overlap = training_overlap_audit(
        args.veji_training_pool,
        protocol["training_pool"],
        {
            track: populations[name]
            for track, name in DEVELOPMENT_TRACKS.items()
        },
    )
    if overlap["exact_state_overlaps"]:
        raise ValueError("VEJI training pool overlaps public development states")
    stages = {}
    missing = []
    for name, (track, split) in TRACKS.items():
        stage_path = work_directory / f"{name}.json"
        split_sha256 = manifests[name]["files"][f"{split}.jsonl"]["sha256"]
        if stage_path.exists():
            stages[name] = load_stage(
                stage_path, name, protocol, split_sha256, inventories[name],
            )
        else:
            missing.append(name)
    runtime = None
    if missing:
        required = protocol["execution"]["required_device"]
        if args.device != required:
            raise ValueError(f"protocol requires device {required}")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required by the frozen VEJI protocol")
        model, runtime = load_veji_model(
            args.veji_model, args.veji_encoder, protocol, args.device,
        )
        cached_runtimes = {
            canonical_sha256(stage["runtime"]): stage["runtime"]
            for stage in stages.values()
        }
        if cached_runtimes and (
            len(cached_runtimes) != 1
            or canonical_sha256(runtime) not in cached_runtimes
        ):
            raise ValueError("cached VEJI stages differ from the active runtime")
        for name, (_, split) in TRACKS.items():
            if name not in missing:
                continue
            predictions = infer_requests(
                model,
                populations[name],
                progress_every=args.progress_every,
            )
            validate_population(predictions, inventories[name], f"VEJI {name}")
            split_sha256 = manifests[name]["files"][f"{split}.jsonl"]["sha256"]
            stages[name] = write_stage(
                work_directory / f"{name}.json",
                name,
                predictions,
                protocol,
                split_sha256,
                runtime,
            )
            print(f"completed stage={name} sha256={stages[name]['stage_sha256']}", flush=True)
    if runtime is None:
        runtimes = {canonical_sha256(stage["runtime"]): stage["runtime"] for stage in stages.values()}
        if len(runtimes) != 1:
            raise ValueError("cached VEJI stages have different runtime identities")
        runtime = next(iter(runtimes.values()))
    calibration_predictions = stages["calibration"]["predictions"]
    calibration_clean, calibration_audit = training_clean_calibration_predictions(
        calibration_predictions, comparison_plan,
    )
    temperature = fit_source_balanced_temperature(calibration_clean)
    development = {}
    comparisons = {}
    for index, (track, name) in enumerate(DEVELOPMENT_TRACKS.items()):
        predictions = stages[name]["predictions"]
        clean, clean_audit = common_clean_predictions(
            predictions, comparison_plan, track,
        )
        reference_predictions = reference["predictions"][name]
        reference_clean, reference_audit = common_clean_predictions(
            reference_predictions, comparison_plan, track,
        )
        if clean_audit != reference_audit:
            raise ValueError(f"{track} common-clean memberships differ")
        bindings = parent_bindings(populations[name], clean)
        seed = protocol["comparison"]["bootstrap_seed"] + index * 100
        development[track] = {
            "manifest_sha256": file_sha256(suite_paths[track] / "manifest.json"),
            "split_sha256": protocol["inputs"]["populations"][name]["sha256"],
            **development_report(predictions, clean, clean_audit, temperature),
        }
        comparisons[track] = {
            "common_clean": clean_audit,
            "raw": paired_summary(
                reference_clean, clean, bindings, 1.0, 1.0,
                protocol["comparison"]["bootstrap_samples"], seed,
            ),
            "scaled": paired_summary(
                reference_clean, clean, bindings,
                float(reference["temperature"]["value"]), temperature,
                protocol["comparison"]["bootstrap_samples"], seed + 50,
            ),
        }
    runtime_bytes = sum(
        descriptor["bytes"]
        for descriptor in protocol["model"]["files"].values()
        if descriptor.get("runtime")
    ) + sum(
        descriptor["bytes"]
        for descriptor in protocol["encoder"]["files"].values()
        if descriptor.get("runtime")
    )
    report = {
        "version": 1,
        "status": "evaluated",
        "protocol_sha256": protocol["protocol_sha256"],
        "comparison_plan_sha256": comparison_plan["plan_sha256"],
        "analyzer": analyzer_identity(),
        "model": {
            "id": protocol["model"]["id"],
            "revision": protocol["model"]["revision"],
            "encoder_id": protocol["encoder"]["id"],
            "encoder_revision": protocol["encoder"]["revision"],
            "runtime": runtime,
            "required_runtime_file_bytes": runtime_bytes,
            "size_scope": "head, encoder weights, tokenizer, and required configuration",
        },
        "inference": {
            "device": args.device,
            "semantic_type_policy": "omitted; only the public primitive type is passed",
            "state_policy": "compile exactly once per request and reuse for its questions",
            "encoder_fallback": False,
        },
        "training_overlap_audit": overlap,
        "temperature": {
            "value": temperature,
            "fit_manifest_sha256": file_sha256(suite_paths["decision"] / "manifest.json"),
            "fit_split_sha256": protocol["inputs"]["populations"]["calibration"]["sha256"],
            "objective": (
                "source-balanced hard-target cross-entropy after the frozen "
                "Haetae training-state exclusions"
            ),
            "common_clean": calibration_audit,
        },
        "development": development,
        "haetae_reference": {
            "path": str(reference_path),
            "file_sha256": file_sha256(reference_path),
            "report_sha256": reference["report_sha256"],
            "temperature": reference["temperature"]["value"],
        },
        "difference_definition": (
            "VEJI minus Haetae; positive favors VEJI for accuracy and negative "
            "favors VEJI for NLL and Brier metrics"
        ),
        "uncertainty_scope": (
            "pointwise paired parent bootstrap for fixed checkpoints and fixed fitted "
            "temperatures; excludes training-seed and calibration-fit uncertainty"
        ),
        "comparisons": comparisons,
        "stages": {
            name: {
                "path": str(work_directory / f"{name}.json"),
                "file_sha256": file_sha256(work_directory / f"{name}.json"),
                "stage_sha256": stage["stage_sha256"],
            }
            for name, stage in stages.items()
        },
        "locked_test_opened": False,
    }
    report["report_sha256"] = canonical_report_sha256(report)
    atomic_json_save(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--comparison-plan", required=True)
    parser.add_argument("--decision-suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--korean-suite", required=True)
    parser.add_argument("--haetae-report", required=True)
    parser.add_argument("--veji-model", required=True)
    parser.add_argument("--veji-encoder", required=True)
    parser.add_argument("--veji-training-pool", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "required_runtime_file_bytes": report["model"]["required_runtime_file_bytes"],
        "temperature": report["temperature"]["value"],
        "development": {
            track: values["common_clean"]["raw"]["source_macro"]
            for track, values in report["development"].items()
        },
        "raw_comparisons": {
            track: values["raw"]["metrics"]
            for track, values in report["comparisons"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
