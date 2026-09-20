"""Freeze evaluation populations before model results are inspected.

The preparation stage reconstructs the run's consumed records, verifies
their immutable fingerprints, excludes exact state overlap, and writes
parent-disjoint temperature-fit (A), conformal-policy (B), and final
certification (C) populations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoint import file_sha256
from .data import LOADERS
from .train import dataset_fingerprint


@dataclass(frozen=True)
class TrackSpec:
    name: str
    source: str
    repository: str
    configuration: str | None
    revision: str
    language: str
    single_split: str | None = None
    calibration_split: str | None = None
    certification_split: str | None = None


TRACKS: dict[str, tuple[TrackSpec, ...]] = {
    "ag_news": (TrackSpec(
        "ag_news", "ag_news", "fancyzhx/ag_news", None,
        "eb185aade064a813bc0b7f42de02595523103ca4", "en",
        single_split="test",
    ),),
    "banking77": (TrackSpec(
        "banking77", "banking77", "mteb/banking77", None,
        "18072d2685ea682290f7b8924d94c62acc19c0b2", "en",
        single_split="test",
    ),),
    "massive": (TrackSpec(
        "massive", "massive", "mteb/massive_intent", "en",
        "feaaf04a19b242736c2d040393e7932a22a0435e", "en",
        calibration_split="validation", certification_split="test",
    ),),
    "mnli": (
        TrackSpec(
            "mnli_matched", "mnli", "nyu-mll/multi_nli", None,
            "da70db2af9d09693783c3320c4249840212ee221", "en",
            single_split="validation_matched",
        ),
        TrackSpec(
            "mnli_mismatched", "mnli", "nyu-mll/multi_nli", None,
            "da70db2af9d09693783c3320c4249840212ee221", "en",
            single_split="validation_mismatched",
        ),
    ),
    "anli": (TrackSpec(
        "anli_r1", "anli", "facebook/anli", "plain_text",
        "8e4813d81f46d313dac7892e1c28076917cfcdf9", "en",
        calibration_split="dev_r1", certification_split="test_r1",
    ),),
    "arc": (TrackSpec(
        "arc_easy", "arc", "allenai/ai2_arc", "ARC-Easy",
        "210d026faf9955653af8916fad021475a3f00453", "en",
        calibration_split="validation", certification_split="test",
    ),),
    "emotion": (TrackSpec(
        "emotion", "emotion", "SetFit/emotion", None,
        "6c362e04d016f6b6a9377e85c3b944140f0b96c9", "en",
        calibration_split="validation", certification_split="test",
    ),),
    "klue_ynat": (TrackSpec(
        "klue_ynat", "klue_ynat",
        "https://github.com/KLUE-benchmark/KLUE", "ynat-v1.1",
        "local-file-sha256", "ko", single_split="validation",
    ),),
    "boolq": (TrackSpec(
        "boolq", "boolq", "google/boolq", None,
        "35b264d03638db9f4ce671b711558bf7ff0f80d5", "en",
        single_split="validation",
    ),),
    "nsmc": (TrackSpec(
        "nsmc", "nsmc", "https://github.com/e9t/nsmc", None,
        "local-file-sha256", "ko", single_split="test",
    ),),
    "civil_toxicity": (TrackSpec(
        "civil_toxicity", "civil_toxicity", "google/civil_comments", None,
        "f2970eb3a55777454c94069077cc8d9b5866312d", "en",
        calibration_split="validation", certification_split="test",
    ),),
    "sst5": (TrackSpec(
        "sst5", "sst5", "SetFit/sst5", None,
        "e51bdcd8cd3a30da231967c1a249ba59361279a3", "en",
        calibration_split="validation", certification_split="test",
    ),),
    "helpsteer2": (TrackSpec(
        "helpsteer2", "helpsteer2", "nvidia/HelpSteer2", None,
        "990b2711a36180dd19d9c94b8627844866f8982a", "en",
        single_split="validation",
    ),),
}

RECORD_FIELDS = (
    "state", "type", "instructions", "options", "label", "soft",
    "source", "parent",
)
ROLES = ("A", "B", "C")


@contextmanager
def _pinned_dataset_revision(spec: TrackSpec):
    """Make a dataset-backed loader use the revision declared by its track."""
    if spec.revision == "local-file-sha256":
        yield
        return
    import datasets

    original = datasets.load_dataset

    def load_dataset(path, *args, **kwargs):
        if path != spec.repository:
            raise ValueError(
                f"track {spec.name} requested unexpected dataset {path}"
            )
        requested = kwargs.get("revision")
        if requested is not None and requested != spec.revision:
            raise ValueError(
                f"track {spec.name} requested revision {requested}"
            )
        kwargs["revision"] = spec.revision
        return original(path, *args, **kwargs)

    datasets.load_dataset = load_dataset
    try:
        yield
    finally:
        datasets.load_dataset = original


def _track_records(spec: TrackSpec, split: str, limit: int | None):
    with _pinned_dataset_revision(spec):
        yield from LOADERS[spec.source](split=split, limit=limit)


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def digest_value(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalized_record(record: dict) -> dict:
    return {field: record.get(field) for field in RECORD_FIELDS}


def record_digest(record: dict) -> str:
    return digest_value(normalized_record(record))


def state_digest(record: dict) -> str:
    return digest_value(record.get("state"))


def ontology_digest(record: dict) -> str:
    return digest_value({
        "source": record.get("source"),
        "type": record.get("type"),
        "options": record.get("options"),
    })


def parent_identity(record: dict) -> str:
    parent = record.get("parent")
    return str(parent) if parent is not None else "state:" + state_digest(record)


def validate_evaluation_record(record: dict, spec: TrackSpec,
                               split: str, row_index: int) -> None:
    options = record.get("options")
    label = record.get("label")
    if (not isinstance(options, list) or len(options) < 2
            or isinstance(label, bool) or not isinstance(label, int)
            or not 0 <= label < len(options)):
        raise ValueError(
            f"unlabeled or invalid evaluation record: "
            f"{spec.name}/{split}/{row_index}"
        )
    soft = record.get("soft")
    if soft is not None:
        if (not isinstance(soft, list) or len(soft) != len(options)
                or any(isinstance(value, bool)
                       or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) or value < 0
                       for value in soft)
                or abs(sum(soft) - 1.0) > 1e-6):
            raise ValueError(
                f"invalid soft evaluation target: "
                f"{spec.name}/{split}/{row_index}"
            )


def selection_digest(seed: int, track: str, split: str, parent: str) -> str:
    return digest_value({
        "seed": seed, "track": track, "split": split, "parent": parent,
    })


def reconstruct_consumed(run: dict) -> dict:
    """Recreate the exact train/internal-validation membership."""
    config = run["spec"]["config"]
    source_names = [name.strip() for name in config["sources"].split(",")]
    train, validation = [], []
    saved_random = random.getstate()
    random.seed(config["seed"])
    try:
        for source in source_names:
            rows = list(LOADERS[source](
                split="train",
                limit=config["per_source"] + config["eval_per_source"],
            ))
            parents: dict[object, list[dict]] = {}
            for index, record in enumerate(rows):
                # train.py uses id(record) for ungrouped rows. Stable row
                # indexes preserve the same insertion order and shuffle.
                key = record.get("parent", index)
                parents.setdefault(key, []).append(record)
            keys = list(parents)
            random.shuffle(keys)
            validation_count = max(
                1,
                int(len(keys) * config["eval_per_source"]
                    / (config["per_source"] + config["eval_per_source"])),
            )
            for key in keys[:validation_count]:
                validation.extend(parents[key])
            for key in keys[validation_count:]:
                train.extend(parents[key])
    finally:
        random.setstate(saved_random)

    observed_train = dataset_fingerprint(train)
    observed_validation = dataset_fingerprint(validation)
    expected_train = run["spec"]["train_fingerprint"]
    expected_validation = run["spec"]["validation_fingerprint"]
    if observed_train != expected_train or observed_validation != expected_validation:
        raise ValueError(
            "reconstructed consumed-data fingerprints do not match run.json"
        )
    all_records = train + validation
    prior_accumulators = {}
    for record in train:
        options = record["options"]
        soft = record.get("soft")
        if soft is None:
            target = [0.0] * len(options)
            target[record["label"]] = 1.0
        else:
            target = [float(value) for value in soft]
        identity = ontology_digest(record)
        accumulator = prior_accumulators.setdefault(identity, {
            "source": record["source"],
            "primitive": record["type"],
            "option_count": len(options),
            "option_text_sha256": digest_value(options),
            "count": 0,
            "target_sum": [0.0] * len(options),
        })
        if len(accumulator["target_sum"]) != len(target):
            raise ValueError("training ontology changed option count")
        accumulator["count"] += 1
        accumulator["target_sum"] = [
            observed + value
            for observed, value in zip(accumulator["target_sum"], target)
        ]
    training_priors = {}
    for identity, accumulator in prior_accumulators.items():
        if accumulator["count"] < 20:
            continue
        denominator = accumulator["count"] + accumulator["option_count"]
        training_priors[identity] = {
            key: value for key, value in accumulator.items()
            if key != "target_sum"
        }
        training_priors[identity]["probabilities"] = [
            (value + 1.0) / denominator
            for value in accumulator["target_sum"]
        ]
        training_priors[identity]["smoothing"] = "add-one"
    return {
        "train_count": len(train),
        "validation_count": len(validation),
        "train_fingerprint": observed_train,
        "validation_fingerprint": observed_validation,
        "record_digests": {record_digest(record) for record in all_records},
        "state_digests": {state_digest(record) for record in all_records},
        "training_only_priors": {
            "ontology": "exact source, primitive, and ordered option texts",
            "minimum_training_records": 20,
            "smoothing": "add-one",
            "values": training_priors,
        },
    }


def _scan_parents(spec: TrackSpec, split: str, forbidden_records: set[str],
                  forbidden_states: set[str]) -> tuple[set[str], set[str], dict]:
    parents, excluded = set(), set()
    rows = 0
    for row_index, record in enumerate(_track_records(spec, split, None)):
        validate_evaluation_record(record, spec, split, row_index)
        rows += 1
        parent = parent_identity(record)
        parents.add(parent)
        if (record_digest(record) in forbidden_records
                or state_digest(record) in forbidden_states):
            excluded.add(parent)
    eligible = parents - excluded
    return eligible, excluded, {
        "emitted_records": rows,
        "parents": len(parents),
        "excluded_overlap_parents": len(excluded),
        "eligible_parents": len(eligible),
    }


def _select_parents(eligible: set[str], count: int, seed: int,
                    track: str, split: str) -> list[str]:
    ordered = sorted(
        eligible,
        key=lambda parent: selection_digest(seed, track, split, parent),
    )
    return ordered[:count]


def _collect(spec: TrackSpec, split: str, selected: set[str],
             role_by_parent: dict[str, str]) -> list[dict]:
    result = []
    for row_index, record in enumerate(_track_records(spec, split, None)):
        validate_evaluation_record(record, spec, split, row_index)
        parent = parent_identity(record)
        if parent not in selected:
            continue
        normalized = normalized_record(record)
        result.append({
            "record": normalized,
            "meta": {
                "track": spec.name,
                "source": spec.source,
                "repository": spec.repository,
                "configuration": spec.configuration,
                "revision": spec.revision,
                "language": spec.language,
                "native_split": split,
                "row_index": row_index,
                "record_sha256": record_digest(record),
                "state_sha256": state_digest(record),
                "parent_id": parent,
                "role": role_by_parent[parent],
            },
        })
    rank = {parent: index for index, parent in enumerate(role_by_parent)}
    result.sort(key=lambda item: (
        ROLES.index(item["meta"]["role"]),
        rank[item["meta"]["parent_id"]],
        item["meta"]["row_index"],
        item["meta"]["record_sha256"],
    ))
    return result


def freeze_track(spec: TrackSpec, seed: int, max_parents: int,
                 forbidden_records: set[str],
                 forbidden_states: set[str]) -> tuple[list[dict], dict]:
    """Freeze one track with parent-disjoint A/B/C roles."""
    if max_parents < 4:
        raise ValueError("max_parents must be at least 4")
    if spec.single_split:
        eligible, excluded, audit = _scan_parents(
            spec, spec.single_split, forbidden_records, forbidden_states,
        )
        selected = _select_parents(
            eligible, min(max_parents, len(eligible)), seed,
            spec.name, spec.single_split,
        )
        if len(selected) < 4:
            raise ValueError(f"{spec.name} has fewer than four eligible parents")
        a_count = len(selected) // 4
        b_count = len(selected) // 4
        role_by_parent = {
            parent: ("A" if index < a_count else
                     "B" if index < a_count + b_count else "C")
            for index, parent in enumerate(selected)
        }
        records = _collect(
            spec, spec.single_split, set(selected), role_by_parent,
        )
        return records, {
            "mode": "partition-single-pool",
            "split_a": spec.single_split,
            "split_b": spec.single_split,
            "split_c": spec.single_split,
            "pool_audit": audit,
            "excluded_parent_ids_sha256": digest_value(sorted(excluded)),
        }

    if not spec.calibration_split or not spec.certification_split:
        raise ValueError(f"incomplete split registry for {spec.name}")
    calibration_eligible, calibration_excluded, calibration_audit = _scan_parents(
        spec, spec.calibration_split, forbidden_records, forbidden_states,
    )
    calibration_limit = min(max_parents // 2, len(calibration_eligible))
    calibration_selected = _select_parents(
        calibration_eligible, calibration_limit, seed,
        spec.name, spec.calibration_split,
    )
    if len(calibration_selected) < 2:
        raise ValueError(f"{spec.name} has fewer than two calibration parents")
    a_count = len(calibration_selected) // 2
    calibration_roles = {
        parent: "A" if index < a_count else "B"
        for index, parent in enumerate(calibration_selected)
    }
    calibration_records = _collect(
        spec, spec.calibration_split, set(calibration_selected),
        calibration_roles,
    )
    calibration_states = {
        item["meta"]["state_sha256"] for item in calibration_records
    }

    certification_eligible, certification_excluded, certification_audit = _scan_parents(
        spec, spec.certification_split, forbidden_records,
        forbidden_states | calibration_states,
    )
    certification_limit = min(max_parents // 2, len(certification_eligible))
    certification_selected = _select_parents(
        certification_eligible, certification_limit, seed,
        spec.name, spec.certification_split,
    )
    if not certification_selected:
        raise ValueError(f"{spec.name} has no certification parents")
    certification_roles = {parent: "C" for parent in certification_selected}
    certification_records = _collect(
        spec, spec.certification_split, set(certification_selected),
        certification_roles,
    )
    return calibration_records + certification_records, {
        "mode": "validation-calibration-and-test-certification",
        "split_a": spec.calibration_split,
        "split_b": spec.calibration_split,
        "split_c": spec.certification_split,
        "calibration_pool_audit": calibration_audit,
        "certification_pool_audit": certification_audit,
        "excluded_calibration_parent_ids_sha256": digest_value(
            sorted(calibration_excluded)
        ),
        "excluded_certification_parent_ids_sha256": digest_value(
            sorted(certification_excluded)
        ),
    }


def validate_role_isolation(records: list[dict]) -> None:
    parent_roles: dict[tuple[str, str], set[str]] = {}
    state_roles: dict[str, set[str]] = {}
    record_roles: dict[str, set[str]] = {}
    for item in records:
        meta = item["meta"]
        parent_roles.setdefault(
            (meta["track"], meta["parent_id"]), set()
        ).add(meta["role"])
        state_roles.setdefault(meta["state_sha256"], set()).add(meta["role"])
        record_roles.setdefault(meta["record_sha256"], set()).add(meta["role"])
    parent_leaks = [key for key, roles in parent_roles.items() if len(roles) > 1]
    state_leaks = [key for key, roles in state_roles.items() if len(roles) > 1]
    record_leaks = [key for key, roles in record_roles.items() if len(roles) > 1]
    if parent_leaks or state_leaks or record_leaks:
        raise ValueError(
            "evaluation roles overlap: "
            f"parents={len(parent_leaks)}, states={len(state_leaks)}, "
            f"records={len(record_leaks)}"
        )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _code_identity(repository: Path) -> dict:
    paths = [
        repository / "src" / "haetae" / name
        for name in ("measure.py", "data.py", "model.py", "checkpoint.py")
    ]
    digest = hashlib.sha256()
    files = {}
    for path in paths:
        relative = str(path.relative_to(repository))
        sha256 = file_sha256(path)
        files[relative] = sha256
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"sha256": digest.hexdigest(), "files": files, "git_commit": commit}


def _local_data_artifacts(repository: Path, sources: list[str]) -> dict:
    names = []
    if "klue_ynat" in sources:
        names.extend(("ynat-v1.1_train.json", "ynat-v1.1_dev.json"))
    if "nsmc" in sources:
        names.extend(("ratings_train.txt", "ratings_test.txt"))
    artifacts = {}
    for name in names:
        path = repository / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"expected local dataset artifact is absent: {path}")
        artifacts[name] = {
            "sha256": file_sha256(path), "bytes": path.stat().st_size,
        }
    return artifacts


def prepare(run_dir: str | Path, output: str | Path, expect_run_id: str,
            max_parents: int = 1600, sources: list[str] | None = None,
            seed: int = 20260920) -> dict:
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation directory {output}")
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("run_id") != expect_run_id:
        raise ValueError("run ID does not match --expect-run-id")
    consumed = reconstruct_consumed(run)
    run_sources = [
        name.strip() for name in run["spec"]["config"]["sources"].split(",")
    ]
    selected_sources = run_sources if sources is None else sources
    unknown = set(selected_sources) - set(run_sources)
    missing_registry = set(selected_sources) - set(TRACKS)
    if unknown:
        raise ValueError(f"sources are absent from the run: {sorted(unknown)}")
    if missing_registry:
        raise ValueError(f"sources lack an evaluation registry: {sorted(missing_registry)}")

    records, track_audits = [], {}
    for source in selected_sources:
        for spec in TRACKS[source]:
            frozen, audit = freeze_track(
                spec, seed, max_parents,
                consumed["record_digests"], consumed["state_digests"],
            )
            records.extend(frozen)
            track_audits[spec.name] = {"spec": asdict(spec), **audit}
    validate_role_isolation(records)

    role_records = {
        role: [item for item in records if item["meta"]["role"] == role]
        for role in ROLES
    }
    temporary = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        files = {}
        for role, frozen in role_records.items():
            path = temporary / f"{role}.jsonl"
            _write_jsonl(path, frozen)
            files[path.name] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "records": len(frozen),
                "parents": len({
                    (item["meta"]["track"], item["meta"]["parent_id"])
                    for item in frozen
                }),
            }
        repository = Path(__file__).resolve().parents[2]
        plan = {
            "version": 1,
            "status": "prepared",
            "run": {
                "directory": str(run_dir),
                "run_id": run["run_id"],
                "spec_sha256": run["spec_sha256"],
                "train_fingerprint": consumed["train_fingerprint"],
                "validation_fingerprint": consumed["validation_fingerprint"],
                "train_records": consumed["train_count"],
                "validation_records": consumed["validation_count"],
            },
            "assignment": {
                "seed": seed,
                "max_parents_per_track": max_parents,
                "roles": {
                    "A": "shared-temperature fitting",
                    "B": "conformal thresholds and decision-policy freezing",
                    "C": "final evaluation and certification",
                },
                "parent_policy": (
                    "explicit dataset parent when present; otherwise exact "
                    "normalized-state SHA-256"
                ),
                "selection": "smallest SHA-256 of seed, track, split, and parent",
                "overlap_policy": (
                    "exclude complete parents with an exact normalized record or "
                    "state match in consumed train/internal-validation data"
                ),
            },
            "tracks": track_audits,
            "files": files,
            "training_only_priors": consumed["training_only_priors"],
            "code": _code_identity(repository),
            "local_data_artifacts": _local_data_artifacts(
                repository, selected_sources,
            ),
        }
        plan["plan_sha256"] = digest_value(plan)
        plan_path = temporary / "plan.json"
        with plan_path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return plan


def load_plan(directory: str | Path, enforce_code: bool = True) -> dict:
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    expected = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if expected != digest_value(unsigned):
        raise ValueError("plan digest mismatch")
    for filename, descriptor in plan["files"].items():
        path = directory / filename
        if (not path.is_file() or path.stat().st_size != descriptor["bytes"]
                or file_sha256(path) != descriptor["sha256"]):
            raise ValueError(f"frozen role file mismatch: {filename}")
    if enforce_code:
        repository = Path(__file__).resolve().parents[2]
        if _code_identity(repository)["sha256"] != plan["code"]["sha256"]:
            raise ValueError("measurement code changed after population freeze")
    return plan


def load_role(directory: str | Path, role: str, plan: dict) -> list[dict]:
    if role not in ROLES:
        raise ValueError(f"unknown evaluation role: {role}")
    path = Path(directory) / f"{role}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    descriptor = plan["files"][path.name]
    if len(records) != descriptor["records"]:
        raise ValueError(f"frozen role count mismatch: {role}")
    if any(item["meta"]["role"] != role for item in records):
        raise ValueError(f"frozen role label mismatch: {role}")
    return records


def target_distribution(record: dict, option_count: int):
    import torch

    soft = record.get("soft")
    if soft is None:
        label = record.get("label")
        if (isinstance(label, bool) or not isinstance(label, int)
                or not 0 <= label < option_count):
            raise ValueError("hard target is outside the option set")
        target = torch.zeros(option_count, dtype=torch.float64)
        target[label] = 1.0
        return target
    target = torch.tensor(soft, dtype=torch.float64)
    if (target.ndim != 1 or target.numel() != option_count
            or not torch.isfinite(target).all() or (target < 0).any()
            or abs(float(target.sum()) - 1.0) > 1e-6):
        raise ValueError("soft target is not a probability distribution")
    return target


def infer_role(run_dir: str | Path, records: list[dict], device: str,
               batch: int) -> list[dict]:
    """Collect detached CPU logits with record alignment and exclusions."""
    import torch
    from .certify import load_model
    from .model import collate, pack_question

    if batch <= 0:
        raise ValueError("inference batch must be positive")
    model, tokenizer, checkpoint, run, manifest = load_model(run_dir, device)
    max_len = int(checkpoint["config"]["max_len"])
    output = []
    with torch.no_grad():
        for start in range(0, len(records), batch):
            items = records[start:start + batch]
            packed = [
                pack_question(
                    tokenizer, item["record"]["state"],
                    item["record"]["instructions"],
                    item["record"]["options"], max_len,
                )
                for item in items
            ]
            keep = [index for index, value in enumerate(packed) if value is not None]
            logits_by_index = {}
            if keep:
                kept_records = [items[index]["record"] for index in keep]
                model_batch = collate(tokenizer, kept_records, max_len)
                logits = model(
                    model_batch["input_ids"].to(device),
                    model_batch["attention_mask"].to(device),
                    model_batch["option_pos"].to(device),
                    model_batch["group_ptr"].to(device),
                )
                logits_by_index = {
                    index: value.detach().float().cpu().tolist()
                    for index, value in zip(keep, logits)
                }
            for index, (item, packed_item) in enumerate(zip(items, packed)):
                prediction = {
                    "meta": item["meta"],
                    "record": item["record"],
                    "status": "ok" if index in logits_by_index else "rejected_context",
                    "logits": logits_by_index.get(index),
                    "packed_tokens": len(packed_item[0]) if packed_item else None,
                }
                output.append(prediction)
    if len(output) != len(records):
        raise RuntimeError("prediction alignment failed")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return output


def _collate_packed(tokenizer, packed: list[tuple[list[int], list[tuple]]]):
    import torch

    if not packed:
        raise ValueError("cannot collate an empty packed batch")
    max_length = max(len(input_ids) for input_ids, _ in packed)
    input_ids = torch.full(
        (len(packed), max_length), tokenizer.pad_token_id, dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    option_positions, group_pointer = [], [0]
    for batch_index, (ids, spans) in enumerate(packed):
        input_ids[batch_index, :len(ids)] = torch.tensor(ids)
        attention_mask[batch_index, :len(ids)] = 1
        option_positions.extend(
            batch_index * max_length + marker for marker, _, _ in spans
        )
        group_pointer.append(group_pointer[-1] + len(spans))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "option_pos": torch.tensor(option_positions, dtype=torch.long),
        "group_ptr": torch.tensor(group_pointer, dtype=torch.long),
    }


def _run_collated(model, collated: dict, device: str):
    return model(
        collated["input_ids"].to(device),
        collated["attention_mask"].to(device),
        collated["option_pos"].to(device),
        collated["group_ptr"].to(device),
    )


def _infer_packed(model, tokenizer, packed: list[tuple], device: str,
                  batch: int):
    import torch

    if batch <= 0:
        raise ValueError("packed inference batch must be positive")
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(packed), batch):
            collated = _collate_packed(tokenizer, packed[start:start + batch])
            output.extend(
                logits.detach().float().cpu()
                for logits in _run_collated(model, collated, device)
            )
    return output


def reorder_packed(packed: tuple[list[int], list[tuple]],
                   order: list[int]):
    """Reorder or remove option blocks while preserving the exact prefix."""
    ids, spans = packed
    if not spans:
        raise ValueError("packed question has no option blocks")
    if len(set(order)) != len(order) or any(
            isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < len(spans) for index in order):
        raise ValueError("option order must contain unique valid indexes")
    if len(order) < 2:
        raise ValueError("a decision needs at least two retained options")
    expected_marker = spans[0][0]
    blocks = []
    for marker, start, end in spans:
        if marker != expected_marker or start != marker + 1 or not start <= end:
            raise ValueError("packed option blocks are not contiguous")
        blocks.append(ids[marker:end])
        expected_marker = end
    if expected_marker != len(ids):
        raise ValueError("packed question has trailing tokens after options")
    reordered_ids = list(ids[:spans[0][0]])
    reordered_spans = []
    for index in order:
        marker = len(reordered_ids)
        reordered_ids.extend(blocks[index])
        reordered_spans.append((marker, marker + 1, len(reordered_ids)))
    return reordered_ids, reordered_spans


def _permutation_orders(option_count: int, identity: str, seed: int,
                        requested: int) -> list[list[int]]:
    if option_count < 2 or requested <= 0:
        return []
    original = tuple(range(option_count))
    generator = random.Random(int(digest_value({
        "identity": identity, "seed": seed, "operation": "permutation",
    })[:16], 16))
    orders = []
    seen = {original}
    attempts = max(20, requested * 20)
    for _ in range(attempts):
        candidate = list(original)
        generator.shuffle(candidate)
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            orders.append(candidate)
            if len(orders) == requested:
                return orders
    for shift in range(1, option_count):
        candidate = list(original[shift:] + original[:shift])
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            orders.append(candidate)
            if len(orders) == requested:
                break
    return orders


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(
        0, math.ceil(probability * len(ordered)) - 1,
    ))
    return ordered[index]


def _stress_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "not_applicable", "records": 0, "variants": 0}
    agreements = [row["agreement"] for row in rows]
    distances = [row["mean_tv"] for row in rows]
    return {
        "status": "ok",
        "records": len(rows),
        "variants": sum(row["variants"] for row in rows),
        "semantic_top1_agreement": _mean(agreements),
        "mean_total_variation": _mean(distances),
        "p95_record_total_variation": _quantile(distances, 0.95),
        "max_record_total_variation": max(distances),
    }


def _grouped_stress(rows: list[dict]) -> dict:
    result = {"all": _stress_summary(rows)}
    for field in ("track", "source"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        result[field] = {
            key: _stress_summary(values)
            for key, values in sorted(grouped.items())
        }
    return result


def choice_stress_report(model, tokenizer, records: list[dict], device: str,
                         batch: int, max_len: int, temperature: float,
                         seed: int, permutations: int,
                         removals_per_record: int) -> dict:
    """Measure Choice stability with an identical retained state prefix."""
    import torch
    from .model import pack_question

    units = [
        item for item in independent_formal_units(records)
        if item["record"]["type"] == "choice"
    ]
    packed_units = []
    rejected = 0
    for item in units:
        packed = pack_question(
            tokenizer, item["record"]["state"],
            item["record"]["instructions"], item["record"]["options"],
            max_len,
        )
        if packed is None:
            rejected += 1
        else:
            packed_units.append((item, packed))
    if not packed_units:
        empty = {
            "raw": _grouped_stress([]),
            "temperature_scaled": _grouped_stress([]),
        }
        return {
            "status": "not_applicable",
            "submitted": len(units),
            "rejected_context": rejected,
            "permutation": empty,
            "option_removal": empty,
        }

    base_logits = _infer_packed(
        model, tokenizer, [packed for _, packed in packed_units], device, batch,
    )
    permutation_variants = []
    for unit_index, (item, packed) in enumerate(packed_units):
        orders = _permutation_orders(
            len(item["record"]["options"]),
            item["meta"]["record_sha256"], seed, permutations,
        )
        for order in orders:
            permutation_variants.append({
                "unit_index": unit_index,
                "order": order,
                "packed": reorder_packed(packed, order),
            })
    permutation_logits = _infer_packed(
        model, tokenizer,
        [variant["packed"] for variant in permutation_variants],
        device, batch,
    ) if permutation_variants else []
    removal_variants = []
    skipped_binary = 0
    skipped_no_controlled_removal = 0
    for unit_index, (item, packed) in enumerate(packed_units):
        option_count = len(item["record"]["options"])
        if option_count < 3:
            skipped_binary += 1
            continue
        protected = {
            int(base_logits[unit_index].argmax()),
            int(item["record"]["label"]),
        }
        candidates = [
            index for index in range(option_count) if index not in protected
        ]
        if not candidates:
            skipped_no_controlled_removal += 1
            continue
        candidates.sort(key=lambda index: digest_value({
            "identity": item["meta"]["record_sha256"],
            "seed": seed,
            "operation": "option-removal",
            "index": index,
        }))
        for removed in candidates[:removals_per_record]:
            keep = [index for index in range(option_count) if index != removed]
            removal_variants.append({
                "unit_index": unit_index,
                "removed": removed,
                "keep": keep,
                "packed": reorder_packed(packed, keep),
            })
    removal_logits = _infer_packed(
        model, tokenizer,
        [variant["packed"] for variant in removal_variants], device, batch,
    ) if removal_variants else []

    def summaries(scale: float) -> tuple[dict, dict]:
        base_probabilities = [
            torch.softmax(logits.to(torch.float64) / scale, -1)
            for logits in base_logits
        ]
        permutation_values = defaultdict(list)
        for variant, logits in zip(permutation_variants, permutation_logits):
            unit_index = variant["unit_index"]
            order = variant["order"]
            probabilities = torch.softmax(
                logits.to(torch.float64) / scale, -1,
            )
            aligned = torch.empty_like(probabilities)
            for new_index, original_index in enumerate(order):
                aligned[original_index] = probabilities[new_index]
            baseline = base_probabilities[unit_index]
            permutation_values[unit_index].append({
                "agreement": int(base_logits[unit_index].argmax()
                                 == logits[torch.tensor(order).argsort()].argmax()),
                "tv": float(0.5 * (baseline - aligned).abs().sum()),
            })
        permutation_rows = []
        for unit_index, values in permutation_values.items():
            item = packed_units[unit_index][0]
            permutation_rows.append({
                "track": item["meta"]["track"],
                "source": item["meta"]["source"],
                "variants": len(values),
                "agreement": _mean([
                    value["agreement"] for value in values
                ]),
                "mean_tv": _mean([value["tv"] for value in values]),
            })

        removal_values = defaultdict(list)
        for variant, logits in zip(removal_variants, removal_logits):
            unit_index = variant["unit_index"]
            keep = variant["keep"]
            probabilities = torch.softmax(
                logits.to(torch.float64) / scale, -1,
            )
            baseline = base_probabilities[unit_index]
            survivor_baseline = baseline[keep]
            survivor_baseline = survivor_baseline / survivor_baseline.sum()
            baseline_top = int(base_logits[unit_index].argmax())
            removal_values[unit_index].append({
                "agreement": int(
                    keep[int(logits.argmax())] == baseline_top
                ),
                "tv": float(
                    0.5 * (survivor_baseline - probabilities).abs().sum()
                ),
            })
        removal_rows = []
        for unit_index, values in removal_values.items():
            item = packed_units[unit_index][0]
            removal_rows.append({
                "track": item["meta"]["track"],
                "source": item["meta"]["source"],
                "variants": len(values),
                "agreement": _mean([
                    value["agreement"] for value in values
                ]),
                "mean_tv": _mean([value["tv"] for value in values]),
            })
        return _grouped_stress(permutation_rows), _grouped_stress(removal_rows)

    raw_permutation, raw_removal = summaries(1.0)
    scaled_permutation, scaled_removal = summaries(temperature)

    return {
        "status": "ok",
        "submitted": len(units),
        "rejected_context": rejected,
        "usable": len(packed_units),
        "temperature": temperature,
        "unit": "one deterministic Choice question per parent and formal cell",
        "context_control": (
            "exact retained state and instruction token prefix reused for every "
            "variant"
        ),
        "permutation": {
            "requested_variants_per_record": permutations,
            "raw": raw_permutation,
            "temperature_scaled": scaled_permutation,
        },
        "option_removal": {
            "requested_variants_per_record": removals_per_record,
            "protected_options": "gold label and baseline top prediction",
            "skipped_binary": skipped_binary,
            "skipped_no_controlled_removal": skipped_no_controlled_removal,
            "raw": raw_removal,
            "temperature_scaled": scaled_removal,
        },
    }


def _synchronize_device(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _latency_distribution(milliseconds: list[float], batch_size: int) -> dict:
    return {
        "trials": len(milliseconds),
        "batch_size": batch_size,
        "mean_ms_per_batch": statistics.fmean(milliseconds),
        "median_ms_per_batch": statistics.median(milliseconds),
        "p90_ms_per_batch": _quantile(milliseconds, 0.90),
        "p95_ms_per_batch": _quantile(milliseconds, 0.95),
        "min_ms_per_batch": min(milliseconds),
        "max_ms_per_batch": max(milliseconds),
        "mean_ms_per_record": statistics.fmean(milliseconds) / batch_size,
        "records_per_second": (
            1000.0 * batch_size / statistics.fmean(milliseconds)
        ),
        "raw_ms": milliseconds,
    }


def _time_operations(operations: list, warmup: int, device: str) -> list[float]:
    import torch

    with torch.no_grad():
        for operation in operations[:warmup]:
            operation()
            _synchronize_device(device)
        measured = []
        for operation in operations[warmup:]:
            _synchronize_device(device)
            started = time.perf_counter_ns()
            operation()
            _synchronize_device(device)
            measured.append((time.perf_counter_ns() - started) / 1_000_000)
    return measured


def latency_report(model, tokenizer, predictions: list[dict], device: str,
                   max_len: int, inference_batch: int, seed: int,
                   max_records: int, warmup: int,
                   repetitions: int, independent_runs: int) -> dict:
    """Measure model-only and packing-inclusive latency on a frozen workload."""
    import torch
    from .model import collate, pack_question

    if (inference_batch <= 0 or max_records <= 0 or warmup < 0
            or repetitions <= 0 or independent_runs <= 0):
        raise ValueError("latency configuration contains a nonpositive count")
    units = independent_formal_units(predictions)
    eligible = [
        item for item in units
        if item.get("status") == "ok" and item.get("packed_tokens") is not None
    ]
    eligible.sort(key=lambda item: digest_value({
        "seed": seed,
        "operation": "latency-workload",
        "record": item["meta"]["record_sha256"],
    }))
    selected_by_digest = {}

    def select(item: dict | None, stratum: str) -> None:
        if item is None:
            return
        identity = item["meta"]["record_sha256"]
        if identity not in selected_by_digest and len(selected_by_digest) >= max_records:
            return
        selected_by_digest.setdefault(identity, {
            "item": item, "strata": set(),
        })["strata"].add(stratum)

    select(next((item for item in eligible
                 if item["record"]["type"] == "noul"), None), "noul")
    select(next((item for item in eligible
                 if item["record"]["type"] == "score"), None), "score")
    choices = [item for item in eligible if item["record"]["type"] == "choice"]
    select(min(
        choices,
        key=lambda item: (
            -len(item["record"]["options"]),
            item["meta"]["record_sha256"],
        ),
    ) if choices else None, "highest-option-count-choice")
    select(max(
        eligible,
        key=lambda item: (
            item["packed_tokens"], item["meta"]["record_sha256"],
        ),
    ) if eligible else None, "longest-context")
    for item in eligible:
        select(item, "digest-selected-mixture")
        if len(selected_by_digest) == max_records:
            break

    selected = []
    for descriptor in selected_by_digest.values():
        item = descriptor["item"]
        packed = pack_question(
            tokenizer, item["record"]["state"],
            item["record"]["instructions"], item["record"]["options"],
            max_len,
        )
        if packed is None:
            raise RuntimeError("latency workload changed from accepted to rejected")
        if len(packed[0]) != item["packed_tokens"]:
            raise RuntimeError("latency workload token count changed after inference")
        selected.append((item, packed, sorted(descriptor["strata"])))
    if not selected:
        return {
            "status": "not_applicable", "submitted": len(units),
            "eligible": 0,
        }

    token_lengths = [len(packed[0]) for _, packed, _ in selected]
    option_counts = [len(packed[1]) for _, packed, _ in selected]
    workload_descriptor = [{
        "record_sha256": item["meta"]["record_sha256"],
        "track": item["meta"]["track"],
        "source": item["meta"]["source"],
        "primitive": item["record"]["type"],
        "language": item["meta"]["language"],
        "tokens": len(packed[0]),
        "options": len(packed[1]),
        "strata": strata,
    } for item, packed, strata in selected]
    results = {}
    for requested_batch in sorted({1, inference_batch}):
        batch_size = min(requested_batch, len(selected))
        scopes = {"model_only": [], "packing_and_model": []}
        for run_index in range(independent_runs):
            trial_count = warmup + repetitions
            scheduled = []
            for trial in range(trial_count):
                start = (
                    (run_index * trial_count + trial) * batch_size
                ) % len(selected)
                scheduled.append([
                    (start + offset) % len(selected)
                    for offset in range(batch_size)
                ])
            model_batches = [
                _collate_packed(
                    tokenizer,
                    [selected[index][1] for index in indexes],
                )
                for indexes in scheduled
            ]
            for collated_batch in model_batches:
                for name in (
                    "input_ids", "attention_mask", "option_pos", "group_ptr",
                ):
                    collated_batch[name] = collated_batch[name].to(device)
            model_operations = [
                (lambda collated_batch=collated_batch:
                 _run_collated(model, collated_batch, device))
                for collated_batch in model_batches
            ]
            end_to_end_operations = [
                (lambda indexes=indexes: _run_collated(
                    model,
                    collate(
                        tokenizer,
                        [selected[index][0]["record"] for index in indexes],
                        max_len,
                    ),
                    device,
                ))
                for indexes in scheduled
            ]
            if run_index % 2 == 0:
                scopes["model_only"].append(
                    _time_operations(model_operations, warmup, device)
                )
                scopes["packing_and_model"].append(
                    _time_operations(end_to_end_operations, warmup, device)
                )
            else:
                end_to_end_times = _time_operations(
                    end_to_end_operations, warmup, device,
                )
                model_times = _time_operations(
                    model_operations, warmup, device,
                )
                scopes["model_only"].append(model_times)
                scopes["packing_and_model"].append(end_to_end_times)
            del model_batches
        results[str(batch_size)] = {}
        for scope, run_samples in scopes.items():
            flattened = [value for samples in run_samples for value in samples]
            results[str(batch_size)][scope] = {
                "aggregate": _latency_distribution(flattened, batch_size),
                "independent_runs": [
                    _latency_distribution(samples, batch_size)
                    for samples in run_samples
                ],
            }

    memory = None
    if device == "mps":
        memory = {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }
    return {
        "status": "ok",
        "submitted": len(units),
        "eligible": len(eligible),
        "workload_records": len(selected),
        "workload_sha256": digest_value(workload_descriptor),
        "workload_manifest": workload_descriptor,
        "selection": "predeclared strata followed by digest-selected mixture",
        "warmup_trials": warmup,
        "measured_trials": repetitions,
        "independent_runs": independent_runs,
        "timed_scopes": {
            "model_only": "pretokenized and resident input tensors plus model",
            "packing_and_model": (
                "tokenization, dynamic collation, device transfer, and model"
            ),
            "excluded": "checkpoint loading and HTTP transport",
            "order": "alternated between independent runs",
        },
        "request_semantics": (
            "each batch entry is one independent question; this is not an HTTP "
            "or shared-state multi-question latency measurement"
        ),
        "synchronization": (
            "torch.mps.synchronize before and after every timed MPS trial"
            if device == "mps" else "synchronous CPU execution"
        ),
        "tokens": {
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "p90": _quantile(token_lengths, 0.90),
            "max": max(token_lengths),
        },
        "options": {
            "min": min(option_counts),
            "median": statistics.median(option_counts),
            "p90": _quantile(option_counts, 0.90),
            "max": max(option_counts),
        },
        "timings": results,
        "provenance": {
            "device": device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "model_dtype": str(next(model.parameters()).dtype),
            "attention_implementation": getattr(
                model.backbone.config, "_attn_implementation", None,
            ),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_memory": memory,
            "host_state_requirement": (
                "run on an idle host with no trainer or competing accelerator "
                "workload; power and thermal state are not machine-verified"
            ),
        },
    }


def fit_source_balanced_temperature(predictions: list[dict]) -> float:
    """Fit one deployable scalar with equal weight per training source."""
    import torch
    import torch.nn.functional as functional

    grouped = defaultdict(list)
    for prediction in predictions:
        if prediction["status"] != "ok":
            continue
        logits = torch.tensor(prediction["logits"], dtype=torch.float64)
        target = target_distribution(prediction["record"], logits.numel())
        grouped[prediction["meta"]["source"]].append((logits, target))
    if not grouped or any(not values for values in grouped.values()):
        raise ValueError("temperature fitting has no usable source populations")
    log_temperature = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=150,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        source_losses = []
        for values in grouped.values():
            losses = [
                -(target * functional.log_softmax(logits / temperature, -1)).sum()
                for logits, target in values
            ]
            source_losses.append(torch.stack(losses).mean())
        loss = torch.stack(source_losses).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature fitting produced an invalid value")
    return temperature


def formal_cell(prediction: dict) -> str:
    if prediction["meta"]["source"] == "helpsteer2":
        return prediction["record"]["source"]
    return prediction["meta"]["track"]


def independent_formal_units(predictions: list[dict]) -> list[dict]:
    """Choose one deterministic question per parent and formal cell."""
    chosen = {}
    for prediction in predictions:
        key = (formal_cell(prediction), prediction["meta"]["parent_id"])
        identity = prediction["meta"]["record_sha256"]
        current = chosen.get(key)
        if current is None or identity < current["meta"]["record_sha256"]:
            chosen[key] = prediction
    return [chosen[key] for key in sorted(chosen)]


def calibrated_probabilities(prediction: dict, temperature: float):
    import torch

    logits = torch.tensor(prediction["logits"], dtype=torch.float64)
    return torch.softmax(logits / temperature, -1)


def conformal_by_cell(predictions: list[dict], temperature: float,
                      alpha: float) -> dict[str, dict]:
    from .calibrate import conformal_threshold

    grouped = defaultdict(list)
    for prediction in independent_formal_units(predictions):
        if prediction["status"] != "ok":
            continue
        label = prediction["record"].get("label")
        if not isinstance(label, int) or isinstance(label, bool):
            continue
        grouped[formal_cell(prediction)].append(
            (calibrated_probabilities(prediction, temperature), label)
        )
    thresholds = {}
    for cell, values in grouped.items():
        threshold = conformal_threshold(
            [probabilities for probabilities, _ in values],
            [label for _, label in values], alpha=alpha,
        )
        thresholds[cell] = {
            "q": None if math.isinf(threshold) else threshold,
            "full_set": math.isinf(threshold),
            "n": len(values),
        }
    if not thresholds:
        raise ValueError("no conformal cells have usable observations")
    return thresholds


def _write_prediction_file(path: Path, predictions: list[dict]) -> dict:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions {path}")
    _write_jsonl(path, predictions)
    return {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "records": len(predictions),
        "accepted": sum(item["status"] == "ok" for item in predictions),
    }


def fit_policy(plan_dir: str | Path, run_dir: str | Path, device: str,
               batch: int, alpha: float, thresholds: list[float],
               confidence: float = 0.95, stress_seed: int = 20260921,
               permutations: int = 3, removals_per_record: int = 1,
               latency_records: int = 64, latency_warmup: int = 30,
               latency_repetitions: int = 200,
               latency_independent_runs: int = 3) -> dict:
    from .calibrate import save_temperature_artifact
    from .checkpoint import load_completed_checkpoint

    plan_dir = Path(plan_dir).resolve()
    run_dir = Path(run_dir).resolve()
    policy_path = plan_dir / "frozen-policy.json"
    if policy_path.exists():
        raise FileExistsError(f"refusing to overwrite policy {policy_path}")
    if not 0 < alpha < 1 or not 0 < confidence < 1:
        raise ValueError("alpha and confidence must be between zero and one")
    if (not thresholds or any(not 0 <= value <= 1 for value in thresholds)
            or len(set(thresholds)) != len(thresholds)):
        raise ValueError("selective thresholds must be unique values in [0, 1]")
    if (permutations <= 0 or removals_per_record <= 0
            or latency_records <= 0 or latency_warmup < 0
            or latency_repetitions <= 0 or latency_independent_runs <= 0):
        raise ValueError("stress and latency counts must be positive")
    plan = load_plan(plan_dir)
    _, run, manifest = load_completed_checkpoint(run_dir)
    if (run["run_id"] != plan["run"]["run_id"]
            or run["spec_sha256"] != plan["run"]["spec_sha256"]):
        raise ValueError("completed run does not match the frozen plan")

    role_a = load_role(plan_dir, "A", plan)
    role_b = load_role(plan_dir, "B", plan)
    predictions_a = infer_role(run_dir, role_a, device, batch)
    predictions_b = infer_role(run_dir, role_b, device, batch)
    temperature = fit_source_balanced_temperature(predictions_a)
    conformal = conformal_by_cell(predictions_b, temperature, alpha)
    a_file = _write_prediction_file(
        plan_dir / "predictions-A.jsonl", predictions_a,
    )
    b_file = _write_prediction_file(
        plan_dir / "predictions-B.jsonl", predictions_b,
    )
    formal_cells = sorted({
        formal_cell(item) for item in independent_formal_units(predictions_b)
        if item["status"] == "ok"
    })
    family_size = len(formal_cells) * len(thresholds)
    simultaneous_confidence = 1.0 - (1.0 - confidence) / family_size
    current = manifest["current"]
    policy = {
        "version": 1,
        "status": "frozen",
        "plan_sha256": plan["plan_sha256"],
        "run": {
            "run_id": run["run_id"],
            "spec_sha256": run["spec_sha256"],
            "generation": current["generation"],
            "checkpoint_sha256": current["sha256"],
        },
        "inference": {
            "max_len": run["spec"]["config"]["max_len"],
            "device_used_for_fit": device,
            "batch_used_for_fit": batch,
        },
        "temperature": {
            "value": temperature,
            "objective": "source-balanced soft-target cross-entropy",
            "role": "A",
            "predictions": a_file,
        },
        "conformal": {
            "alpha": alpha,
            "role": "B",
            "unit": "one deterministic question per parent and formal cell",
            "target": (
                "stored hard label; for soft-label sources this is the declared "
                "majority or consensus label, not an individual annotation"
            ),
            "thresholds_by_cell": conformal,
            "predictions": b_file,
        },
        "selective_risk": {
            "score": "maximum temperature-scaled probability",
            "action": "argmax label",
            "thresholds": sorted(thresholds),
            "formal_cells": formal_cells,
            "family_size": family_size,
            "family_confidence": confidence,
            "per_bound_confidence": simultaneous_confidence,
            "correction": "Bonferroni",
        },
        "stress": {
            "role": "C",
            "primitive": "choice",
            "seed": stress_seed,
            "permutations_per_record": permutations,
            "removals_per_record": removals_per_record,
            "unit": "one deterministic question per parent and formal cell",
            "option_removal_protects": "gold label and baseline top prediction",
            "context_control": (
                "reuse the exact retained state and instruction token prefix"
            ),
        },
        "latency": {
            "role": "C",
            "seed": stress_seed,
            "max_records": latency_records,
            "warmup_trials": latency_warmup,
            "measured_trials": latency_repetitions,
            "independent_runs": latency_independent_runs,
            "batch_sizes": sorted({1, batch}),
            "modes": ["model_only", "packing_and_model"],
        },
        "code": plan["code"],
    }
    policy["policy_sha256"] = digest_value(policy)
    _atomic_json(policy_path, policy)
    save_temperature_artifact(
        run_dir, temperature, run, manifest,
        {
            "dataset": "haetae-frozen-source-balanced-mixture-v1",
            "split": f"{plan['plan_sha256']}:A",
            "fingerprint": plan["files"]["A.jsonl"]["sha256"],
            "n": a_file["accepted"],
        },
        int(run["spec"]["config"]["max_len"]),
    )
    return policy


def load_policy(plan_dir: str | Path, plan: dict) -> dict:
    policy = json.loads((Path(plan_dir) / "frozen-policy.json").read_text())
    expected = policy.get("policy_sha256")
    unsigned = dict(policy)
    unsigned.pop("policy_sha256", None)
    if expected != digest_value(unsigned):
        raise ValueError("frozen policy digest mismatch")
    if policy.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("frozen policy belongs to another plan")
    return policy


def reliability_bins(confidences: list[float], outcomes: list[float],
                     bins: int = 15) -> dict:
    if not confidences or len(confidences) != len(outcomes) or bins <= 0:
        raise ValueError("reliability inputs must have equal nonzero length")
    details, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            item for item, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (index == bins - 1 and confidence == 1.0)
        ]
        if members:
            mean_confidence = sum(confidences[item] for item in members) / len(members)
            mean_outcome = sum(outcomes[item] for item in members) / len(members)
            ece += len(members) / len(confidences) * abs(
                mean_confidence - mean_outcome
            )
        else:
            mean_confidence = mean_outcome = None
        details.append({
            "lower": lower,
            "upper": upper,
            "right_closed": index == bins - 1,
            "n": len(members),
            "mean_prediction": mean_confidence,
            "mean_outcome": mean_outcome,
        })
    return {"ece": ece, "bins": details}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def cluster_bootstrap_interval(values_by_parent: dict[str, list[float]],
                               samples: int = 500) -> dict | None:
    parents = sorted(values_by_parent)
    if len(parents) < 2:
        return None
    seed = int(digest_value(parents)[:16], 16)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = generator.choices(parents, k=len(parents))
        values = [
            value for parent in selected for value in values_by_parent[parent]
        ]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    lower = estimates[max(0, math.floor(0.025 * samples) - 1)]
    upper = estimates[min(samples - 1, math.ceil(0.975 * samples) - 1)]
    return {
        "lower": lower, "upper": upper, "samples": samples,
        "unit": "parent",
    }


def metric_summary(predictions: list[dict], temperature: float) -> dict:
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import f1_score

    nll, brier_histogram, brier_annotation = [], [], []
    correct, confidences, hard_labels, predicted_labels = [], [], [], []
    rps, expected_mae = [], []
    yes_probabilities, yes_targets = [], []
    ordinal_thresholds = defaultdict(lambda: ([], []))
    option_counts = set()
    clustered = defaultdict(lambda: defaultdict(list))
    accepted = [item for item in predictions if item["status"] == "ok"]
    for prediction in accepted:
        logits = torch.tensor(prediction["logits"], dtype=torch.float64)
        target = target_distribution(prediction["record"], logits.numel())
        log_probabilities = functional.log_softmax(logits / temperature, -1)
        probabilities = log_probabilities.exp()
        label = int(prediction["record"]["label"])
        predicted = int(probabilities.argmax())
        one_hot = functional.one_hot(
            torch.tensor(label), logits.numel()
        ).to(torch.float64)
        nll.append(float(-(target * log_probabilities).sum()))
        brier_histogram.append(float(((probabilities - target) ** 2).sum()))
        brier_annotation.append(float(
            1.0 + (probabilities ** 2).sum()
            - 2.0 * (probabilities * target).sum()
        ))
        correct.append(float(predicted == label))
        confidences.append(float(probabilities.max()))
        hard_labels.append(label)
        predicted_labels.append(predicted)
        option_counts.add(logits.numel())
        parent = prediction["meta"]["parent_id"]
        clustered["nll"][parent].append(nll[-1])
        clustered["brier_histogram"][parent].append(brier_histogram[-1])
        clustered["brier_annotation"][parent].append(brier_annotation[-1])
        clustered["accuracy"][parent].append(correct[-1])
        if prediction["record"]["type"] == "score":
            cumulative_probability = probabilities.cumsum(0)[:-1]
            cumulative_target = target.cumsum(0)[:-1]
            rps.append(float(
                ((cumulative_probability - cumulative_target) ** 2).mean()
            ))
            levels = torch.arange(logits.numel(), dtype=torch.float64)
            expected_mae.append(abs(float((probabilities * levels).sum()) - label))
            clustered["ranked_probability_score"][parent].append(rps[-1])
            clustered["expected_index_mae"][parent].append(expected_mae[-1])
            for threshold, (predicted_cumulative, target_cumulative) in enumerate(
                    zip(cumulative_probability, cumulative_target)):
                predicted_values, target_values = ordinal_thresholds[
                    (logits.numel(), threshold)
                ]
                predicted_values.append(float(predicted_cumulative))
                target_values.append(float(target_cumulative))
        if prediction["record"]["type"] == "noul":
            yes_probabilities.append(float(probabilities[0]))
            yes_targets.append(float(target[0]))
    if not accepted:
        return {
            "status": "no_usable_records", "submitted": len(predictions),
            "n": 0, "rejected": len(predictions),
        }
    top_reliability = reliability_bins(confidences, correct)
    result = {
        "status": "ok",
        "submitted": len(predictions),
        "n": len(accepted),
        "rejected": len(predictions) - len(accepted),
        "temperature": temperature,
        "nll": _mean(nll),
        "brier_histogram": _mean(brier_histogram),
        "brier_annotation": _mean(brier_annotation),
        "accuracy": _mean(correct),
        "macro_f1": float(f1_score(
            hard_labels, predicted_labels,
            labels=(
                list(range(next(iter(option_counts))))
                if len(option_counts) == 1
                else sorted(set(hard_labels) | set(predicted_labels))
            ),
            average="macro", zero_division=0,
        )),
        "mean_confidence": _mean(confidences),
        "confidence_bias": _mean(confidences) - _mean(correct),
        "top_label_reliability": top_reliability,
        "option_counts": sorted(option_counts),
        "cluster_bootstrap_ci95": {
            metric: interval
            for metric, values in clustered.items()
            if (interval := cluster_bootstrap_interval(values)) is not None
        },
    }
    if rps:
        result["ranked_probability_score"] = _mean(rps)
        result["expected_index_mae"] = _mean(expected_mae)
        result["ordinal_threshold_reliability"] = {
            f"K={option_count}:Y<={threshold}": reliability_bins(
                predicted_values, target_values,
            )
            for (option_count, threshold),
            (predicted_values, target_values) in sorted(
                ordinal_thresholds.items()
            )
        }
    if yes_probabilities:
        result["yes_probability_reliability"] = reliability_bins(
            yes_probabilities, yes_targets,
        )
    return result


def grouped_metric_report(predictions: list[dict], temperature: float) -> dict:
    groupings = {
        "cell": lambda item: formal_cell(item),
        "source": lambda item: item["meta"]["source"],
        "primitive": lambda item: item["record"]["type"],
        "language": lambda item: item["meta"]["language"],
        "option_count": lambda item: str(len(item["record"]["options"])),
        "token_length": lambda item: (
            "rejected" if item["packed_tokens"] is None else
            "0-256" if item["packed_tokens"] <= 256 else
            "257-768" if item["packed_tokens"] <= 768 else
            "769-1536" if item["packed_tokens"] <= 1536 else "1537+"
        ),
    }
    report = {"all": metric_summary(predictions, temperature)}
    for grouping, key_function in groupings.items():
        groups = defaultdict(list)
        for prediction in predictions:
            groups[key_function(prediction)].append(prediction)
        report[grouping] = {
            key: metric_summary(values, temperature)
            for key, values in sorted(groups.items())
        }
    macro_fields = (
        "nll", "brier_histogram", "brier_annotation", "accuracy",
        "macro_f1", "confidence_bias", "ranked_probability_score",
        "expected_index_mae",
    )
    report["source_macro"] = {
        field: _mean([
            metrics[field] for metrics in report["source"].values()
            if metrics.get("status") == "ok" and field in metrics
        ])
        for field in macro_fields
    }
    return report


def baseline_metric_reports(predictions: list[dict],
                            training_priors: dict) -> dict:
    uniform, empirical = [], []
    empirical_available = 0
    for prediction in predictions:
        uniform_item = dict(prediction)
        empirical_item = dict(prediction)
        if prediction["status"] != "ok":
            uniform.append(uniform_item)
            empirical.append(empirical_item)
            continue
        option_count = len(prediction["record"]["options"])
        uniform_item["logits"] = [0.0] * option_count
        uniform.append(uniform_item)
        prior = training_priors.get(ontology_digest(prediction["record"]))
        if (prior is None
                or len(prior.get("probabilities", [])) != option_count):
            empirical_item["status"] = "training_prior_not_available"
            empirical_item["logits"] = None
        else:
            empirical_available += 1
            empirical_item["logits"] = [
                math.log(probability)
                for probability in prior["probabilities"]
            ]
        empirical.append(empirical_item)
    return {
        "uniform": {
            "description": "equal probability over each record's options",
            "metrics": grouped_metric_report(uniform, 1.0),
        },
        "training_only_empirical_prior": {
            "description": (
                "add-one-smoothed target frequencies from consumed training "
                "records with an exact source, primitive, and option ontology"
            ),
            "available": empirical_available,
            "submitted": len(predictions),
            "metrics": grouped_metric_report(empirical, 1.0),
        },
    }


def paired_temperature_differences(predictions: list[dict],
                                   temperature: float) -> dict:
    import torch
    import torch.nn.functional as functional

    differences = defaultdict(list)
    clustered = defaultdict(lambda: defaultdict(list))
    for prediction in predictions:
        if prediction["status"] != "ok":
            continue
        logits = torch.tensor(prediction["logits"], dtype=torch.float64)
        target = target_distribution(prediction["record"], logits.numel())
        raw_log = functional.log_softmax(logits, -1)
        scaled_log = functional.log_softmax(logits / temperature, -1)
        raw_probability = raw_log.exp()
        scaled_probability = scaled_log.exp()
        values = {
            "nll": float(
                -(target * scaled_log).sum() + (target * raw_log).sum()
            ),
            "brier_histogram": float(
                ((scaled_probability - target) ** 2).sum()
                - ((raw_probability - target) ** 2).sum()
            ),
            "brier_annotation": float(
                (scaled_probability ** 2).sum()
                - 2.0 * (scaled_probability * target).sum()
                - (raw_probability ** 2).sum()
                + 2.0 * (raw_probability * target).sum()
            ),
        }
        parent = prediction["meta"]["parent_id"]
        for metric, value in values.items():
            differences[metric].append(value)
            clustered[metric][parent].append(value)
    return {
        "definition": "temperature_scaled minus raw; negative favors scaling",
        "n": len(next(iter(differences.values()), [])),
        "mean_difference": {
            metric: _mean(values) for metric, values in differences.items()
        },
        "parent_cluster_bootstrap_ci95": {
            metric: cluster_bootstrap_interval(values)
            for metric, values in clustered.items()
            if len(values) >= 2
        },
    }


def conformal_report(predictions: list[dict], policy: dict,
                     temperature: float) -> dict:
    from .calibrate import prediction_set

    grouped = defaultdict(list)
    for prediction in independent_formal_units(predictions):
        grouped[formal_cell(prediction)].append(prediction)
    result = {}
    thresholds = policy["conformal"]["thresholds_by_cell"]
    for cell, values in sorted(grouped.items()):
        descriptor = thresholds.get(cell)
        if descriptor is None:
            result[cell] = {
                "status": "not_predeclared", "submitted": len(values),
            }
            continue
        sizes, normalized_sizes, covered = [], [], []
        rejected = 0
        for prediction in values:
            if prediction["status"] != "ok":
                rejected += 1
                continue
            probabilities = calibrated_probabilities(prediction, temperature)
            selected = (
                list(range(probabilities.numel()))
                if descriptor["full_set"]
                else prediction_set(probabilities, descriptor["q"])
            )
            sizes.append(len(selected))
            normalized_sizes.append(
                len(selected) / probabilities.numel()
            )
            covered.append(int(prediction["record"]["label"] in selected))
        if not sizes:
            result[cell] = {
                "status": "no_usable_records", "submitted": len(values),
                "rejected": rejected,
            }
            continue
        ordered_sizes = sorted(sizes)
        ordered_normalized_sizes = sorted(normalized_sizes)
        p90_index = min(len(sizes) - 1, math.ceil(0.9 * len(sizes)) - 1)
        result[cell] = {
            "status": "ok",
            "submitted": len(values),
            "n": len(sizes),
            "rejected": rejected,
            "coverage": _mean(covered),
            "mean_set_size": _mean(sizes),
            "median_set_size": ordered_sizes[len(sizes) // 2],
            "p90_set_size": ordered_sizes[p90_index],
            "mean_normalized_set_size": _mean(normalized_sizes),
            "median_normalized_set_size": ordered_normalized_sizes[
                len(normalized_sizes) // 2
            ],
            "p90_normalized_set_size": ordered_normalized_sizes[p90_index],
            "empty_rate": sum(size == 0 for size in sizes) / len(sizes),
            "singleton_rate": sum(size == 1 for size in sizes) / len(sizes),
            "full_set_rate": sum(
                size == len(prediction["record"]["options"])
                for size, prediction in zip(sizes, [
                    item for item in values if item["status"] == "ok"
                ])
            ) / len(sizes),
            "calibration_n": descriptor["n"],
            "q": descriptor["q"],
            "forced_full_set": descriptor["full_set"],
        }
    return result


def selective_risk_report(predictions: list[dict], policy: dict,
                          temperature: float) -> dict:
    from .calibrate import binomial_upper_bound

    grouped = defaultdict(list)
    for prediction in independent_formal_units(predictions):
        grouped[formal_cell(prediction)].append(prediction)
    confidence = policy["selective_risk"]["per_bound_confidence"]
    result = {}
    for cell in policy["selective_risk"]["formal_cells"]:
        values = grouped.get(cell, [])
        eligible = [item for item in values if item["status"] == "ok"]
        cells = {}
        for threshold in policy["selective_risk"]["thresholds"]:
            accepted = []
            for prediction in eligible:
                probabilities = calibrated_probabilities(
                    prediction, temperature,
                )
                if float(probabilities.max()) >= threshold:
                    accepted.append((probabilities, prediction["record"]["label"]))
            errors = sum(
                int(probabilities.argmax()) != label
                for probabilities, label in accepted
            )
            if not accepted:
                status, upper = "not_certified", None
            else:
                status = "bounded"
                upper = binomial_upper_bound(
                    errors, len(accepted), confidence=confidence,
                )
            cells[str(threshold)] = {
                "status": status,
                "submitted": len(values),
                "eligible": len(eligible),
                "accepted": len(accepted),
                "errors": errors,
                "coverage_of_eligible": (
                    len(accepted) / len(eligible) if eligible else 0.0
                ),
                "coverage_of_submitted": (
                    len(accepted) / len(values) if values else 0.0
                ),
                "risk_upper_bound": upper,
                "confidence": confidence,
            }
        result[cell] = cells
    return result


def evaluate_policy(plan_dir: str | Path, run_dir: str | Path, device: str,
                    batch: int) -> dict:
    from .checkpoint import load_completed_checkpoint
    from .certify import load_model

    plan_dir = Path(plan_dir).resolve()
    run_dir = Path(run_dir).resolve()
    metrics_path = plan_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite metrics {metrics_path}")
    plan = load_plan(plan_dir)
    policy = load_policy(plan_dir, plan)
    _, run, manifest = load_completed_checkpoint(run_dir)
    current = manifest["current"]
    expected_run = policy["run"]
    if (run["run_id"] != expected_run["run_id"]
            or run["spec_sha256"] != expected_run["spec_sha256"]
            or current["generation"] != expected_run["generation"]
            or current["sha256"] != expected_run["checkpoint_sha256"]):
        raise ValueError("completed checkpoint does not match frozen policy")
    role_c = load_role(plan_dir, "C", plan)
    predictions = infer_role(run_dir, role_c, device, batch)
    prediction_file = _write_prediction_file(
        plan_dir / "predictions-C.jsonl", predictions,
    )
    temperature = policy["temperature"]["value"]
    model, tokenizer, checkpoint, _, _ = load_model(run_dir, device)
    stress_policy = policy["stress"]
    latency_policy = policy["latency"]
    try:
        stress = choice_stress_report(
            model, tokenizer, role_c, device, batch,
            int(checkpoint["config"]["max_len"]), temperature,
            stress_policy["seed"],
            stress_policy["permutations_per_record"],
            stress_policy["removals_per_record"],
        )
        latency = latency_report(
            model, tokenizer, predictions, device,
            int(checkpoint["config"]["max_len"]), batch,
            latency_policy["seed"], latency_policy["max_records"],
            latency_policy["warmup_trials"],
            latency_policy["measured_trials"],
            latency_policy["independent_runs"],
        )
    finally:
        del model
        gc.collect()
        if device == "mps":
            import torch

            torch.mps.empty_cache()
    report = {
        "version": 1,
        "status": "evaluated",
        "plan_sha256": plan["plan_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "run": expected_run,
        "predictions": prediction_file,
        "metrics": {
            "raw": grouped_metric_report(predictions, 1.0),
            "temperature_scaled": grouped_metric_report(
                predictions, temperature,
            ),
        },
        "baselines": baseline_metric_reports(
            predictions, plan["training_only_priors"]["values"],
        ),
        "paired_temperature_differences": paired_temperature_differences(
            predictions, temperature,
        ),
        "conformal": conformal_report(predictions, policy, temperature),
        "selective_risk": selective_risk_report(
            predictions, policy, temperature,
        ),
        "choice_stress": stress,
        "latency": latency,
        "statistical_policy": {
            "formal_unit": (
                "one deterministic question per parent and formal cell"
            ),
            "selective_family_size": policy["selective_risk"]["family_size"],
            "family_confidence": policy["selective_risk"]["family_confidence"],
            "per_bound_confidence": policy["selective_risk"]["per_bound_confidence"],
            "multiplicity_correction": "Bonferroni",
        },
    }
    report["metrics_sha256"] = digest_value(report)
    _atomic_json(metrics_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", required=True)
    prepare_parser.add_argument("--expect-run-id", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--max-parents", type=int, default=1600)
    prepare_parser.add_argument("--sources")
    prepare_parser.add_argument("--seed", type=int, default=20260920)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--plan", required=True)
    fit_parser.add_argument("--run-dir", required=True)
    fit_parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    fit_parser.add_argument("--batch", type=int, default=4)
    fit_parser.add_argument("--alpha", type=float, default=0.1)
    fit_parser.add_argument("--thresholds", default="0.5,0.7,0.9")
    fit_parser.add_argument("--stress-seed", type=int, default=20260921)
    fit_parser.add_argument("--permutations", type=int, default=3)
    fit_parser.add_argument("--removals-per-record", type=int, default=1)
    fit_parser.add_argument("--latency-records", type=int, default=64)
    fit_parser.add_argument("--latency-warmup", type=int, default=30)
    fit_parser.add_argument("--latency-repetitions", type=int, default=200)
    fit_parser.add_argument("--latency-independent-runs", type=int, default=3)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", required=True)
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    evaluate_parser.add_argument("--batch", type=int, default=4)
    evaluate_parser.add_argument(
        "--allow-certification", action="store_true",
        help="explicitly permit reading the frozen C population",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        source_names = (
            [name.strip() for name in args.sources.split(",")]
            if args.sources else None
        )
        plan = prepare(
            args.run_dir, args.out, args.expect_run_id,
            max_parents=args.max_parents, sources=source_names, seed=args.seed,
        )
        print(json.dumps({
            "plan_sha256": plan["plan_sha256"],
            "files": plan["files"],
        }, indent=2))
    elif args.command == "fit":
        thresholds = [
            float(value) for value in args.thresholds.split(",") if value
        ]
        policy = fit_policy(
            args.plan, args.run_dir, args.device, args.batch, args.alpha,
            thresholds, stress_seed=args.stress_seed,
            permutations=args.permutations,
            removals_per_record=args.removals_per_record,
            latency_records=args.latency_records,
            latency_warmup=args.latency_warmup,
            latency_repetitions=args.latency_repetitions,
            latency_independent_runs=args.latency_independent_runs,
        )
        print(json.dumps({
            "policy_sha256": policy["policy_sha256"],
            "run": policy["run"],
            "temperature": policy["temperature"]["value"],
            "formal_cells": policy["selective_risk"]["formal_cells"],
        }, indent=2))
    elif args.command == "evaluate":
        if not args.allow_certification:
            parser.error("evaluate requires --allow-certification")
        report = evaluate_policy(
            args.plan, args.run_dir, args.device, args.batch,
        )
        print(json.dumps({
            "metrics_sha256": report["metrics_sha256"],
            "predictions": report["predictions"],
        }, indent=2))


if __name__ == "__main__":
    main()
