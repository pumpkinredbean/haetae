"""Freeze the public Kev recipe with disjoint Korean training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path

from haetae.checkpoint import atomic_json_save
from haetae.data import LOADERS
from haetae.measure import digest_value, state_digest
from haetae.train import dataset_fingerprint
from experiments.freeze_korean_suite import nsmc_request, ynat_request
from experiments.kev_adapter import file_sha256, normalize_request


EXPECTED_KEV_MANIFEST_SHA256 = (
    "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"
)
EXPECTED_KOREAN_MANIFEST_SHA256 = (
    "614a7c0a2febc617f2aaf34bf1906cc8bf0ed958e9138052a37da5ba5b04d5c1"
)
EXPECTED_TRANSFER_MANIFEST_SHA256 = (
    "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1"
)
KOREAN_SOURCES = ("klue_ynat", "nsmc")


def code_identity(repository: Path) -> dict:
    files = [
        repository / "experiments" / "freeze_shared_suite.py",
        repository / "experiments" / "freeze_korean_suite.py",
        repository / "experiments" / "kev_adapter.py",
        repository / "src" / "haetae" / "checkpoint.py",
        repository / "src" / "haetae" / "data.py",
        repository / "src" / "haetae" / "measure.py",
        repository / "src" / "haetae" / "train.py",
    ]
    digest = hashlib.sha256()
    values = {}
    for path in files:
        relative = str(path.relative_to(repository))
        sha256 = file_sha256(path)
        values[relative] = sha256
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": values}


def read_raw_split(suite: Path, manifest: dict, split: str) -> list[dict]:
    filename = f"{split}.jsonl"
    descriptor = manifest["files"][filename]
    path = suite / filename
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Kev suite digest mismatch: {filename}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            normalize_request(record)
            records.append(record)
    if len(records) != descriptor["records"]:
        raise ValueError(f"Kev suite record count mismatch: {filename}")
    questions = sum(len(record["questions"]) for record in records)
    if questions != descriptor["questions"]:
        raise ValueError(f"Kev suite question count mismatch: {filename}")
    return records


def reconstruct_partitions(run: dict) -> tuple[list[dict], list[dict]]:
    config = run["spec"]["config"]
    sources = [name.strip() for name in config["sources"].split(",")]
    train, validation = [], []
    rng = random.Random(config["seed"])
    for source in sources:
        rows = list(LOADERS[source](
            split="train",
            limit=config["per_source"] + config["eval_per_source"],
        ))
        parents = {}
        for index, record in enumerate(rows):
            parents.setdefault(record.get("parent", index), []).append(record)
        keys = list(parents)
        rng.shuffle(keys)
        validation_count = max(
            1,
            int(len(keys) * config["eval_per_source"]
                / (config["per_source"] + config["eval_per_source"])),
        )
        for key in keys[:validation_count]:
            validation.extend(parents[key])
        for key in keys[validation_count:]:
            train.extend(parents[key])
    if dataset_fingerprint(train) != run["spec"]["train_fingerprint"]:
        raise ValueError("reconstructed training fingerprint differs from run")
    if dataset_fingerprint(validation) != run["spec"]["validation_fingerprint"]:
        raise ValueError("reconstructed validation fingerprint differs from run")
    return train, validation


def korean_request(record: dict) -> dict:
    identity = state_digest(record)
    if record["source"] == "klue_ynat":
        return ynat_request(record, identity)
    if record["source"] == "nsmc":
        return nsmc_request(record, identity)
    raise ValueError(f"unsupported Korean source: {record['source']}")


def select_korean_training(
    records: list[dict],
    calibration_records: list[dict],
) -> tuple[list[dict], dict]:
    calibration_states = {
        (record["source"], state_digest(record))
        for record in calibration_records
        if record["source"] in KOREAN_SOURCES
    }
    groups = {}
    for record in records:
        if record["source"] not in KOREAN_SOURCES:
            continue
        key = (record["source"], state_digest(record))
        groups.setdefault(key, []).append(record)
    duplicate_records = sum(len(group) - 1 for group in groups.values())
    ambiguous = {
        key for key, group in groups.items()
        if len({int(record["label"]) for record in group}) > 1
    }
    calibration_overlap = set(groups) & calibration_states
    selected = [
        korean_request(group[0])
        for key, group in groups.items()
        if key not in ambiguous and key not in calibration_overlap
    ]
    selected.sort(key=lambda record: record["_meta"]["id"])
    return selected, {
        "raw_records": sum(len(group) for group in groups.values()),
        "unique_states": len(groups),
        "duplicate_records": duplicate_records,
        "ambiguous_states": len(ambiguous),
        "calibration_overlap_states": len(calibration_overlap),
        "selected_requests": len(selected),
    }


def request_state_identity(record: dict) -> str:
    return digest_value(normalize_request(record)["state"])


def assert_disjoint(partitions: dict[str, list[dict]]) -> dict:
    state_roles = {}
    parent_roles = {}
    for split, records in partitions.items():
        for record in records:
            meta = record.get("_meta", {})
            source = str(meta.get("source", ""))
            state_roles.setdefault(request_state_identity(record), set()).add(split)
            parent = meta.get("group_id") or meta.get("id")
            parent_roles.setdefault((source, str(parent)), set()).add(split)
    state_overlap = {
        identity: roles for identity, roles in state_roles.items()
        if len(roles) > 1
    }
    parent_overlap = {
        identity: roles for identity, roles in parent_roles.items()
        if len(roles) > 1
    }
    if state_overlap or parent_overlap:
        raise ValueError(
            "combined train, calibration, and development partitions overlap"
        )
    return {
        "unique_states": len(state_roles),
        "unique_source_parents": len(parent_roles),
        "state_overlap": 0,
        "source_parent_overlap": 0,
    }


def write_jsonl(path: Path, records: list[dict]) -> dict:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "records": len(records),
        "questions": sum(len(record["questions"]) for record in records),
    }


def freeze(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    kev_suite = Path(args.kev_suite).resolve()
    kev_manifest_path = kev_suite / "manifest.json"
    kev_manifest_sha256 = file_sha256(kev_manifest_path)
    if kev_manifest_sha256 != args.expect_kev_manifest_sha256:
        raise ValueError("Kev suite manifest differs from the declared revision")
    kev_manifest = json.loads(kev_manifest_path.read_text())
    kev = {
        split: read_raw_split(kev_suite, kev_manifest, split)
        for split in ("train", "calibration", "development")
    }

    run_dir = Path(args.run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    baseline_train, baseline_validation = reconstruct_partitions(run)
    korean_train, korean_train_audit = select_korean_training(
        baseline_train, baseline_validation,
    )
    korean_calibration = sorted(
        (korean_request(record) for record in baseline_validation
         if record["source"] in KOREAN_SOURCES),
        key=lambda record: record["_meta"]["id"],
    )
    if not korean_train or not korean_calibration:
        raise ValueError("the baseline run has no Korean train/calibration data")

    korean_manifest_path = Path(args.korean_suite).resolve() / "manifest.json"
    korean_manifest_sha256 = file_sha256(korean_manifest_path)
    if korean_manifest_sha256 != args.expect_korean_manifest_sha256:
        raise ValueError("Korean suite manifest differs from the frozen revision")
    korean_manifest = json.loads(korean_manifest_path.read_text())
    if (korean_manifest["run"]["run_id"] != run["run_id"]
            or korean_manifest["run"]["spec_sha256"] != run["spec_sha256"]):
        raise ValueError("Korean suite is bound to a different baseline run")
    korean_development = read_raw_split(
        korean_manifest_path.parent, korean_manifest, "development",
    )

    transfer_suite = Path(args.transfer_suite).resolve()
    transfer_manifest_path = transfer_suite / "manifest.json"
    transfer_manifest_sha256 = file_sha256(transfer_manifest_path)
    if transfer_manifest_sha256 != args.expect_transfer_manifest_sha256:
        raise ValueError("transfer suite manifest differs from the frozen revision")
    transfer_manifest = json.loads(transfer_manifest_path.read_text())
    transfer_development = read_raw_split(
        transfer_suite, transfer_manifest, "development",
    )

    partitions = {
        "train": kev["train"] + korean_train,
        "calibration": kev["calibration"] + korean_calibration,
        "development": kev["development"],
    }
    overlap_audit = assert_disjoint({
        "train": partitions["train"],
        "calibration": partitions["calibration"],
        "decision_development": partitions["development"],
        "transfer_development": transfer_development,
        "korean_development": korean_development,
    })
    repository = Path(__file__).resolve().parents[1]
    temporary = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        files = {
            f"{split}.jsonl": write_jsonl(
                temporary / f"{split}.jsonl", records,
            )
            for split, records in partitions.items()
        }
        manifest = {
            "version": 1,
            "status": "frozen",
            "sources": {
                "kev": {
                    "manifest_sha256": kev_manifest_sha256,
                    "files": {
                        name: kev_manifest["files"][name]
                        for name in (
                            "train.jsonl", "calibration.jsonl",
                            "development.jsonl", "test.jsonl",
                        )
                    },
                    "locked_test_materialized": False,
                },
                "korean": {
                    "manifest_sha256": korean_manifest_sha256,
                    "development": korean_manifest["files"][
                        "development.jsonl"
                    ],
                    "locked_test_opened": False,
                },
                "transfer": {
                    "manifest_sha256": transfer_manifest_sha256,
                    "development": transfer_manifest["files"][
                        "development.jsonl"
                    ],
                    "locked_test_materialized": False,
                },
                "baseline_run": {
                    "run_id": run["run_id"],
                    "spec_sha256": run["spec_sha256"],
                    "train_fingerprint": run["spec"]["train_fingerprint"],
                    "validation_fingerprint": run["spec"][
                        "validation_fingerprint"
                    ],
                },
            },
            "files": files,
            "trainable_sources": sorted(
                set(kev_manifest["trainable_sources"])
                | {"korean_ynat", "korean_nsmc"}
            ),
            "holdout_sources": transfer_manifest.get("holdout_sources", []),
            "context": {"max_packed": 2048, "truncate": False},
            "audit": {
                "kev_train_requests": len(kev["train"]),
                "korean_train_requests": len(korean_train),
                "korean_train_selection": korean_train_audit,
                "kev_calibration_requests": len(kev["calibration"]),
                "korean_calibration_requests": len(korean_calibration),
                **overlap_audit,
            },
            "selection": (
                "preserve the frozen Kev v7 public partitions; append only "
                "the exact Korean train and internal-validation memberships "
                "bound to the immutable baseline run specification"
            ),
            "locked_policy": (
                "no locked test file is materialized in this suite; use the "
                "separately gated upstream Kev and Korean suites only after "
                "model selection"
            ),
            "code": code_identity(repository),
        }
        manifest["manifest_sha256"] = digest_value(manifest)
        atomic_json_save(manifest, temporary / "manifest.json")
        os.rename(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kev-suite", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--korean-suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expect-kev-manifest-sha256",
        default=EXPECTED_KEV_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expect-korean-manifest-sha256",
        default=EXPECTED_KOREAN_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expect-transfer-manifest-sha256",
        default=EXPECTED_TRANSFER_MANIFEST_SHA256,
    )
    args = parser.parse_args()
    manifest = freeze(args)
    print(json.dumps({
        "manifest_sha256": manifest["manifest_sha256"],
        "files": manifest["files"],
        "audit": manifest["audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
