"""Materialize only the public Kev partitions used by the experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download

from haetae.checkpoint import atomic_json_save
from experiments.kev_adapter import file_sha256, load_frozen_split


DATASET = "jaredpalmer/kev-suites"
REVISION = "57a3ffd3951432855c96eceeef4362617f6a057d"
PUBLIC_FILES = (
    "manifest.json", "train.jsonl", "calibration.jsonl", "development.jsonl",
)
SUITES = {
    "kev-decision-v7": {
        "remote": "v7/decision-v7",
        "manifest_sha256": (
            "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"
        ),
    },
    "kev-transfer-v4": {
        "remote": "v4/transfer-v4",
        "manifest_sha256": (
            "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1"
        ),
    },
}


def verify_staged(directory: Path, expected_manifest_sha256: str) -> dict:
    manifest_path = directory / "manifest.json"
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Kev manifest differs from the pinned suite")
    manifest = json.loads(manifest_path.read_text())
    for split in ("train", "calibration", "development"):
        load_frozen_split(directory, split)
    if (directory / "test.jsonl").exists():
        raise ValueError("locked Kev test was materialized unexpectedly")
    return manifest


def copy_and_sync(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def materialize(output_root: Path) -> dict:
    results = {}
    for name, specification in SUITES.items():
        output = output_root / name
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        temporary = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
        temporary.mkdir(parents=True)
        try:
            for filename in PUBLIC_FILES:
                remote = f"{specification['remote']}/{filename}"
                cached = Path(hf_hub_download(
                    DATASET,
                    remote,
                    repo_type="dataset",
                    revision=REVISION,
                ))
                copy_and_sync(cached, temporary / filename)
            manifest = verify_staged(
                temporary, specification["manifest_sha256"],
            )
            source = {
                "dataset": DATASET,
                "revision": REVISION,
                "remote": specification["remote"],
                "manifest_sha256": specification["manifest_sha256"],
                "locked_test_materialized": False,
            }
            atomic_json_save(source, temporary / "source.json")
            os.rename(temporary, output)
            directory_fd = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        results[name] = {
            "manifest_sha256": specification["manifest_sha256"],
            "files": {
                filename: manifest["files"][filename]
                for filename in PUBLIC_FILES if filename != "manifest.json"
            },
            "locked_test_materialized": False,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="evaluations")
    args = parser.parse_args()
    results = materialize(Path(args.out_root).resolve())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
