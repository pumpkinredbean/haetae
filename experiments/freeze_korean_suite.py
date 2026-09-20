"""Freeze a Korean development and locked suite before model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from haetae.checkpoint import atomic_json_save
from haetae.data import LOADERS
from haetae.measure import (
    digest_value,
    load_plan,
    load_role,
    reconstruct_consumed,
    state_digest,
)
from experiments.kev_adapter import file_sha256


SUITE_VERSION = 1
DEFAULT_SEED = 20260920
YNAT_TOPICS = (
    ("it_science", "IT, science, and technology"),
    ("economy", "economy, companies, finance, and markets"),
    ("society", "society, education, labor, and public affairs"),
    ("culture", "daily life, culture, health, and entertainment"),
    ("world", "international affairs and world news"),
    ("sports", "sports, athletes, games, and results"),
    ("politics", "politics, government, elections, and policy"),
)


def source_code_identity(repository: Path) -> dict:
    files = [
        repository / "experiments" / "freeze_korean_suite.py",
        repository / "experiments" / "kev_adapter.py",
        repository / "src" / "haetae" / "checkpoint.py",
        repository / "src" / "haetae" / "data.py",
        repository / "src" / "haetae" / "measure.py",
        repository / "src" / "haetae" / "train.py",
    ]
    result = {}
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(repository))
        sha256 = file_sha256(path)
        result[relative] = sha256
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": result}


def noul_question(identifier: str, instructions: str, label: bool,
                  source: str) -> tuple[str, dict]:
    return identifier, {
        "type": "noul",
        "instructions": instructions,
        "criteria": {
            "true": "주어진 텍스트가 이 설명을 지지한다.",
            "false": "주어진 텍스트가 이 설명을 지지하지 않는다.",
        },
        "label": label,
        "src": source,
    }


def ynat_request(record: dict, state_sha256: str) -> dict:
    label = int(record["label"])
    keys = [key for key, _ in YNAT_TOPICS]
    criteria = {
        key: description for key, description in YNAT_TOPICS
    }
    selector = int(digest_value({
        "state": state_sha256, "operation": "balanced-topic-check",
    })[:16], 16)
    false_index = selector % (len(keys) - 1)
    if false_index >= label:
        false_index += 1
    if selector % 2:
        checks = ((label, True), (false_index, False))
    else:
        checks = ((false_index, False), (label, True))
    questions = {
        "topic": {
            "type": "choice",
            "instructions": "이 뉴스 기사의 주제를 고르세요.",
            "criteria": criteria,
            "label": keys[label],
            "src": "korean_ynat_topic",
        }
    }
    for index, (topic_index, truth) in enumerate(checks, start=1):
        key, _ = YNAT_TOPICS[topic_index]
        identifier, question = noul_question(
            f"topic_check_{index}",
            f"이 뉴스는 '{key}' 주제인가요?",
            truth,
            "korean_ynat_check",
        )
        questions[identifier] = question
    return {
        "state": record["state"],
        "questions": questions,
        "_meta": {
            "id": f"korean_ynat/{state_sha256}",
            "group_id": f"korean_ynat/{state_sha256}",
            "source": "korean_ynat",
            "text_sha256": state_sha256,
        },
    }


def nsmc_request(record: dict, state_sha256: str) -> dict:
    positive = int(record["label"]) == 0
    identifier, sentiment = noul_question(
        "positive",
        "이 영화 리뷰는 전반적으로 긍정적인가요?",
        positive,
        "korean_nsmc_noul",
    )
    return {
        "state": record["state"],
        "questions": {
            identifier: sentiment,
            "polarity": {
                "type": "choice",
                "instructions": "이 영화 리뷰의 감정 방향을 고르세요.",
                "criteria": {
                    "positive": "전반적으로 긍정적인 리뷰",
                    "negative": "전반적으로 부정적인 리뷰",
                },
                "label": "positive" if positive else "negative",
                "src": "korean_nsmc_choice",
            },
        },
        "_meta": {
            "id": f"korean_nsmc/{state_sha256}",
            "group_id": f"korean_nsmc/{state_sha256}",
            "source": "korean_nsmc",
            "text_sha256": state_sha256,
        },
    }


def select_records(records: list[dict], source: str, forbidden_states: set[str],
                   seed: int, total: int) -> tuple[list[dict], dict]:
    unique = {}
    ambiguous_states = set()
    excluded_overlap = 0
    duplicate_records = 0
    for record in records:
        identity = state_digest(record)
        if identity in forbidden_states:
            excluded_overlap += 1
            continue
        previous = unique.get(identity)
        if previous is None:
            unique[identity] = record
        else:
            duplicate_records += 1
            if int(previous["label"]) != int(record["label"]):
                ambiguous_states.add(identity)
    for identity in ambiguous_states:
        unique.pop(identity, None)
    by_label = {}
    for identity, record in unique.items():
        by_label.setdefault(int(record["label"]), []).append((identity, record))
    labels = sorted(by_label)
    if len(labels) < 2:
        raise ValueError(f"{source} needs at least two labels")
    quotas = {
        label: total // len(labels) + (index < total % len(labels))
        for index, label in enumerate(labels)
    }
    ranked = {}
    for label in labels:
        ranked[label] = sorted(
            by_label[label],
            key=lambda item: digest_value({
                "seed": seed, "source": source, "label": label,
                "state": item[0],
            }),
        )
        if len(ranked[label]) < quotas[label]:
            raise ValueError(
                f"{source} label {label} has only {len(ranked[label])} "
                f"eligible unique states; need {quotas[label]}"
            )
    ordered = []
    for index in range(max(quotas.values())):
        for label in labels:
            if index < quotas[label]:
                ordered.append(ranked[label][index])
    if len(ordered) != total:
        raise ValueError(
            f"{source} selected {len(ordered)} states; expected {total}"
        )
    selected = [
        (ynat_request(record, identity) if source == "korean_ynat"
         else nsmc_request(record, identity))
        for identity, record in ordered[:total]
    ]
    return selected, {
        "raw_records": len(records),
        "excluded_overlap_records": excluded_overlap,
        "duplicate_records": duplicate_records,
        "ambiguous_states": len(ambiguous_states),
        "eligible_unique_states": len(unique),
        "eligible_by_label": {
            str(label): len(by_label[label]) for label in labels
        },
        "selected_by_label": {
            str(label): quotas[label] for label in labels
        },
        "selected": len(selected),
    }


def primary_label_counts(requests: list[dict], source: str) -> dict[str, int]:
    question_id = "topic" if source == "korean_ynat" else "polarity"
    counts = {}
    for request in requests:
        if request["_meta"]["source"] != source:
            continue
        label = request["questions"][question_id]["label"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


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
    run_dir = Path(args.run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    consumed = reconstruct_consumed(run)
    baseline_plan = load_plan(args.baseline_plan)
    frozen_states = set(consumed["state_digests"])
    for role in ("A", "B", "C"):
        frozen_states.update(
            item["meta"]["state_sha256"]
            for item in load_role(args.baseline_plan, role, baseline_plan)
        )

    development_count = args.development_per_source
    test_count = args.test_per_source
    total = development_count + test_count
    ynat, ynat_audit = select_records(
        list(LOADERS["klue_ynat"](split="validation", limit=None)),
        "korean_ynat", frozen_states, args.seed, total,
    )
    nsmc, nsmc_audit = select_records(
        list(LOADERS["nsmc"](split="test", limit=None)),
        "korean_nsmc", frozen_states, args.seed, total,
    )
    development = (
        ynat[:development_count] + nsmc[:development_count]
    )
    locked = ynat[development_count:] + nsmc[development_count:]
    development.sort(key=lambda item: item["_meta"]["id"])
    locked.sort(key=lambda item: item["_meta"]["id"])
    if ({item["_meta"]["group_id"] for item in development}
            & {item["_meta"]["group_id"] for item in locked}):
        raise RuntimeError("Korean development and locked parents overlap")

    repository = Path(__file__).resolve().parents[1]
    temporary = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        files = {
            "development.jsonl": write_jsonl(
                temporary / "development.jsonl", development,
            ),
            "test.jsonl": write_jsonl(temporary / "test.jsonl", locked),
        }
        empty_sha = hashlib.sha256(b"").hexdigest()
        for name in ("train.jsonl", "calibration.jsonl"):
            (temporary / name).write_bytes(b"")
            files[name] = {
                "sha256": empty_sha, "bytes": 0,
                "records": 0, "questions": 0,
            }
        manifest = {
            "version": SUITE_VERSION,
            "status": "frozen",
            "seed": args.seed,
            "run": {
                "run_id": run["run_id"],
                "spec_sha256": run["spec_sha256"],
                "train_fingerprint": consumed["train_fingerprint"],
                "validation_fingerprint": consumed["validation_fingerprint"],
            },
            "excluded_baseline_plan_sha256": baseline_plan["plan_sha256"],
            "excluded_baseline_role_files": baseline_plan["files"],
            "source_artifacts": {
                "ynat-v1.1_dev.json": file_sha256(
                    repository / "data" / "ynat-v1.1_dev.json"
                ),
                "ratings_test.txt": file_sha256(
                    repository / "data" / "ratings_test.txt"
                ),
            },
            "audit": {
                "forbidden_state_count": len(frozen_states),
                "korean_ynat": ynat_audit,
                "korean_nsmc": nsmc_audit,
                "development_label_counts": {
                    "korean_ynat": primary_label_counts(
                        development, "korean_ynat",
                    ),
                    "korean_nsmc": primary_label_counts(
                        development, "korean_nsmc",
                    ),
                },
                "locked_label_counts": {
                    "korean_ynat": primary_label_counts(
                        locked, "korean_ynat",
                    ),
                    "korean_nsmc": primary_label_counts(
                        locked, "korean_nsmc",
                    ),
                },
                "development_parents": len(development),
                "locked_parents": len(locked),
                "parent_overlap": 0,
            },
            "files": files,
            "trainable_sources": [],
            "holdout_sources": ["korean_ynat", "korean_nsmc"],
            "context": {"max_packed": 2048, "truncate": False},
            "selection": (
                "exclude consumed and baseline A/B/C states, deduplicate exact "
                "states, exclude conflicting labels, balance each source by "
                "label, then rank by seed/source/label/state SHA-256"
            ),
            "locked_policy": (
                "development may be used for model selection; test requires "
                "the adapter's explicit allow_test gate"
            ),
            "code": source_code_identity(repository),
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
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--development-per-source", type=int, default=1000)
    parser.add_argument("--test-per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.development_per_source <= 0 or args.test_per_source <= 0:
        raise ValueError("development and test counts must be positive")
    manifest = freeze(args)
    print(json.dumps({
        "manifest_sha256": manifest["manifest_sha256"],
        "files": manifest["files"],
        "audit": manifest["audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
