"""Freeze and verify the fixed-weight T04 diagnostic before inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from experiments.coverage_v1.registry import (
    PARTITION_EVIDENCE,
    ArtifactStore,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_json_bytes,
)
from experiments.kev_adapter import normalize_request
from tools.evidence_registry import resolve_explicit_artifact


SEED = 20260921
FIT_PER_CLASS = 256
CALIBRATION_PER_CLASS = 64
SEQUENCE_LIMIT = 20_000
BOOTSTRAP_DRAWS = 20_000
OUTPUT_FILES = (
    "development.jsonl",
    "exclusions.json",
    "protocol.json",
    "semantic-variants.jsonl",
    "support-calibration.jsonl",
    "support-fit.jsonl",
)
DIAGNOSTIC_CODE_FILES = (
    "experiments/coverage_v1/diagnostic_v1.py",
    "experiments/coverage_v1/run_diagnostic_v1.py",
    "experiments/coverage_v1/registry.py",
    "experiments/kev_adapter.py",
    "experiments/shared_state.py",
    "src/haetae/checkpoint.py",
)
NATIVE_LABELS = {
    "emotion": ("sadness", "joy", "love", "anger", "fear", "surprise"),
    "tweet_offensive": ("false", "true"),
}
SUPPORT_SOURCES = {
    "emotion": {
        "dataset": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "repository_file": "split/train-00000-of-00001.parquet",
        "sha256": "10817f0f2ea42358bc62f69a09dfb8bd71701727df6d5a387bea742f3ea06417",
        "size_bytes": 1_030_740,
        "rows": 16_000,
    },
    "tweet_offensive": {
        "dataset": "cardiffnlp/tweet_eval",
        "revision": "39925773dee306b2b4f90ba0ed3e60da4e830ada",
        "repository_file": "offensive/train/0000.parquet",
        "sha256": "05afbd6d4027c59464b8098cd85681ba5a7870cb52240c4034975c3b3220015a",
        "size_bytes": 1_019_132,
        "rows": 11_916,
    },
}
ORIGINAL_MODEL_FILES = {
    "config.json": {
        "sha256": "3b0eea47c4734682370b66dbd73d655f1100387f683d7c69015f33aa1bac1be9",
        "size_bytes": 1_187,
    },
    "pytorch_model.bin": {
        "sha256": "be7c9ce2af947fb5bb17a3bc919552f898c5ef4164bf0a7190cf25a0f98f85b6",
        "size_bytes": 563_633_326,
    },
    "special_tokens_map.json": {
        "sha256": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
        "size_bytes": 636,
    },
    "tokenizer.json": {
        "sha256": "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
        "size_bytes": 17_525_329,
    },
    "tokenizer_config.json": {
        "sha256": "ea349495764b24570b4560fe2e4a557e5c0a47e6bce782c2409b1d6445cf8633",
        "size_bytes": 46_440,
    },
}
SHARED_TOKENIZER_FILES = {
    "tokenizer.json": {
        "sha256": "fb6235c8bbe93ae2ebaab5a3f3c201f7a69f04fa2491273aa05ef882d1539504",
        "size_bytes": 34_363_962,
    },
    "tokenizer_config.json": {
        "sha256": "55fa4253842c520aa71caf546bb34a2dacaba65d3127e24dc268305bf4d68715",
        "size_bytes": 627,
    },
}
LEXICON = {
    "sadness": "feeling sorrow, unhappiness, or loss",
    "joy": "feeling happiness, delight, or pleasure",
    "love": "feeling affection, care, or attachment",
    "anger": "feeling irritation, rage, or hostility",
    "fear": "feeling worry, danger, or anxiety",
    "surprise": "feeling startled by something unexpected",
    "none_of_these": "none of the listed emotions describes the answer",
    "false": "non-offensive language without insults, threats, targeted profanity, or hateful content",
    "true": "offensive language containing insults, threats, targeted profanity, or hateful content",
}


def _descriptor(path: Path) -> dict:
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _verify_descriptor(path: Path, expected: dict, label: str) -> None:
    if not path.resolve(strict=True).is_file() or _descriptor(path) != expected:
        raise ValueError(f"{label} differs from its fixed identity")


def diagnostic_code_identity(repository: Path) -> dict:
    files = {name: file_sha256(repository / name) for name in DIAGNOSTIC_CODE_FILES}
    digest = hashlib.sha256()
    for name, sha256 in files.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return {"sha256": digest.hexdigest(), "files": files}


def _load_local_paths(path: Path) -> dict:
    values = load_json_bytes(path.read_bytes(), "diagnostic local path map")
    if set(values) != {
        "schema_version", "emotion_parquet", "original_model_directory",
        "shared_run_directory", "tweet_offensive_parquet",
    } or values["schema_version"] != 1:
        raise ValueError("diagnostic local path map fields differ")
    return values


def _support_rows(source: str, path: Path) -> list[dict]:
    expected = SUPPORT_SOURCES[source]
    data = path.read_bytes()
    if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"{source} support file differs")
    import pyarrow.parquet as parquet

    table = parquet.read_table(io.BytesIO(data), columns=["text", "label"])
    values = table.to_pylist()
    if len(values) != expected["rows"]:
        raise ValueError(f"{source} support row count differs")
    labels = NATIVE_LABELS[source]
    rows = []
    for index, value in enumerate(values):
        text = value.get("text")
        label = value.get("label")
        if not isinstance(text, str) or not text or type(label) is not int or not 0 <= label < len(labels):
            raise ValueError(f"{source} support row is malformed")
        parent_id = f"{source}/train/{index}"
        rows.append({
            "source": source,
            "split": "train",
            "row": index,
            "parent_id": parent_id,
            "state": text,
            "state_sha256": canonical_sha256(text),
            "label": labels[label],
        })
    return rows


def exclusion_index(partitions: dict[str, list[dict]]) -> dict:
    states: dict[str, set[str]] = defaultdict(set)
    parents: dict[tuple[str, str], set[str]] = defaultdict(set)
    for role, rows in partitions.items():
        for raw in rows:
            normalized = normalize_request(raw)
            state_sha256 = canonical_sha256(normalized["state"])
            states[state_sha256].add(role)
            meta = raw.get("_meta", {})
            source = meta.get("source")
            parent = meta.get("group_id") or meta.get("id")
            if not isinstance(source, str) or not source or not isinstance(parent, str) or not parent:
                raise ValueError("exclusion partition has incomplete parent identity")
            parents[(source, parent)].add(role)
    return {
        "state_sha256": [
            {"sha256": value, "roles": sorted(roles)}
            for value, roles in sorted(states.items())
        ],
        "source_parents": [
            {"source": source, "parent_id": parent, "roles": sorted(roles)}
            for (source, parent), roles in sorted(parents.items())
        ],
    }


def select_support(
    rows: list[dict],
    excluded_states: set[str],
    excluded_parents: set[tuple[str, str]],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    by_state: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_state[row["state_sha256"]].append(row)
    candidates = []
    removals = []
    duplicate_rows = 0
    conflicting_states = 0
    for state_sha256, group in sorted(by_state.items()):
        labels = {row["label"] for row in group}
        if len(labels) != 1:
            conflicting_states += 1
            removals.extend({"reason": "conflicting_labels", **row} for row in group)
            continue
        representative = min(group, key=lambda row: row["row"])
        for row in group:
            if row is not representative:
                duplicate_rows += 1
                removals.append({"reason": "same_label_duplicate", **row})
        if state_sha256 in excluded_states:
            removals.append({"reason": "rendered_state_exclusion", **representative})
            continue
        parent_key = (representative["source"], representative["parent_id"])
        if parent_key in excluded_parents:
            removals.append({"reason": "source_parent_exclusion", **representative})
            continue
        representative = dict(representative)
        representative["selection_sha256"] = canonical_sha256({
            "seed": SEED,
            "source": representative["source"],
            "parent_id": representative["parent_id"],
        })
        candidates.append(representative)

    fit, calibration = [], []
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_label[row["label"]].append(row)
    expected_labels = set(NATIVE_LABELS[rows[0]["source"]])
    if set(by_label) != expected_labels:
        raise ValueError("support labels differ from the native ontology")
    for label in NATIVE_LABELS[rows[0]["source"]]:
        ordered = sorted(by_label[label], key=lambda row: row["selection_sha256"])
        needed = FIT_PER_CLASS + CALIBRATION_PER_CLASS
        if len(ordered) < needed:
            raise ValueError(f"insufficient support rows for {rows[0]['source']}:{label}")
        fit.extend(ordered[:FIT_PER_CLASS])
        calibration.extend(ordered[FIT_PER_CLASS:needed])
    fit.sort(key=lambda row: (row["source"], row["label"], row["selection_sha256"]))
    calibration.sort(key=lambda row: (row["source"], row["label"], row["selection_sha256"]))
    summary = {
        "raw_rows": len(rows),
        "unique_rendered_states": len(by_state),
        "same_label_duplicate_rows": duplicate_rows,
        "conflicting_states": conflicting_states,
        "eligible_rows": len(candidates),
        "fit_rows": len(fit),
        "calibration_rows": len(calibration),
        "fit_label_counts": dict(sorted(Counter(row["label"] for row in fit).items())),
        "calibration_label_counts": dict(sorted(Counter(row["label"] for row in calibration).items())),
    }
    return fit, calibration, removals, summary


def development_rows(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for raw in raw_rows:
        normalized = normalize_request(raw)
        for question in normalized["questions"]:
            target = question.get("source")
            if target not in NATIVE_LABELS:
                continue
            meta = raw["_meta"]
            parent_id = meta.get("group_id") or meta["id"]
            semantic_label = question["option_keys"][question["label"]]
            if target == "emotion":
                keys = question["option_keys"]
                if set(keys) == set(NATIVE_LABELS[target]) and semantic_label in NATIVE_LABELS[target]:
                    population = "native_six_class"
                elif semantic_label == "none_of_these":
                    population = "augmented_none_target"
                elif "none_of_these" in keys:
                    population = "augmented_none_candidate"
                else:
                    raise ValueError("emotion development ontology differs")
            else:
                if question["option_keys"] != ["true", "false"]:
                    raise ValueError("offensive Noul polarity differs")
                population = "native_binary"
            row = {
                "development_id": str(meta["id"]),
                "source": str(meta["source"]),
                "parent_id": str(parent_id),
                "target": target,
                "population": population,
                "state": normalized["state"],
                "state_sha256": canonical_sha256(normalized["state"]),
                "question": question,
                "semantic_label": semantic_label,
            }
            row["row_sha256"] = canonical_sha256(row)
            rows.append(row)
    rows.sort(key=lambda row: (row["target"], row["development_id"]))
    counts = Counter((row["target"], row["population"]) for row in rows)
    expected = {
        ("emotion", "native_six_class"): 92,
        ("emotion", "augmented_none_candidate"): 12,
        ("emotion", "augmented_none_target"): 12,
        ("tweet_offensive", "native_binary"): 80,
    }
    if counts != expected:
        raise ValueError(f"target development membership differs: {dict(counts)}")
    return rows


def _variant_row(row: dict, family: str, options: list[str], keys: list[str]) -> dict:
    semantic_label = row["semantic_label"]
    if semantic_label not in keys:
        raise ValueError("semantic label is absent after remapping")
    question = row["question"]
    value = {
        "development_id": row["development_id"],
        "source": row["source"],
        "parent_id": row["parent_id"],
        "target": row["target"],
        "population": row["population"],
        "family": family,
        "state": row["state"],
        "state_sha256": row["state_sha256"],
        "question": {
            "id": question["id"],
            "type": question["type"],
            "instructions": question["instructions"],
            "options": options,
            "option_keys": keys,
            "label": keys.index(semantic_label),
        },
        "semantic_label": semantic_label,
    }
    value["input_sha256"] = canonical_sha256({
        "state": value["state"],
        "question": value["question"],
    })
    value["variant_id"] = canonical_sha256({
        "development_id": value["development_id"],
        "family": family,
        "input_sha256": value["input_sha256"],
    })
    return value


def semantic_variants(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        question = row["question"]
        options = list(question["options"])
        keys = list(question["option_keys"])
        output.append(_variant_row(row, "original", options, keys))
        if question["type"] == "choice":
            for shift in range(1, len(options)):
                output.append(_variant_row(
                    row,
                    f"cyclic_rotation_{shift}",
                    options[shift:] + options[:shift],
                    keys[shift:] + keys[:shift],
                ))
            opaque = [
                f"label_{index + 1}: {option}"
                for index, option in enumerate(options)
            ]
            output.append(_variant_row(row, "opaque_keys", opaque, keys))
            lexicon = [f"{key}: {LEXICON[key]}" for key in keys]
            output.append(_variant_row(row, "native_taxonomy_lexicon", lexicon, keys))
        elif question["type"] == "noul":
            if keys != ["true", "false"]:
                raise ValueError("Noul variants cannot change polarity")
            lexicon = [
                f"yes: {LEXICON['true']}",
                f"no: {LEXICON['false']}",
            ]
            output.append(_variant_row(row, "native_taxonomy_lexicon", lexicon, keys))
        else:
            raise ValueError("T04 target primitive is unsupported")
    output.sort(key=lambda row: (row["target"], row["development_id"], row["family"]))
    if len({row["variant_id"] for row in output}) != len(output):
        raise ValueError("semantic variant inventory differs")
    return output


def _write_json(path: Path, value: dict) -> dict:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    data = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "rows": len(rows),
    }


def _model_bindings(store: ArtifactStore, local: dict) -> dict:
    original = Path(local["original_model_directory"]).resolve(strict=True)
    shared = Path(local["shared_run_directory"]).resolve(strict=True)
    for name, descriptor in ORIGINAL_MODEL_FILES.items():
        _verify_descriptor(original / name, descriptor, f"original model {name}")
    for name, descriptor in SHARED_TOKENIZER_FILES.items():
        _verify_descriptor(shared / name, descriptor, f"shared tokenizer {name}")
    checkpoint_path = resolve_explicit_artifact(store.paths, "shared_checkpoint_g79")
    checkpoint_entry = store.registry.get("artifacts", {}).get("shared_checkpoint_g79")
    if (
        not isinstance(checkpoint_entry, dict)
        or checkpoint_entry.get("kind") != "original"
        or checkpoint_entry.get("contains_locked_data") is not False
    ):
        raise ValueError("shared checkpoint is not registered as unlocked original evidence")
    checkpoint = {
        "evidence_id": "shared_checkpoint_g79",
        "sha256": checkpoint_entry["sha256"],
        "size_bytes": checkpoint_entry["size_bytes"],
    }
    _verify_descriptor(
        checkpoint_path,
        {"sha256": checkpoint["sha256"], "size_bytes": checkpoint["size_bytes"]},
        "shared checkpoint",
    )
    run = store.read_json("shared_run_spec")
    manifest = store.read_json("shared_run_manifest")
    if (
        run.get("run_id") != "71d06d48-a3cf-4ddc-bfe2-2315dbe66da8"
        or run.get("spec_sha256") != "2797fc5a985c315f035658790c6ba582d634fe2ceeaa2d8fc5d35e78a100e9d5"
        or manifest.get("current", {}).get("generation") != 79
        or manifest.get("current", {}).get("sha256") != checkpoint["sha256"]
    ):
        raise ValueError("shared generation 79 identity differs")
    return {
        "original_pretrained": {
            "model": "jhu-clsp/mmBERT-small",
            "revision": "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
            "files": ORIGINAL_MODEL_FILES,
            "model_config_fingerprint": "c0dd1c06807070ca39d94cf8ed7aefe2b938cfb9a1e15ab55e664e01b9358598",
            "tokenizer_fingerprint": "a19e6948e936996e602b44fd26f2f327f3a4b150aab15d99e86d13a8f8d54e2b",
        },
        "shared_generation_79": {
            "run_id": run["run_id"],
            "spec_sha256": run["spec_sha256"],
            "checkpoint": checkpoint,
            "model_config_fingerprint": run["spec"]["model_config_fingerprint"],
            "tokenizer_fingerprint": run["spec"]["tokenizer_fingerprint"],
            "tokenizer_files": SHARED_TOKENIZER_FILES,
            "generation": 79,
            "maximum_length": 2048,
            "attention_implementation": "eager",
        },
    }


def freeze(
    evidence_paths: Path,
    local_paths: Path,
    output: Path,
    registry_path: Path,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    repository = Path(__file__).resolve().parents[2]
    store = ArtifactStore.from_files(registry_path, evidence_paths)
    local = _load_local_paths(local_paths)
    partitions = {
        role: store.read_jsonl(evidence_id)
        for role, evidence_id in PARTITION_EVIDENCE.items()
    }
    exclusions = exclusion_index(partitions)
    excluded_states = {row["sha256"] for row in exclusions["state_sha256"]}
    excluded_parents = {
        (row["source"], row["parent_id"])
        for row in exclusions["source_parents"]
    }

    support_fit, support_calibration, support_removals = [], [], []
    support_summary = {}
    for source, path_key in (
        ("emotion", "emotion_parquet"),
        ("tweet_offensive", "tweet_offensive_parquet"),
    ):
        rows = _support_rows(source, Path(local[path_key]).resolve(strict=True))
        fit, calibration, removals, summary = select_support(
            rows, excluded_states, excluded_parents,
        )
        support_fit.extend(fit)
        support_calibration.extend(calibration)
        support_removals.extend(removals)
        support_summary[source] = summary

    development = development_rows(partitions["transfer_development"])
    variants = semantic_variants(development)
    if len(variants) != 1_100:
        raise ValueError("semantic variant count differs")
    unique_development_states = len({row["state_sha256"] for row in development})
    feature_sequences = 2 * (
        len(support_fit) + len(support_calibration) + unique_development_states
    )
    sequence_budget = {
        "original_backbone_feature_sequences": len(support_fit) + len(support_calibration) + unique_development_states,
        "shared_backbone_feature_sequences": len(support_fit) + len(support_calibration) + unique_development_states,
        "shared_pointer_semantic_sequences": len(variants),
        "aggregate_encoded_sequences": feature_sequences + len(variants),
        "planned_model_forward_calls": (
            2 * math.ceil(
                (len(support_fit) + len(support_calibration) + unique_development_states)
                / 2
            )
            + math.ceil(len(variants) / 2)
        ),
        "limit": SEQUENCE_LIMIT,
    }
    if sequence_budget["aggregate_encoded_sequences"] > SEQUENCE_LIMIT:
        raise ValueError("diagnostic sequence budget exceeds its fixed limit")

    exclusions["support_removals"] = support_removals
    exclusions["summary"] = {
        "partition_state_identities": len(exclusions["state_sha256"]),
        "partition_source_parents": len(exclusions["source_parents"]),
        "support_removals": len(support_removals),
    }
    models = _model_bindings(store, local)
    protocol = {
        "schema_version": 3,
        "status": "frozen_before_inference",
        "evidence_class": "exploratory fixed-weight diagnostic",
        "seed": SEED,
        "purpose": "Separate supervision, semantic binding, accessible features, ranking, thresholds, and calibration for emotion and offensive-tweet transfer.",
        "models": models,
        "runtime": {
            "device": "mps",
            "parameter_dtype": "float32",
            "attention_implementation": "eager",
            "hardware": {
                "model_identifier": "Mac15,7",
                "chip": "Apple M3 Pro",
                "memory_gib": 36,
            },
            "software": {
                "python": "3.13.2",
                "macos": "26.6.2",
                "torch": "2.14.0",
                "transformers": "5.17.0",
                "scikit_learn": "1.9.1",
                "scipy": "1.18.1",
                "numpy": "2.5.3",
                "pyarrow": "25.0.1",
            },
        },
        "support_sources": SUPPORT_SOURCES,
        "support_selection": {
            "fit_parents_per_class": FIT_PER_CLASS,
            "calibration_parents_per_class": CALIBRATION_PER_CLASS,
            "order": "ascending canonical SHA-256 of seed, source, and source-namespaced parent ID",
            "duplicate_policy": "remove every conflicting rendered state; retain the lowest source row for same-label duplicates",
            "exclusions": "all shared train, calibration, decision development, transfer development, and Korean development rendered states and source-namespaced parents",
            "summary": support_summary,
        },
        "development": {
            "evidence": store.binding("transfer_suite_development"),
            "emotion_native_six_class_questions": 92,
            "emotion_augmented_none_candidate_questions": 12,
            "emotion_augmented_none_target_questions": 12,
            "offensive_native_binary_questions": 80,
            "unique_source_parents": unique_development_states,
            "feature_probe_scoring": "emotion native_six_class and offensive native_binary only; augmented emotion strata receive no six-class probe probability",
        },
        "feature_rendering": {
            "tokenizer": "original pinned mmBERT tokenizer without decision delimiters",
            "sequence": "[CLS], complete rendered state tokens, [SEP]",
            "truncation": "reject",
            "maximum_length": 2048,
            "representation": "mean final-layer hidden state over state tokens only, excluding CLS, SEP, and padding",
            "attention": "the same eager full and sliding state masks for both fixed backbones",
        },
        "semantic_rendering": {
            "model": "shared_generation_79 pointer readout only",
            "families": ["original", "all nonzero cyclic Choice rotations", "opaque Choice keys with original option text", "one fixed native-taxonomy lexicon"],
            "choice_mapping": "the semantic key travels with its rendered option and the gold index is remapped exactly",
            "noul_mapping": "yes-first true,false polarity is fixed; no rotation or opaque-key variant",
            "score_mapping": "not present in the target population; ordinal rotation remains prohibited",
            "development_label_fitting": False,
        },
        "fitting": {
            "standardizer": {"class": "StandardScaler", "with_mean": True, "with_std": True, "fit_role": "support_fit"},
            "classifier": {"class": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "fit_intercept": True, "class_weight": None, "random_state": SEED, "initialization": "zero", "hyperparameter_search": False, "fit_role": "support_fit"},
            "emotion_calibrator": {"class": "positive scalar temperature", "parameter": "exp(log_temperature)", "bounds_log_temperature": [-5.0, 5.0], "method": "scipy bounded minimize_scalar", "xatol": 1e-12, "maxiter": 1000, "fit_role": "support_calibration"},
            "offensive_calibrator": {"class": "positive-slope logistic with intercept", "parameters": ["exp(log_slope)", "intercept"], "bounds": [[-5.0, 5.0], [-20.0, 20.0]], "method": "scipy L-BFGS-B", "ftol": 1e-12, "maxiter": 1000, "fit_role": "support_calibration"},
        },
        "metrics": {
            "probe": ["accuracy", "macro_f1", "nll", "multiclass_brier", "offensive_auroc"],
            "semantic": ["accuracy", "nll", "multiclass_brier", "offensive_auroc", "semantic_action_stability", "total_variation"],
            "ranking_threshold_calibration": "report raw ranking, raw threshold action, and support-calibrated action and probability scores separately",
            "historical_accuracy_gaps": {"emotion": 0.2844827586206896, "tweet_offensive": 0.13749999999999996},
        },
        "uncertainty": {
            "draws": BOOTSTRAP_DRAWS,
            "unit": "source-namespaced development parent",
            "interval": "one-sided 97.5% lower bound and descriptive two-sided 95% interval for original-minus-shared probe accuracy; other diagnostic metrics are descriptive point estimates",
            "quantiles": "inverse empirical CDF with nearest-rank endpoints",
            "seed": SEED,
        },
        "diagnostic_flags": {
            "accessible_feature_regression": "original-minus-shared probe accuracy is at least 0.05 and its one-sided 97.5% parent-bootstrap lower bound is positive",
            "task_or_readout_sensitivity": "a predeclared probe or semantic family recovers at least half its target's historical accuracy gap",
            "calibration_only": "support calibration improves NLL or Brier without improving ranking or AUROC",
        },
        "compute": {
            **sequence_budget,
            "accelerator_processes": 1,
            "microbatch": 2,
            "optimizer_updates": 0,
            "model_forwards_before_M3_review": 0,
            "locked_test_opened": False,
        },
        "code": diagnostic_code_identity(repository),
        "prohibited": ["locked test access", "development-label fitting", "prompt or hyperparameter search", "backbone or pointer-head updates", "claims of causal proof or zero-shot transfer after support fitting"],
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        descriptors = {
            "development.jsonl": _write_jsonl(temporary / "development.jsonl", development),
            "exclusions.json": _write_json(temporary / "exclusions.json", exclusions),
            "protocol.json": _write_json(temporary / "protocol.json", protocol),
            "semantic-variants.jsonl": _write_jsonl(temporary / "semantic-variants.jsonl", variants),
            "support-calibration.jsonl": _write_jsonl(temporary / "support-calibration.jsonl", support_calibration),
            "support-fit.jsonl": _write_jsonl(temporary / "support-fit.jsonl", support_fit),
        }
        manifest = {
            "schema_version": 1,
            "status": "frozen_before_inference",
            "protocol_sha256": protocol["protocol_sha256"],
            "files": descriptors,
            "model_forwards": 0,
            "optimizer_updates": 0,
            "locked_test_opened": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(temporary / "manifest.json", manifest)
        output.mkdir()
        try:
            for name in OUTPUT_FILES:
                os.rename(temporary / name, output / name)
            os.rename(temporary / "manifest.json", output / "manifest.json")
        except BaseException:
            shutil.rmtree(output, ignore_errors=True)
            raise
        temporary.rmdir()
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    try:
        if verify_frozen(output) != manifest:
            raise RuntimeError("published diagnostic manifest differs")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return manifest


def _safe_file(root: Path, name: str) -> Path:
    if name not in set(OUTPUT_FILES) | {"manifest.json"}:
        raise ValueError("diagnostic filename is not allowed")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"diagnostic file cannot be a symlink: {name}")
    resolved = path.resolve(strict=True)
    if resolved.parent != root or not resolved.is_file():
        raise ValueError(f"diagnostic file escapes its directory: {name}")
    return resolved


def _read_jsonl(path: Path, label: str) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if line:
            rows.append(load_json_bytes(line, f"{label} line {number}"))
    return rows


def verify_frozen(directory: Path) -> dict:
    if directory.is_symlink():
        raise ValueError("diagnostic directory cannot be a symlink")
    root = directory.resolve(strict=True)
    if {path.name for path in root.iterdir()} != set(OUTPUT_FILES) | {"manifest.json"}:
        raise ValueError("diagnostic file inventory differs")
    manifest = load_json_bytes(_safe_file(root, "manifest.json").read_bytes(), "diagnostic manifest")
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if set(manifest) != {
        "schema_version", "status", "protocol_sha256", "files",
        "model_forwards", "optimizer_updates", "locked_test_opened",
        "manifest_sha256",
    }:
        raise ValueError("diagnostic manifest fields differ")
    if expected != canonical_sha256(unsigned):
        raise ValueError("diagnostic manifest self-digest differs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "frozen_before_inference"
        or manifest.get("model_forwards") != 0
        or manifest.get("optimizer_updates") != 0
        or manifest.get("locked_test_opened") is not False
        or set(manifest.get("files", {})) != set(OUTPUT_FILES)
    ):
        raise ValueError("diagnostic manifest boundary differs")
    for name, descriptor in manifest["files"].items():
        path = _safe_file(root, name)
        actual = _descriptor(path)
        expected_fields = {"sha256", "size_bytes", "rows"} if name.endswith(".jsonl") else {"sha256", "size_bytes"}
        if not isinstance(descriptor, dict) or set(descriptor) != expected_fields:
            raise ValueError(f"diagnostic descriptor fields differ: {name}")
        if any(actual.get(key) != descriptor.get(key) for key in ("sha256", "size_bytes")):
            raise ValueError(f"diagnostic file differs: {name}")
        if name.endswith(".jsonl"):
            rows = sum(bool(line) for line in path.read_bytes().splitlines())
            if rows != descriptor.get("rows"):
                raise ValueError(f"diagnostic row count differs: {name}")
    protocol = load_json_bytes(_safe_file(root, "protocol.json").read_bytes(), "diagnostic protocol")
    protocol_sha256 = protocol.get("protocol_sha256")
    protocol_unsigned = dict(protocol)
    protocol_unsigned.pop("protocol_sha256", None)
    if (
        protocol_sha256 != canonical_sha256(protocol_unsigned)
        or protocol_sha256 != manifest["protocol_sha256"]
        or protocol.get("status") != "frozen_before_inference"
        or protocol.get("compute", {}).get("model_forwards_before_M3_review") != 0
        or protocol.get("compute", {}).get("optimizer_updates") != 0
        or protocol.get("compute", {}).get("locked_test_opened") is not False
    ):
        raise ValueError("diagnostic protocol identity differs")
    repository = Path(__file__).resolve().parents[2]
    if protocol.get("code") != diagnostic_code_identity(repository):
        raise ValueError("diagnostic code identity differs")
    if (
        protocol.get("support_sources") != SUPPORT_SOURCES
        or protocol.get("models", {}).get("original_pretrained", {}).get("files")
        != ORIGINAL_MODEL_FILES
        or protocol.get("models", {}).get("shared_generation_79", {}).get("tokenizer_files")
        != SHARED_TOKENIZER_FILES
    ):
        raise ValueError("diagnostic source identity differs")

    support_fit = _read_jsonl(_safe_file(root, "support-fit.jsonl"), "support fit")
    support_calibration = _read_jsonl(
        _safe_file(root, "support-calibration.jsonl"), "support calibration",
    )
    memberships = set()
    for role, rows, per_class in (
        ("fit", support_fit, FIT_PER_CLASS),
        ("calibration", support_calibration, CALIBRATION_PER_CLASS),
    ):
        counts = Counter((row.get("source"), row.get("label")) for row in rows)
        expected_counts = {
            (source, label): per_class
            for source, labels in NATIVE_LABELS.items()
            for label in labels
        }
        if counts != expected_counts:
            raise ValueError(f"diagnostic {role} class balance differs")
        for row in rows:
            required = {
                "source", "split", "row", "parent_id", "state",
                "state_sha256", "label", "selection_sha256",
            }
            if set(row) != required:
                raise ValueError(f"diagnostic {role} row fields differ")
            if row["state_sha256"] != canonical_sha256(row["state"]):
                raise ValueError(f"diagnostic {role} state digest differs")
            selection = canonical_sha256({
                "seed": SEED,
                "source": row["source"],
                "parent_id": row["parent_id"],
            })
            if row["selection_sha256"] != selection:
                raise ValueError(f"diagnostic {role} selection digest differs")
            identity = (row["source"], row["parent_id"])
            if identity in memberships:
                raise ValueError("diagnostic support roles overlap")
            memberships.add(identity)

    development = _read_jsonl(
        _safe_file(root, "development.jsonl"), "diagnostic development",
    )
    population_counts = Counter(
        (row.get("target"), row.get("population")) for row in development
    )
    if population_counts != {
        ("emotion", "native_six_class"): 92,
        ("emotion", "augmented_none_candidate"): 12,
        ("emotion", "augmented_none_target"): 12,
        ("tweet_offensive", "native_binary"): 80,
    }:
        raise ValueError("diagnostic development populations differ")
    for row in development:
        unsigned_row = dict(row)
        expected_row_sha256 = unsigned_row.pop("row_sha256", None)
        if (
            expected_row_sha256 != canonical_sha256(unsigned_row)
            or row.get("state_sha256") != canonical_sha256(row.get("state"))
        ):
            raise ValueError("diagnostic development row digest differs")
    variants = _read_jsonl(
        _safe_file(root, "semantic-variants.jsonl"), "semantic variants",
    )
    if variants != semantic_variants(development):
        raise ValueError("diagnostic semantic variants do not reproduce")
    unique_development_states = len({row["state_sha256"] for row in development})
    feature_count = len(support_fit) + len(support_calibration) + unique_development_states
    compute = protocol.get("compute", {})
    microbatch = compute.get("microbatch")
    if type(microbatch) is not int or microbatch <= 0:
        raise ValueError("diagnostic microbatch is invalid")
    if (
        compute.get("original_backbone_feature_sequences") != feature_count
        or compute.get("shared_backbone_feature_sequences") != feature_count
        or compute.get("shared_pointer_semantic_sequences") != len(variants)
        or compute.get("aggregate_encoded_sequences") != 2 * feature_count + len(variants)
        or compute.get("aggregate_encoded_sequences") > compute.get("limit", 0)
        or compute.get("planned_model_forward_calls")
        != 2 * math.ceil(feature_count / microbatch)
        + math.ceil(len(variants) / microbatch)
    ):
        raise ValueError("diagnostic compute budget does not reproduce")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--registry", default=Path("research/evidence/index.json"), type=Path)
    args = parser.parse_args()
    result = freeze(args.paths, args.local_paths, args.out, args.registry)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
