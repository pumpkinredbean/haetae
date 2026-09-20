import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from haetae.measure import (
    TrackSpec,
    baseline_metric_reports,
    choice_stress_report,
    conformal_by_cell,
    fit_source_balanced_temperature,
    freeze_track,
    independent_formal_units,
    metric_summary,
    ontology_digest,
    prepare,
    reorder_packed,
    selective_risk_report,
    state_digest,
    validate_role_isolation,
)
from haetae.train import dataset_fingerprint


def row(index, parent=None):
    record = {
        "state": f"state-{index // 2}",
        "type": "choice",
        "instructions": "pick",
        "options": ["a", "b"],
        "label": index % 2,
        "soft": None,
        "source": "fixture",
    }
    if parent is not None:
        record["parent"] = parent
    return record


def prediction(index, logits, label, *, soft=None, parent=None,
               source="fixture", track="fixture", type_="choice"):
    return {
        "status": "ok",
        "logits": logits,
        "packed_tokens": 32,
        "record": {
            "state": f"state-{index}", "type": type_,
            "instructions": "pick",
            "options": [str(option) for option in range(len(logits))],
            "label": label, "soft": soft, "source": source,
            "parent": parent,
        },
        "meta": {
            "source": "helpsteer2" if source.startswith("helpsteer2_") else source,
            "track": track, "language": "en",
            "parent_id": parent or f"parent-{index}",
            "record_sha256": f"{index:064x}",
        },
    }


class MeasurementPreparationTest(unittest.TestCase):
    def setUp(self):
        self.spec = TrackSpec(
            "fixture_track", "fixture", "fixture/repository", None,
            "fixture-revision", "en", single_split="validation",
        )

    def test_parent_assignment_is_disjoint_and_deterministic(self):
        records = [row(index, f"parent-{index // 2}") for index in range(16)]

        def loader(split, limit):
            return iter(copy.deepcopy(records))

        with mock.patch.dict("haetae.measure.LOADERS", {"fixture": loader}):
            first, _ = freeze_track(self.spec, 31, 8, set(), set())
            random.seed(999)
            second, _ = freeze_track(self.spec, 31, 8, set(), set())

        first_membership = {
            (item["meta"]["parent_id"], item["meta"]["role"])
            for item in first
        }
        second_membership = {
            (item["meta"]["parent_id"], item["meta"]["role"])
            for item in second
        }
        self.assertEqual(first_membership, second_membership)
        roles_by_parent = {}
        for item in first:
            roles_by_parent.setdefault(item["meta"]["parent_id"], set()).add(
                item["meta"]["role"]
            )
        self.assertTrue(all(len(roles) == 1 for roles in roles_by_parent.values()))
        self.assertEqual(
            {role: sum(role in roles for roles in roles_by_parent.values())
             for role in ("A", "B", "C")},
            {"A": 2, "B": 2, "C": 4},
        )

    def test_overlap_excludes_the_complete_parent(self):
        records = [row(index, f"parent-{index // 2}") for index in range(12)]
        forbidden = state_digest(records[0])

        def loader(split, limit):
            return iter(copy.deepcopy(records))

        with mock.patch.dict("haetae.measure.LOADERS", {"fixture": loader}):
            frozen, audit = freeze_track(
                self.spec, 31, 8, set(), {forbidden},
            )
        self.assertEqual(audit["pool_audit"]["excluded_overlap_parents"], 1)
        self.assertNotIn("parent-0", {
            item["meta"]["parent_id"] for item in frozen
        })

    def test_role_validator_rejects_shared_state(self):
        base = {
            "record": row(0),
            "meta": {
                "track": "fixture", "parent_id": "p0",
                "state_sha256": "a" * 64, "record_sha256": "b" * 64,
                "role": "A",
            },
        }
        other = copy.deepcopy(base)
        other["meta"].update(parent_id="p1", record_sha256="c" * 64,
                             role="C")
        with self.assertRaisesRegex(ValueError, "roles overlap"):
            validate_role_isolation([base, other])

    def test_prepare_writes_digest_bound_role_files(self):
        training = [row(index) for index in range(5)]
        evaluation = [row(index + 100, f"eval-{index // 2}")
                      for index in range(16)]
        seed = 17
        keys = list(range(len(training)))
        random.Random(seed).shuffle(keys)
        validation = [training[keys[0]]]
        train = [training[index] for index in keys[1:]]

        def loader(split, limit):
            records = training if split == "train" else evaluation
            records = records[:limit] if limit is not None else records
            return iter(copy.deepcopy(records))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            run = {
                "run_id": "fixture-run",
                "spec_sha256": "d" * 64,
                "spec": {
                    "config": {
                        "sources": "fixture", "per_source": 4,
                        "eval_per_source": 1, "seed": seed,
                    },
                    "train_fingerprint": dataset_fingerprint(train),
                    "validation_fingerprint": dataset_fingerprint(validation),
                },
            }
            (run_dir / "run.json").write_text(json.dumps(run))
            output = root / "evaluation"
            with (mock.patch.dict("haetae.measure.LOADERS", {"fixture": loader}),
                  mock.patch.dict("haetae.measure.TRACKS", {
                      "fixture": (self.spec,)
                  })):
                plan = prepare(
                    run_dir, output, "fixture-run", max_parents=8, seed=31,
                )
            self.assertEqual(plan["status"], "prepared")
            self.assertEqual(len(plan["plan_sha256"]), 64)
            for role in ("A", "B", "C"):
                self.assertTrue((output / f"{role}.jsonl").is_file())
                self.assertEqual(
                    plan["files"][f"{role}.jsonl"]["records"],
                    len((output / f"{role}.jsonl").read_text().splitlines()),
                )
            with self.assertRaises(FileExistsError):
                prepare(run_dir, output, "fixture-run", max_parents=8, seed=31)


class MeasurementMetricTest(unittest.TestCase):
    def test_reorder_packed_preserves_exact_prefix_and_option_blocks(self):
        packed = (
            [101, 7, 102, 8, 102, 11, 12, 102, 21],
            [(4, 5, 7), (7, 8, 9)],
        )
        reordered = reorder_packed(packed, [1, 0])
        self.assertEqual(reordered[0], [101, 7, 102, 8, 102, 21, 102, 11, 12])
        self.assertEqual(reordered[1], [(4, 5, 6), (6, 7, 9)])

    def test_choice_stress_aligns_semantic_options_and_skips_binary_removal(self):
        import torch

        class Tokenizer:
            cls_token_id = 101
            sep_token_id = 102
            pad_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                token = int(text) + 1 if text in {"0", "1", "2"} else 9
                return {"input_ids": [token]}

        class OptionTokenModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask, option_pos, group_ptr):
                flat = input_ids.reshape(-1)
                scores = flat[option_pos + 1].float()
                return [
                    scores[group_ptr[index]:group_ptr[index + 1]]
                    for index in range(len(group_ptr) - 1)
                ]

        records = [
            {
                "record": prediction(
                    1, [0.0, 1.0, 2.0], 2, parent="three"
                )["record"],
                "meta": prediction(
                    1, [0.0, 1.0, 2.0], 2, parent="three"
                )["meta"],
            },
            {
                "record": prediction(
                    2, [0.0, 1.0], 1, parent="binary"
                )["record"],
                "meta": prediction(
                    2, [0.0, 1.0], 1, parent="binary"
                )["meta"],
            },
        ]
        report = choice_stress_report(
            OptionTokenModel(), Tokenizer(), records, "cpu", batch=2,
            max_len=32, temperature=1.0, seed=7, permutations=3,
            removals_per_record=1,
        )
        self.assertEqual(
            report["permutation"]["raw"]["all"]
            ["semantic_top1_agreement"], 1.0
        )
        self.assertAlmostEqual(
            report["permutation"]["raw"]["all"]["mean_total_variation"],
            0.0,
        )
        self.assertEqual(report["option_removal"]["skipped_binary"], 1)
        self.assertEqual(
            report["option_removal"]["raw"]["all"]
            ["semantic_top1_agreement"], 1.0
        )

    def test_nll_uses_stable_logits_without_probability_floor(self):
        summary = metric_summary([
            prediction(1, [100.0, -100.0], 1),
        ], temperature=1.0)
        self.assertAlmostEqual(summary["nll"], 200.0, places=6)

    def test_soft_target_metrics_keep_annotation_uncertainty(self):
        import math

        summary = metric_summary([
            prediction(1, [math.log(0.6), math.log(0.4)], 0,
                       soft=[0.6, 0.4]),
        ], temperature=1.0)
        self.assertAlmostEqual(summary["brier_histogram"], 0.0, places=12)
        self.assertAlmostEqual(summary["brier_annotation"], 0.48, places=12)

    def test_score_metrics_include_each_ordinal_threshold(self):
        summary = metric_summary([
            prediction(1, [0.0, 1.0, 2.0], 2, type_="score"),
        ], temperature=1.0)
        self.assertEqual(
            set(summary["ordinal_threshold_reliability"]),
            {"K=3:Y<=0", "K=3:Y<=1"},
        )

    def test_baselines_use_uniform_and_matching_training_ontology(self):
        item = prediction(1, [2.0, 0.0], 0)
        identity = ontology_digest(item["record"])
        reports = baseline_metric_reports([item], {
            identity: {"probabilities": [0.75, 0.25]},
        })
        self.assertEqual(
            reports["training_only_empirical_prior"]["available"], 1
        )
        self.assertAlmostEqual(
            reports["uniform"]["metrics"]["all"]["nll"],
            0.6931471805599453,
        )

    def test_temperature_fit_uses_soft_targets_and_source_balance(self):
        import math

        predictions = [
            prediction(1, [math.log(0.6), math.log(0.4)], 0,
                       soft=[0.6, 0.4], source="source-a"),
            prediction(2, [math.log(0.2), math.log(0.8)], 1,
                       soft=[0.2, 0.8], source="source-b"),
        ]
        temperature = fit_source_balanced_temperature(predictions)
        self.assertAlmostEqual(temperature, 1.0, places=4)

    def test_formal_units_choose_one_question_per_parent_and_cell(self):
        predictions = [
            prediction(2, [0.0, 1.0], 1, parent="shared"),
            prediction(1, [0.0, 1.0], 1, parent="shared"),
            prediction(3, [1.0, 0.0], 0, parent="other"),
        ]
        formal = independent_formal_units(predictions)
        self.assertEqual(len(formal), 2)
        self.assertEqual(
            {item["meta"]["record_sha256"] for item in formal},
            {f"{1:064x}", f"{3:064x}"},
        )

    def test_small_conformal_cell_records_forced_full_set(self):
        predictions = [prediction(1, [2.0, 0.0], 0)]
        thresholds = conformal_by_cell(predictions, 1.0, alpha=0.1)
        self.assertTrue(thresholds["fixture"]["full_set"])
        self.assertIsNone(thresholds["fixture"]["q"])

    def test_zero_acceptance_is_not_certified(self):
        predictions = [prediction(1, [0.0, 0.0], 0)]
        policy = {
            "selective_risk": {
                "per_bound_confidence": 0.95,
                "formal_cells": ["fixture"],
                "thresholds": [0.9],
            }
        }
        report = selective_risk_report(predictions, policy, 1.0)
        cell = report["fixture"]["0.9"]
        self.assertEqual(cell["status"], "not_certified")
        self.assertIsNone(cell["risk_upper_bound"])


if __name__ == "__main__":
    unittest.main()
