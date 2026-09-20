import unittest

from experiments.compare_development import (
    inventory_from_requests,
    paired_summary,
    prediction_key,
    quantile_interval,
    validate_baseline_run,
    validate_population,
    validate_report_plan,
)


def prediction(source, request_id, label, logits):
    return {
        "request_id": request_id,
        "group_id": request_id,
        "state_sha256": f"state-{request_id}",
        "question_id": "choice",
        "source": source,
        "type": "choice",
        "options": ["left", "right"],
        "option_keys": ["left", "right"],
        "label": label,
        "soft": None,
        "logits": logits,
    }


def parent_bindings(predictions, origin="fixture"):
    return {
        prediction_key(item): {
            "parent_namespace": origin,
            "parent": f"{origin}\0{item['group_id']}",
        }
        for item in predictions
    }


class DevelopmentComparisonTests(unittest.TestCase):
    def test_paired_parent_bootstrap_detects_consistent_accuracy_gain(self):
        baseline, shared = [], []
        for source in ("alpha", "beta"):
            for index in range(2):
                request_id = f"{source}-{index}"
                baseline.append(prediction(source, request_id, 1, [2.0, 0.0]))
                shared.append(prediction(source, request_id, 1, [0.0, 2.0]))
        report = paired_summary(
            baseline, shared, parent_bindings(baseline),
            1.0, 1.0, 100, 7,
        )
        accuracy = report["metrics"]["accuracy"]
        self.assertEqual(report["questions"], 4)
        self.assertEqual(report["parents"], 4)
        self.assertEqual(accuracy["shared_minus_baseline"], 1.0)
        self.assertEqual(
            accuracy["parent_bootstrap_ci95"]["lower"], 1.0,
        )
        self.assertEqual(
            accuracy["source_macro_parent_bootstrap_ci95"]["upper"],
            1.0,
        )

    def test_paired_membership_must_match(self):
        baseline = [prediction("alpha", "one", 0, [1.0, 0.0])]
        shared = [prediction("alpha", "two", 0, [1.0, 0.0])]
        with self.assertRaisesRegex(ValueError, "membership differs"):
            paired_summary(
                baseline, shared, parent_bindings(baseline),
                1.0, 1.0, 10, 3,
            )

    def test_cross_task_questions_share_parent_multiplicity(self):
        baseline, shared, bindings = [], [], {}
        for index in range(40):
            parent = f"review-{index}"
            for source in ("sentiment_choice", "sentiment_noul"):
                base = prediction(source, parent, 1, [2.0, 0.0])
                candidate = prediction(source, parent, 1, [0.0, 2.0])
                if index >= 20:
                    base["logits"], candidate["logits"] = (
                        candidate["logits"], base["logits"],
                    )
                baseline.append(base)
                shared.append(candidate)
                bindings[prediction_key(base)] = {
                    "parent_namespace": "nsmc",
                    "parent": f"nsmc\0{parent}",
                }
        report = paired_summary(
            baseline, shared, bindings, 1.0, 1.0, 5000, 11,
        )
        accuracy = report["metrics"]["accuracy"]
        self.assertEqual(report["parents"], 40)
        self.assertEqual(report["task_sources"], 2)
        self.assertAlmostEqual(accuracy["shared_minus_baseline"], 0.0)
        self.assertLessEqual(
            accuracy["parent_bootstrap_ci95"]["lower"], -0.25,
        )
        self.assertGreaterEqual(
            accuracy["parent_bootstrap_ci95"]["upper"], 0.25,
        )

    def test_invalid_negative_label_is_rejected(self):
        baseline = [
            prediction("alpha", f"row-{index}", -1, [1.0, 0.0])
            for index in range(2)
        ]
        shared = [dict(item) for item in baseline]
        with self.assertRaisesRegex(ValueError, "invalid hard label"):
            paired_summary(
                baseline, shared, parent_bindings(baseline),
                1.0, 1.0, 10, 3,
            )

    def test_quantile_interval_uses_one_nearest_rank_convention(self):
        interval = quantile_interval([float(index) for index in range(100)])
        self.assertEqual(interval["lower"], 2.0)
        self.assertEqual(interval["upper"], 97.0)

    def test_frozen_request_maps_task_sources_to_one_origin_parent(self):
        request = {
            "state": "review",
            "meta": {"source": "nsmc", "id": "nsmc/1", "group_id": "nsmc/1"},
            "questions": [
                {
                    "id": "choice", "source": "nsmc_choice", "type": "choice",
                    "options": ["negative", "positive"],
                    "option_keys": ["negative", "positive"],
                    "label": 1, "soft": None,
                },
                {
                    "id": "noul", "source": "nsmc_noul", "type": "noul",
                    "options": ["yes", "no"], "option_keys": ["yes", "no"],
                    "label": 0, "soft": None,
                },
            ],
        }
        inventory = inventory_from_requests([request])
        self.assertEqual(len(inventory), 2)
        self.assertEqual(
            {item["parent"] for item in inventory.values()}, {"nsmc\0nsmc/1"},
        )

    def test_population_completeness_is_checked_against_frozen_inventory(self):
        first = prediction("alpha", "one", 0, [1.0, 0.0])
        second = prediction("alpha", "two", 0, [1.0, 0.0])
        fields = (
            "request_id", "group_id", "state_sha256", "question_id", "source",
            "type", "options", "option_keys", "label", "soft",
        )
        inventory = {
            prediction_key(item): {
                "expected": {field: item[field] for field in fields},
                "parent_namespace": "alpha",
                "parent": f"alpha\0{item['group_id']}",
            }
            for item in (first, second)
        }
        with self.assertRaisesRegex(ValueError, "frozen population"):
            validate_population([first], inventory, "fixture")

    def test_report_bindings_are_checked_against_plan(self):
        plan = {
            "plan_sha256": "plan",
            "baseline_run": {
                "run_id": "run", "spec_sha256": "spec", "generation": 3,
                "checkpoint_sha256": "checkpoint",
            },
            "suites": {
                track: {
                    "manifest_sha256": f"{track}-manifest",
                    "files": {
                        "development.jsonl": {"sha256": f"{track}-split"},
                        **({"calibration.jsonl": {"sha256": "calibration"}}
                           if track == "decision" else {}),
                    },
                }
                for track in ("decision", "transfer", "korean")
            },
        }
        report = {
            "comparison_plan_sha256": "plan",
            "model": "baseline-separate-question",
            "run": {
                "run_id": "run", "spec_sha256": "spec", "generation": 3,
                "checkpoint_sha256": "checkpoint",
            },
            "temperature": {
                "fit_manifest_sha256": "decision-manifest",
                "fit_split_sha256": "calibration",
            },
            **{
                f"{track}_development": {
                    "manifest_sha256": f"{track}-manifest",
                    "split_sha256": f"{track}-split",
                }
                for track in ("decision", "transfer", "korean")
            },
        }
        validate_report_plan(report, plan, "baseline")
        validate_baseline_run(report, plan)
        report["decision_development"]["split_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs from the plan"):
            validate_report_plan(report, plan, "baseline")
        report["decision_development"]["split_sha256"] = "decision-split"
        report["run"]["generation"] = 999
        with self.assertRaisesRegex(ValueError, "run differs"):
            validate_baseline_run(report, plan)


if __name__ == "__main__":
    unittest.main()
