import unittest

from experiments.compare_development import paired_summary


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


class DevelopmentComparisonTests(unittest.TestCase):
    def test_paired_parent_bootstrap_detects_consistent_accuracy_gain(self):
        baseline, shared = [], []
        for source in ("alpha", "beta"):
            for index in range(2):
                request_id = f"{source}-{index}"
                baseline.append(prediction(source, request_id, 1, [2.0, 0.0]))
                shared.append(prediction(source, request_id, 1, [0.0, 2.0]))
        report = paired_summary(baseline, shared, 1.0, 1.0, 100, 7)
        accuracy = report["metrics"]["accuracy"]
        self.assertEqual(report["questions"], 4)
        self.assertEqual(report["parents"], 4)
        self.assertEqual(accuracy["shared_minus_baseline"], 1.0)
        self.assertEqual(
            accuracy["parent_bootstrap_ci95"]["lower"], 1.0,
        )
        self.assertEqual(
            accuracy["source_stratified_parent_bootstrap_ci95"]["upper"],
            1.0,
        )

    def test_paired_membership_must_match(self):
        baseline = [prediction("alpha", "one", 0, [1.0, 0.0])]
        shared = [prediction("alpha", "two", 0, [1.0, 0.0])]
        with self.assertRaisesRegex(ValueError, "membership differs"):
            paired_summary(baseline, shared, 1.0, 1.0, 10, 3)


if __name__ == "__main__":
    unittest.main()
