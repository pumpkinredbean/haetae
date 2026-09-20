import unittest

import torch

from experiments.evaluate_shared import (
    fit_source_balanced_temperature,
    grouped_metrics,
    metric_summary,
)


def prediction(source, logits, label, type_="choice"):
    return {
        "source": source,
        "type": type_,
        "options": [str(index) for index in range(len(logits))],
        "label": label,
        "soft": None,
        "logits": logits,
    }


class SharedEvaluationTest(unittest.TestCase):
    def test_stable_nll_retains_confident_error(self):
        metrics = metric_summary([
            prediction("fixture", [100.0, -100.0], 1),
        ], 1.0)
        self.assertAlmostEqual(metrics["nll"], 200.0, places=8)

    def test_temperature_is_source_balanced(self):
        values = [prediction("large", [4.0, 0.0], 0) for _ in range(20)]
        values += [prediction("small", [4.0, 0.0], 1)]
        temperature = fit_source_balanced_temperature(values)
        self.assertGreater(temperature, 1.0)
        report = grouped_metrics(values, temperature)
        expected = sum(
            metrics["nll"] for metrics in report["source"].values()
        ) / 2
        self.assertAlmostEqual(report["source_macro"]["nll"], expected)

    def test_score_reports_ranked_probability_score(self):
        metrics = metric_summary([
            prediction("score", [0.0, 1.0, 2.0], 2, "score"),
        ], 1.0)
        self.assertIn("ranked_probability_score", metrics)
        self.assertIn("expected_index_mae", metrics)

    def test_soft_target_distinguishes_two_brier_definitions(self):
        value = prediction("soft", [0.0, 0.0], 0)
        value["soft"] = [0.5, 0.5]
        metrics = metric_summary([value], 1.0)
        self.assertAlmostEqual(metrics["brier_histogram"], 0.0)
        self.assertAlmostEqual(metrics["brier_annotation"], 0.5)


if __name__ == "__main__":
    unittest.main()
