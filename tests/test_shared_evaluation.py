import unittest

import torch

from experiments.evaluate_shared import (
    fit_source_balanced_temperature,
    grouped_metrics,
    logits_to_list,
    macro_f1,
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
    def test_logits_move_to_cpu_before_float64_conversion(self):
        class DeviceGuard:
            def __init__(self):
                self.device = "mps"
                self.events = []

            def detach(self):
                self.events.append("detach")
                return self

            def cpu(self):
                self.events.append("cpu")
                self.device = "cpu"
                return self

            def to(self, *, dtype):
                if self.device != "cpu":
                    raise AssertionError(
                        "float64 conversion happened before CPU transfer"
                    )
                self.events.append(("to", dtype))
                return self

            def tolist(self):
                self.events.append("tolist")
                return [1.0, 2.0]

        guarded = DeviceGuard()
        self.assertEqual(logits_to_list(guarded), [1.0, 2.0])
        self.assertEqual(
            guarded.events,
            ["detach", "cpu", ("to", torch.float64), "tolist"],
        )

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

    def test_macro_f1_requires_one_ordered_ontology(self):
        same = [
            prediction("fixed", [2.0, 0.0], 0),
            prediction("fixed", [0.0, 2.0], 1),
        ]
        for value in same:
            value["option_keys"] = ["no", "yes"]
        self.assertEqual(macro_f1(same, [0, 1]), {
            "status": "ok", "value": 1.0,
        })
        mixed = [dict(same[0]), dict(same[1])]
        mixed[1]["option_keys"] = ["yes", "no"]
        self.assertEqual(
            macro_f1(mixed, [0, 1])["status"], "not_applicable",
        )


if __name__ == "__main__":
    unittest.main()
