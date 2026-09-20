import unittest

import torch

from haetae.calibrate import (
    binomial_upper_bound,
    certify_selective_risk,
    conformal_threshold,
    fit_temperature,
    prediction_set,
)


class CalibrationValidationTest(unittest.TestCase):
    def setUp(self):
        self.probs = [torch.tensor([0.8, 0.2]), torch.tensor([0.4, 0.6])]
        self.labels = [0, 1]

    def test_nominal_paths(self):
        temperature = fit_temperature(
            [p.log() for p in self.probs], self.labels)
        self.assertGreater(temperature, 0)
        self.assertGreaterEqual(
            conformal_threshold(self.probs, self.labels, alpha=0.5), 0)
        result = certify_selective_risk(
            self.probs, self.labels, threshold=0.5)
        self.assertEqual(result["errors"], 0)

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "equal nonzero length"):
            fit_temperature([self.probs[0].log()], self.labels)
        with self.assertRaisesRegex(ValueError, "equal nonzero length"):
            conformal_threshold(self.probs, self.labels[:1])
        with self.assertRaisesRegex(ValueError, "equal nonzero length"):
            certify_selective_risk(self.probs, self.labels[:1], 0.5)

    def test_binomial_validates_before_empty_case(self):
        with self.assertRaises(ValueError):
            binomial_upper_bound(5, 0)
        with self.assertRaises(ValueError):
            binomial_upper_bound(0, 0, confidence=1.0)
        self.assertEqual(binomial_upper_bound(0, 0), 1.0)

    def test_prediction_set_keeps_inclusive_floating_point_boundary(self):
        probabilities = torch.full((9,), 1 / 9, dtype=torch.float64)
        threshold = conformal_threshold(
            [probabilities] * 20, [0] * 20, alpha=0.5,
        )
        self.assertEqual(prediction_set(probabilities, threshold), list(range(9)))


if __name__ == "__main__":
    unittest.main()
