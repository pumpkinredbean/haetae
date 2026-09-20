import json
import tempfile
import unittest
from pathlib import Path

from experiments.comparison_protocol import (
    canonical_sha256,
    common_clean_predictions,
    load_comparison_plan,
    training_clean_calibration_predictions,
)


class ComparisonProtocolTest(unittest.TestCase):
    def plan(self):
        return {
            "excluded_development_requests": {
                "decision": [{
                    "state_sha256": "seen",
                    "questions": 2,
                }],
                "transfer": [],
                "korean": [],
            },
            "excluded_calibration_requests": [{
                "state_sha256": "trained",
                "questions": 1,
            }],
        }

    def test_common_clean_removes_complete_request(self):
        predictions = [
            {"state_sha256": "seen", "question_id": "a"},
            {"state_sha256": "kept", "question_id": "b"},
            {"state_sha256": "seen", "question_id": "c"},
        ]
        clean, audit = common_clean_predictions(
            predictions, self.plan(), "decision",
        )
        self.assertEqual(clean, [predictions[1]])
        self.assertEqual(audit, {
            "excluded_requests": 1,
            "excluded_questions": 2,
            "retained_questions": 1,
        })

    def test_exclusion_count_must_match_predictions(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            common_clean_predictions(
                [{"state_sha256": "seen"}], self.plan(), "decision",
            )

    def test_training_clean_calibration_uses_separate_exclusions(self):
        predictions = [
            {"state_sha256": "trained"},
            {"state_sha256": "validation"},
        ]
        clean, audit = training_clean_calibration_predictions(
            predictions, self.plan(),
        )
        self.assertEqual(clean, [predictions[1]])
        self.assertEqual(audit["excluded_requests"], 1)

    def test_plan_digest_is_verified(self):
        plan = {
            "version": 1,
            "status": "frozen",
            "suites": {
                "decision": {}, "transfer": {}, "korean": {},
            },
            "excluded_development_requests": {
                "decision": [], "transfer": [], "korean": [],
            },
            "excluded_calibration_requests": [],
            "rendered_state_audit": {},
            "code": {},
        }
        plan["plan_sha256"] = canonical_sha256(plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(
                load_comparison_plan(path, enforce_code=False)["plan_sha256"],
                plan["plan_sha256"],
            )
            plan["status"] = "changed"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_comparison_plan(path, enforce_code=False)


if __name__ == "__main__":
    unittest.main()
