import unittest

from experiments.freeze_korean_suite import (
    nsmc_request,
    select_records,
    ynat_request,
)
from haetae.measure import state_digest


class KoreanSuiteTest(unittest.TestCase):
    def test_ynat_request_has_one_true_and_one_false_check(self):
        request = ynat_request({"state": "기사", "label": 3}, "a" * 64)
        questions = request["questions"]
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions["topic"]["label"], "culture")
        labels = {
            questions["topic_check_1"]["label"],
            questions["topic_check_2"]["label"],
        }
        self.assertEqual(labels, {True, False})

    def test_nsmc_request_preserves_positive_semantics(self):
        positive = nsmc_request({"state": "좋아요", "label": 0}, "b" * 64)
        negative = nsmc_request({"state": "별로", "label": 1}, "c" * 64)
        self.assertTrue(positive["questions"]["positive"]["label"])
        self.assertEqual(
            positive["questions"]["polarity"]["label"], "positive",
        )
        self.assertFalse(negative["questions"]["positive"]["label"])
        self.assertEqual(
            negative["questions"]["polarity"]["label"], "negative",
        )

    def test_selection_balances_labels_and_excludes_ambiguous_states(self):
        records = [
            {"state": f"state-{label}-{index}", "label": label}
            for label in range(2)
            for index in range(5)
        ]
        ambiguous = {"state": "ambiguous", "label": 0}
        records.extend([ambiguous, {"state": "ambiguous", "label": 1}])
        forbidden = {state_digest(records[0])}
        selected, audit = select_records(
            records, "korean_nsmc", forbidden, seed=7, total=6,
        )
        labels = [
            request["questions"]["polarity"]["label"]
            for request in selected
        ]
        self.assertEqual(labels.count("positive"), 3)
        self.assertEqual(labels.count("negative"), 3)
        self.assertEqual(audit["excluded_overlap_records"], 1)
        self.assertEqual(audit["ambiguous_states"], 1)


if __name__ == "__main__":
    unittest.main()
