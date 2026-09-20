import unittest

from experiments.freeze_shared_suite import assert_disjoint, select_korean_training


def request(identifier, source="source", state=None):
    return {
        "state": state or identifier,
        "questions": {"q": {}},
        "_meta": {
            "id": identifier,
            "group_id": identifier,
            "source": source,
            "text_sha256": state or identifier,
        },
    }


class SharedSuiteTest(unittest.TestCase):
    def test_disjoint_audit_accepts_namespaced_parents(self):
        audit = assert_disjoint({
            "train": [request("same", "one", "train-state")],
            "calibration": [request("same", "two", "cal-state")],
            "development": [request("dev", "one", "dev-state")],
        })
        self.assertEqual(audit["state_overlap"], 0)
        self.assertEqual(audit["source_parent_overlap"], 0)

    def test_disjoint_audit_rejects_cross_split_state(self):
        with self.assertRaisesRegex(ValueError, "partitions overlap"):
            assert_disjoint({
                "train": [request("train", state="shared-state")],
                "calibration": [request("cal", state="shared-state")],
                "development": [],
            })

    def test_korean_training_removes_conflicts_and_calibration_overlap(self):
        training = [
            {"state": "kept", "source": "nsmc", "label": 0},
            {"state": "duplicate", "source": "nsmc", "label": 1},
            {"state": "duplicate", "source": "nsmc", "label": 1},
            {"state": "conflict", "source": "nsmc", "label": 0},
            {"state": "conflict", "source": "nsmc", "label": 1},
            {"state": "held", "source": "nsmc", "label": 0},
        ]
        calibration = [
            {"state": "held", "source": "nsmc", "label": 0},
        ]
        selected, audit = select_korean_training(training, calibration)
        self.assertEqual(len(selected), 2)
        self.assertEqual(audit["duplicate_records"], 2)
        self.assertEqual(audit["ambiguous_states"], 1)
        self.assertEqual(audit["calibration_overlap_states"], 1)


if __name__ == "__main__":
    unittest.main()
