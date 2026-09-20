import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.kev_adapter import flatten_for_baseline, load_frozen_split


class KevAdapterTest(unittest.TestCase):
    def test_loads_digest_bound_typed_requests(self):
        request = {
            "state": {"passage": "evidence\u2028continued", "count": 2},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": None, "b": "second"},
                    "label": "b",
                    "src": "fixture_choice",
                },
                "valid": {
                    "type": "noul",
                    "instructions": "Is it valid?",
                    "criteria": {"true": "supported", "false": "unsupported"},
                    "label": True,
                    "src": "fixture_noul",
                },
                "rating": {
                    "type": "score",
                    "instructions": "Rate",
                    "criteria": ["low", "high"],
                    "label": 1,
                    "src": "fixture_score",
                },
            },
            "_meta": {
                "id": "fixture/dev/1", "group_id": "group-1",
                "source": "fixture",
            },
        }
        encoded = (json.dumps(request, sort_keys=True) + "\n").encode()
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            (suite / "development.jsonl").write_bytes(encoded)
            manifest = {
                "files": {
                    "development.jsonl": {
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "records": 1,
                        "questions": 3,
                    }
                }
            }
            (suite / "manifest.json").write_text(json.dumps(manifest))
            requests, loaded_manifest = load_frozen_split(
                suite, "development",
            )
        self.assertEqual(loaded_manifest, manifest)
        questions = {
            question["id"]: question for question in requests[0]["questions"]
        }
        self.assertEqual(questions["route"]["options"], ["a", "b: second"])
        self.assertEqual(questions["route"]["label"], 1)
        self.assertEqual(questions["valid"]["options"], [
            "yes: supported", "no: unsupported",
        ])
        flattened = flatten_for_baseline(requests)
        self.assertEqual(len(flattened), 3)
        self.assertTrue(all(record["parent"] == "group-1" for record in flattened))

    def test_rejects_changed_suite_bytes(self):
        encoded = b"{}\n"
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            (suite / "development.jsonl").write_bytes(encoded)
            (suite / "manifest.json").write_text(json.dumps({
                "files": {
                    "development.jsonl": {
                        "sha256": "0" * 64,
                        "records": 1,
                        "questions": 0,
                    }
                }
            }))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_frozen_split(suite, "development")

    def test_locked_test_requires_explicit_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            encoded = b""
            (suite / "test.jsonl").write_bytes(encoded)
            manifest = {
                "files": {
                    "test.jsonl": {
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "records": 0,
                        "questions": 0,
                    }
                }
            }
            (suite / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "locked test"):
                load_frozen_split(suite, "test")
            requests, _ = load_frozen_split(
                suite, "test", allow_test=True,
            )
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
