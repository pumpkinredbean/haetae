import json
import tempfile
import unittest
from pathlib import Path

from haetae.calibrate import save_temperature_artifact
from haetae.checkpoint import CheckpointError
from haetae.serve import _load_calibration


class CalibrationArtifactTest(unittest.TestCase):
    def setUp(self):
        self.run = {
            "run_id": "run-1",
            "spec": {
                "tokenizer_fingerprint": "tokenizer-digest",
                "config": {"max_len": 1536},
            },
        }
        self.manifest = {
            "current": {"generation": 7, "sha256": "checkpoint-digest"}
        }

    def write(self, directory, **changes):
        artifact = {
            "version": 1,
            "run_id": "run-1",
            "checkpoint_generation": 7,
            "checkpoint_sha256": "checkpoint-digest",
            "tokenizer_fingerprint": "tokenizer-digest",
            "inference_policy": {
                "packing": "packed-preceding-separator-v1",
                "max_len": 1536,
                "temperature_scaling": "scalar-v1",
            },
            "calibration_data": {
                "dataset": "fixture",
                "split": "calibration",
                "fingerprint": "a" * 64,
                "n": 100,
            },
            "temperature": 1.25,
        }
        artifact.update(changes)
        Path(directory, "calibration.json").write_text(json.dumps(artifact))

    def test_missing_artifact_is_explicitly_uncalibrated(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                _load_calibration(temporary, self.run, self.manifest),
                (1.0, False),
            )

    def test_bound_artifact_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write(temporary)
            self.assertEqual(
                _load_calibration(temporary, self.run, self.manifest),
                (1.25, True),
            )

    def test_writer_produces_a_loadable_bound_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_temperature_artifact(
                temporary, 0.9, self.run, self.manifest,
                {
                    "dataset": "fixture",
                    "split": "calibration",
                    "fingerprint": "b" * 64,
                    "n": 40,
                },
                1536,
            )
            self.assertEqual(
                _load_calibration(temporary, self.run, self.manifest),
                (0.9, True),
            )

    def test_stale_checkpoint_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write(temporary, checkpoint_sha256="stale")
            with self.assertRaisesRegex(CheckpointError, "digest mismatch"):
                _load_calibration(temporary, self.run, self.manifest)

    def test_nonfinite_temperature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.write(temporary, temperature=float("inf"))
            with self.assertRaisesRegex(CheckpointError, "finite and positive"):
                _load_calibration(temporary, self.run, self.manifest)


if __name__ == "__main__":
    unittest.main()
