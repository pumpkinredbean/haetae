import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_shared_execution import (
    aggregate_equivalence,
    assign_performance_panels,
    batch_schedule,
    build_schedules,
    comparison_metrics,
    encode_and_validate,
    load_design,
    paired_ratio_bootstrap,
    request_identity,
    workload_row,
)
from experiments.shared_state import DELIMITER_TOKENS


class TinyTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    unk_token_id = 3

    def __init__(self):
        self.vocab = {
            token: index + 4 for index, token in enumerate(DELIMITER_TOKENS)
        }
        self.next_id = 20

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        identifiers = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            identifiers.append(self.vocab[word])
        return {"input_ids": identifiers}


def make_request(identifier, question_count, suite="decision"):
    return {
        "state": f"state {identifier}",
        "questions": [
            {
                "id": f"q-{identifier}-{index}",
                "type": "choice",
                "instructions": f"choose {index}",
                "options": ["alpha", "beta"],
                "option_keys": ["a", "b"],
                "label": index % 2,
                "soft": None,
                "source": f"{suite}_source",
            }
            for index in range(question_count)
        ],
        "meta": {
            "id": f"{suite}-{identifier}",
            "group_id": f"group-{identifier}",
            "source": f"{suite}_source",
        },
    }


class SharedExecutionFreezeTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.delimiters = {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in DELIMITER_TOKENS
        }

    def row(self, identifier, question_count, suite="decision"):
        request = make_request(identifier, question_count, suite)
        packed, singles = encode_and_validate(
            self.tokenizer, self.delimiters, request, 128,
        )
        return workload_row(suite, request, packed, singles)

    def test_packed_and_separate_token_arrays_are_bound(self):
        request = make_request(1, 3)
        packed, singles = encode_and_validate(
            self.tokenizer, self.delimiters, request, 128,
        )
        row = workload_row("decision", request, packed, singles)
        self.assertEqual(row["question_count"], 3)
        self.assertEqual(row["packed_length"], sum(row["branch_lengths"]) + row["state_length"])
        self.assertEqual(len(row["separate_token_sha256"]), 3)
        self.assertEqual(len(request_identity("decision", request)), 64)

    def test_result_independent_panels_and_schedules_are_deterministic(self):
        rows = []
        for suite, counts in (
            ("decision", (1, 2, 3)),
            ("transfer", (1,)),
            ("korean", (2, 3)),
        ):
            for count in counts:
                for index in range(70):
                    rows.append(self.row(f"{suite}-{count}-{index}", count, suite))
        design = load_design()
        assign_performance_panels(rows, design)
        first = build_schedules(rows, design)
        second = build_schedules(rows, design)
        self.assertEqual(first, second)
        self.assertEqual(
            sum("multi_primary_pool" in row["performance_panels"] for row in rows),
            256,
        )
        self.assertEqual(
            sum("single_control_pool" in row["performance_panels"] for row in rows),
            64,
        )
        repetition = first["repetitions"]["0"]
        self.assertEqual(
            sum(
                len(batch["workload_ids"])
                for batch in repetition["multi_primary"]["1"]["measured"]
            ),
            200,
        )
        self.assertEqual(
            sum(
                len(batch["workload_ids"])
                for batch in repetition["single_control"]["2"]["measured"]
            ),
            64,
        )

    def test_batch_schedule_never_mixes_suites(self):
        rows = [self.row(1, 2, "decision"), self.row(2, 2, "korean")]
        mapping = {row["workload_id"]: row for row in rows}
        batches = batch_schedule(list(mapping), mapping, 2)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch["workload_ids"]) == 1 for batch in batches))


class SharedExecutionAnalysisTest(unittest.TestCase):
    def test_probability_comparison_preserves_real_argmax(self):
        result = comparison_metrics([0.0, 0.001], [0.001, 0.0], 1.0)
        self.assertTrue(result["action_changed"])
        self.assertEqual(result["reference_action"], 1)
        self.assertEqual(result["candidate_action"], 0)
        self.assertGreater(result["total_variation"], 0)

    def test_equivalence_gate_rejects_material_action_change(self):
        design = load_design()
        value = comparison_metrics([0.0, 1.0], [1.0, 0.0], 1.0)
        row = {
            "kind": "base_equivalence",
            "workload_id": "w",
            "comparisons": {"B": [value], "S": [value]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.jsonl"
            path.write_text(json.dumps(row) + "\n")
            report = aggregate_equivalence(path, design)
        self.assertFalse(report["base_equivalence/B/1.0"]["passed"])
        self.assertEqual(
            report["base_equivalence/B/1.0"]["material_action_changes"], 1,
        )

    def test_parent_bootstrap_averages_repetitions_before_resampling(self):
        rows = []
        for repetition in range(3):
            for parent, packed, batched in (
                ("a", 80, 100), ("b", 90, 100),
            ):
                for arm, duration in (("P", packed), ("B", batched)):
                    rows.append({
                        "request_batch_size": 1,
                        "repetition": repetition,
                        "workload_ids": [parent],
                        "arm": arm,
                        "duration_ns": duration,
                    })
        report = paired_ratio_bootstrap(rows, draws=200, seed=7)
        self.assertEqual(report["parents"], 2)
        self.assertAlmostEqual(report["ratio"], 0.85)
        self.assertLess(report["interval_95"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
