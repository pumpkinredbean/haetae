import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_shared_execution import (
    ARMS,
    aggregate_equivalence,
    arm_order,
    artifact_row_binding,
    assign_performance_panels,
    batch_schedule,
    build_schedules,
    canonical_conditions,
    clean_incomplete_equivalence,
    comparison_metrics,
    encode_and_validate,
    load_design,
    ordered_conditions,
    paired_ratio_bootstrap,
    request_identity,
    validate_comparison_value,
    validate_equivalence_coverage,
    validate_memory_coverage,
    validate_runtime_compatibility,
    validate_timing_coverage,
    value_sha256,
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
        self.assertEqual(repetition, first["repetitions"]["1"])
        self.assertEqual(repetition, first["repetitions"]["2"])
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
        canonical = canonical_conditions(repetition)
        for condition_index in range(len(canonical)):
            orders = [arm_order(value, condition_index) for value in range(3)]
            for position in range(3):
                self.assertEqual({order[position] for order in orders}, set(ARMS))
        self.assertEqual(first["memory"]["requests"], 1000)

    def test_performance_panels_use_distinct_parents_and_cover_high_option_case(self):
        rows = []
        for parent in range(70):
            for sibling in range(2):
                row = self.row(f"parent-{parent}-sibling-{sibling}", 2, "decision")
                row["parent_id"] = f"parent-{parent}"
                rows.append(row)
        rows[0]["candidate_counts"] = [99, 2]
        rows[0]["packed_length"] = 1
        design = load_design()
        assign_performance_panels(rows, design)
        primary = [
            row for row in rows if "multi_primary_pool" in row["performance_panels"]
        ]
        diagnostic = [
            row for row in rows if "longest_diagnostic" in row["performance_panels"]
        ]
        self.assertEqual(len(primary), 64)
        self.assertEqual(len({row["parent_id"] for row in primary}), 64)
        self.assertEqual(len(diagnostic), 16)
        self.assertEqual(len({row["parent_id"] for row in diagnostic}), 16)
        self.assertIn(rows[0], diagnostic)

    def test_batch_schedule_never_mixes_suites(self):
        rows = [self.row(1, 2, "decision"), self.row(2, 2, "korean")]
        mapping = {row["workload_id"]: row for row in rows}
        batches = batch_schedule(list(mapping), mapping, 2)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch["workload_ids"]) == 1 for batch in batches))
        self.assertTrue(all(len(batch["parent_ids"]) == 1 for batch in batches))


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
                        "parent_ids": [parent],
                        "arm": arm,
                        "duration_ns": duration,
                    })
        report = paired_ratio_bootstrap(rows, draws=200, seed=7)
        self.assertEqual(report["parents"], 2)
        self.assertEqual(report["workloads"], 2)
        self.assertAlmostEqual(report["ratio"], 0.85)
        self.assertLess(report["interval_95"][1], 1.0)

    def test_parent_bootstrap_accumulates_repeated_observations(self):
        rows = []
        for repetition in range(3):
            for observation in range(20):
                for arm, duration in (
                    ("P", 80 if observation == 19 else 104),
                    ("B", 100),
                ):
                    rows.append({
                        "request_batch_size": 1,
                        "repetition": repetition,
                        "workload_ids": ["workload"],
                        "parent_ids": ["parent"],
                        "arm": arm,
                        "duration_ns": duration,
                    })
        report = paired_ratio_bootstrap(rows, draws=200, seed=7)
        reversed_report = paired_ratio_bootstrap(
            list(reversed(rows)), draws=200, seed=7,
        )
        self.assertAlmostEqual(report["ratio"], 1.028)
        self.assertEqual(report, reversed_report)
        self.assertEqual(report["paired_observations"], 60)

    def test_parent_bootstrap_clusters_sibling_workloads(self):
        rows = []
        for repetition in range(3):
            for workload, parent, packed in (
                ("a1", "a", 80), ("a2", "a", 90),
                ("b1", "b", 110), ("b2", "b", 120),
            ):
                for arm, duration in (("P", packed), ("B", 100)):
                    rows.append({
                        "request_batch_size": 1,
                        "repetition": repetition,
                        "workload_ids": [workload],
                        "parent_ids": [parent],
                        "arm": arm,
                        "duration_ns": duration,
                    })
        report = paired_ratio_bootstrap(rows, draws=200, seed=7)
        self.assertEqual(report["parents"], 2)
        self.assertEqual(report["workloads"], 4)
        self.assertAlmostEqual(report["ratio"], 1.0)


class SharedExecutionArtifactValidationTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "workload_id": "workload",
            "parent_id": "parent",
            "suite": "decision",
            "question_ids": ["question"],
            "candidate_counts": [2],
            "question_count": 1,
            "packed_length": 5,
            "separate_lengths": [6],
            "performance_panels": [],
        }
        self.protocol = {
            "protocol_sha256": "protocol",
            "run": {"checkpoint_sha256": "checkpoint"},
            "design": {
                "equivalence": {"temperatures": [1.0]},
                "memory": {
                    "steady_state_requests": 1,
                    "request_batch_size": 1,
                    "scope": "complete_request",
                    "maximum_residual_live_tensor_increase_bytes": 64,
                },
            },
        }
        self.workload_meta = {"workloads_file_sha256": "workloads"}

    def equivalence_records(self, binding):
        packed_logits = [0.25, -0.25]
        observations = [{
            "kind": "base_equivalence", **binding,
            "workload_id": "workload", "parent_id": "parent",
            "suite": "decision", "question_id": "question",
            "question_index": 0, "packed_logits": packed_logits,
            "comparisons": {
                arm: [comparison_metrics(packed_logits, packed_logits, 1.0)]
                for arm in ("B", "S")
            },
        }]
        return observations + [{
            "kind": "request_complete", **binding,
            "workload_id": "workload", "parent_id": "parent",
            "suite": "decision",
            "observations_sha256": value_sha256(observations),
        }]

    def test_equivalence_journal_recovers_only_an_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.jsonl"
            completed = clean_incomplete_equivalence(
                path, self.protocol, self.workload_meta, [self.row], "cpu",
            )
            self.assertEqual(completed, set())
            binding = artifact_row_binding(
                self.protocol, self.workload_meta, "cpu",
            )
            records = self.equivalence_records(binding)
            with path.open("ab") as handle:
                for record in records:
                    handle.write((json.dumps(record) + "\n").encode())
                handle.write(b'{"kind":"base_equivalence"')
            completed = clean_incomplete_equivalence(
                path, self.protocol, self.workload_meta, [self.row], "cpu",
            )
            self.assertEqual(completed, {"workload"})
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_equivalence_journal_rejects_retained_foreign_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.jsonl"
            clean_incomplete_equivalence(
                path, self.protocol, self.workload_meta, [self.row], "cpu",
            )
            binding = artifact_row_binding(
                self.protocol, self.workload_meta, "cpu",
            )
            foreign = dict(binding, protocol_sha256="old")
            records = self.equivalence_records(foreign)
            with path.open("ab") as handle:
                for record in records:
                    handle.write((json.dumps(record) + "\n").encode())
            with self.assertRaisesRegex(ValueError, "binding differs"):
                clean_incomplete_equivalence(
                    path, self.protocol, self.workload_meta, [self.row], "cpu",
                )

    def test_equivalence_journal_rejects_inconsistent_numerical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.jsonl"
            clean_incomplete_equivalence(
                path, self.protocol, self.workload_meta, [self.row], "cpu",
            )
            binding = artifact_row_binding(
                self.protocol, self.workload_meta, "cpu",
            )
            records = self.equivalence_records(binding)
            records[0]["packed_logits"] = [100.0, -100.0]
            records[-1]["observations_sha256"] = value_sha256(records[:-1])
            with path.open("ab") as handle:
                for record in records:
                    handle.write((json.dumps(record) + "\n").encode())
            with self.assertRaisesRegex(ValueError, "differ from packed logits"):
                clean_incomplete_equivalence(
                    path, self.protocol, self.workload_meta, [self.row], "cpu",
                )

    def test_equivalence_summary_requires_frozen_candidate_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.jsonl"
            binding = artifact_row_binding(
                self.protocol, self.workload_meta, "cpu",
            )
            records = [
                {"kind": "journal_header", **binding},
                *self.equivalence_records(binding),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            metadata = {
                "device": "cpu", "records": len(records), "requests": 1,
                "kind_counts": {
                    "journal_header": 1, "base_equivalence": 1,
                    "request_complete": 1,
                },
            }
            frozen_row = dict(self.row, candidate_counts=[3])
            with self.assertRaisesRegex(ValueError, "packed logits are invalid"):
                validate_equivalence_coverage(
                    path, metadata, [frozen_row], self.protocol["design"],
                    self.protocol, self.workload_meta,
                )

    def test_equivalence_summary_recomputes_metrics_from_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.jsonl"
            binding = artifact_row_binding(
                self.protocol, self.workload_meta, "cpu",
            )
            transaction = self.equivalence_records(binding)
            transaction[0]["comparisons"]["B"][0]["reference_logits"] = [
                100.0, -100.0,
            ]
            transaction[-1]["observations_sha256"] = value_sha256(transaction[:-1])
            records = [{"kind": "journal_header", **binding}, *transaction]
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            metadata = {
                "device": "cpu", "records": len(records), "requests": 1,
                "kind_counts": {
                    "journal_header": 1, "base_equivalence": 1,
                    "request_complete": 1,
                },
            }
            with self.assertRaisesRegex(ValueError, "inconsistent with logits"):
                validate_equivalence_coverage(
                    path, metadata, [self.row], self.protocol["design"],
                    self.protocol, self.workload_meta,
                )

    def test_logit_validation_handles_probability_underflow(self):
        valid = comparison_metrics([0.0, -744.0], [0.0, -744.00001], 1.0)
        validate_comparison_value(valid, 1.0, 2)
        inconsistent = comparison_metrics([0.0, -1000.0], [0.0, -1100.0], 1.0)
        inconsistent["max_absolute_log_probability_difference"] = 0.0
        with self.assertRaisesRegex(ValueError, "inconsistent with logits"):
            validate_comparison_value(inconsistent, 1.0, 2)

    def test_timing_validation_rejects_nonpositive_duration(self):
        item = {
            "suite": "decision", "workload_ids": ["workload"],
            "parent_ids": ["parent"],
        }
        populations = {"multi_primary": {"1": {"measured": [item]}}}
        schedules = {
            "schedules_sha256": "schedules",
            "repetitions": {"0": populations},
        }
        metadata = {
            "device": "cpu", "repetition": 0, "process_id": "process",
            "records": 6,
        }
        binding = artifact_row_binding(
            self.protocol, self.workload_meta, "cpu", schedules,
        )
        rows = []
        canonical = canonical_conditions(populations)
        for condition in ordered_conditions(0, populations):
            population, batch, scope = condition
            for arm in arm_order(0, canonical.index(condition)):
                shape = (
                    {"input_sequences": 1, "maximum_sequence_length": 5,
                     "unpadded_tokens": 5, "padded_tokens": 5, "padding_tokens": 0}
                    if arm == "P" else
                    {"input_sequences": 1, "maximum_sequence_length": 6,
                     "unpadded_tokens": 6, "padded_tokens": 6, "padding_tokens": 0}
                )
                rows.append({
                    **binding,
                    "process_id": "process", "repetition": 0,
                    "population": population, "suite": "decision",
                    "request_batch_size": batch, "actual_requests": 1,
                    "scope": scope, "arm": arm, "schedule_index": 0,
                    "workload_ids": ["workload"], "parent_ids": ["parent"],
                    "duration_ns": 100, "ns_per_request": 100.0,
                    **shape,
                })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timing.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            validate_timing_coverage(
                path, metadata, schedules, self.protocol,
                self.workload_meta, [self.row],
            )
            rows[0]["duration_ns"] = -1
            rows[0]["ns_per_request"] = -1.0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                validate_timing_coverage(
                    path, metadata, schedules, self.protocol,
                    self.workload_meta, [self.row],
                )

    def test_memory_validation_uses_frozen_count_and_recomputes_residual(self):
        batch = {
            "suite": "decision", "workload_ids": ["workload"],
            "parent_ids": ["parent"],
        }
        schedules = {
            "schedules_sha256": "schedules",
            "memory": {
                "requests": 1, "request_batch_size": 1, "batches": [batch],
            },
        }
        binding = artifact_row_binding(
            self.protocol, self.workload_meta, "mps", schedules,
        )
        snapshot = {
            "rss_high_water_bytes": 10,
            "mps_current_allocated_bytes": 10,
            "mps_driver_allocated_bytes": 20,
            "mps_recommended_max_bytes": 100,
        }
        row = {
            **binding, "arm": "P", "batch_index": 0,
            **batch, **snapshot,
        }
        metadata = {
            "device": "mps", "arm": "P", "scope": "complete_request",
            "requests": 1, "request_batch_size": 1,
            "baseline": snapshot,
            "observed_maximum": snapshot,
            "before_cache_release": snapshot,
            "after_cache_release": dict(snapshot, mps_current_allocated_bytes=1000),
            "residual_live_tensor_increase_bytes": 990,
            "residual_flag": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "residual flag"):
                validate_memory_coverage(
                    path, metadata, schedules, self.protocol, self.workload_meta,
                )
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "request count"):
                validate_memory_coverage(
                    path, dict(metadata, requests=0), schedules,
                    self.protocol, self.workload_meta,
                )

    def test_runtime_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent runtimes"):
            validate_runtime_compatibility([
                {"runtime": {"torch": "one"}},
                {"runtime": {"torch": "two"}},
            ])


if __name__ == "__main__":
    unittest.main()
