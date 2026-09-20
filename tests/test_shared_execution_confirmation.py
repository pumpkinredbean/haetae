import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.confirm_shared_execution import (
    apply_gates,
    build_schedule,
    complementary_triplets,
    population_statistics,
    selected_source_rows,
    validate_process,
    validate_schedule,
)
from experiments.benchmark_shared_execution import atomic_json, jsonl_bytes, value_sha256
from experiments.kev_adapter import file_sha256


class SharedExecutionConfirmationTests(unittest.TestCase):
    def design(self):
        return {
            "fresh_processes": 6,
            "arms": ["P", "B"],
            "populations": {"decision": 100, "korean": 100, "single_control": 64},
            "warmup": {"complete_passes": 2},
            "primary_gate": {
                "populations": ["decision", "korean"],
                "maximum_equal_workload_mean_ratio": 0.9,
                "maximum_parent_bootstrap_95_percent_upper_bound": 1.0,
                "maximum_p95_latency_ratio": 1.05,
                "minimum_processes_below_one": 5,
                "require_both_arm_order_strata_below_one": True,
            },
            "negative_control": {
                "population": "single_control",
                "material_ratio_lower": 0.9,
                "material_ratio_upper": 1.1,
            },
            "uncertainty": {
                "parent_bootstrap_draws": 100,
                "parent_bootstrap_seed": 7,
                "process_parent_bootstrap_draws": 100,
                "process_parent_bootstrap_seed": 11,
            },
        }

    def selected(self):
        result = []
        for population, count in self.design()["populations"].items():
            for index in range(count):
                result.append({
                    "population": population,
                    "suite": population if population != "single_control" else "decision",
                    "workload_id": f"{population}-{index}",
                    "parent_id": f"parent-{population}-{index}",
                })
        return result

    def protocol_and_schedule(self):
        design = self.design()
        schedule = build_schedule(self.selected(), design)
        schedule["source_schedules_sha256"] = "source-schedule"
        schedule["schedule_sha256"] = value_sha256(schedule)
        protocol = {
            "design": design,
            "source": {
                "schedules_sha256": "source-schedule",
                "protocol_sha256": "source-protocol",
                "workloads_file_sha256": "source-workloads",
                "checkpoint_sha256": "checkpoint",
            },
            "runtime": {"runtime": "test"},
            "schedule_sha256": schedule["schedule_sha256"],
            "protocol_sha256": "confirmation-protocol",
        }
        validate_schedule(schedule, protocol)
        return protocol, schedule

    def test_complementary_triplets_balance_every_process(self):
        triplets = complementary_triplets()
        self.assertEqual(len(triplets), 20)
        for start in range(0, len(triplets), 2):
            self.assertEqual(set(triplets[start]).union(triplets[start + 1]), set(range(6)))
            self.assertFalse(set(triplets[start]).intersection(triplets[start + 1]))
        self.assertEqual(
            [sum(process in group for group in triplets) for process in range(6)],
            [10] * 6,
        )

    def test_schedule_balances_every_request_and_process(self):
        protocol, schedule = self.protocol_and_schedule()
        for entries in schedule["processes"].values():
            for population, count in protocol["design"]["populations"].items():
                subset = [entry for entry in entries if entry["population"] == population]
                self.assertEqual(len(subset), count)
                self.assertEqual(sum(entry["first_arm"] == "P" for entry in subset), count // 2)
        counts = {}
        for entries in schedule["processes"].values():
            for entry in entries:
                counts[entry["workload_id"]] = counts.get(entry["workload_id"], 0) + (
                    entry["first_arm"] == "P"
                )
        self.assertEqual(set(counts.values()), {3})

    def rows_for_population(self, population, ratio, control_first=False):
        rows = []
        for process_index in range(6):
            for workload_index in range(4):
                first = "P" if (process_index + workload_index) % 2 == 0 else "B"
                durations = {"P": 100 * ratio, "B": 100}
                for position, arm in enumerate((first, "B" if first == "P" else "P")):
                    rows.append({
                        "process_index": process_index,
                        "workload_id": f"{population}-{workload_index}",
                        "parent_id": f"parent-{workload_index // 2}",
                        "population": population,
                        "first_arm": first,
                        "pair_position": position,
                        "arm": arm,
                        "duration_ns": durations[arm],
                    })
        return rows

    def test_crossed_statistics_preserve_process_and_parent_pairing(self):
        result = population_statistics(
            self.rows_for_population("decision", 0.8), "decision", self.design()
        )
        self.assertAlmostEqual(result["equal_workload_ratio"], 0.8)
        self.assertEqual(result["processes_below_one"], 6)
        self.assertEqual(result["parents"], 2)
        self.assertEqual(result["workloads"], 4)
        self.assertEqual(result["parent_bootstrap_interval_95"], [0.8, 0.8])
        self.assertEqual(result["process_parent_bootstrap_interval_95"], [0.8, 0.8])
        self.assertAlmostEqual(result["order_strata"]["P_first"]["ratio"], 0.8)
        self.assertAlmostEqual(result["order_strata"]["B_first"]["ratio"], 0.8)

    def test_gates_reject_primary_order_dependence_and_material_control(self):
        def item(ratio, interval, processes=6, p_first=None, b_first=None):
            return {
                "equal_workload_ratio": ratio,
                "parent_bootstrap_interval_95": interval,
                "process_parent_bootstrap_interval_95": interval,
                "p95_latency_ratio": ratio,
                "processes_below_one": processes,
                "order_strata": {
                    "P_first": {"ratio": ratio if p_first is None else p_first},
                    "B_first": {"ratio": ratio if b_first is None else b_first},
                },
            }
        values = {
            "decision": item(0.8, [0.75, 0.85]),
            "korean": item(0.85, [0.8, 0.9]),
            "single_control": item(1.0, [0.95, 1.05]),
        }
        self.assertTrue(all(x["passed"] for x in apply_gates(values, self.design())["primary"].values()))
        changed = copy.deepcopy(values)
        changed["decision"]["order_strata"]["B_first"]["ratio"] = 1.01
        self.assertFalse(apply_gates(changed, self.design())["primary"]["decision"]["passed"])
        changed = copy.deepcopy(values)
        changed["single_control"] = item(1.2, [1.1, 1.3])
        self.assertTrue(apply_gates(changed, self.design())["negative_control"]["failure"])
        self.assertFalse(any(x["passed"] for x in apply_gates(changed, self.design())["primary"].values()))
        changed = copy.deepcopy(values)
        changed["single_control"] = item(1.2, [0.8, 1.3])
        self.assertTrue(apply_gates(changed, self.design())["negative_control"]["failure"])
        changed = copy.deepcopy(values)
        changed["single_control"] = item(1.05, [1.01, 1.09])
        self.assertTrue(apply_gates(changed, self.design())["negative_control"]["failure"])

    def test_process_validator_rejects_swapped_pair(self):
        protocol, schedule = self.protocol_and_schedule()
        entries = schedule["processes"]["0"]
        rows = []
        for entry in entries:
            pair = (entry["first_arm"], "B" if entry["first_arm"] == "P" else "P")
            for position, arm in enumerate(pair):
                rows.append({
                    "version": 1,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "schedule_sha256": schedule["schedule_sha256"],
                    "source_protocol_sha256": protocol["source"]["protocol_sha256"],
                    "source_workloads_sha256": protocol["source"]["workloads_file_sha256"],
                    "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
                    "device": "mps",
                    "scope": "complete_request",
                    "request_batch_size": 1,
                    "process_index": 0,
                    "process_id": "process",
                    "schedule_index": entry["schedule_index"],
                    "population": entry["population"],
                    "suite": entry["suite"],
                    "workload_id": entry["workload_id"],
                    "parent_id": entry["parent_id"],
                    "first_arm": entry["first_arm"],
                    "pair_position": position,
                    "arm": arm,
                    "duration_ns": 100,
                })
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "process-0.jsonl"
            result_path.write_bytes(jsonl_bytes(rows))
            metadata = {
                "version": 1,
                "status": "completed",
                "protocol_sha256": protocol["protocol_sha256"],
                "schedule_sha256": schedule["schedule_sha256"],
                "source_protocol_sha256": protocol["source"]["protocol_sha256"],
                "source_workloads_sha256": protocol["source"]["workloads_file_sha256"],
                "checkpoint_sha256": protocol["source"]["checkpoint_sha256"],
                "device": "mps",
                "scope": "complete_request",
                "request_batch_size": 1,
                "process_index": 0,
                "process_id": "process",
                "process_pid": 1,
                "process_started_unix_ns": 1,
                "measurement_started_unix": 1.0,
                "process_finished_unix": 2.0,
                "warmup_complete_passes": 2,
                "warmup_operations": len(entries) * 4,
                "records": len(rows),
                "file_sha256": file_sha256(result_path),
                "runtime": protocol["runtime"],
                "load_average_at_finish": [0.0, 0.0, 0.0],
                "locked_test_opened": False,
            }
            metadata["report_sha256"] = value_sha256(metadata)
            meta_path = Path(temporary) / "process-0.meta.json"
            atomic_json(meta_path, metadata)
            validate_process(result_path, meta_path, 0, protocol, schedule)
            rows[0]["arm"], rows[1]["arm"] = rows[1]["arm"], rows[0]["arm"]
            result_path.write_bytes(jsonl_bytes(rows))
            metadata["file_sha256"] = file_sha256(result_path)
            metadata.pop("report_sha256")
            metadata["report_sha256"] = value_sha256(metadata)
            atomic_json(meta_path, metadata)
            with self.assertRaisesRegex(ValueError, "arm differs"):
                validate_process(result_path, meta_path, 0, protocol, schedule)

    def test_source_selection_rejects_repeated_parent(self):
        design = self.design()
        primary = []
        source_rows = []
        for population, suite, count, questions in (
            ("decision", "decision", 100, 2),
            ("korean", "korean", 100, 2),
        ):
            for index in range(count):
                workload = f"{population}-{index}"
                parent = f"{population}-parent-{index}"
                primary.append({
                    "suite": suite,
                    "workload_ids": [workload],
                    "parent_ids": [parent],
                })
                source_rows.append({
                    "workload_id": workload,
                    "parent_id": parent,
                    "question_count": questions,
                })
        controls = []
        for index in range(64):
            workload = f"control-{index}"
            parent = f"control-parent-{index}"
            controls.append({
                "suite": "decision",
                "workload_ids": [workload],
                "parent_ids": [parent],
            })
            source_rows.append({
                "workload_id": workload,
                "parent_id": parent,
                "question_count": 1,
            })
        schedules = {
            "repetitions": {
                "0": {
                    "multi_primary": {"1": {"measured": primary}},
                    "single_control": {"1": {"measured": controls}},
                }
            }
        }
        selected_source_rows(schedules, source_rows, design)
        schedules["repetitions"]["0"]["multi_primary"]["1"]["measured"][1][
            "parent_ids"
        ] = ["decision-parent-0"]
        source_rows[1]["parent_id"] = "decision-parent-0"
        with self.assertRaisesRegex(ValueError, "repeats a parent"):
            selected_source_rows(schedules, source_rows, design)


if __name__ == "__main__":
    unittest.main()
