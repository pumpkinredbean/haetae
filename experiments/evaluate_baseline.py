"""Evaluate the completed baseline on the frozen public development suites."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from haetae.checkpoint import atomic_json_save
from haetae.certify import load_model
from haetae.model import collate, pack_question
from experiments.comparison_protocol import (
    common_clean_predictions,
    load_comparison_plan,
    request_state_sha256,
    training_clean_calibration_predictions,
    validate_suite_binding,
)
from experiments.evaluate_shared import (
    canonical_report_sha256,
    development_report,
    fit_source_balanced_temperature,
)
from experiments.kev_adapter import file_sha256, load_frozen_split


def packed_question_length(tokenizer, record: dict, max_length: int) -> int:
    state_tokens = tokenizer(
        record["state"], add_special_tokens=False,
    )["input_ids"]
    instruction_tokens = tokenizer(
        record["instructions"], add_special_tokens=False,
    )["input_ids"]
    option_tokens = [
        tokenizer(option, add_special_tokens=False)["input_ids"]
        for option in record["options"]
    ]
    complete_length = (
        2 + len(state_tokens) + len(instruction_tokens)
        + sum(1 + len(tokens) for tokens in option_tokens)
    )
    packed = pack_question(
        tokenizer,
        record["state"],
        record["instructions"],
        record["options"],
        max_length,
    )
    if packed is None:
        raise ValueError("baseline evaluation question exceeds the content budget")
    if len(packed[0]) != complete_length:
        raise ValueError("baseline evaluation would truncate a state")
    return complete_length


def flatten_requests(tokenizer, requests: list[dict], max_length: int) -> list[dict]:
    flattened = []
    for request in requests:
        state_sha256 = request_state_sha256(request)
        for question in request["questions"]:
            record = {
                "state": request["state"],
                "type": question["type"],
                "instructions": question["instructions"],
                "options": question["options"],
                "label": question["label"],
                "soft": question["soft"],
                "source": question["source"],
            }
            flattened.append({
                "record": record,
                "request_id": request["meta"].get("id"),
                "group_id": request["meta"].get("group_id"),
                "state_sha256": state_sha256,
                "question_id": question["id"],
                "option_keys": question["option_keys"],
                "questions_in_request": len(request["questions"]),
                "packed_question_tokens": packed_question_length(
                    tokenizer, record, max_length,
                ),
            })
    return flattened


def infer_requests(
    model,
    tokenizer,
    requests: list[dict],
    device: str,
    max_length: int,
    batch: int,
) -> list[dict]:
    if batch <= 0:
        raise ValueError("evaluation batch must be positive")
    flattened = flatten_requests(tokenizer, requests, max_length)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flattened), batch):
            items = flattened[start:start + batch]
            records = [item["record"] for item in items]
            packed = collate(tokenizer, records, max_length)
            if any(packed["dropped_options"]):
                raise ValueError("validated baseline question was rejected by collation")
            logits = model(
                packed["input_ids"].to(device),
                packed["attention_mask"].to(device),
                packed["option_pos"].to(device),
                packed["group_ptr"].to(device),
            )
            if len(logits) != len(items):
                raise ValueError("baseline model output count differs from questions")
            for item, question_logits in zip(items, logits):
                record = item["record"]
                if question_logits.numel() != len(record["options"]):
                    raise ValueError("baseline logits do not match the option count")
                predictions.append({
                    "request_id": item["request_id"],
                    "group_id": item["group_id"],
                    "state_sha256": item["state_sha256"],
                    "question_id": item["question_id"],
                    "source": record["source"],
                    "type": record["type"],
                    "options": record["options"],
                    "option_keys": item["option_keys"],
                    "label": record["label"],
                    "soft": record["soft"],
                    "packed_question_tokens": item["packed_question_tokens"],
                    "questions_in_request": item["questions_in_request"],
                    "logits": question_logits.detach().cpu().to(
                        dtype=torch.float64,
                    ).tolist(),
                })
    return predictions


def separate_packing_report(predictions: list[dict]) -> dict:
    return {
        "questions": len(predictions),
        "separate_question_tokens": sum(
            prediction["packed_question_tokens"]
            for prediction in predictions
        ),
        "scope": "un-padded input tokens with the state encoded per question",
    }


def evaluate(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    comparison_plan = load_comparison_plan(args.comparison_plan)
    decision_manifest = validate_suite_binding(
        comparison_plan, "decision", args.decision_suite,
    )
    transfer_manifest = validate_suite_binding(
        comparison_plan, "transfer", args.transfer_suite,
    )
    korean_manifest = validate_suite_binding(
        comparison_plan, "korean", args.korean_suite,
    )
    device = args.device or (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, tokenizer, checkpoint, run, manifest = load_model(args.run, device)
    expected_run = comparison_plan["baseline_run"]
    current = manifest["current"]
    actual_run = {
        "run_id": run["run_id"],
        "spec_sha256": run["spec_sha256"],
        "generation": current["generation"],
        "checkpoint_sha256": current["sha256"],
    }
    if any(actual_run[key] != expected_run[key] for key in actual_run):
        raise ValueError("baseline run differs from the comparison plan")
    max_length = int(checkpoint["config"]["max_len"])

    decision_calibration, _ = load_frozen_split(
        args.decision_suite, "calibration",
    )
    decision_development, _ = load_frozen_split(
        args.decision_suite, "development",
    )
    transfer_development, _ = load_frozen_split(
        args.transfer_suite, "development",
    )
    korean_development, _ = load_frozen_split(
        args.korean_suite, "development",
    )
    try:
        calibration_predictions = infer_requests(
            model, tokenizer, decision_calibration,
            device, max_length, args.batch,
        )
        calibration_clean, calibration_clean_audit = (
            training_clean_calibration_predictions(
                calibration_predictions, comparison_plan,
            )
        )
        temperature = fit_source_balanced_temperature(calibration_clean)
        decision_predictions = infer_requests(
            model, tokenizer, decision_development,
            device, max_length, args.batch,
        )
        transfer_predictions = infer_requests(
            model, tokenizer, transfer_development,
            device, max_length, args.batch,
        )
        korean_predictions = infer_requests(
            model, tokenizer, korean_development,
            device, max_length, args.batch,
        )
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    decision_clean, decision_clean_audit = common_clean_predictions(
        decision_predictions, comparison_plan, "decision",
    )
    transfer_clean, transfer_clean_audit = common_clean_predictions(
        transfer_predictions, comparison_plan, "transfer",
    )
    korean_clean, korean_clean_audit = common_clean_predictions(
        korean_predictions, comparison_plan, "korean",
    )
    report = {
        "version": 1,
        "status": "evaluated",
        "model": "baseline-separate-question",
        "comparison_plan_sha256": comparison_plan["plan_sha256"],
        "run": actual_run,
        "inference": {
            "device": device,
            "batch": args.batch,
            "max_length": max_length,
        },
        "temperature": {
            "value": temperature,
            "fit_manifest_sha256": file_sha256(
                Path(args.decision_suite) / "manifest.json"
            ),
            "fit_split_sha256": decision_manifest["files"][
                "calibration.jsonl"
            ]["sha256"],
            "objective": (
                "source-balanced soft-target cross-entropy after excluding "
                "baseline-training state overlap"
            ),
            "common_clean": calibration_clean_audit,
        },
        "decision_development": {
            "manifest_sha256": comparison_plan["suites"]["decision"][
                "manifest_sha256"
            ],
            "split_sha256": decision_manifest["files"][
                "development.jsonl"
            ]["sha256"],
            **development_report(
                decision_predictions, decision_clean,
                decision_clean_audit, temperature,
            ),
            "packing": separate_packing_report(decision_predictions),
        },
        "transfer_development": {
            "manifest_sha256": comparison_plan["suites"]["transfer"][
                "manifest_sha256"
            ],
            "split_sha256": transfer_manifest["files"][
                "development.jsonl"
            ]["sha256"],
            **development_report(
                transfer_predictions, transfer_clean,
                transfer_clean_audit, temperature,
            ),
            "packing": separate_packing_report(transfer_predictions),
        },
        "korean_development": {
            "manifest_sha256": comparison_plan["suites"]["korean"][
                "manifest_sha256"
            ],
            "split_sha256": korean_manifest["files"][
                "development.jsonl"
            ]["sha256"],
            **development_report(
                korean_predictions, korean_clean,
                korean_clean_audit, temperature,
            ),
            "packing": separate_packing_report(korean_predictions),
        },
        "locked_test_opened": False,
        "predictions": {
            "calibration": calibration_predictions,
            "decision_development": decision_predictions,
            "transfer_development": transfer_predictions,
            "korean_development": korean_predictions,
        },
    }
    report["report_sha256"] = canonical_report_sha256(report)
    atomic_json_save(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--comparison-plan", required=True)
    parser.add_argument("--decision-suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--korean-suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "temperature": report["temperature"]["value"],
        "decision_clean_raw": report["decision_development"][
            "common_clean"
        ]["raw"]["source_macro"],
        "transfer_raw": report["transfer_development"]["raw"][
            "source_macro"
        ],
        "korean_raw": report["korean_development"]["raw"][
            "source_macro"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
