"""Evaluate a completed shared-state run without opening locked test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from haetae.checkpoint import (
    atomic_json_save,
    load_completed_checkpoint,
    model_config_fingerprint,
    tokenizer_fingerprint,
)
from experiments.comparison_protocol import (
    common_clean_predictions,
    load_comparison_plan,
    request_state_sha256,
    training_clean_calibration_predictions,
    validate_suite_binding,
)
from experiments.kev_adapter import file_sha256, load_frozen_split
from experiments.shared_state import encode_request, load_shared_state_model
from experiments.train_shared import source_identity


def logits_to_list(logits: torch.Tensor) -> list[float]:
    return logits.detach().cpu().to(dtype=torch.float64).tolist()


def target_distribution(question: dict, option_count: int) -> torch.Tensor:
    if question["soft"] is None:
        target = torch.zeros(option_count, dtype=torch.float64)
        target[question["label"]] = 1.0
        return target
    target = torch.tensor(question["soft"], dtype=torch.float64)
    if (target.ndim != 1 or target.numel() != option_count
            or not torch.isfinite(target).all() or torch.any(target < 0)
            or abs(float(target.sum()) - 1.0) > 1e-6):
        raise ValueError("evaluation target is not a probability distribution")
    return target


def load_model(run_dir: str | Path, device: str):
    run_dir = Path(run_dir).resolve()
    payload, run, manifest = load_completed_checkpoint(run_dir)
    repository = Path(__file__).resolve().parents[1]
    current_code = source_identity(repository)
    if current_code["sha256"] != run["spec"]["training_code"]["sha256"]:
        raise ValueError("shared-state training code differs from the completed run")
    experiment = run["spec"]["experiment"]
    config = run["spec"]["config"]
    tokenizer = AutoTokenizer.from_pretrained(run_dir, local_files_only=True)
    model, delimiters = load_shared_state_model(
        config["backbone"],
        tokenizer,
        device,
        revision=experiment["backbone_revision"],
        pointer_size=experiment["pointer_size"],
        attention_implementation=experiment["attention_implementation"],
        local_files_only=True,
    )
    if tokenizer_fingerprint(tokenizer) != run["spec"]["tokenizer_fingerprint"]:
        raise ValueError("saved tokenizer differs from the completed run")
    if model_config_fingerprint(model) != run["spec"]["model_config_fingerprint"]:
        raise ValueError("model configuration differs from the completed run")
    model.backbone.load_state_dict(payload["backbone"])
    model.head.load_state_dict(payload["head"])
    model.eval()
    return model, tokenizer, delimiters, payload, run, manifest


def infer_requests(
    model,
    tokenizer,
    delimiters: dict[str, int],
    requests: list[dict],
    max_length: int,
    batch: int,
) -> list[dict]:
    if batch <= 0:
        raise ValueError("evaluation batch must be positive")
    predictions = []
    with torch.no_grad():
        for start in range(0, len(requests), batch):
            request_batch = requests[start:start + batch]
            encodings = [
                encode_request(
                    tokenizer,
                    request["state"],
                    request["questions"],
                    max_length,
                    delimiters,
                )
                for request in request_batch
            ]
            if any(encoding is None for encoding in encodings):
                raise ValueError("evaluation request exceeds the fixed content budget")
            if any(
                encoding.retained_state_tokens != encoding.state_tokens
                for encoding in encodings
            ):
                raise ValueError("evaluation request would truncate its state")
            outputs = model(encodings)
            for request, encoding, request_outputs in zip(
                    request_batch, encodings, outputs):
                if len(request_outputs) != len(request["questions"]):
                    raise ValueError("model question count differs from request")
                for question, logits in zip(
                        request["questions"], request_outputs):
                    predictions.append({
                        "request_id": request["meta"].get("id"),
                        "group_id": request["meta"].get("group_id"),
                        "state_sha256": request_state_sha256(request),
                        "question_id": question["id"],
                        "source": question["source"],
                        "type": question["type"],
                        "options": question["options"],
                        "option_keys": question["option_keys"],
                        "label": question["label"],
                        "soft": question["soft"],
                        "packed_request_tokens": len(encoding.input_ids),
                        "questions_in_request": len(request["questions"]),
                        "logits": logits_to_list(logits),
                    })
    return predictions


def fit_source_balanced_temperature(predictions: list[dict]) -> float:
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["source"]].append(prediction)
    if not grouped:
        raise ValueError("temperature fitting needs predictions")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=150,
        line_search_fn="strong_wolfe",
    )

    def objective():
        source_losses = []
        temperature = log_temperature.exp()
        for values in grouped.values():
            losses = []
            for prediction in values:
                logits = torch.tensor(prediction["logits"], dtype=torch.float64)
                target = target_distribution(prediction, logits.numel())
                losses.append(
                    -(target * F.log_softmax(logits / temperature, -1)).sum()
                )
            source_losses.append(torch.stack(losses).mean())
        return torch.stack(source_losses).mean()

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature fitting produced an invalid value")
    return temperature


def reliability(confidences: list[float], correct: list[float], bins: int = 15):
    details, expected_error = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            item for item, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (index == bins - 1 and confidence == 1.0)
        ]
        mean_confidence = (
            sum(confidences[item] for item in members) / len(members)
            if members else None
        )
        mean_correct = (
            sum(correct[item] for item in members) / len(members)
            if members else None
        )
        if members:
            expected_error += (
                len(members) / len(confidences)
                * abs(mean_confidence - mean_correct)
            )
        details.append({
            "lower": lower,
            "upper": upper,
            "n": len(members),
            "mean_confidence": mean_confidence,
            "mean_accuracy": mean_correct,
        })
    return {"ece": expected_error, "bins": details}


def macro_f1(predictions: list[dict], predicted_labels: list[int]) -> dict:
    ontologies = {
        tuple(prediction.get("option_keys", prediction["options"]))
        for prediction in predictions
    }
    if len(ontologies) != 1:
        return {"status": "not_applicable", "reason": "mixed option ontology"}
    labels = range(len(next(iter(ontologies))))
    values = []
    for label in labels:
        true_positive = sum(
            predicted == label and prediction["label"] == label
            for prediction, predicted in zip(predictions, predicted_labels)
        )
        false_positive = sum(
            predicted == label and prediction["label"] != label
            for prediction, predicted in zip(predictions, predicted_labels)
        )
        false_negative = sum(
            predicted != label and prediction["label"] == label
            for prediction, predicted in zip(predictions, predicted_labels)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return {"status": "ok", "value": sum(values) / len(values)}


def metric_summary(predictions: list[dict], temperature: float) -> dict:
    if not predictions:
        return {"status": "no_records", "n": 0}
    nll, brier, annotation_brier, accuracy, confidence = [], [], [], [], []
    predicted_labels = []
    ranked_probability_score, expected_index_error = [], []
    for prediction in predictions:
        logits = torch.tensor(prediction["logits"], dtype=torch.float64)
        target = target_distribution(prediction, logits.numel())
        log_probabilities = F.log_softmax(logits / temperature, -1)
        probabilities = log_probabilities.exp()
        predicted = int(probabilities.argmax())
        predicted_labels.append(predicted)
        nll.append(float(-(target * log_probabilities).sum()))
        brier.append(float(((probabilities - target) ** 2).sum()))
        annotation_brier.append(float(
            1.0 + probabilities.square().sum()
            - 2.0 * (probabilities * target).sum()
        ))
        accuracy.append(float(predicted == prediction["label"]))
        confidence.append(float(probabilities.max()))
        if prediction["type"] == "score":
            ranked_probability_score.append(float((
                probabilities.cumsum(0)[:-1]
                - target.cumsum(0)[:-1]
            ).square().mean()))
            levels = torch.arange(probabilities.numel(), dtype=torch.float64)
            expected_index_error.append(abs(
                float((probabilities * levels).sum()) - prediction["label"]
            ))
    result = {
        "status": "ok",
        "n": len(predictions),
        "temperature": temperature,
        "nll": sum(nll) / len(nll),
        "brier_histogram": sum(brier) / len(brier),
        "brier_annotation": sum(annotation_brier) / len(annotation_brier),
        "accuracy": sum(accuracy) / len(accuracy),
        "mean_confidence": sum(confidence) / len(confidence),
        "top_label_reliability": reliability(confidence, accuracy),
        "hard_label_macro_f1": macro_f1(predictions, predicted_labels),
    }
    if ranked_probability_score:
        result["ranked_probability_score"] = (
            sum(ranked_probability_score) / len(ranked_probability_score)
        )
        result["expected_index_mae"] = (
            sum(expected_index_error) / len(expected_index_error)
        )
    return result


def grouped_metrics(predictions: list[dict], temperature: float) -> dict:
    sources, primitives = defaultdict(list), defaultdict(list)
    for prediction in predictions:
        sources[prediction["source"]].append(prediction)
        primitives[prediction["type"]].append(prediction)
    source_metrics = {
        key: metric_summary(values, temperature)
        for key, values in sorted(sources.items())
    }
    return {
        "all": metric_summary(predictions, temperature),
        "source": source_metrics,
        "primitive": {
            key: metric_summary(values, temperature)
            for key, values in sorted(primitives.items())
        },
        "source_macro": {
            field: sum(metrics[field] for metrics in source_metrics.values())
            / len(source_metrics)
            for field in (
                "nll", "brier_histogram", "brier_annotation", "accuracy",
            )
        },
    }


def packing_report(
    tokenizer,
    delimiters: dict[str, int],
    requests: list[dict],
    max_length: int,
) -> dict:
    shared_tokens, separate_tokens = 0, 0
    multi_question_requests = 0
    for request in requests:
        packed = encode_request(
            tokenizer, request["state"], request["questions"],
            max_length, delimiters,
        )
        if packed is None:
            raise ValueError("packing report encountered a rejected request")
        if packed.retained_state_tokens != packed.state_tokens:
            raise ValueError("packing report encountered state truncation")
        shared_tokens += len(packed.input_ids)
        if len(request["questions"]) > 1:
            multi_question_requests += 1
        for question in request["questions"]:
            separate = encode_request(
                tokenizer, request["state"], [question],
                max_length, delimiters,
            )
            if separate is None:
                raise ValueError("separate packing encountered a rejected question")
            if separate.retained_state_tokens != separate.state_tokens:
                raise ValueError("separate packing encountered state truncation")
            separate_tokens += len(separate.input_ids)
    return {
        "requests": len(requests),
        "multi_question_requests": multi_question_requests,
        "shared_tokens": shared_tokens,
        "separate_tokens": separate_tokens,
        "token_ratio_shared_over_separate": shared_tokens / separate_tokens,
        "token_reduction": 1.0 - shared_tokens / separate_tokens,
        "scope": "un-padded input tokens; no latency claim",
    }


def development_report(
    predictions: list[dict],
    clean_predictions: list[dict],
    clean_audit: dict,
    temperature: float,
) -> dict:
    return {
        "raw": grouped_metrics(predictions, 1.0),
        "scaled": grouped_metrics(predictions, temperature),
        "common_clean": {
            **clean_audit,
            "raw": grouped_metrics(clean_predictions, 1.0),
            "scaled": grouped_metrics(clean_predictions, temperature),
        },
    }


def evaluate(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    comparison_plan = load_comparison_plan(args.comparison_plan)
    device = args.device or (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, tokenizer, delimiters, payload, run, manifest = load_model(
        args.run, device,
    )
    max_length = int(payload["config"]["max_len"])
    decision_manifest_sha256 = file_sha256(
        Path(args.decision_suite) / "manifest.json"
    )
    if decision_manifest_sha256 != run["spec"]["experiment"][
            "suite_manifest_sha256"]:
        raise ValueError("decision suite differs from the training-bound suite")
    validate_suite_binding(
        comparison_plan, "decision", args.decision_suite,
    )
    transfer_manifest_sha256 = file_sha256(
        Path(args.transfer_suite) / "manifest.json"
    )
    validate_suite_binding(
        comparison_plan, "transfer", args.transfer_suite,
    )
    korean_manifest_sha256 = file_sha256(
        Path(args.korean_suite) / "manifest.json"
    )
    validate_suite_binding(
        comparison_plan, "korean", args.korean_suite,
    )
    decision_calibration, decision_manifest = load_frozen_split(
        args.decision_suite, "calibration",
    )
    decision_development, decision_manifest_again = load_frozen_split(
        args.decision_suite, "development",
    )
    if decision_manifest_again != decision_manifest:
        raise ValueError("decision suite manifest changed during evaluation")
    transfer_development, transfer_manifest = load_frozen_split(
        args.transfer_suite, "development",
    )
    korean_development, korean_manifest = load_frozen_split(
        args.korean_suite, "development",
    )
    calibration_predictions = infer_requests(
        model, tokenizer, delimiters, decision_calibration,
        max_length, args.batch,
    )
    calibration_clean, calibration_clean_audit = (
        training_clean_calibration_predictions(
            calibration_predictions, comparison_plan,
        )
    )
    temperature = fit_source_balanced_temperature(calibration_clean)
    decision_predictions = infer_requests(
        model, tokenizer, delimiters, decision_development,
        max_length, args.batch,
    )
    transfer_predictions = infer_requests(
        model, tokenizer, delimiters, transfer_development,
        max_length, args.batch,
    )
    korean_predictions = infer_requests(
        model, tokenizer, delimiters, korean_development,
        max_length, args.batch,
    )
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
        "comparison_plan_sha256": comparison_plan["plan_sha256"],
        "run": {
            "run_id": run["run_id"],
            "spec_sha256": run["spec_sha256"],
            "generation": manifest["current"]["generation"],
            "checkpoint_sha256": manifest["current"]["sha256"],
        },
        "inference": {"device": device, "batch": args.batch},
        "temperature": {
            "value": temperature,
            "fit_suite": str(Path(args.decision_suite).resolve()),
            "fit_manifest_sha256": decision_manifest_sha256,
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
            "manifest_sha256": decision_manifest_sha256,
            "split_sha256": decision_manifest["files"][
                "development.jsonl"
            ]["sha256"],
            **development_report(
                decision_predictions, decision_clean,
                decision_clean_audit, temperature,
            ),
            "packing": packing_report(
                tokenizer, delimiters, decision_development, max_length,
            ),
        },
        "transfer_development": {
            "manifest_sha256": transfer_manifest_sha256,
            "split_sha256": transfer_manifest["files"][
                "development.jsonl"
            ]["sha256"],
            **development_report(
                transfer_predictions, transfer_clean,
                transfer_clean_audit, temperature,
            ),
            "packing": packing_report(
                tokenizer, delimiters, transfer_development, max_length,
            ),
        },
        "korean_development": {
            "manifest_sha256": korean_manifest_sha256,
            "split_sha256": korean_manifest["files"][
                "development.jsonl"
            ]["sha256"],
            **development_report(
                korean_predictions, korean_clean,
                korean_clean_audit, temperature,
            ),
            "packing": packing_report(
                tokenizer, delimiters, korean_development, max_length,
            ),
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


def canonical_report_sha256(report: dict) -> str:
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--comparison-plan", required=True)
    parser.add_argument("--decision-suite", required=True)
    parser.add_argument("--transfer-suite", required=True)
    parser.add_argument("--korean-suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "temperature": report["temperature"]["value"],
        "decision_raw": report["decision_development"]["raw"]["source_macro"],
        "transfer_raw": report["transfer_development"]["raw"]["source_macro"],
        "korean_raw": report["korean_development"]["raw"]["source_macro"],
    }, indent=2))


if __name__ == "__main__":
    main()
