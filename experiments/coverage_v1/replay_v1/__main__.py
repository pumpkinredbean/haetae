"""Command line interface for the reviewed T05 paired-replay protocol."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def _add_review_gate(parser: argparse.ArgumentParser, *, training: bool = False) -> None:
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument("--reviewed-protocol-sha256", required=True)
    parser.add_argument("--reviewed-manifest-sha256", required=True)
    parser.add_argument(
        "--allow-reviewed-training" if training else "--allow-reviewed-inference",
        action="store_true",
        required=True,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m experiments.coverage_v1.replay_v1")
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--paths", required=True)
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--out", required=True)
    verify = commands.add_parser("verify-plan")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--paths", required=True)
    verify.add_argument("--registry", required=True)
    package = commands.add_parser("package-review")
    package.add_argument("--plan", required=True)
    package.add_argument("--paths", required=True)
    package.add_argument("--registry", required=True)
    package.add_argument("--out", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--plan", required=True)
    initialize.add_argument("--paths", required=True)
    initialize.add_argument("--seed", type=int, choices=(17, 23), required=True)
    initialize.add_argument("--arm", choices=("control", "targeted"), required=True)
    initialize.add_argument("--run", required=True)
    initialize.add_argument("--pilot-gate")
    _add_review_gate(initialize)
    pair = commands.add_parser("verify-initial-pair")
    pair.add_argument("--plan", required=True)
    pair.add_argument("--seed", type=int, choices=(17, 23), required=True)
    pair.add_argument("--control-run", required=True)
    pair.add_argument("--targeted-run", required=True)
    pair.add_argument("--out", required=True)
    train = commands.add_parser("train")
    train.add_argument("--plan", required=True)
    train.add_argument("--paths", required=True)
    train.add_argument("--seed", type=int, choices=(17, 23), required=True)
    train.add_argument("--arm", choices=("control", "targeted"), required=True)
    train.add_argument("--run", required=True)
    train.add_argument("--initial-pair", required=True)
    train.add_argument("--pilot-gate")
    train.add_argument("--resume-current", action="store_true")
    _add_review_gate(train, training=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--plan", required=True)
    evaluate.add_argument("--paths", required=True)
    evaluate.add_argument("--seed", type=int, choices=(17, 23), required=True)
    evaluate.add_argument("--arm", choices=("control", "targeted"), required=True)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--pair-terminal-bindings", required=True)
    evaluate.add_argument("--out", required=True)
    _add_review_gate(evaluate)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--plan", required=True)
    analyze.add_argument("--seed", type=int, choices=(17, 23), required=True)
    analyze.add_argument("--control-report", required=True)
    analyze.add_argument("--targeted-report", required=True)
    analyze.add_argument("--out", required=True)
    result = commands.add_parser("verify-result")
    result.add_argument("--plan", required=True)
    result.add_argument("--seed", type=int, choices=(17, 23), required=True)
    result.add_argument("--result", required=True)
    result.add_argument("--control-report", required=True)
    result.add_argument("--targeted-report", required=True)
    return root


def _package_inputs(material: dict, registry: Path) -> tuple[dict, dict, dict]:
    paths = material["private_material"]["paths"]
    captured = material["private_material"]["captured"]
    groups = {
        "comparison_plan": "comparison/plan.json",
        "baseline_development_report": "comparison/baseline-development.json",
        "shared_development_report": "comparison/shared-development.json",
        "development_comparison": "comparison/development-comparison.json",
        "korean_suite_manifest": "korean/manifest.json",
        "korean_suite_development": "korean/development.jsonl",
        "mps_copy_result": "mps-copy/result.json",
        "shared_run_manifest": "shared/latest.json",
        "shared_run_spec": "shared/run.json",
        "shared_suite_manifest": "shared/suite-manifest.json",
        "shared_suite_train": "shared/train.jsonl",
        "shared_suite_calibration": "shared/calibration.jsonl",
        "shared_suite_development": "shared/development.jsonl",
        "shared_suite_rendered_state_audit": "shared/rendered-state-audit.json",
        "transfer_suite_manifest": "transfer/manifest.json",
        "transfer_suite_development": "transfer/development.jsonl",
        "t04_corrected_measured_archive": "t04/measured.zip",
        "t04_corrected_output_manifest": "t04/output-manifest.json",
        "t04_corrected_result": "t04/result.json",
        "t04_corrected_original_parity": "t04/original-parity.json",
        "t04_corrected_semantic_observations": "t04/semantic-logits.jsonl",
        "t04_original_plan_manifest": "t04/plan-manifest.json",
        "t04_plan/development.jsonl": "t04/development.jsonl",
        "t04_plan/exclusions.json": "t04/exclusions.json",
        "t04_plan/protocol.json": "t04/protocol.json",
        "t04_plan/semantic-variants.jsonl": "t04/semantic-variants.jsonl",
        "t04_plan/support-calibration.jsonl": "t04/support-calibration.jsonl",
        "t04_plan/support-fit.jsonl": "t04/support-fit.jsonl",
    }
    evidence = {member: captured[artifact].path for artifact, member in groups.items()}
    evidence["registry.json"] = registry
    metadata = {
        "tokenizer.json": captured["g79_tokenizer/tokenizer.json"].path,
        "tokenizer_config.json": captured["g79_tokenizer/tokenizer_config.json"].path,
        "backbone-config.json": captured["backbone_config.json"].path,
    }
    archive_map = {
        "schema_version": 1,
        "artifacts": {
            artifact: {"archive_member": f"evidence/{member}"}
            for artifact, member in sorted(groups.items())
        },
        "checkpoint": {
            "included": False,
            "descriptor": material["input_bindings"]["shared_checkpoint_g79"],
        },
        "owner_lock": {"identity_only": True},
        "output_root": {"included": False},
    }
    del paths
    return evidence, metadata, archive_map


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[3]
    if args.command == "freeze":
        from experiments.coverage_v1.replay_v1.freeze import freeze, materialize_freeze

        material = materialize_freeze(
            repository=repository, paths_file=args.paths, registry_file=args.registry,
        )
        material.pop("private_material")
        manifest = freeze(out=args.out, **material)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    elif args.command == "verify-plan":
        from experiments.coverage_v1.replay_v1.freeze import (
            materialize_freeze,
            verify_plan,
        )

        manifest = verify_plan(args.plan, expected_producer_commit=None)
        material = materialize_freeze(
            repository=repository, paths_file=args.paths, registry_file=args.registry,
        )
        with tempfile.TemporaryDirectory(prefix="haetae-t05-verify-") as temporary:
            replay = Path(temporary) / "freeze"
            private = material.pop("private_material")
            freeze_manifest = __import__(
                "experiments.coverage_v1.replay_v1.freeze", fromlist=["freeze"],
            ).freeze(out=replay, **material)
            material["private_material"] = private
            if freeze_manifest != manifest:
                raise SystemExit("freeze manifest differs from full reconstruction")
            for source in Path(args.plan).iterdir():
                if source.read_bytes() != (replay / source.name).read_bytes():
                    raise SystemExit(f"freeze byte replay differs: {source.name}")
        if manifest["code"] != __import__(
            "experiments.coverage_v1.replay_v1.freeze", fromlist=["freeze_source_identity"],
        ).freeze_source_identity(repository):
            raise SystemExit("freeze code identity differs from the current producer")
        if manifest["population_counts"]["original_requests"] != len(material["original_pool"]):
            raise SystemExit("freeze input reconstruction differs")
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    elif args.command == "package-review":
        from experiments.coverage_v1.replay_v1.freeze import materialize_freeze
        from experiments.coverage_v1.replay_v1.package import build_m4_package

        material = materialize_freeze(
            repository=repository, paths_file=args.paths, registry_file=args.registry,
        )
        evidence, metadata_files, archive_map = _package_inputs(
            material, Path(args.registry),
        )
        result = build_m4_package(
            repository=repository, plan=args.plan, out=args.out,
            evidence_files=evidence, model_metadata_files=metadata_files,
            archive_map=archive_map,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        from experiments.coverage_v1.replay_v1.train import dispatch_reviewed_command

        dispatch_reviewed_command(args, repository)


if __name__ == "__main__":
    main()
