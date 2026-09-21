import json
import os
import py_compile
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.comparison_protocol import canonical_sha256
from experiments.evaluate_veji import (
    EvaluationDirectoryLock,
    atomic_json_save_no_clobber,
    canonical_device,
    import_veji,
    infer_requests,
    load_protocol,
    stage_digest,
    validate_destinations,
    validate_prediction_numbers,
    validate_runtime,
    veji_question,
)


class FakeVEJI:
    def __init__(self):
        self.cfg = SimpleNamespace(max_context_chars=1000)
        self.compiled = []
        self.questions = []

    def compile_state(self, state):
        self.compiled.append(state)
        return SimpleNamespace(state=state, chunks=[state])

    def forward_question(self, compiled, question):
        self.questions.append((compiled.state, question))
        return {"logits": torch.tensor([0.25, -0.5])}


def request():
    return {
        "state": "shared state",
        "meta": {"id": "request-1", "group_id": "parent-1"},
        "questions": [
            {
                "id": "q1",
                "type": "choice",
                "instructions": "choose",
                "options": ["a", "b"],
                "option_keys": ["a", "b"],
                "label": 0,
                "soft": None,
                "source": "source-1",
            },
            {
                "id": "q2",
                "type": "noul",
                "instructions": "verify",
                "options": ["yes", "no"],
                "option_keys": ["true", "false"],
                "label": 1,
                "soft": None,
                "source": "source-1",
            },
        ],
    }


def test_request_compiles_state_once_and_omits_task_hint():
    model = FakeVEJI()
    predictions = infer_requests(model, [request()], progress_every=0)
    assert model.compiled == ["shared state"]
    assert len(predictions) == 2
    assert all("semantic_type" not in question for _, question in model.questions)
    assert model.questions[0][1] == {
        "id": "q1",
        "type": "choice",
        "instruction": "choose",
        "options": ["a", "b"],
    }
    assert predictions[1]["group_id"] == "parent-1"
    assert predictions[1]["logits"] == [0.25, -0.5]


def test_veji_question_does_not_mutate_normalized_question():
    original = request()["questions"][0]
    converted = veji_question(original)
    converted["options"][0] = "changed"
    assert original["options"][0] == "a"


def test_protocol_digest_is_checked(tmp_path: Path):
    value = {"version": 1, "status": "frozen", "locked_test_opened": False}
    value["protocol_sha256"] = canonical_sha256(value)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value))
    assert load_protocol(path)["status"] == "frozen"
    value["status"] = "changed"
    path.write_text(json.dumps(value))
    try:
        load_protocol(path)
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("changed protocol was accepted")


def test_stage_digest_excludes_only_its_signature():
    stage = {"status": "complete", "predictions": [{"logits": [1.0, 2.0]}]}
    digest = stage_digest(stage)
    stage["stage_sha256"] = digest
    assert stage_digest(stage) == digest
    stage["predictions"][0]["logits"][0] = 3.0
    assert stage_digest(stage) != digest


def runtime_protocol():
    return {
        "model": {
            "reported_trainable_parameters": 3,
            "files": {"head": {"bytes": 4, "sha256": "a" * 64}},
        },
        "encoder": {
            "parameters": 7,
            "files": {"weights": {"bytes": 8, "sha256": "b" * 64}},
        },
        "execution": {
            "software": {"python": "3.13.2"},
        },
    }


def runtime_identity():
    return {
        "device": "mps",
        "encoder_backend": "sentence-transformers",
        "head_parameters": 3,
        "encoder_parameters": 7,
        "total_parameters": 10,
        "model_files": {"head": {"bytes": 4, "sha256": "a" * 64}},
        "encoder_files": {"weights": {"bytes": 8, "sha256": "b" * 64}},
        "software": {"python": "3.13.2"},
    }


def test_runtime_validation_binds_files_counts_device_and_software():
    protocol = runtime_protocol()
    runtime = runtime_identity()
    validate_runtime(runtime, protocol, "mps")
    for field, value in (
        ("model_files", {}),
        ("total_parameters", 1),
        ("device", "cpu"),
        ("software", {"python": "other"}),
    ):
        changed = dict(runtime)
        changed[field] = value
        try:
            validate_runtime(changed, protocol, "mps")
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"changed runtime field was accepted: {field}")


def test_default_mps_device_has_one_canonical_identity():
    assert canonical_device("mps") == "mps"
    assert canonical_device("mps:0") == "mps"


def test_stage_logits_reject_booleans_and_numeric_strings():
    validate_prediction_numbers([{"logits": [0, 0.5]}], "valid")
    for invalid in (
        [{"logits": [True, 0.5]}],
        [{"logits": ["0.75", 0.5]}],
        None,
        ["not-an-object"],
    ):
        try:
            validate_prediction_numbers(invalid, "invalid")
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid predictions were accepted: {invalid}")


def test_import_executes_verified_bytes_instead_of_stale_bytecode(tmp_path: Path):
    source = tmp_path / "module.py"
    source.write_text("IMPORTED_ID = 'stale-B'\n")
    py_compile.compile(str(source), doraise=True)
    timestamp = source.stat().st_mtime
    source.write_text("IMPORTED_ID = 'bound-A'\n")
    os.utime(source, (timestamp, timestamp))
    payload = source.read_bytes()
    assert import_veji(source, payload).IMPORTED_ID == "bound-A"


def test_no_clobber_json_publication_preserves_complete_file(tmp_path: Path):
    output = tmp_path / "stage.json"
    atomic_json_save_no_clobber({"generation": 1}, output)
    try:
        atomic_json_save_no_clobber({"generation": 2}, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("complete stage was overwritten")
    assert json.loads(output.read_text()) == {"generation": 1}


def test_work_directory_lock_rejects_second_writer(tmp_path: Path):
    with EvaluationDirectoryLock(tmp_path):
        try:
            with EvaluationDirectoryLock(tmp_path):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("second evaluation writer acquired the lock")


def test_final_report_cannot_alias_a_stage(tmp_path: Path):
    work_directory = tmp_path / "stages"
    arguments = SimpleNamespace(
        out=work_directory / "calibration.json",
        protocol=tmp_path / "protocol.json",
        comparison_plan=tmp_path / "plan.json",
        haetae_report=tmp_path / "haetae.json",
        veji_training_pool=tmp_path / "train.jsonl",
        decision_suite=tmp_path / "decision",
        transfer_suite=tmp_path / "transfer",
        korean_suite=tmp_path / "korean",
        veji_model=tmp_path / "model",
        veji_encoder=tmp_path / "encoder",
    )
    protocol = {
        "model": {"files": {}},
        "encoder": {"files": {}},
    }
    try:
        validate_destinations(arguments, protocol, work_directory)
    except ValueError as error:
        assert "stage" in str(error)
    else:
        raise AssertionError("final report aliased a stage")
