import argparse
import copy
import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from transformers import get_cosine_schedule_with_warmup

import haetae.checkpoint as checkpoint_module
from haetae.checkpoint import (
    CheckpointError,
    CheckpointStore,
    RunDirectoryLock,
    TRAINING_CONFIG_FIELDS,
    checkpoint_payload,
    load_completed_checkpoint,
    optimizer_signature,
    restore_checkpoint,
    tokenizer_fingerprint,
    validate_checkpoint,
)
from haetae.train import permute_choice, train


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)


class TinyDecisionModel(TinyModel):
    def forward(self, input_ids, attention_mask, option_pos, group_ptr):
        scores = self.head(self.backbone(input_ids.float()))
        return [scores[index] for index in range(scores.shape[0])]


class TinyTokenizer:
    special_tokens_map = {"pad_token": "[PAD]"}
    model_max_length = 32
    padding_side = "right"
    truncation_side = "right"

    def get_vocab(self):
        return {"[PAD]": 0, "x": 1}

    def save_pretrained(self, output):
        return (str(Path(output) / "tokenizer.json"),)


def make_args(output, resume="none", steps=10):
    return argparse.Namespace(
        out=str(output), resume=resume, backbone="tiny", sources="fixture",
        per_source=1, eval_per_source=1, steps=steps, batch=1, microbatch=1,
        lr=1e-3,
        head_lr=2e-3, brier_w=0.5, max_len=32, seed=17,
        save_every=2,
    )


def make_training(args):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / args.steps)
    )
    return model, optimizer, scheduler


def make_entrypoint_training(args, model):
    decay = [model.backbone.weight]
    no_decay = [model.backbone.bias]
    optimizer = torch.optim.AdamW(
        [
            {"name": "backbone_decay", "params": decay,
             "lr": args.lr, "weight_decay": 0.01},
            {"name": "backbone_no_decay", "params": no_decay,
             "lr": args.lr, "weight_decay": 0.0},
            {"name": "head", "params": list(model.head.parameters()),
             "lr": args.head_lr, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.06 * args.steps), args.steps
    )
    return optimizer, scheduler


def make_spec(args, model, optimizer, device="cpu"):
    return {
        "config": {key: getattr(args, key) for key in TRAINING_CONFIG_FIELDS},
        "train_fingerprint": "train-digest",
        "validation_fingerprint": "validation-digest",
        "backend": device,
        "optimizer_class": (
            f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
        ),
        "optimizer_signature": optimizer_signature(model, optimizer),
        "tokenizer_fingerprint": "tokenizer-digest",
        "model_config_fingerprint": "model-digest",
        "code": {"sha256": "code-digest", "files": []},
        "software": {"python": "test", "torch": "test",
                     "transformers": "test", "datasets": "test"},
    }


def update_once(model, optimizer, scheduler):
    loss = model.head(model.backbone(torch.ones(1, 2))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def publish_fixture(store, payload, status):
    return store.publish(payload, status, lambda _: None)


def run_fixture_updates(model, optimizer, scheduler, records, data_state,
                        start_step, target_step):
    order = list(data_state["order"])
    cursor = data_state["cursor"]
    epoch = data_state["epoch"]
    trace = []
    step = start_step
    while step < target_step:
        if cursor >= len(order):
            epoch += 1
            order = list(range(len(records)))
            random.shuffle(order)
            cursor = 0
        indices = order[cursor:cursor + 2]
        cursor += len(indices)
        batch = [permute_choice(records[index], random) for index in indices]
        rng_draw = random.random()
        torch_draw = float(torch.rand(()))
        features = []
        for record in batch:
            option_value = sum(
                (position + 1) * (ord(option[-1]) - ord("a") + 1)
                for position, option in enumerate(record["options"])
            )
            features.append([float(record["row"]), float(option_value)])
        inputs = torch.tensor(features) + torch_draw
        loss = model.head(model.backbone(inputs)).square().mean() * (1.0 + rng_draw)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        trace.append({
            "step": step,
            "records": [record["row"] for record in batch],
            "options": [record["options"] for record in batch],
            "python_rng": rng_draw,
            "torch_rng": torch_draw,
            "loss": float(loss.detach()),
            "lrs": [group["lr"] for group in optimizer.param_groups],
        })
    return trace, {"epoch": epoch, "order": order, "cursor": cursor}


class ResumeTest(unittest.TestCase):
    def test_microbatch_accumulation_matches_full_batch_update(self):
        records = [
            {"row": index + 1, "state": str(index), "type": "choice",
             "instructions": "pick", "options": ["a", "b"],
             "label": index % 2, "soft": None, "source": "fixture",
             "parent": f"row-{index}"}
            for index in range(5)
        ]

        def fixture_collate(tokenizer, batch_records, max_len):
            values = torch.tensor([
                [float(record["row"]), 1.0] for record in batch_records
            ])
            return {
                "input_ids": values,
                "attention_mask": torch.ones_like(values),
                "option_pos": torch.empty(0, dtype=torch.long),
                "group_ptr": torch.arange(len(batch_records) + 1),
                "n_options": [2] * len(batch_records),
                "dropped_options": [False] * len(batch_records),
            }

        def run_once(output, microbatch):
            args = make_args(output, steps=2)
            args.per_source = 4
            args.eval_per_source = 1
            args.batch = 2
            args.microbatch = microbatch
            template = TinyDecisionModel()
            optimizer, _ = make_entrypoint_training(args, template)
            spec = make_spec(args, template, optimizer)
            patches = (
                mock.patch("haetae.train.AutoTokenizer.from_pretrained",
                           return_value=TinyTokenizer()),
                mock.patch("haetae.train.HaetaeModel",
                           side_effect=lambda _: TinyDecisionModel()),
                mock.patch.dict("haetae.train.LOADERS", {
                    "fixture": lambda split, limit: iter(copy.deepcopy(records))
                }),
                mock.patch("haetae.train.collate",
                           side_effect=fixture_collate),
                mock.patch("haetae.train.make_run_spec", return_value=spec),
                mock.patch("haetae.train.torch.backends.mps.is_available",
                           return_value=False),
                mock.patch("haetae.eval.evaluate"),
            )
            with (patches[0], patches[1], patches[2], patches[3], patches[4],
                  patches[5], patches[6] as evaluate_mock):
                train(args)
            self.assertEqual(
                evaluate_mock.call_args.kwargs["batch"], microbatch
            )
            payload, _, _ = load_completed_checkpoint(output)
            return payload

        with tempfile.TemporaryDirectory() as full_dir, \
                tempfile.TemporaryDirectory() as micro_dir:
            full = run_once(full_dir, 2)
            micro = run_once(micro_dir, 1)
            for section in ("backbone", "head"):
                self.assertEqual(full[section].keys(), micro[section].keys())
                for name in full[section]:
                    torch.testing.assert_close(
                        full[section][name], micro[section][name],
                        rtol=1e-6, atol=1e-7,
                    )
            self.assertEqual(full["scheduler"], micro["scheduler"])

    def test_tokenizer_fingerprint_covers_pipeline_not_only_vocab(self):
        class Backend:
            def __init__(self, normalizer):
                self.normalizer = normalizer

            def to_str(self):
                return json.dumps({
                    "model": {"vocab": {"x": 0}},
                    "normalizer": self.normalizer,
                })

        first = TinyTokenizer()
        first.backend_tokenizer = Backend("lowercase")
        second = TinyTokenizer()
        second.backend_tokenizer = Backend("identity")
        self.assertNotEqual(
            tokenizer_fingerprint(first), tokenizer_fingerprint(second)
        )

    def test_resume_after_rejected_epoch_tail_reaches_next_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary, resume="auto", steps=2)
            args.per_source = 2
            args.eval_per_source = 1
            args.save_every = 50
            records = [
                {"row": 1, "state": "validation", "type": "choice",
                 "instructions": "pick", "options": ["x", "y"],
                 "label": 0, "soft": None, "source": "fixture",
                 "parent": "validation"},
                {"row": 2, "state": "valid", "type": "choice",
                 "instructions": "pick", "options": ["x", "y"],
                 "label": 0, "soft": None, "source": "fixture",
                 "parent": "valid"},
                {"row": 3, "state": "reject", "type": "choice",
                 "instructions": "pick", "options": ["x", "y"],
                 "label": 1, "soft": None, "source": "fixture",
                 "parent": "reject"},
            ]
            model = TinyDecisionModel()
            optimizer, scheduler = make_entrypoint_training(args, model)
            loss = sum(parameter.sum() for parameter in model.parameters())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                {"epoch": 0, "order": [0, 1], "cursor": 1},
                "interrupted",
            ), "interrupted")

            def fixture_collate(tokenizer, batch_records, max_len):
                dropped = [record["state"] == "reject" for record in batch_records]
                if all(dropped):
                    return {"dropped_options": dropped, "n_options": [0]}
                values = torch.tensor([
                    [float(record["row"]), 1.0] for record in batch_records
                ])
                return {
                    "input_ids": values,
                    "attention_mask": torch.ones_like(values),
                    "option_pos": torch.tensor([0, 1]),
                    "group_ptr": torch.tensor([0, 2]),
                    "n_options": [2],
                    "dropped_options": dropped,
                }

            patches = (
                mock.patch("haetae.train.AutoTokenizer.from_pretrained",
                           return_value=TinyTokenizer()),
                mock.patch("haetae.train.HaetaeModel",
                           side_effect=lambda _: TinyDecisionModel()),
                mock.patch.dict("haetae.train.LOADERS",
                                {"fixture": lambda split, limit: iter(records)}),
                mock.patch("haetae.train.collate", side_effect=fixture_collate),
                mock.patch("haetae.train.make_run_spec", return_value=spec),
                mock.patch("haetae.train.torch.backends.mps.is_available",
                           return_value=False),
                mock.patch("haetae.eval.evaluate"),
            )
            with (patches[0], patches[1], patches[2], patches[3], patches[4],
                  patches[5], patches[6]):
                train(args)

            progress = json.loads(Path(temporary, "progress.json").read_text())
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["step"], 2)
            self.assertGreaterEqual(progress["epoch"], 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_native_mps_restores_moments_and_keeps_step_counter_on_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            torch.manual_seed(31)
            model = TinyModel().to("mps")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: max(0.0, 1.0 - step / args.steps)
            )
            spec = make_spec(args, model, optimizer, "mps")
            store = CheckpointStore(temporary)
            run = store.start(spec)
            inputs = torch.ones(1, 2, device="mps")

            loss = model.head(model.backbone(inputs)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                {"epoch": 0, "order": [0], "cursor": 1}, "interrupted",
            ), "interrupted")

            second_loss = model.head(model.backbone(inputs)).sum()
            second_loss.backward()
            optimizer.step()
            scheduler.step()
            expected = copy.deepcopy(model.state_dict())

            resumed_model = TinyModel().to("mps")
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters(), lr=args.lr
            )
            resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
                resumed_optimizer,
                lambda step: max(0.0, 1.0 - step / args.steps),
            )
            resumed_store = CheckpointStore(temporary)
            resumed_run = resumed_store.open(spec)
            payload, _, _ = resumed_store.load(
                lambda candidate: validate_checkpoint(
                    candidate, resumed_run, args, resumed_model,
                    resumed_optimizer, resumed_scheduler, 1, "mps"
                )
            )
            restore_checkpoint(
                payload, resumed_model, resumed_optimizer, resumed_scheduler, "mps"
            )
            for parameter in resumed_model.parameters():
                state = resumed_optimizer.state[parameter]
                self.assertEqual(state["exp_avg"].device.type, "mps")
                self.assertEqual(state["exp_avg_sq"].device.type, "mps")
                self.assertEqual(state["step"].device.type, "cpu")

            resumed_loss = resumed_model.head(resumed_model.backbone(inputs)).sum()
            resumed_loss.backward()
            resumed_optimizer.step()
            resumed_scheduler.step()
            for key, value in expected.items():
                self.assertTrue(torch.equal(resumed_model.state_dict()[key], value))

    def test_interrupted_run_matches_uninterrupted_stream_and_parameters(self):
        records = [
            {"row": index + 1, "state": str(index), "type": "choice",
             "instructions": "pick", "options": ["a", "b", "c"],
             "label": index % 3, "soft": None}
            for index in range(7)
        ]
        initial_order = list(range(len(records)))

        random.seed(73)
        torch.manual_seed(73)
        baseline_model, baseline_optimizer, baseline_scheduler = make_training(
            make_args("unused")
        )
        random.shuffle(initial_order)
        baseline_trace, baseline_data = run_fixture_updates(
            baseline_model, baseline_optimizer, baseline_scheduler, records,
            {"epoch": 0, "order": initial_order, "cursor": 0}, 0, 9,
        )
        baseline_model_state = copy.deepcopy(baseline_model.state_dict())
        baseline_optimizer_state = copy.deepcopy(baseline_optimizer.state_dict())

        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            random.seed(73)
            torch.manual_seed(73)
            split_model, split_optimizer, split_scheduler = make_training(args)
            split_order = list(range(len(records)))
            random.shuffle(split_order)
            before, split_data = run_fixture_updates(
                split_model, split_optimizer, split_scheduler, records,
                {"epoch": 0, "order": split_order, "cursor": 0}, 0, 4,
            )
            spec = make_spec(args, split_model, split_optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            publish_fixture(store, checkpoint_payload(
                run, args, split_model, split_optimizer, split_scheduler,
                4, 0, split_data, "interrupted",
            ), "interrupted")

            resumed_model, resumed_optimizer, resumed_scheduler = make_training(args)
            resumed_store = CheckpointStore(temporary)
            resumed_run = resumed_store.open(spec)
            payload, role, failures = resumed_store.load(
                lambda candidate: validate_checkpoint(
                    candidate, resumed_run, args, resumed_model,
                    resumed_optimizer, resumed_scheduler, len(records), "cpu"
                )
            )
            step, _, restored_data = restore_checkpoint(
                payload, resumed_model, resumed_optimizer, resumed_scheduler, "cpu"
            )
            after, resumed_data = run_fixture_updates(
                resumed_model, resumed_optimizer, resumed_scheduler, records,
                restored_data, step, 9,
            )

            self.assertEqual(role, "current")
            self.assertEqual(failures, [])
            self.assertEqual(before + after, baseline_trace)
            self.assertEqual(resumed_data, baseline_data)
            for key, expected in baseline_model_state.items():
                self.assertTrue(torch.equal(resumed_model.state_dict()[key], expected))
            self.assertEqual(
                resumed_optimizer.state_dict()["param_groups"],
                baseline_optimizer_state["param_groups"],
            )
            for parameter_id, expected_state in baseline_optimizer_state["state"].items():
                actual_state = resumed_optimizer.state_dict()["state"][parameter_id]
                for key, expected in expected_state.items():
                    self.assertTrue(torch.equal(actual_state[key], expected))

    def test_round_trip_restores_training_rng_and_data_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)

            update_once(model, optimizer, scheduler)
            expected_weights = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            data_state = {"epoch": 3, "order": [2, 0, 1], "cursor": 2}
            payload = checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 4,
                data_state, "running",
            )
            publish_fixture(store, payload, "running")
            expected_python_random = random.random()
            expected_torch_random = torch.rand(3)

            for parameter in model.parameters():
                parameter.data.zero_()
            random.seed(999)
            torch.manual_seed(999)

            resumed_store = CheckpointStore(temporary)
            resumed_run = resumed_store.open(spec)
            loaded, role, failures = resumed_store.load(
                lambda candidate: validate_checkpoint(
                    candidate, resumed_run, args, model, optimizer,
                    scheduler, 3, "cpu"
                )
            )
            step, skipped, restored_data = restore_checkpoint(
                loaded, model, optimizer, scheduler, "cpu"
            )

            self.assertEqual(role, "current")
            self.assertEqual(failures, [])
            self.assertEqual((step, skipped), (1, 4))
            self.assertEqual(restored_data, data_state)
            self.assertEqual(random.random(), expected_python_random)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch_random))
            self.assertTrue(all(
                torch.equal(model.state_dict()[key], value)
                for key, value in expected_weights.items()
            ))
            self.assertEqual(scheduler.last_epoch, step)

    def test_corrupt_current_generation_falls_back_to_previous(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            data_state = {"epoch": 0, "order": [0, 1], "cursor": 1}

            update_once(model, optimizer, scheduler)
            first = publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                data_state, "running",
            ), "running")
            update_once(model, optimizer, scheduler)
            data_state["cursor"] = 2
            second = publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 2, 0,
                data_state, "running",
            ), "running")

            current_path = Path(temporary, second["path"])
            with current_path.open("r+b") as handle:
                handle.write(b"corrupt")

            new_model, new_optimizer, new_scheduler = make_training(args)
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            loaded, role, failures = resumed.load(
                lambda candidate: validate_checkpoint(
                    candidate, resumed_run, args, new_model, new_optimizer,
                    new_scheduler, 2, "cpu"
                )
            )
            self.assertEqual(role, "previous")
            self.assertEqual(loaded["step"], 1)
            self.assertEqual(resumed.active_descriptor["path"], first["path"])
            self.assertIn("digest mismatch", failures[0])
            events = json.loads(Path(temporary, "recovery_events.json").read_text())
            self.assertEqual(events[-1]["rejected_generation"], second["generation"])
            self.assertEqual(events[-1]["selected_generation"], first["generation"])

    def test_recovery_ignores_derived_progress_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            data_state = {"epoch": 0, "order": [0, 1], "cursor": 1}
            update_once(model, optimizer, scheduler)
            publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                data_state, "running",
            ), "running")
            update_once(model, optimizer, scheduler)
            data_state["cursor"] = 2
            current = publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 2, 0,
                data_state, "running",
            ), "running")
            Path(temporary, current["path"]).unlink()

            resumed_model, resumed_optimizer, resumed_scheduler = make_training(args)
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            real_atomic_save = checkpoint_module.atomic_json_save

            def fail_progress(payload, path):
                if Path(path).name == "progress.json":
                    raise OSError("injected progress failure")
                return real_atomic_save(payload, path)

            with mock.patch("haetae.checkpoint.atomic_json_save",
                            side_effect=fail_progress):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded, role, _ = resumed.load(
                        lambda candidate: validate_checkpoint(
                            candidate, resumed_run, args, resumed_model,
                            resumed_optimizer, resumed_scheduler, 2, "cpu"
                        )
                    )
            self.assertTrue(any(
                "recovery succeeded" in str(item.message) for item in caught
            ))
            self.assertEqual(role, "previous")
            self.assertEqual(loaded["step"], 1)

    def test_semantically_invalid_current_generation_does_not_roll_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            data_state = {"epoch": 0, "order": [0, 1], "cursor": 1}

            update_once(model, optimizer, scheduler)
            publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                data_state, "running",
            ), "running")
            update_once(model, optimizer, scheduler)
            data_state["cursor"] = 2
            invalid = checkpoint_payload(
                run, args, model, optimizer, scheduler, 2, 0,
                data_state, "running",
            )
            first_head_key = next(iter(invalid["head"]))
            invalid["head"][first_head_key] = invalid["head"][first_head_key].clone()
            invalid["head"][first_head_key].reshape(-1)[0] = float("nan")
            publish_fixture(store, invalid, "running")

            new_model, new_optimizer, new_scheduler = make_training(args)
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            with self.assertRaisesRegex(CheckpointError, "nonfinite tensor"):
                resumed.load(
                    lambda candidate: validate_checkpoint(
                        candidate, resumed_run, args, new_model, new_optimizer,
                        new_scheduler, 2, "cpu"
                    )
                )
            self.assertFalse(Path(temporary, "recovery_events.json").exists())

    def test_malformed_descriptor_does_not_roll_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            data_state = {"epoch": 0, "order": [0, 1], "cursor": 1}
            update_once(model, optimizer, scheduler)
            publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                data_state, "running",
            ), "running")
            update_once(model, optimizer, scheduler)
            data_state["cursor"] = 2
            publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 2, 0,
                data_state, "running",
            ), "running")
            manifest_path = Path(temporary, "latest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["current"]["size"] = "not-an-integer"
            manifest_path.write_text(json.dumps(manifest))

            new_model, new_optimizer, new_scheduler = make_training(args)
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            with self.assertRaisesRegex(CheckpointError, "size is invalid"):
                resumed.load(
                    lambda candidate: validate_checkpoint(
                        candidate, resumed_run, args, new_model, new_optimizer,
                        new_scheduler, 2, "cpu"
                    )
                )
            self.assertFalse(Path(temporary, "recovery_events.json").exists())

    def test_completed_loader_uses_run_target_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            update_once(model, optimizer, scheduler)
            payload = checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                {"epoch": 0, "order": [0], "cursor": 1}, "completed",
            )
            payload["config"] = dict(payload["config"])
            payload["config"]["steps"] = 1
            publish_fixture(store, payload, "completed")
            with self.assertRaisesRegex(CheckpointError, "configuration differs"):
                load_completed_checkpoint(temporary)

    def test_manifest_failure_keeps_prior_generation_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            data_state = {"epoch": 0, "order": [0, 1], "cursor": 1}

            update_once(model, optimizer, scheduler)
            first = publish_fixture(store, checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                data_state, "running",
            ), "running")
            update_once(model, optimizer, scheduler)
            data_state["cursor"] = 2
            second_payload = checkpoint_payload(
                run, args, model, optimizer, scheduler, 2, 0,
                data_state, "running",
            )
            real_atomic_save = checkpoint_module.atomic_json_save

            def fail_manifest(payload, path):
                if Path(path).name == "latest.json":
                    raise OSError("injected manifest failure")
                return real_atomic_save(payload, path)

            with mock.patch("haetae.checkpoint.atomic_json_save",
                            side_effect=fail_manifest):
                with self.assertRaisesRegex(OSError, "injected manifest"):
                    publish_fixture(store, second_payload, "running")

            manifest = json.loads(Path(temporary, "latest.json").read_text())
            self.assertEqual(manifest["current"]["path"], first["path"])
            self.assertEqual(len(list(Path(temporary).glob("checkpoint-g*.pt"))), 2)
            resumed_model, resumed_optimizer, resumed_scheduler = make_training(args)
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            loaded, role, failures = resumed.load(
                lambda candidate: validate_checkpoint(
                    candidate, resumed_run, args, resumed_model,
                    resumed_optimizer, resumed_scheduler, 2, "cpu"
                )
            )
            self.assertEqual(role, "current")
            self.assertEqual(failures, [])
            self.assertEqual(loaded["step"], 1)

    def test_invalid_rng_is_rejected_before_live_state_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            update_once(model, optimizer, scheduler)
            payload = checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                {"epoch": 0, "order": [0, 1], "cursor": 1}, "running",
            )
            payload["rng"]["python"] = ("invalid",)
            publish_fixture(store, payload, "running")

            target_model, target_optimizer, target_scheduler = make_training(args)
            before = copy.deepcopy(target_model.state_dict())
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            with self.assertRaisesRegex(CheckpointError, "Python RNG"):
                resumed.load(
                    lambda candidate: validate_checkpoint(
                        candidate, resumed_run, args, target_model,
                        target_optimizer, target_scheduler, 2, "cpu"
                    )
                )
            for key, value in before.items():
                self.assertTrue(torch.equal(target_model.state_dict()[key], value))
            self.assertEqual(target_optimizer.state, {})

    def test_invalid_scheduler_is_rejected_before_live_state_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            update_once(model, optimizer, scheduler)
            payload = checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                {"epoch": 0, "order": [0, 1], "cursor": 1}, "running",
            )
            payload["scheduler"]["lr_lambdas"] = ["invalid"]
            publish_fixture(store, payload, "running")

            target_model, target_optimizer, target_scheduler = make_training(args)
            before = copy.deepcopy(target_model.state_dict())
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            with self.assertRaisesRegex(CheckpointError, "function state"):
                resumed.load(
                    lambda candidate: validate_checkpoint(
                        candidate, resumed_run, args, target_model,
                        target_optimizer, target_scheduler, 2, "cpu"
                    )
                )
            for key, value in before.items():
                self.assertTrue(torch.equal(target_model.state_dict()[key], value))
            self.assertEqual(target_optimizer.state, {})

    def test_boolean_adam_counter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary)
            model, optimizer, scheduler = make_training(args)
            spec = make_spec(args, model, optimizer)
            store = CheckpointStore(temporary)
            run = store.start(spec)
            update_once(model, optimizer, scheduler)
            payload = checkpoint_payload(
                run, args, model, optimizer, scheduler, 1, 0,
                {"epoch": 0, "order": [0, 1], "cursor": 1}, "running",
            )
            first_state = next(iter(payload["optimizer"]["state"].values()))
            first_state["step"] = torch.tensor(True)
            publish_fixture(store, payload, "running")

            target_model, target_optimizer, target_scheduler = make_training(args)
            resumed = CheckpointStore(temporary)
            resumed_run = resumed.open(spec)
            with self.assertRaisesRegex(CheckpointError, "step counter"):
                resumed.load(
                    lambda candidate: validate_checkpoint(
                        candidate, resumed_run, args, target_model,
                        target_optimizer, target_scheduler, 2, "cpu"
                    )
                )

    def test_fresh_run_rejects_any_stale_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "calibration.json").write_text("{}")
            args = make_args(temporary)
            model, optimizer, _ = make_training(args)
            store = CheckpointStore(temporary)
            with self.assertRaisesRegex(CheckpointError, "populated"):
                store.start(make_spec(args, model, optimizer))

    def test_run_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            with RunDirectoryLock(temporary):
                with self.assertRaisesRegex(CheckpointError, "another training"):
                    with RunDirectoryLock(temporary):
                        self.fail("second lock unexpectedly acquired")

    def test_no_progress_is_failed_and_never_published_as_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(temporary, steps=2)
            records = [
                {"state": "a", "type": "choice", "instructions": "pick",
                 "options": ["x", "y"], "label": 0, "soft": None,
                 "source": "fixture", "parent": "a"},
                {"state": "b", "type": "choice", "instructions": "pick",
                 "options": ["x", "y"], "label": 1, "soft": None,
                 "source": "fixture", "parent": "b"},
            ]
            temp_model, temp_optimizer, _ = make_training(args)
            fixed_spec = make_spec(args, temp_model, temp_optimizer)
            dropped_batch = {
                "dropped_options": [True],
                "n_options": [0],
            }
            patches = (
                mock.patch("haetae.train.AutoTokenizer.from_pretrained",
                           return_value=TinyTokenizer()),
                mock.patch("haetae.train.HaetaeModel", side_effect=lambda _: TinyModel()),
                mock.patch.dict("haetae.train.LOADERS",
                                {"fixture": lambda split, limit: iter(records)}),
                mock.patch("haetae.train.collate", return_value=dropped_batch),
                mock.patch("haetae.train.make_run_spec", return_value=fixed_spec),
                mock.patch("haetae.train.torch.backends.mps.is_available",
                           return_value=False),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaisesRegex(CheckpointError, "no trainable records"):
                    train(args)

            manifest = json.loads(Path(temporary, "latest.json").read_text())
            progress = json.loads(Path(temporary, "progress.json").read_text())
            self.assertEqual(manifest["status"], "failed_no_progress")
            self.assertEqual(progress["status"], "failed_no_progress")
            self.assertEqual(progress["step"], 0)
            self.assertFalse(Path(temporary, "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
