import argparse
import random
import tempfile
import unittest
from pathlib import Path

import torch

from haetae.train import load_checkpoint, save_checkpoint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)


def make_args(out):
    return argparse.Namespace(
        out=str(out), resume="none", backbone="tiny", sources="fixture",
        per_source=3, eval_per_source=1, steps=10, batch=1, lr=1e-3,
        head_lr=2e-3, brier_w=0.5, max_len=32, seed=17,
        save_every=2,
    )


class ResumeTest(unittest.TestCase):
    def test_round_trip_restores_training_and_data_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp)
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _: 1.0)

            loss = model.head(model.backbone(torch.ones(1, 2))).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            expected_weights = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            data_state = {"epoch": 3, "order": [2, 0, 1], "cursor": 2}
            save_checkpoint(args, model, optimizer, scheduler, 1, 4,
                            data_state, "fixture-digest")
            expected_python_random = random.random()
            expected_torch_random = torch.rand(3)

            for parameter in model.parameters():
                parameter.data.zero_()
            random.seed(999)
            torch.manual_seed(999)
            args.resume = "auto"
            step, skipped, restored_data = load_checkpoint(
                args, model, optimizer, scheduler, "cpu",
                "fixture-digest", 3)

            self.assertEqual((step, skipped), (1, 4))
            self.assertEqual(restored_data, data_state)
            self.assertEqual(random.random(), expected_python_random)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch_random))
            self.assertTrue(all(
                torch.equal(model.state_dict()[key], value)
                for key, value in expected_weights.items()
            ))
            self.assertEqual(scheduler.last_epoch, step)
            self.assertTrue(Path(tmp, "progress.json").exists())

    def test_rejects_wrong_data_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp)
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _: 1.0)
            data_state = {"epoch": 0, "order": [0, 1, 2], "cursor": 0}
            save_checkpoint(args, model, optimizer, scheduler, 0, 0,
                            data_state, "right-digest")
            args.resume = "auto"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_checkpoint(args, model, optimizer, scheduler, "cpu",
                                "wrong-digest", 3)


if __name__ == "__main__":
    unittest.main()
