import unittest
from unittest import mock

import torch

from haetae.certify import collect_logits


class TinyLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.grad_enabled = None

    def forward(self, input_ids, attention_mask, option_pos, group_ptr):
        self.grad_enabled = torch.is_grad_enabled()
        return [self.weight * torch.tensor([1.0, -1.0])]


class CertificationTest(unittest.TestCase):
    def test_logit_collection_does_not_retain_graphs(self):
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "option_pos": torch.tensor([0, 1]),
            "group_ptr": torch.tensor([0, 2]),
            "dropped_options": [False],
        }
        model = TinyLogitModel()
        with mock.patch("haetae.model.collate", return_value=batch):
            logits = collect_logits(model, object(), [{}], "cpu", 1536, 1)
        self.assertFalse(model.grad_enabled)
        self.assertFalse(logits[0][1].requires_grad)
        self.assertIsNone(logits[0][1].grad_fn)


if __name__ == "__main__":
    unittest.main()
