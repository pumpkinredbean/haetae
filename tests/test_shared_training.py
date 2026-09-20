import random
import unittest

import torch

from experiments.train_shared import (
    permute_choices,
    question_loss,
    request_batch_loss,
)


class SharedTrainingTest(unittest.TestCase):
    def test_choice_permutation_keeps_semantic_target(self):
        request = {
            "state": "evidence",
            "questions": [{
                "id": "choice",
                "type": "choice",
                "instructions": "choose",
                "options": ["alpha", "beta", "gamma"],
                "option_keys": ["a", "b", "c"],
                "label": 1,
                "soft": [0.1, 0.7, 0.2],
                "source": "fixture",
            }],
            "meta": {},
        }
        shuffled = permute_choices(request, random.Random(3))
        question = shuffled["questions"][0]
        self.assertEqual(question["options"][question["label"]], "beta")
        self.assertEqual(question["soft"][question["label"]], 0.7)
        self.assertEqual(
            request["questions"][0]["options"], ["alpha", "beta", "gamma"],
        )

    def test_request_loss_gives_each_request_equal_weight(self):
        first = {
            "state": "one",
            "questions": [{
                "type": "noul", "options": ["yes", "no"],
                "label": 0, "soft": None,
            }],
        }
        second = {
            "state": "two",
            "questions": [
                {
                    "type": "noul", "options": ["yes", "no"],
                    "label": 1, "soft": None,
                },
                {
                    "type": "score", "options": ["low", "high"],
                    "label": 1, "soft": None,
                },
            ],
        }
        outputs = [
            [torch.tensor([2.0, -1.0], requires_grad=True)],
            [
                torch.tensor([0.4, 0.6], requires_grad=True),
                torch.tensor([-0.5, 1.0], requires_grad=True),
            ],
        ]
        loss = request_batch_loss(outputs, [first, second], 0.5, 0.25)
        expected_first = question_loss(outputs[0][0], first["questions"][0], 0.5, 0.25)
        expected_second = torch.stack([
            question_loss(outputs[1][0], second["questions"][0], 0.5, 0.25),
            question_loss(outputs[1][1], second["questions"][1], 0.5, 0.25),
        ]).mean()
        torch.testing.assert_close(loss, (expected_first + expected_second) / 2)
        loss.backward()
        self.assertTrue(all(
            logits.grad is not None
            for request_outputs in outputs for logits in request_outputs
        ))


if __name__ == "__main__":
    unittest.main()
