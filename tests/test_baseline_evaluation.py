import unittest

import torch

from experiments.evaluate_baseline import infer_requests, packed_question_length


class TinyTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2

    def __init__(self):
        self.ids = {}

    def __call__(self, text, add_special_tokens=False):
        values = []
        for token in text.split():
            if token not in self.ids:
                self.ids[token] = len(self.ids) + 3
            values.append(self.ids[token])
        return {"input_ids": values}


class FakeBaseline(torch.nn.Module):
    def forward(self, input_ids, attention_mask, option_pos, group_ptr):
        return [
            torch.arange(
                int(group_ptr[index + 1] - group_ptr[index]),
                device=input_ids.device,
                dtype=torch.float32,
            )
            for index in range(len(group_ptr) - 1)
        ]


def request():
    return {
        "state": "shared evidence",
        "questions": [
            {
                "id": "first",
                "type": "choice",
                "instructions": "choose one",
                "options": ["alpha", "beta"],
                "option_keys": ["a", "b"],
                "label": 1,
                "soft": None,
                "source": "fixture",
            },
            {
                "id": "second",
                "type": "noul",
                "instructions": "is true",
                "options": ["yes", "no"],
                "option_keys": ["true", "false"],
                "label": 0,
                "soft": None,
                "source": "fixture",
            },
        ],
        "meta": {"id": "request-1", "group_id": "group-1"},
    }


class BaselineEvaluationTest(unittest.TestCase):
    def test_inference_keeps_request_identity_for_separate_questions(self):
        tokenizer = TinyTokenizer()
        predictions = infer_requests(
            FakeBaseline(), tokenizer, [request()], "cpu", 64, 2,
        )
        self.assertEqual(len(predictions), 2)
        self.assertEqual(
            [item["question_id"] for item in predictions],
            ["first", "second"],
        )
        self.assertEqual(
            {item["state_sha256"] for item in predictions},
            {predictions[0]["state_sha256"]},
        )
        self.assertEqual(predictions[0]["logits"], [0.0, 1.0])
        self.assertTrue(all(
            item["questions_in_request"] == 2 for item in predictions
        ))

    def test_state_truncation_is_rejected(self):
        tokenizer = TinyTokenizer()
        record = {
            "state": "one two three four five six",
            "instructions": "choose",
            "options": ["a", "b"],
        }
        with self.assertRaisesRegex(ValueError, "truncate"):
            packed_question_length(tokenizer, record, 7)


if __name__ == "__main__":
    unittest.main()
