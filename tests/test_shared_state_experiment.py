import unittest

import torch
from transformers import ModernBertConfig, ModernBertModel

from experiments.shared_state import (
    DELIMITER_TOKENS,
    SharedStateDecisionModel,
    branch_attention_masks,
    encode_request,
)


class TinyTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    unk_token_id = 3

    def __init__(self):
        self.vocab = {
            token: index + 4 for index, token in enumerate(DELIMITER_TOKENS)
        }
        self.next_id = 20

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[word])
        return {"input_ids": ids}


def question(instructions, options, label=0):
    return {
        "instructions": instructions,
        "options": options,
        "label": label,
    }


class SharedStateExperimentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = ModernBertConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
            layer_types=["full_attention", "sliding_attention"],
            local_attention=8,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
        )
        config._attn_implementation = "eager"
        self.tokenizer = TinyTokenizer()
        self.model = SharedStateDecisionModel(
            ModernBertModel(config), pad_token_id=0, pointer_size=16,
        ).eval()
        self.delimiters = {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in DELIMITER_TOKENS
        }

    def encode(self, questions):
        return encode_request(
            self.tokenizer,
            "shared state evidence",
            questions,
            max_length=96,
            delimiters=self.delimiters,
        )

    def test_packed_questions_match_separate_execution(self):
        first = question("first decision", ["alpha", "beta"], 0)
        second = question("second decision", ["gamma", "delta", "epsilon"], 2)
        packed = self.encode([first, second])
        solo_first = self.encode([first])
        solo_second = self.encode([second])
        with torch.no_grad():
            packed_logits = self.model([packed])[0]
            first_logits = self.model([solo_first])[0][0]
            second_logits = self.model([solo_second])[0][0]
        torch.testing.assert_close(
            packed_logits[0], first_logits, rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            packed_logits[1], second_logits, rtol=1e-5, atol=1e-6,
        )

    def test_sibling_content_cannot_change_another_question(self):
        second = question("stable decision", ["gamma", "delta"], 1)
        original = self.encode([
            question("first decision", ["alpha", "beta"], 0), second,
        ])
        changed = self.encode([
            question("other wording", ["theta", "omega"], 0), second,
        ])
        with torch.no_grad():
            original_logits = self.model([original])[0][1]
            changed_logits = self.model([changed])[0][1]
        torch.testing.assert_close(
            original_logits, changed_logits, rtol=0.0, atol=1e-6,
        )

    def test_mask_blocks_siblings_and_state_feedback(self):
        encoded = self.encode([
            question("first", ["a", "b"]),
            question("second", ["c", "d"]),
        ])
        masks = branch_attention_masks([encoded], "cpu", torch.float32, 8)
        allowed = masks["full_attention"][0, 0] == 0
        state_index = encoded.segments.index(0)
        first_index = encoded.segments.index(1)
        second_index = encoded.segments.index(2)
        self.assertTrue(allowed[first_index, state_index])
        self.assertFalse(allowed[first_index, second_index])
        self.assertFalse(allowed[state_index, first_index])

    def test_sliding_mask_uses_modernbert_half_window(self):
        encoded = self.encode([question("first", ["a", "b"])])
        radius = int(self.model.backbone.config.sliding_window)
        self.assertEqual(radius, 4)
        mask = branch_attention_masks(
            [encoded], "cpu", torch.float32, radius,
        )["sliding_attention"][0, 0]
        branch = [
            index for index, segment in enumerate(encoded.segments)
            if segment == 1
        ]
        query, boundary, outside = branch[5], branch[1], branch[0]
        self.assertEqual(
            encoded.position_ids[query] - encoded.position_ids[boundary], 4,
        )
        self.assertEqual(
            encoded.position_ids[query] - encoded.position_ids[outside], 5,
        )
        self.assertEqual(float(mask[query, boundary]), 0.0)
        self.assertLess(float(mask[query, outside]), 0.0)

    def test_state_is_stored_once_and_fixed_content_is_never_truncated(self):
        encoded = encode_request(
            self.tokenizer,
            " ".join(f"state{i}" for i in range(100)),
            [
                question("first", ["alpha", "beta"]),
                question("second", ["gamma", "delta"]),
            ],
            max_length=40,
            delimiters=self.delimiters,
        )
        self.assertLess(encoded.retained_state_tokens, encoded.state_tokens)
        self.assertEqual(len(encoded.option_indices), 2)
        self.assertEqual([len(indexes) for indexes in encoded.option_indices], [2, 2])


if __name__ == "__main__":
    unittest.main()
