"""Packed decision model.

One forward pass scores every option of one question. The sequence
is packed as

    [CLS] state [SEP] instructions [SEP] opt1 [SEP] opt2 [SEP] ...

and each option is read from the hidden state of the [SEP] token
that *precedes* it. The first two [SEP]s delimit state and question;
every later [SEP] is an option marker. A shared scalar head maps each
marker position to a logit; softmax over a question's own option
logits gives the probability distribution.

This mirrors what TypeSafe discloses about Jev: the state is read
once, all answers come back in parallel, and no text is generated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def pack_question(tokenizer, state, instructions, options, max_len=8192):
    """Tokenize one packed question.

    Returns input_ids, attention_mask, and the token index of each
    option's marker [SEP].
    """
    enc_state = tokenizer(state, add_special_tokens=False)["input_ids"]
    enc_instr = tokenizer(instructions, add_special_tokens=False)["input_ids"]
    enc_opts = [tokenizer(o, add_special_tokens=False)["input_ids"] for o in options]

    sep = tokenizer.sep_token_id
    cls = tokenizer.cls_token_id

    ids = [cls] + enc_state + [sep] + enc_instr
    option_pos = []
    for opt in enc_opts:
        ids.append(sep)
        option_pos.append(len(ids) - 1)
        ids += opt
    ids = ids[:max_len]
    option_pos = [p for p in option_pos if p < max_len]
    return ids, option_pos


class HaetaeModel(nn.Module):
    def __init__(self, backbone="answerdotai/ModernBERT-base", head_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, head_dim), nn.GELU(), nn.Linear(head_dim, 1)
        )

    def forward(self, input_ids, attention_mask, option_pos, group_ptr):
        """Score all options in a batch of packed questions.

        option_pos: LongTensor [total_options] — flat index into the
            batch's (batch * seq) hidden states.
        group_ptr: LongTensor [batch+1] — option i of sample s occupies
            flat positions group_ptr[s]:group_ptr[s+1].

        Returns list of per-question logit tensors (variable length).
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, T, H]
        flat = h.reshape(-1, h.size(-1))
        scores = self.head(flat[option_pos]).squeeze(-1)  # [total_options]
        return [scores[group_ptr[i]:group_ptr[i + 1]] for i in range(len(group_ptr) - 1)]


def collate(tokenizer, records, max_len=8192):
    """Pack a list of records into model inputs (dynamic padding)."""
    packed = [pack_question(tokenizer, r["state"], r["instructions"],
                            r["options"], max_len) for r in records]
    maxlen = max(len(ids) for ids, _ in packed)
    B = len(packed)
    input_ids = torch.full((B, maxlen), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    flat_pos, group_ptr = [], [0]
    for b, (ids, pos) in enumerate(packed):
        input_ids[b, : len(ids)] = torch.tensor(ids)
        attn[b, : len(ids)] = 1
        flat_pos += [b * maxlen + p for p in pos]
        group_ptr.append(group_ptr[-1] + len(pos))
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "option_pos": torch.tensor(flat_pos, dtype=torch.long),
        "group_ptr": torch.tensor(group_ptr, dtype=torch.long),
        "n_options": [len(p) for _, p in packed],
    }


def confidence_from_probs(probs: torch.Tensor) -> float:
    """1 - H(p)/ln K : 1 when certain, 0 when uniform. Jev-compatible shape."""
    k = probs.numel()
    if k <= 1:
        return 1.0
    h = -(probs * probs.clamp_min(1e-9).log()).sum().item()
    return max(0.0, min(1.0, 1.0 - h / torch.log(torch.tensor(float(k))).item()))
