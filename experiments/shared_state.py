"""Shared-state, sibling-isolated decision model prototype.

The state is encoded once. Every question is a bidirectional branch that can
attend to the state and itself, but never to another question. Branch position
IDs restart after the state so packed and single-question execution have the
same logical positions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

import torch
import torch.nn as nn


DELIMITER_TOKENS = (
    "<haetae_question>",
    "<haetae_option>",
    "<haetae_option_end>",
    "<haetae_decide>",
)
_DELIMITER_PATTERN = re.compile(r"<haetae_(question|option|option_end|decide)>")


@dataclass(frozen=True)
class PackedRequest:
    input_ids: list[int]
    segments: list[int]
    position_ids: list[int]
    decision_indices: list[int]
    option_indices: list[list[int]]
    labels: list[int | None]
    state_tokens: int
    retained_state_tokens: int


def canonical_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def add_delimiters(tokenizer) -> dict[str, int]:
    tokenizer.add_special_tokens({
        "additional_special_tokens": list(DELIMITER_TOKENS),
    })
    result = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in DELIMITER_TOKENS
    }
    if len(set(result.values())) != len(DELIMITER_TOKENS):
        raise ValueError("decision delimiters do not have distinct token IDs")
    if tokenizer.unk_token_id in result.values():
        raise ValueError("decision delimiter resolved to the unknown token")
    return result


def user_token_ids(tokenizer, value) -> list[int]:
    text = canonical_state(value)
    text = _DELIMITER_PATTERN.sub(
        lambda match: f"⟦haetae_{match.group(1)}⟧", text,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_request(tokenizer, state, questions: list[dict], max_length: int,
                   delimiters: dict[str, int] | None = None
                   ) -> PackedRequest | None:
    if not questions:
        raise ValueError("a request needs at least one question")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    delimiters = delimiters or {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in DELIMITER_TOKENS
    }
    question_token, option_token, end_token, decide_token = (
        delimiters[token] for token in DELIMITER_TOKENS
    )

    branches = []
    for question in questions:
        options = question.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("each question needs at least two options")
        branch = [question_token]
        branch.extend(user_token_ids(tokenizer, question["instructions"]))
        branch.append(tokenizer.sep_token_id)
        option_ends = []
        for option in options:
            branch.append(option_token)
            branch.extend(user_token_ids(tokenizer, option))
            branch.append(end_token)
            option_ends.append(len(branch) - 1)
        branch.append(decide_token)
        branches.append((branch, option_ends, question.get("label")))

    fixed_tokens = 2 + sum(len(branch) for branch, _, _ in branches)
    if fixed_tokens > max_length:
        return None
    state = user_token_ids(tokenizer, state)
    state_budget = max_length - fixed_tokens
    state_prefix = [tokenizer.cls_token_id]
    state_prefix.extend(state[:state_budget])
    state_prefix.append(tokenizer.sep_token_id)
    state_length = len(state_prefix)

    input_ids = list(state_prefix)
    segments = [0] * state_length
    positions = list(range(state_length))
    decision_indices, option_indices, labels = [], [], []
    for segment, (branch, option_ends, label) in enumerate(branches, start=1):
        base = len(input_ids)
        input_ids.extend(branch)
        segments.extend([segment] * len(branch))
        positions.extend(range(state_length, state_length + len(branch)))
        decision_indices.append(base + len(branch) - 1)
        option_indices.append([base + index for index in option_ends])
        labels.append(label)
    return PackedRequest(
        input_ids=input_ids,
        segments=segments,
        position_ids=positions,
        decision_indices=decision_indices,
        option_indices=option_indices,
        labels=labels,
        state_tokens=len(state),
        retained_state_tokens=min(len(state), state_budget),
    )


def branch_attention_masks(encodings: list[PackedRequest], device,
                           dtype: torch.dtype, sliding_window: int) -> dict:
    if not encodings:
        raise ValueError("cannot mask an empty batch")
    if sliding_window < 0:
        raise ValueError("sliding_window must be nonnegative")
    length = max(len(encoding.input_ids) for encoding in encodings)
    segments = torch.full(
        (len(encodings), length), -1, dtype=torch.long, device=device,
    )
    positions = torch.zeros_like(segments)
    for batch_index, encoding in enumerate(encodings):
        size = len(encoding.input_ids)
        segments[batch_index, :size] = torch.tensor(
            encoding.segments, device=device,
        )
        positions[batch_index, :size] = torch.tensor(
            encoding.position_ids, device=device,
        )

    query_segment = segments[:, :, None]
    key_segment = segments[:, None, :]
    valid_query = query_segment >= 0
    valid_key = key_segment >= 0
    state_to_state = (query_segment == 0) & (key_segment == 0)
    branch_to_visible = (query_segment > 0) & (
        (key_segment == 0) | (key_segment == query_segment)
    )
    allowed = valid_query & valid_key & (state_to_state | branch_to_visible)
    diagonal = torch.eye(length, dtype=torch.bool, device=device)[None]
    allowed = allowed | diagonal

    query_position = positions[:, :, None]
    key_position = positions[:, None, :]
    local = (query_position - key_position).abs() <= sliding_window
    sliding_allowed = allowed & local
    sliding_allowed = sliding_allowed | diagonal

    def additive(mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            mask.shape, dtype=dtype, device=device,
        ).masked_fill(~mask, torch.finfo(dtype).min)[:, None]

    return {
        "full_attention": additive(allowed),
        "sliding_attention": additive(sliding_allowed),
    }


def collate_requests(encodings: list[PackedRequest], pad_token_id: int,
                     device) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(encoding.input_ids) for encoding in encodings)
    input_ids = torch.full(
        (len(encodings), length), pad_token_id,
        dtype=torch.long, device=device,
    )
    position_ids = torch.zeros_like(input_ids)
    for batch_index, encoding in enumerate(encodings):
        size = len(encoding.input_ids)
        input_ids[batch_index, :size] = torch.tensor(
            encoding.input_ids, device=device,
        )
        position_ids[batch_index, :size] = torch.tensor(
            encoding.position_ids, device=device,
        )
    return input_ids, position_ids


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, pointer_size: int = 128):
        super().__init__()
        self.query = nn.Linear(hidden_size, pointer_size)
        self.key = nn.Linear(hidden_size, pointer_size)
        self.scale = pointer_size ** -0.5

    def forward(self, decision: torch.Tensor,
                options: torch.Tensor) -> torch.Tensor:
        return (self.key(options) @ self.query(decision)) * self.scale


class SharedStateDecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, pad_token_id: int,
                 pointer_size: int = 128):
        super().__init__()
        self.backbone = backbone
        self.pad_token_id = pad_token_id
        self.head = PointerHead(backbone.config.hidden_size, pointer_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, encodings: list[PackedRequest]) -> list[list[torch.Tensor]]:
        input_ids, position_ids = collate_requests(
            encodings, self.pad_token_id, self.device,
        )
        dtype = next(self.backbone.parameters()).dtype
        masks = branch_attention_masks(
            encodings, self.device, dtype,
            int(self.backbone.config.local_attention),
        )
        hidden = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=masks,
        ).last_hidden_state
        output = []
        for batch_index, encoding in enumerate(encodings):
            questions = []
            for decision_index, option_indexes in zip(
                    encoding.decision_indices, encoding.option_indices):
                options = hidden[
                    batch_index,
                    torch.tensor(option_indexes, device=self.device),
                ]
                questions.append(self.head(
                    hidden[batch_index, decision_index], options,
                ))
            output.append(questions)
        return output


def load_shared_state_model(name: str, tokenizer, device: str,
                            revision: str | None = None,
                            pointer_size: int = 128,
                            attention_implementation: str = "eager",
                            local_files_only: bool = False):
    from transformers import AutoModel

    delimiters = add_delimiters(tokenizer)
    backbone = AutoModel.from_pretrained(
        name,
        revision=revision,
        attn_implementation=attention_implementation,
        local_files_only=local_files_only,
    )
    backbone.resize_token_embeddings(len(tokenizer))
    model = SharedStateDecisionModel(
        backbone, tokenizer.pad_token_id, pointer_size,
    ).to(device)
    return model, delimiters
