"""The pieces of the network that no model in this study varies.

Feed-forward block, sinusoidal positions, the tied embedding, and the attention masks.
Masks are boolean rather than additive: two additive masks of size `finfo.min` sum to
negative infinity in half precision, which is a quiet way to produce NaN.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeedForward(nn.Module):
    """Widen to `d_ff`, apply GELU, narrow back. Where most of the parameters live."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.dropout(F.gelu(self.up(x))))


class TiedEmbedding(nn.Module):
    """One table for the source, the target and the output head.

    The corpus is tokenised with a single joint vocabulary, so one table serves all three.
    Embeddings are scaled by the square root of the width, as in the original transformer.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_id: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

    def embed(self, token_ids: Tensor) -> Tensor:
        return self.table(token_ids) * self.scale

    def project(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.table.weight)


def sinusoidal_positions(max_positions: int, d_model: int) -> Tensor:
    """The fixed position table. Not a parameter, and not saved with the model."""
    position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
    step = torch.arange(0, d_model, 2, dtype=torch.float32)
    rate = torch.exp(-math.log(10_000.0) * step / d_model)

    table = torch.zeros(max_positions, d_model)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate)
    return table


def padding_mask(kv_mask: Tensor) -> Tensor:
    """(batch, key_len) of real-token flags -> (batch, 1, 1, key_len) of disallowed flags."""
    return ~kv_mask.bool()[:, None, None, :]


def causal_mask(length: int, device: torch.device) -> Tensor:
    """(1, 1, query_len, key_len): a query may not attend to a later key."""
    square = torch.ones(length, length, dtype=torch.bool, device=device)
    return torch.triu(square, diagonal=1)[None, None]
