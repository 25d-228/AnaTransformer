"""Multi-head attention. It owns the heads, the softmax and the output projection.

How the queries, keys and values are produced is not its business: that is handed in as a
`QKVProjection`. Swapping that object is the only thing any model in this study changes.

The module can also run one position at a time, which is what generation needs. That is
only sound because every role operator that reaches the decoder is positionwise — a token's
query, key and value depend on that token and nothing else — so a key computed at step three
is still the right key at step three hundred. The one operator that is not positionwise,
mixing over four consecutive tokens, is refused at any decoder site when the model is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ana.config import Site
from ana.nn.projection import QKVProjection


@dataclass
class AttentionCache:
    """Keys and values kept between generation steps, already split into heads.

    For decoder self-attention the pair grows by one position each step. For cross-attention
    it is computed once from the encoder output and never changes. A shared key-value
    projection stores its one tensor in `key` and leaves `value` empty.
    """

    key: Tensor | None = None
    value: Tensor | None = None


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product attention over `n_heads` heads.

    A module is told which site it occupies when it is built, rather than working it out at
    call time from whether a key-value tensor happens to have been passed. Where an operator
    may legally run has to be settled when the model is constructed, not once it is training.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        projection: QKVProjection,
        site: Site,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.site = site

        self.projection = projection
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def _join_heads(self, x: Tensor) -> Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, self.n_heads * self.head_dim)

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, blocked: Tensor | None) -> Tensor:
        scores = torch.matmul(query * self.scale, key.transpose(-2, -1))
        if blocked is not None:
            scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)

        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        return self.output(self._join_heads(context))

    def forward(
        self,
        query_input: Tensor,
        kv_input: Tensor,
        query_mask: Tensor,
        kv_mask: Tensor,
        blocked: Tensor | None = None,
    ) -> Tensor:
        """`blocked` is a boolean mask, True where a query may not read a key."""
        query, key, value = self.projection(query_input, kv_input, query_mask, kv_mask)
        return self._attend(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            blocked,
        )

    def step(
        self,
        query_input: Tensor,
        kv_input: Tensor | None,
        query_mask: Tensor,
        kv_mask: Tensor | None,
        cache: AttentionCache,
        blocked: Tensor | None = None,
    ) -> Tensor:
        """One query position, reading keys and values that were computed earlier.

        `query_input` is a single position, shape (batch, 1, d_model). For cross-attention
        `kv_input` is the encoder output, and its keys and values are computed on the first
        step and reused. For decoder self-attention the new position's key and value are
        appended to what is already in the cache.
        """
        query = self._split_heads(self.projection.query_only(query_input, query_mask))

        if self.projection.shares_key_value_cache:
            cache.value = None
            if self.site is Site.CROSS:
                if cache.key is None:
                    key, _ = self.projection.key_value(kv_input, kv_mask)
                    cache.key = self._split_heads(key)
            else:
                key, _ = self.projection.key_value(query_input, query_mask)
                key = self._split_heads(key)
                cache.key = key if cache.key is None else torch.cat([cache.key, key], dim=2)
            cached_value = cache.key
        else:
            if self.site is Site.CROSS:
                if cache.key is None:
                    key, value = self.projection.key_value(kv_input, kv_mask)
                    cache.key = self._split_heads(key)
                    cache.value = self._split_heads(value)
            else:
                key, value = self.projection.key_value(query_input, query_mask)
                key = self._split_heads(key)
                value = self._split_heads(value)
                if cache.key is None:
                    cache.key, cache.value = key, value
                else:
                    cache.key = torch.cat([cache.key, key], dim=2)
                    cache.value = torch.cat([cache.value, value], dim=2)
            cached_value = cache.value

        return self._attend(query, cache.key, cached_value, blocked)
