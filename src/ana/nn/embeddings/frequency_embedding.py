"""Tied lexical vectors with full rows for selected frequent vocabulary IDs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FrequencyAdaptiveEmbedding(nn.Module):
    """Keep selected rows dense and represent all remaining rows by a basis.

    Selection is supplied by the training-data runner, never inferred here.
    Persistent ID mappings retain the original tokenizer/output ordering.
    Lookup uses only requested rows; vocabulary prediction reconstructs one
    temporary dense table. There is no second learned output table or cache.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
        full_token_ids: Sequence[int] | Tensor,
        code_dim: int,
    ) -> None:
        super().__init__()
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (vocab_size, d_model, pad_id, code_dim)
        ):
            raise TypeError("vocabulary, model/code widths and padding ID must be integers")
        if vocab_size < 2 or d_model < 1 or not 1 <= code_dim <= d_model:
            raise ValueError("require vocabulary >= 2 and 1 <= code width <= model width")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("padding ID must belong to the vocabulary")
        full_ids = torch.as_tensor(full_token_ids, device="cpu")
        if full_ids.is_floating_point() or full_ids.is_complex() or full_ids.dtype == torch.bool:
            raise TypeError("full token IDs must be integers")
        if full_ids.ndim != 1 or not 1 <= full_ids.numel() < vocab_size:
            raise ValueError("full token IDs must be a nonempty vector leaving compressed rows")
        full_ids = full_ids.to(dtype=torch.long).clone()
        if (full_ids < 0).any() or (full_ids >= vocab_size).any():
            raise ValueError("full token ID outside the vocabulary")
        if full_ids.unique().numel() != full_ids.numel():
            raise ValueError("full token IDs must be distinct")
        if not full_ids.eq(pad_id).any():
            raise ValueError("reserve the padding token in the full-vector allocation")

        self.vocab_size, self.d_model, self.pad_id = vocab_size, d_model, pad_id
        self.code_dim, self.n_full = code_dim, full_ids.numel()
        self.scale = math.sqrt(d_model)
        self.code_scale = d_model**-0.5
        is_rare = torch.ones(vocab_size, dtype=torch.bool)
        is_rare[full_ids] = False
        rare_ids = torch.arange(vocab_size)[is_rare]
        full_index = torch.full((vocab_size,), -1, dtype=torch.long)
        rare_index = torch.full_like(full_index, -1)
        full_index[full_ids] = torch.arange(self.n_full)
        rare_index[rare_ids] = torch.arange(rare_ids.numel())
        self.register_buffer("full_token_ids", full_ids)
        self.register_buffer("rare_token_ids", rare_ids)
        self.register_buffer("full_index", full_index)
        self.register_buffer("rare_index", rare_index)

        self.full_rows = nn.Parameter(torch.randn(self.n_full, d_model) * self.code_scale)
        self.codes = nn.Parameter(torch.randn(rare_ids.numel(), code_dim) * self.code_scale)
        self.basis = nn.Parameter(torch.empty(code_dim, d_model))
        nn.init.orthogonal_(self.basis, gain=math.sqrt(d_model / code_dim))
        with torch.no_grad():
            self.full_rows[full_index[pad_id]].zero_()

    def weight(self) -> Tensor:
        """Reconstruct tied weights in original vocabulary order."""
        compressed = self.codes @ self.basis
        weight = self.full_rows.new_empty(self.vocab_size, self.d_model)
        weight.index_copy_(0, self.full_token_ids, self.full_rows)
        weight.index_copy_(0, self.rare_token_ids, compressed.to(weight))
        return weight

    def embed(self, token_ids: Tensor) -> Tensor:
        """Look up requested rows without constructing the vocabulary table.

        Fixed-shape masked branches avoid selecting variable-size GPU subsets.
        Padding retains its current value but supplies no lookup gradient;
        output prediction may update that row, as with ordinary tied embeddings.
        """
        flat_ids = token_ids.reshape(-1)
        full_ids = self.full_index.index_select(0, flat_ids).reshape_as(token_ids)
        rare_ids = self.rare_index.index_select(0, flat_ids).reshape_as(token_ids)
        full = F.embedding(full_ids.clamp_min(0), self.full_rows)
        full = torch.where(token_ids.eq(self.pad_id).unsqueeze(-1), full.detach(), full)
        compressed = F.embedding(rare_ids.clamp_min(0), self.codes) @ self.basis
        selected = torch.where(full_ids.ge(0).unsqueeze(-1), full, compressed)
        return selected * self.scale

    def project(self, hidden: Tensor) -> Tensor:
        """Use the same lexical weights with the ordinary global softmax."""
        return F.linear(hidden, self.weight())
