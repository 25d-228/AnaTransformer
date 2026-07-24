"""The encoder-decoder itself.

The model does not know which variant it is. It asks a factory for an attention module at
each of the twelve sites and is otherwise identical across the whole study. Normalisation
is applied before each sublayer rather than after: post-norm is what the original paper
used, but it is sensitive to the warmup schedule in a way that is easy to mistake for a
property of the model being tested.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ana.config import ModelConfig, Site
from ana.nn.attention import AttentionCache, MultiHeadAttention
from ana.nn.layers import (
    FeedForward,
    TiedEmbedding,
    causal_mask,
    padding_mask,
    sinusoidal_positions,
)

AttentionFactory = Callable[[Site], MultiHeadAttention]


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, attention: AttentionFactory) -> None:
        super().__init__()
        self.self_attention = attention(Site.ENCODER_SELF)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.norm_feed_forward = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor, blocked: Tensor) -> Tensor:
        h = self.norm_attention(x)
        x = x + self.dropout(self.self_attention(h, h, mask, mask, blocked))

        h = self.norm_feed_forward(x)
        return x + self.dropout(self.feed_forward(h))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, attention: AttentionFactory) -> None:
        super().__init__()
        self.self_attention = attention(Site.DECODER_SELF)
        self.cross_attention = attention(Site.CROSS)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm_self = nn.LayerNorm(config.d_model)
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.norm_feed_forward = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        target_mask: Tensor,
        source_mask: Tensor,
        blocked_self: Tensor,
        blocked_cross: Tensor,
    ) -> Tensor:
        h = self.norm_self(x)
        x = x + self.dropout(self.self_attention(h, h, target_mask, target_mask, blocked_self))

        h = self.norm_cross(x)
        x = x + self.dropout(
            self.cross_attention(h, memory, target_mask, source_mask, blocked_cross)
        )

        h = self.norm_feed_forward(x)
        return x + self.dropout(self.feed_forward(h))

    def step(
        self,
        x: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        blocked_cross: Tensor,
        self_cache: AttentionCache,
        cross_cache: AttentionCache,
    ) -> Tensor:
        """One position. `x` is (batch, 1, d_model).

        No mask is needed on the decoder's own attention. The cache holds exactly the
        positions that came before this one, so a position cannot reach a later one: the
        later one has not been produced yet.
        """
        present = x.new_ones(x.size(0), 1, dtype=torch.long)

        h = self.norm_self(x)
        x = x + self.dropout(self.self_attention.step(h, None, present, None, self_cache))

        h = self.norm_cross(x)
        x = x + self.dropout(
            self.cross_attention.step(h, memory, present, source_mask, cross_cache, blocked_cross)
        )

        h = self.norm_feed_forward(x)
        return x + self.dropout(self.feed_forward(h))


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: ModelConfig, attention: AttentionFactory) -> None:
        super().__init__()
        self.config = config

        self.embedding = TiedEmbedding(config.vocab_size, config.d_model, config.pad_id)
        self.register_buffer(
            "positions",
            sinusoidal_positions(config.max_positions, config.d_model),
            persistent=False,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = nn.ModuleList(
            EncoderLayer(config, attention) for _ in range(config.n_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(config, attention) for _ in range(config.n_decoder_layers)
        )
        self.norm_encoder = nn.LayerNorm(config.d_model)
        self.norm_decoder = nn.LayerNorm(config.d_model)

        self._initialise()

    def _initialise(self) -> None:
        """Xavier-uniform on every matrix, then let each submodule restore its own choices.

        The role operators initialise themselves deliberately — a diagonal of ones, a
        uniform router, a unit magnitude — and a blanket pass over the whole model would
        overwrite exactly those choices without saying so.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model**-0.5)
                with torch.no_grad():
                    module.weight[self.config.pad_id].zero_()

        for module in self.modules():
            reset = getattr(module, "reset_role_parameters", None)
            if callable(reset):
                reset()

    def _embed(self, token_ids: Tensor) -> Tensor:
        length = token_ids.size(1)
        if length > self.config.max_positions:
            raise ValueError(f"sequence of {length} exceeds max_positions")
        return self.dropout(self.embedding.embed(token_ids) + self.positions[:length])

    def encode(self, source_ids: Tensor, source_mask: Tensor) -> Tensor:
        blocked = padding_mask(source_mask)
        x = self._embed(source_ids)
        for layer in self.encoder:
            x = layer(x, source_mask, blocked)
        return self.norm_encoder(x)

    def decode(
        self,
        target_ids: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        length = target_ids.size(1)
        blocked_self = padding_mask(target_mask) | causal_mask(length, target_ids.device)
        blocked_cross = padding_mask(source_mask)

        x = self._embed(target_ids)
        for layer in self.decoder:
            x = layer(x, memory, target_mask, source_mask, blocked_self, blocked_cross)
        return self.norm_decoder(x)

    def logits(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        memory = self.encode(source_ids, source_mask)
        hidden = self.decode(target_ids, memory, source_mask, target_mask)
        return self.embedding.project(hidden)

    def new_cache(self) -> list[tuple[AttentionCache, AttentionCache]]:
        """One pair of caches per decoder layer: its own attention, and cross-attention."""
        return [(AttentionCache(), AttentionCache()) for _ in self.decoder]

    def decode_step(
        self,
        token_ids: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        cache: list[tuple[AttentionCache, AttentionCache]],
        position: int,
    ) -> Tensor:
        """Score the next position from a single token and what came before it.

        `token_ids` is (batch, 1). Reruns nothing: the keys and values of earlier positions
        are already in the cache, so a step costs the same whether it is the third or the
        three-hundredth. Rerunning the whole prefix, which is what the batched path does,
        makes generation quadratic in the length of the output and blows up the memory the
        attention matrix needs.
        """
        x = self.dropout(self.embedding.embed(token_ids) + self.positions[position : position + 1])
        blocked_cross = padding_mask(source_mask)

        for layer, (self_cache, cross_cache) in zip(self.decoder, cache, strict=True):
            x = layer.step(x, memory, source_mask, blocked_cross, self_cache, cross_cache)

        return self.embedding.project(self.norm_decoder(x))[:, -1]

    def shift_right(self, labels: Tensor) -> Tensor:
        """Teacher forcing: prepend the start symbol, drop the final position."""
        target = labels.masked_fill(labels == -100, self.config.pad_id)
        start = target.new_full((target.size(0), 1), self.config.bos_id)
        return torch.cat([start, target[:, :-1]], dim=1)

    def forward(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target_ids = self.shift_right(labels)
        target_mask = (target_ids != self.config.pad_id).long()
        target_mask[:, 0] = 1  # the start symbol is always a real position

        scores = self.logits(source_ids, source_mask, target_ids, target_mask)
        loss = F.cross_entropy(
            scores.reshape(-1, scores.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )
        return loss, scores
