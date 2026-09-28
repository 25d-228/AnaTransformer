"""Causal four-term translation scores on an ordinary compact Transformer.

For an earlier source context A and its known target token B, compare A:B
with the current source context C and candidate target token D. Positive
features use fixed p=0.5. The ordinary signed backbone remains unrestricted;
the additional logits are the exact weighted negative squared analogy defect,
up to a constant shared by all vocabulary candidates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ana.config import ModelConfig, Site
from ana.model import Seq2SeqTransformer
from ana.nn.attention import AttentionCache, MultiHeadAttention
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding
from ana.nn.layers import causal_mask, padding_mask
from ana.nn.projection import SeparateQKV


def _work_dtype(value: Tensor) -> torch.dtype:
    return torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype


class CausalTranslationHead(nn.Module):
    """Two small positive feature maps and one positive vocabulary-score gain.

    Public feature methods return T(u)=2*(sqrt(u)-1), where
    u=softplus(map(LayerNorm(x)))/log(2)+1e-6. LayerNorm has no parameters.
    Feature width is a parameter budget, not the number of analogy roles.
    """

    power = 0.5

    def __init__(self, d_model: int, feature_dim: int = 4) -> None:
        super().__init__()
        if d_model < 1 or feature_dim < 1:
            raise ValueError("model and analogy feature widths must be positive")
        self.d_model, self.feature_dim = d_model, feature_dim
        self.source_map = nn.Linear(d_model, feature_dim)
        self.target_map = nn.Linear(d_model, feature_dim)
        self.raw_gain = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        for mapping in (self.source_map, self.target_map):
            nn.init.xavier_uniform_(mapping.weight)
            nn.init.zeros_(mapping.bias)

    def gain(self) -> Tensor:
        """Return the positive learned scalar, initially 0.1."""
        return F.softplus(self.raw_gain.to(dtype=_work_dtype(self.raw_gain)))

    def _features(self, value: Tensor, mapping: nn.Linear) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            work = value.to(dtype=_work_dtype(value))
            normalized = F.layer_norm(work, (self.d_model,))
            mapped = F.linear(normalized, mapping.weight.to(work), mapping.bias.to(work))
            positive = F.softplus(mapped) / math.log(2.0) + 1e-6
            return 2.0 * (positive.sqrt() - 1.0)

    def source_features(self, context: Tensor) -> Tensor:
        """Return powered source features, preserving leading dimensions."""
        return self._features(context, self.source_map)

    def vocabulary_features(self, effective_weight: Tensor) -> Tensor:
        """Return powered target features from the current tied lexical table."""
        return self._features(effective_weight, self.target_map)

    def reference_correction(
        self,
        current: Tensor,
        reference_source: Tensor,
        reference_relation: Tensor,
        reference_valid: Tensor,
        vocabulary: Tensor,
        query_valid: Tensor | None = None,
    ) -> Tensor:
        """Score [B,Q,r] current features against [B,S,r] earlier examples.

        Relations are T(B)-T(A); vocabulary is [V,r]. Reference validity is
        [B,S] or broadcastable to [B,Q,S]. No reference-by-vocabulary tensor
        is formed. Rows without references return exactly zero correction.
        """
        with torch.autocast(device_type=current.device.type, enabled=False):
            current = current.to(dtype=_work_dtype(current))
            source = reference_source.to(current)
            relation = reference_relation.to(current)
            words = vocabulary.to(current)
            if source.size(-2) == 0:
                return current.new_zeros(*current.shape[:-1], words.size(0))

            distances = (
                current.square().sum(-1, keepdim=True)
                + source.square().sum(-1).unsqueeze(-2)
                - 2.0 * torch.matmul(current, source.transpose(-2, -1))
            ).clamp_min(0.0) / self.feature_dim
            valid = reference_valid.bool()
            if valid.ndim == 2:
                valid = valid.unsqueeze(-2)
            valid = torch.broadcast_to(valid, distances.shape)
            if query_valid is not None:
                valid = valid & query_valid.bool().unsqueeze(-1)
            active = valid.any(-1)
            scores = (-distances).masked_fill(~valid, -torch.inf)
            # An empty masked row must not enter softmax as all -infinity.
            scores = scores.masked_fill(~active.unsqueeze(-1), 0.0)
            routes = scores.softmax(-1).masked_fill(~valid, 0.0)
            completed = current + torch.matmul(routes, relation)
            correction = (
                2.0 * torch.matmul(completed, words.transpose(-2, -1)) - words.square().sum(-1)
            ) * (self.gain().to(current) / self.feature_dim)
            return correction.masked_fill(~active.unsqueeze(-1), 0.0)

    def causal_correction(
        self,
        source: Tensor,
        vocabulary: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
        pad_id: int,
    ) -> Tensor:
        """Return [B,T,V] scores using only known prefix tokens and s<t.

        Source position s predicts the token stored at target_ids[:,s+1].
        Thus the final source position cannot yet form a reference pair.
        Inputs source/vocabulary already contain the powered feature maps.
        """
        length = source.size(1)
        valid_query = target_mask.bool() & target_ids.ne(pad_id)
        reference_source = source[:, :-1]
        reference_relation = F.embedding(target_ids[:, 1:], vocabulary) - reference_source
        valid_reference = valid_query[:, :-1] & valid_query[:, 1:]
        positions = torch.arange(length, device=source.device)
        earlier = positions[:-1].unsqueeze(0) < positions.unsqueeze(1)
        return self.reference_correction(
            source,
            reference_source,
            reference_relation,
            valid_reference.unsqueeze(1) & earlier.unsqueeze(0),
            vocabulary,
            valid_query,
        )


class CausalTranslationTransformer(Seq2SeqTransformer):
    """Linear lexical compression plus causal analogy vocabulary scores.

    ``forward`` uses the inherited ordinary translation loss through ``logits``.
    ``decode_step`` supports the existing greedy/beam decoders. The inherited
    ``decode`` returns ordinary hidden states only: directly projecting those
    hidden states bypasses this head. Activation checkpointing is not supported
    in this pilot's full scoring path.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.n_decoder_layers < 1:
            raise ValueError("causal translation analogy needs a decoder layer")

        def attention(site: Site) -> MultiHeadAttention:
            return MultiHeadAttention(
                config.d_model,
                config.n_heads,
                SeparateQKV(config.d_model),
                site,
                config.dropout,
            )

        # Match build_model('embedding_linear', config) through the completed
        # embedding replacement before drawing any new head parameters.
        super().__init__(config, attention)
        self.embedding = LinearCompressedEmbedding(
            config.vocab_size,
            config.d_model,
            config.pad_id,
            normalize_rows=False,
        )
        self.analogy_head = CausalTranslationHead(config.d_model)

    def _decode_with_context(
        self,
        target_ids: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        blocked_self = padding_mask(target_mask) | causal_mask(
            target_ids.size(1), target_ids.device
        )
        blocked_cross = padding_mask(source_mask)
        x = self._embed(target_ids)
        for layer in self.decoder[:-1]:
            x = layer(x, memory, target_mask, source_mask, blocked_self, blocked_cross)

        # Run the final ordinary layer once, retaining its cross-attention
        # output before residual dropout. No attention weights or hooks change.
        layer = self.decoder[-1]
        h = layer.norm_self(x)
        x = x + layer.dropout(layer.self_attention(h, h, target_mask, target_mask, blocked_self))
        h = layer.norm_cross(x)
        context = layer.cross_attention(h, memory, target_mask, source_mask, blocked_cross)
        x = x + layer.dropout(context)
        h = layer.norm_feed_forward(x)
        x = x + layer.dropout(layer.feed_forward(h))
        return self.norm_decoder(x), context

    def logits(
        self,
        source_ids: Tensor,
        source_mask: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        if self.activation_checkpointing:
            raise ValueError("causal translation pilot does not support activation checkpointing")
        memory = self.encode(source_ids, source_mask)
        hidden, context = self._decode_with_context(target_ids, memory, source_mask, target_mask)
        weight = self.embedding.weight()
        ordinary = F.linear(hidden, weight)
        correction = self.analogy_head.causal_correction(
            self.analogy_head.source_features(context),
            self.analogy_head.vocabulary_features(weight),
            target_ids,
            target_mask,
            self.config.pad_id,
        )
        return ordinary + correction.to(ordinary)

    def new_cache(self) -> list[tuple[AttentionCache, AttentionCache]]:
        """Add history/pending caches that existing beam backpointers reorder.

        History key: [B,S,r] source features. History value: [B,S,r+1]
        relation features plus validity. Pending key/value: [B,1,r] current
        source features and [B,1,1] query validity, awaiting the next token.
        """
        return super().new_cache() + [(AttentionCache(), AttentionCache())]

    def decode_step(
        self,
        token_ids: Tensor,
        memory: Tensor,
        source_mask: Tensor,
        cache: list[tuple[AttentionCache, AttentionCache]],
        position: int,
    ) -> Tensor:
        if len(cache) != len(self.decoder) + 1:
            raise ValueError("use this model's new_cache() for causal translation decoding")
        x = self.dropout(self.embedding.embed(token_ids) + self.positions[position : position + 1])
        blocked_cross = padding_mask(source_mask)
        for layer, (self_cache, cross_cache) in zip(self.decoder[:-1], cache[:-2], strict=True):
            x = layer.step(x, memory, source_mask, blocked_cross, self_cache, cross_cache)

        layer = self.decoder[-1]
        self_cache, cross_cache = cache[-2]
        present = x.new_ones(x.size(0), 1, dtype=torch.long)
        h = layer.norm_self(x)
        x = x + layer.dropout(layer.self_attention.step(h, None, present, None, self_cache))
        h = layer.norm_cross(x)
        context = layer.cross_attention.step(
            h, memory, present, source_mask, cross_cache, blocked_cross
        )
        x = x + layer.dropout(context)
        h = layer.norm_feed_forward(x)
        x = x + layer.dropout(layer.feed_forward(h))

        weight = self.embedding.weight()
        ordinary = F.linear(self.norm_decoder(x), weight)
        current = self.analogy_head.source_features(context)
        vocabulary = self.analogy_head.vocabulary_features(weight)
        history, pending = cache[-1]
        query_valid = token_ids.ne(self.config.pad_id)
        if history.key is None:
            history.key = current.new_empty(current.size(0), 0, self.analogy_head.feature_dim)
            history.value = current.new_empty(
                current.size(0), 0, self.analogy_head.feature_dim + 1
            )
        if pending.key is not None:
            relation = F.embedding(token_ids, vocabulary) - pending.key
            valid = pending.value.bool() & query_valid.unsqueeze(-1)
            history.key = torch.cat((history.key, pending.key), dim=1)
            completed = torch.cat((relation, valid.to(relation)), dim=-1)
            history.value = torch.cat((history.value, completed), dim=1)

        correction = self.analogy_head.reference_correction(
            current,
            history.key,
            history.value[..., :-1],
            history.value[..., -1].bool(),
            vocabulary,
            query_valid,
        )
        pending.key = current
        pending.value = query_valid.unsqueeze(-1).to(current)
        return (ordinary + correction.to(ordinary))[:, -1]
