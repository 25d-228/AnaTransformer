"""Five comparison rows with a corpus-specific compact lexical budget.

The two embedding rows construct exactly the same model. Only their training
objective differs, in the runner. No model is added to the global registry.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace

from ana.config import ModelConfig
from ana.model import Seq2SeqTransformer
from ana.nn.attention import MultiHeadAttention
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding
from ana.nn.layers import FeedForward
from ana.nn.projection import SeparateQKV, SharedQKV
from ana.registry import (
    baseline_parameters,
    build_model,
    count_parameters,
    matched_config,
    model_parameters,
)

MODEL_NAMES = (
    "baseline",
    "baseline_matched",
    "shared_qkv",
    "embedding_linear",
    "embedding_power_consistency",
)
EMBEDDING_MODELS = frozenset(MODEL_NAMES[-2:])


def _compressed_parameters(shape: ModelConfig, code_dim: int) -> int:
    return (
        baseline_parameters(shape)
        - shape.vocab_size * shape.d_model
        + code_dim * (shape.vocab_size + shape.d_model)
    )


def _compact_config(corpus_name: str, full_shape: ModelConfig) -> tuple[ModelConfig, int]:
    if corpus_name in ("enfr", "multi30k_enfr", "multi30k-en-fr"):
        # The published full shape is d128/FF256, giving d128/FF232/r96.
        # Expressing the same proportion also permits a tiny wiring check.
        shape = replace(full_shape, d_ff=max(1, round(full_shape.d_ff * 29 / 32)))
        code_dim = 3 * full_shape.d_model // 4
    elif corpus_name == "cogs":
        code_dim = 160
        initial = replace(full_shape, d_model=400, d_ff=1)
        target = baseline_parameters(matched_config(full_shape))
        unit = (initial.n_encoder_layers + initial.n_decoder_layers) * (2 * initial.d_model + 1)
        ideal = 1 + (target - _compressed_parameters(initial, code_dim)) / unit
        widths = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
        width = min(
            widths,
            key=lambda value: (
                abs(_compressed_parameters(replace(initial, d_ff=value), code_dim) - target),
                value,
            ),
        )
        shape = replace(initial, d_ff=width)
        if abs(_compressed_parameters(shape, code_dim) - target) > unit / 2 + 1:
            raise ValueError("COGS compact width cannot reach the matched parameter budget")
    else:
        raise ValueError(f"unsupported comparison corpus {corpus_name!r}")
    if code_dim * (shape.vocab_size + shape.d_model) >= shape.vocab_size * shape.d_model:
        raise ValueError("the selected lexical rank must save parameters on this vocabulary")
    return shape, code_dim


def _specification(
    corpus_name: str,
    model_name: str,
    full_shape: ModelConfig,
) -> tuple[ModelConfig, int | None, int]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown comparison model {model_name!r}")
    if model_name in EMBEDDING_MODELS:
        shape, code_dim = _compact_config(corpus_name, full_shape)
        return shape, code_dim, _compressed_parameters(shape, code_dim)
    shape = matched_config(full_shape) if model_name == "baseline_matched" else full_shape
    return shape, None, model_parameters(model_name, full_shape)


def model_summary(corpus_name: str, model_name: str, full_shape: ModelConfig) -> dict:
    """Return the resolved shape and analytical counts without constructing it."""
    shape, code_dim, parameters = _specification(corpus_name, model_name, full_shape)
    full_parameters = baseline_parameters(full_shape)
    matched_parameters = baseline_parameters(matched_config(full_shape))
    return {
        "corpus": corpus_name,
        "model": model_name,
        "model_config": asdict(shape),
        "code_dim": code_dim,
        "parameters": parameters,
        "full_parameters": full_parameters,
        "saved_parameters": full_parameters - parameters,
        "parameter_saving_fraction": (full_parameters - parameters) / full_parameters,
        "matched_parameters": matched_parameters,
        "difference_from_matched": parameters - matched_parameters,
        "uses_power_consistency": model_name == "embedding_power_consistency",
    }


def construct(corpus_name: str, model_name: str, full_shape: ModelConfig) -> Seq2SeqTransformer:
    """Construct one row; embedding rows preserve identical initialization."""
    shape, code_dim, expected = _specification(corpus_name, model_name, full_shape)
    if model_name in EMBEDDING_MODELS:
        # Same ordinary initialization/replacement order as embedding_linear.
        model = build_model("baseline", shape)
        model.embedding = LinearCompressedEmbedding(
            shape.vocab_size,
            shape.d_model,
            shape.pad_id,
            code_dim=code_dim,
        )
    else:
        model = build_model(model_name, full_shape)
    actual = count_parameters(model)
    if actual != expected:
        raise RuntimeError(f"parameter count mismatch: constructed {actual}, expected {expected}")
    if model_name in EMBEDDING_MODELS and actual >= baseline_parameters(full_shape):
        raise ValueError("the compact model must be smaller than the full Transformer")
    projection_kind = SharedQKV if model_name == "shared_qkv" else SeparateQKV
    for layer in (*model.encoder, *model.decoder):
        if type(layer.feed_forward) is not FeedForward:
            raise TypeError("the comparison requires ordinary feed-forward layers")
        attentions = [layer.self_attention]
        if hasattr(layer, "cross_attention"):
            attentions.append(layer.cross_attention)
        if any(
            type(attention) is not MultiHeadAttention
            or type(attention.projection) is not projection_kind
            for attention in attentions
        ):
            raise TypeError(
                "the comparison requires ordinary attention and its declared Q/K/V roles"
            )
    return model
