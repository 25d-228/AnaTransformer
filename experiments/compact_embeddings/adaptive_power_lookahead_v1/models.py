"""The existing compact model with multi-batch lookahead power selection."""

from __future__ import annotations

from dataclasses import asdict, replace

from ana.config import ModelConfig
from ana.model import Seq2SeqTransformer
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding
from ana.registry import baseline_parameters, build_model, count_parameters

MODEL_NAMES = ("embedding_adaptive_power_lookahead",)


def _specification(corpus_name: str, model_name: str, full_shape: ModelConfig):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown training variant {model_name!r}")
    if corpus_name in ("multi30k", "multi30k_enfr"):
        if full_shape.d_model != 128 or full_shape.d_ff != 256:
            raise ValueError("Retain the original Multi30k architecture.")
        shape, code_dim = replace(full_shape, d_ff=232), 96
    elif corpus_name == "cogs":
        shape, code_dim = replace(full_shape, d_model=400, d_ff=509), 160
    else:
        raise ValueError(f"unsupported corpus {corpus_name!r}")
    expected = (
        baseline_parameters(shape)
        - shape.vocab_size * shape.d_model
        + code_dim * (shape.vocab_size + shape.d_model)
    )
    return shape, code_dim, expected


def model_summary(corpus_name: str, model_name: str, full_shape: ModelConfig) -> dict:
    shape, code_dim, parameters = _specification(corpus_name, model_name, full_shape)
    full_parameters = baseline_parameters(full_shape)
    return {
        "corpus": corpus_name,
        "model": model_name,
        "model_config": asdict(shape),
        "code_dim": code_dim,
        "parameters": parameters,
        "full_parameters": full_parameters,
        "saved_parameters": full_parameters - parameters,
        "parameter_saving_fraction": (full_parameters - parameters) / full_parameters,
    }


def construct(corpus_name: str, model_name: str, full_shape: ModelConfig) -> Seq2SeqTransformer:
    """Keep the exact baseline-then-embedding-replacement initialization order."""
    shape, code_dim, expected = _specification(corpus_name, model_name, full_shape)
    model = build_model("baseline", shape)
    model.embedding = LinearCompressedEmbedding(
        shape.vocab_size,
        shape.d_model,
        shape.pad_id,
        code_dim=code_dim,
    )
    actual = count_parameters(model)
    if actual != expected or actual >= baseline_parameters(full_shape):
        raise RuntimeError(f"invalid compact parameter count: {actual}, expected {expected}")
    return model
