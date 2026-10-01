"""Refine model D without changing its numerical-analogy operation or size."""

from __future__ import annotations

from torch import nn

from ana.config import ModelConfig
from ana.model import Seq2SeqTransformer
from experiments.permutations.analogy_combined_v4 import models as previous


MODEL_LABELS = {
    "d_router_03": "D1: model D with controller learning rate 0.3x",
    "d_router_10": "D2: model D with controller learning rate 1.0x",
    "d_cross_focus": "D3: model D with cross-attention-focused branches",
}
MODEL_NAMES = tuple(MODEL_LABELS)
CORPUS_NAMES = ("multi30k", "multi30k_enfr", "cogs")
ROUTING_LR_SCALES = {
    "d_router_03": 0.3,
    "d_router_10": 1.0,
    "d_cross_focus": 0.1,
}

collect_diagnostics = previous.collect_diagnostics
reset_diagnostics = previous.reset_diagnostics


def models_for_corpus(corpus_name: str) -> tuple[str, ...]:
    """IWSLT14 is deliberately excluded from this focused nine-run batch."""
    if corpus_name not in CORPUS_NAMES:
        raise ValueError(f"unsupported refinement corpus {corpus_name!r}")
    return MODEL_NAMES


def model_details(model_name: str) -> dict:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"unknown model-D refinement {model_name!r}")
    details = previous.model_details("compact_qkv")
    details.update({
        "name": model_name,
        "label": MODEL_LABELS[model_name],
        "parent_model": "analogy_combined_v4/compact_qkv",
        "routing_learning_rate_scale": ROUTING_LR_SCALES[model_name],
        "residual_budget": "same total parameter count as model D",
        "initialization": "same initial weights as model D",
    })
    if model_name == "d_cross_focus":
        details.update({
            "low_rank_residual": "d_model / 16",
            "decoder_residual_rank": "3 * d_model / 16",
            "initialization": (
                "retain model D backbone and analogy weights; replace only "
                "residual matrices, with zero-initialized up projections"
            ),
        })
    return details


def construct(
    corpus_name: str, model_name: str, full_shape: ModelConfig
) -> Seq2SeqTransformer:
    models_for_corpus(corpus_name)
    model_details(model_name)
    rank_overrides = {}
    if model_name == "d_cross_focus":
        d_model = full_shape.d_model
        if d_model % 16:
            raise ValueError("cross-focused ranks require width divisible by 16")
        encoder_rank, cross_rank = d_model // 16, 3 * d_model // 16
        n_encoder = full_shape.n_encoder_layers
        n_decoder = full_shape.n_decoder_layers
        original_budget = (n_encoder + n_decoder) * (d_model // 8)
        updated_budget = n_encoder * encoder_rank + n_decoder * cross_rank
        if updated_budget != original_budget:
            raise ValueError("cross-focused branches must keep model D's size")
        rank_overrides = {
            "encoder_rank": encoder_rank,
            "cross_rank": cross_rank,
        }
    model = previous.construct(
        corpus_name, "compact_qkv", full_shape, **rank_overrides
    )
    model.analogy_routing_lr_scale = ROUTING_LR_SCALES[model_name]
    return model


def optimizer_parameter_groups(model: nn.Module) -> list[dict]:
    groups = previous.optimizer_parameter_groups(model)
    for group in groups:
        if group["group_name"] == "routing_controllers":
            group["lr_scale"] = model.analogy_routing_lr_scale
    return groups
