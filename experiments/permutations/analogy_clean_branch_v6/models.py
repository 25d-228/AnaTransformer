"""Keep model D's small encoder branches on unchanged attention inputs."""

from __future__ import annotations

from ana.config import ModelConfig
from ana.model import Seq2SeqTransformer
from experiments.permutations.analogy_combined_v4 import models as previous


MODEL_LABELS = {
    "compact_qkv_clean": "D-clean: unchanged-input small branches",
}
MODEL_NAMES = tuple(MODEL_LABELS)
CORPUS_NAMES = ("multi30k", "multi30k_enfr", "cogs")

collect_diagnostics = previous.collect_diagnostics
reset_diagnostics = previous.reset_diagnostics
optimizer_parameter_groups = previous.optimizer_parameter_groups


def models_for_corpus(corpus_name: str) -> tuple[str, ...]:
    """Only the two Multi30k directions and COGS belong to this batch."""
    if corpus_name not in CORPUS_NAMES:
        raise ValueError(f"unsupported clean-branch corpus {corpus_name!r}")
    return MODEL_NAMES


def model_details(model_name: str) -> dict:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"unknown clean-branch model {model_name!r}")
    details = previous.model_details("compact_qkv")
    details.update({
        "name": model_name,
        "label": MODEL_LABELS[model_name],
        "parent_model": "analogy_combined_v4/compact_qkv",
        "encoder_shared_projection_input": "role-conditioned analogy input",
        "encoder_residual_input": "original unchanged attention input",
        "residual_budget": "same total parameter count as model D",
        "initialization": "same initial weights as model D",
    })
    return details


def construct(
    corpus_name: str, model_name: str, full_shape: ModelConfig
) -> Seq2SeqTransformer:
    models_for_corpus(corpus_name)
    model_details(model_name)
    model = previous.construct(corpus_name, "compact_qkv", full_shape)
    for layer in model.encoder:
        projection = layer.self_attention.projection
        projection.clean_residual_input = True
    return model
