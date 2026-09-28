"""Compact embeddings with prediction-trained, context-dependent powers.

The two designs retain ordinary independent Q/K/V projections. A changes
prediction sharpness with one power per position; B applies a shared power
to each pair of positive decoder features before vocabulary projection.
Both start at exactly the existing compact model's prediction function.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ana.config import ModelConfig
from ana.data.corpus import Batch
from ana.model import Seq2SeqTransformer
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding
from ana.registry import baseline_parameters, build_model, count_parameters

MODEL_NAMES = (
    "context_power_prediction",
    "context_power_prediction_analogy",
    "context_power_features",
    "context_power_features_analogy",
    "context_power_features_fixed",
)
POSITIVE_EPSILON = 1e-6


def model_family(model_name: str) -> str:
    """Return the design family, validating the complete model identifier."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown context-power variant {model_name!r}")
    return "prediction" if model_name.startswith("context_power_prediction") else "features"


def uses_analogy(model_name: str) -> bool:
    """The fixed-power feature model is also an analogy-loss comparison."""
    model_family(model_name)
    return model_name.endswith("_analogy") or model_name == "context_power_features_fixed"


class ContextPowerEmbedding(LinearCompressedEmbedding):
    """Keep compact token lookup and modify only the output projection.

    Zero-filled head parameters consume no additional random draws. The
    powers are part of the prediction path, so ordinary cross-entropy trains
    them even when the analogy penalty is disabled.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
        *,
        code_dim: int,
        model_name: str,
    ) -> None:
        family = model_family(model_name)
        super().__init__(vocab_size, d_model, pad_id, code_dim=code_dim)
        self.power_family = family
        self.fixed_power = model_name == "context_power_features_fixed"
        self.power_groups = 1 if family == "prediction" else d_model // 2
        if self.fixed_power:
            self.register_parameter("power_weight", None)
            self.register_parameter("power_bias", None)
        else:
            self.power_weight = nn.Parameter(torch.zeros(self.power_groups, d_model))
            self.power_bias = nn.Parameter(torch.zeros(self.power_groups))

    def powers(self, hidden: Tensor) -> Tensor:
        if self.fixed_power:
            return hidden.new_ones(*hidden.shape[:-1], self.power_groups)
        raw_power = F.linear(hidden, self.power_weight, self.power_bias)
        if self.power_family == "prediction":
            return 0.25 + 0.5 * raw_power.sigmoid()
        return 1.0 + 0.5 * raw_power.tanh()

    def project_details(self, hidden: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return logits and pre-transform quantities for the training loss.

        Powers remain attached here: cross-entropy must train their heads.
        The trainer, not this projection, detaches powers in the analogy loss.
        """
        power = self.powers(hidden)
        if self.power_family == "prediction":
            raw_logits = super().project(hidden)
            logits = raw_logits * (power / 0.5)
            return logits, {"raw_logits": raw_logits, "power": power}

        positive = (F.softplus(hidden) + POSITIVE_EPSILON).reshape(
            *hidden.shape[:-1], self.power_groups, 2
        )
        correction = positive * torch.expm1((power.unsqueeze(-1) - 1.0) * positive.log())
        transformed = hidden + correction.flatten(start_dim=-2)
        logits = super().project(transformed)
        return logits, {"positive": positive, "power": power}

    def project(self, hidden: Tensor) -> Tensor:
        """Use the same power operation in full and cached prediction paths."""
        logits, _ = self.project_details(hidden)
        return logits


def _specification(corpus_name: str, model_name: str, full_shape: ModelConfig):
    family = model_family(model_name)
    if corpus_name in ("multi30k", "multi30k_enfr"):
        if full_shape.d_model != 128 or full_shape.d_ff != 256:
            raise ValueError("Retain the original Multi30k architecture.")
        shape, code_dim = replace(full_shape, d_ff=232), 96
    elif corpus_name == "cogs":
        shape, code_dim = replace(full_shape, d_model=400, d_ff=509), 160
    else:
        raise ValueError(f"unsupported corpus {corpus_name!r}")
    compact_parameters = (
        baseline_parameters(shape)
        - shape.vocab_size * shape.d_model
        + code_dim * (shape.vocab_size + shape.d_model)
    )
    power_groups = 1 if family == "prediction" else shape.d_model // 2
    head_parameters = (
        0 if model_name == "context_power_features_fixed" else power_groups * (shape.d_model + 1)
    )
    return shape, code_dim, compact_parameters, head_parameters


def model_summary(corpus_name: str, model_name: str, full_shape: ModelConfig) -> dict:
    shape, code_dim, compact_parameters, head_parameters = _specification(
        corpus_name, model_name, full_shape
    )
    parameters = compact_parameters + head_parameters
    full_parameters = baseline_parameters(full_shape)
    family = model_family(model_name)
    return {
        "corpus": corpus_name,
        "model": model_name,
        "model_config": asdict(shape),
        "code_dim": code_dim,
        "parameters": parameters,
        "compact_parameters": compact_parameters,
        "power_head_parameters": head_parameters,
        "full_parameters": full_parameters,
        "saved_parameters": full_parameters - parameters,
        "parameter_saving_fraction": (full_parameters - parameters) / full_parameters,
        "power_family": family,
        "learned_power": model_name != "context_power_features_fixed",
        "analogy_loss": uses_analogy(model_name),
        "initial_power": 0.5 if family == "prediction" else 1.0,
        "power_bounds": (
            [1.0, 1.0]
            if model_name == "context_power_features_fixed"
            else ([0.25, 0.75] if family == "prediction" else [0.5, 1.5])
        ),
        "inference_power": model_name != "context_power_features_fixed",
    }


def construct(corpus_name: str, model_name: str, full_shape: ModelConfig) -> Seq2SeqTransformer:
    """Preserve the existing baseline-then-compact-embedding initialization."""
    shape, code_dim, compact_parameters, head_parameters = _specification(
        corpus_name, model_name, full_shape
    )
    model = build_model("baseline", shape)
    model.embedding = ContextPowerEmbedding(
        shape.vocab_size,
        shape.d_model,
        shape.pad_id,
        code_dim=code_dim,
        model_name=model_name,
    )
    expected = compact_parameters + head_parameters
    actual = count_parameters(model)
    if actual != expected or actual >= baseline_parameters(full_shape):
        raise RuntimeError(f"invalid compact parameter count: {actual}, expected {expected}")
    return model


def power_forward(model: Seq2SeqTransformer, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
    """Match ordinary teacher forcing and cross-entropy, returning power details."""
    if not isinstance(model.embedding, ContextPowerEmbedding):
        raise TypeError("power_forward requires a context-power embedding")
    target_ids = model.shift_right(batch.labels)
    target_mask = (target_ids != model.config.pad_id).long()
    target_mask[:, 0] = 1
    memory = model.encode(batch.source_ids, batch.source_mask)
    hidden = model.decode(target_ids, memory, batch.source_mask, target_mask)
    logits, details = model.embedding.project_details(hidden)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=model.config.label_smoothing,
    )
    return loss, details
