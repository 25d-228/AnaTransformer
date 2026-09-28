"""Combine preprojection analogies with selective projection flexibility.

The positive-quartet operation is unchanged from analogy_projection_v3.
Low-rank branches and independently learned cross-attention projections are
ordinary readouts outside the numerical-analogy preservation claim.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from analogy_core import (
    ROLE_NAMES,
    AnalogyProjection,
    IndependentCrossProjection,
    collect_diagnostics,
    optimizer_parameter_groups,
    reset_diagnostics,
)

from ana.config import ModelConfig
from ana.model import Seq2SeqTransformer
from ana.nn.projection import QKVProjection, SharedQKV
from ana.registry import baseline_parameters, build_model, count_parameters


MODEL_LABELS = {
    "combo": "A: encoder role mixing + independent cross-attention Q",
    "combo_wide": "B: wider encoder role mixing + independent cross-attention Q",
    "compact_q": "C: encoder role mixing + compact cross-attention Q",
    "compact_qkv": "D: encoder role mixing + compact cross-attention Q/K/V",
    "combo_crosskv": "E: combined model + compact cross-attention K/V",
    "combo_selfqk": "F: combined model + compact decoder self-attention Q/K",
    "balanced_qkv": (
        "G: redistributed encoder and cross-attention Q/K/V mixing"
    ),
    "gated_qkv": "H: token-controlled compact cross-attention Q/K/V",
    "balanced_gated": "I: redistributed branches with token controls",
    "shared_bottleneck": "J: redistributed branches with shared bottlenecks",
    "diagonal_shortcuts": "K: analogy inputs with diagonal shortcuts",
    "compact_qkv_no_analogy": "L: model D without the analogy module",
    "balanced_qkv_no_analogy": "M: model G without the analogy module",
    "pre_crossq": "Preprojection analogy + independent cross-attention Q",
    "pre_lowrank": "Preprojection analogy + small independent role residuals",
}
MODEL_NAMES = tuple(MODEL_LABELS)
NEW_MODEL_NAMES = MODEL_NAMES[:13]
CORPUS_WIDTHS = {
    "multi30k": 128,
    "multi30k_enfr": 128,
    "cogs": 512,
    "iwslt14": 512,
}
INDEPENDENT_CROSS_QUERY = frozenset({
    "combo", "combo_wide", "combo_crosskv", "combo_selfqk", "pre_crossq",
})
CROSS_RESIDUAL_ROLES = {
    "compact_q": ("query",),
    "compact_qkv": ROLE_NAMES,
    "combo_crosskv": ("key", "value"),
    "balanced_qkv": ROLE_NAMES,
    "gated_qkv": ROLE_NAMES,
    "balanced_gated": ROLE_NAMES,
    "shared_bottleneck": ROLE_NAMES,
    "diagonal_shortcuts": ROLE_NAMES,
}
NO_ANALOGY_CONTROLS = {
    "compact_qkv_no_analogy": "compact_qkv",
    "balanced_qkv_no_analogy": "balanced_qkv",
}


def models_for_corpus(corpus_name: str) -> tuple[str, ...]:
    """Only IWSLT14 adds the two previously tested architectures."""
    if corpus_name not in CORPUS_WIDTHS:
        raise ValueError(f"unsupported corpus {corpus_name!r}")
    return MODEL_NAMES if corpus_name == "iwslt14" else NEW_MODEL_NAMES


def model_details(model_name: str) -> dict:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"unknown analogy-combination model {model_name!r}")
    if model_name in NO_ANALOGY_CONTROLS:
        details = model_details(NO_ANALOGY_CONTROLS[model_name])
        details.update({
            "name": model_name,
            "label": MODEL_LABELS[model_name],
            "sites": "none",
            "global_routing": None,
            "global_passes": None,
            "positive_encoding": None,
            "canonical_order": None,
            "local_routing": None,
            "common_multiplier": None,
            "invalid_group_action": None,
            "readout": (
                "shared projection plus ordinary low-rank role residuals"
            ),
            "correction_gain_initial": None,
            "encoder_residual_input": (
                "ordinary attention input; no analogy block"
            ),
            "routing_learning_rate_scale": None,
            "preservation_scope": None,
            "control_for": NO_ANALOGY_CONTROLS[model_name],
            "initialization": (
                "retain paired model weights; remove analogy modules"
            ),
        })
        details.pop("residual_budget", None)
        return details
    encoder_rank = (
        None if model_name in {"pre_crossq", "diagonal_shortcuts"}
        else "d_model / 8"
    )
    if model_name == "combo_wide":
        encoder_rank = "d_model / 4"
    elif model_name in {"balanced_qkv", "balanced_gated", "shared_bottleneck"}:
        encoder_rank = "d_model / 16"
    details = {
        "name": model_name,
        "label": MODEL_LABELS[model_name],
        "sites": "encoder",
        "roles": list(ROLE_NAMES),
        "global_routing": "input-dependent hard Beneš, positions restored",
        "global_passes": "one per encoder input stream, shared by roles",
        "positive_encoding": "softplus(x)+1e-6 and softplus(-x)+1e-6",
        "canonical_order": "0 < a < b <= c < d on both rails",
        "local_routing": "hard D4; negative rail uses reversal conjugation",
        "common_multiplier": "exp(0.1*tanh(logit)), shared by both rails",
        "invalid_group_action": "both rails bypass D4 and multiplier",
        "readout": "small signed input correction before the shared projection",
        "correction_gain_initial": 0.1,
        "independent_cross_attention": (
            "Q only; K/V shared"
            if model_name in INDEPENDENT_CROSS_QUERY else "none"
        ),
        "low_rank_residual": encoder_rank,
        "encoder_residual_input": (
            "role-conditioned analogy input" if encoder_rank else None
        ),
        "cross_residual_roles": list(CROSS_RESIDUAL_ROLES.get(model_name, ())),
        "decoder_self_residual_roles": (
            ["query", "key"] if model_name == "combo_selfqk" else []
        ),
        "decoder_residual_rank": (
            "d_model / 8"
            if model_name in CROSS_RESIDUAL_ROLES or model_name == "combo_selfqk"
            else None
        ),
        "decoder_residual_input": "ordinary attention input; no analogy block",
        "residual_bias": False,
        "residual_up_initial": "zero",
        "initial_function": "ordinary shared-QKV",
        "routing_learning_rate_scale": 0.1,
        "learned_power": False,
        "computed_power": False,
        "ordinary_embeddings": True,
        "preservation_scope": "each accepted canonical positive rail only",
    }
    if model_name == "balanced_qkv":
        details.update({
            "decoder_residual_rank": "d_model / 16",
            "residual_budget": "same total parameter count as pre_lowrank",
        })
    if model_name in {"gated_qkv", "balanced_gated"}:
        details.update({
            "cross_residual_gate": "2 * sigmoid(role-specific linear(input))",
            "cross_residual_gate_shape": "one scalar per role and input token",
            "cross_residual_gate_input": "Q: query input; K/V: source input",
            "cross_residual_gate_initial": 1.0,
            "cross_residual_gate_weight_initial": "zero",
            "cross_residual_gate_bias_initial": "zero",
        })
    if model_name in {"balanced_gated", "shared_bottleneck"}:
        details["decoder_residual_rank"] = "d_model / 16"
    if model_name == "shared_bottleneck":
        details.update({
            "residual_down_sharing": (
                "one Q/K/V down projection per attention site"
            ),
            "residual_up_sharing": (
                "three separate role-specific up projections"
            ),
        })
    elif model_name == "diagonal_shortcuts":
        details.update({
            "decoder_residual_rank": None,
            "residual_up_initial": None,
            "diagonal_residual_initial": "zero",
            "diagonal_residual_sites": ["encoder", "cross_attention"],
            "diagonal_residual_roles": list(ROLE_NAMES),
            "encoder_residual_input": "role-conditioned analogy input",
            "diagonal_residual": "role-specific channel scale times input",
            "existing_output_diagonals": "retained unchanged",
        })
    return details


class RankedAnalogyProjection(AnalogyProjection):
    """The unchanged analogy projection with an explicit role-residual rank."""

    def __init__(self, original: SharedQKV, rank: int) -> None:
        super().__init__(original, low_rank=False)
        d_model = self.shared.in_features
        if not 1 <= rank <= d_model:
            raise ValueError("residual rank must be between one and d_model")
        self.low_rank = True
        self.down = nn.ModuleDict({
            name: nn.Linear(d_model, rank, bias=False) for name in ROLE_NAMES
        })
        self.up = nn.ModuleDict({
            name: nn.Linear(rank, d_model, bias=False) for name in ROLE_NAMES
        })
        for projection in self.up.values():
            nn.init.zeros_(projection.weight)


class LightweightAnalogyProjection(AnalogyProjection):
    """Unchanged protected inputs with shared bottlenecks or diagonal skips."""

    def __init__(self, original: SharedQKV, rank: int | None) -> None:
        super().__init__(original, low_rank=False)
        d_model = self.shared.in_features
        self.rank = rank
        if rank is None:
            self.shortcuts = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(d_model)) for name in ROLE_NAMES
            })
        else:
            if not 1 <= rank <= d_model:
                raise ValueError(
                    "residual rank must be between one and d_model"
                )
            self.down = nn.Linear(d_model, rank, bias=False)
            self.up = nn.ModuleDict({
                name: nn.Linear(rank, d_model, bias=False)
                for name in ROLE_NAMES
            })
            for projection in self.up.values():
                nn.init.zeros_(projection.weight)

    def _project(
        self, values: Tensor, mask: Tensor, role_names: tuple[str, ...]
    ) -> tuple[Tensor, ...]:
        prepared = self.preparation(values, mask)
        outputs = []
        for name in role_names:
            change = self.changes[name](prepared).to(values.dtype)
            conditioned = values + change
            output = self.roles[name](self.shared(conditioned), mask)
            if self.rank is None:
                residual = self.shortcuts[name] * conditioned
            else:
                residual = self.up[name](self.down(conditioned))
            outputs.append(output + residual)
        return tuple(outputs)


class OrdinaryRankedProjection(QKVProjection):
    """Keep paired-model weights without registering analogy components."""

    def __init__(self, original: RankedAnalogyProjection) -> None:
        super().__init__()
        self.shared = original.shared
        self.roles = original.roles
        self.down = original.down
        self.up = original.up

    def _project(
        self, values: Tensor, mask: Tensor, role_names: tuple[str, ...]
    ) -> tuple[Tensor, ...]:
        projected = self.shared(values)
        return tuple(
            self.roles[name](projected, mask)
            + self.up[name](self.down[name](values))
            for name in role_names
        )

    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor:
        return self._project(query_input, query_mask, ("query",))[0]

    def key_value(
        self, kv_input: Tensor, kv_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._project(kv_input, kv_mask, ("key", "value"))

    def forward(
        self, query_input: Tensor, kv_input: Tensor,
        query_mask: Tensor, kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if kv_input is query_input:
            return self._project(query_input, query_mask, ROLE_NAMES)
        query = self.query_only(query_input, query_mask)
        key, value = self.key_value(kv_input, kv_mask)
        return query, key, value


class ResidualProjection(QKVProjection):
    """Small role-specific residuals with the original generation interface."""

    def __init__(
        self, original: QKVProjection, d_model: int,
        role_names: tuple[str, ...], rank: int,
    ) -> None:
        super().__init__()
        self.original = original
        self.down = nn.ModuleDict({
            name: nn.Linear(d_model, rank, bias=False) for name in role_names
        })
        self.up = nn.ModuleDict({
            name: nn.Linear(rank, d_model, bias=False) for name in role_names
        })
        for projection in self.up.values():
            nn.init.zeros_(projection.weight)

    def _adjust(self, name: str, output: Tensor, values: Tensor) -> Tensor:
        if name in self.down:
            return output + self.up[name](self.down[name](values))
        return output

    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor:
        query = self.original.query_only(query_input, query_mask)
        return self._adjust("query", query, query_input)

    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
        key, value = self.original.key_value(kv_input, kv_mask)
        return (
            self._adjust("key", key, kv_input),
            self._adjust("value", value, kv_input),
        )

    def forward(
        self, query_input: Tensor, kv_input: Tensor,
        query_mask: Tensor, kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query, key, value = self.original(
            query_input, kv_input, query_mask, kv_mask
        )
        return (
            self._adjust("query", query, query_input),
            self._adjust("key", key, kv_input),
            self._adjust("value", value, kv_input),
        )


class GatedResidualProjection(ResidualProjection):
    """Per-token low-rank readout gates, outside the analogy block."""

    def __init__(
        self, original: QKVProjection, d_model: int,
        role_names: tuple[str, ...], rank: int,
    ) -> None:
        super().__init__(original, d_model, role_names, rank)
        # Gate creation must not change later non-gate weights versus model D.
        with torch.random.fork_rng(devices=[]):
            self.gates = nn.ModuleDict({
                name: nn.Linear(d_model, 1) for name in role_names
            })
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def _adjust(self, name: str, output: Tensor, values: Tensor) -> Tensor:
        if name not in self.down:
            return output
        residual = self.up[name](self.down[name](values))
        gate = 2.0 * torch.sigmoid(self.gates[name](values))
        return output + gate * residual


class SharedBottleneckResidualProjection(ResidualProjection):
    """Share a site's input bottleneck; keep separate role-specific outputs."""

    def __init__(
        self, original: QKVProjection, d_model: int, rank: int
    ) -> None:
        # Skip the independent bottlenecks in ResidualProjection.__init__.
        QKVProjection.__init__(self)
        self.original = original
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.ModuleDict({
            name: nn.Linear(rank, d_model, bias=False) for name in ROLE_NAMES
        })
        for projection in self.up.values():
            nn.init.zeros_(projection.weight)

    def _adjust(self, name: str, output: Tensor, values: Tensor) -> Tensor:
        return output + self.up[name](self.down(values))


class DiagonalResidualProjection(ResidualProjection):
    """Add channel-scaled inputs without introducing another dense matrix."""

    def __init__(self, original: QKVProjection, d_model: int) -> None:
        QKVProjection.__init__(self)
        self.original = original
        self.shortcuts = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(d_model)) for name in ROLE_NAMES
        })

    def _adjust(self, name: str, output: Tensor, values: Tensor) -> Tensor:
        return output + self.shortcuts[name] * values


def construct(
    corpus_name: str, model_name: str, full_shape: ModelConfig
) -> Seq2SeqTransformer:
    """Keep full corpus shapes and ordinary embeddings, below full-model size."""
    model_details(model_name)
    if corpus_name not in CORPUS_WIDTHS:
        raise ValueError(f"unsupported corpus {corpus_name!r}")
    if full_shape.d_model != CORPUS_WIDTHS[corpus_name]:
        raise ValueError("this study keeps the original corpus backbone width")
    if model_name in NO_ANALOGY_CONTROLS:
        model = construct(
            corpus_name, NO_ANALOGY_CONTROLS[model_name], full_shape
        )
        for layer in model.encoder:
            original = layer.self_attention.projection
            layer.self_attention.projection = OrdinaryRankedProjection(
                original
            )
        return model
    model = build_model("shared_qkv", full_shape)
    d_model = full_shape.d_model
    with torch.random.fork_rng(devices=[]):
        for layer in model.encoder:
            original = layer.self_attention.projection
            if not isinstance(original, SharedQKV):
                raise TypeError("expected the ordinary shared projection")
            if model_name in {"pre_crossq", "pre_lowrank"}:
                projection = AnalogyProjection(
                    original, low_rank=model_name == "pre_lowrank"
                )
            elif model_name in {"shared_bottleneck", "diagonal_shortcuts"}:
                rank = (
                    d_model // 16 if model_name == "shared_bottleneck" else None
                )
                projection = LightweightAnalogyProjection(original, rank)
            else:
                rank_divisor = 4 if model_name == "combo_wide" else 8
                if model_name in {"balanced_qkv", "balanced_gated"}:
                    rank_divisor = 16
                rank = d_model // rank_divisor
                projection = RankedAnalogyProjection(original, rank)
            layer.self_attention.projection = projection
        for layer in model.decoder:
            cross = layer.cross_attention.projection
            if not isinstance(cross, SharedQKV):
                raise TypeError("expected the ordinary shared cross projection")
            if model_name in INDEPENDENT_CROSS_QUERY:
                cross = IndependentCrossProjection(cross, separate_value=False)
            if model_name == "shared_bottleneck":
                cross = SharedBottleneckResidualProjection(
                    cross, d_model, d_model // 16
                )
            elif model_name == "diagonal_shortcuts":
                cross = DiagonalResidualProjection(cross, d_model)
            elif model_name in CROSS_RESIDUAL_ROLES:
                residual_type = (
                    GatedResidualProjection
                    if model_name in {"gated_qkv", "balanced_gated"}
                    else ResidualProjection
                )
                rank_divisor = (
                    16 if model_name in {"balanced_qkv", "balanced_gated"}
                    else 8
                )
                cross = residual_type(
                    cross, d_model, CROSS_RESIDUAL_ROLES[model_name],
                    d_model // rank_divisor,
                )
            layer.cross_attention.projection = cross
            if model_name == "combo_selfqk":
                layer.self_attention.projection = ResidualProjection(
                    layer.self_attention.projection, d_model,
                    ("query", "key"), d_model // 8,
                )
    if count_parameters(model) >= baseline_parameters(full_shape):
        raise ValueError(
            "the analogy model must stay smaller than the full Transformer"
        )
    return model
