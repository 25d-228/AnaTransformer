"""Give preserved numerical analogies specific jobs in shared attention.

The positive-rail operation is unchanged from model D. Branch placement,
signed or summed rail readouts, and subsequent projections are outside its
preservation claim. No numerical-analogy power is computed or learned.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from analogy_core import (
    ROLE_NAMES,
    AnalogyChange,
    AnalogyProjection,
    CanonicalInput,
)
from ana.config import ModelConfig
from ana.model import Seq2SeqTransformer
from ana.nn.projection import QKVProjection
from ana.registry import baseline_parameters, count_parameters
from experiments.permutations.analogy_combined_v4 import models as previous


MODEL_LABELS = {
    "small_branch": "N1: analogy-specialized encoder branches",
    "small_branch_cross": "N2: analogy-specialized encoder and cross branches",
    "rail_readout": "N3: model D with signed and magnitude rail changes",
    "small_branch_rail": (
        "N4: specialized branches with magnitude rail changes"
    ),
    "decoder_query": "N5: model D plus decoder self-attention query analogy",
    "query_only": "N6: query-only encoder and decoder self-attention analogy",
}
MODEL_NAMES = tuple(MODEL_LABELS)
CORPUS_NAMES = ("multi30k", "multi30k_enfr", "cogs")
SMALL_BRANCH_MODELS = frozenset({
    "small_branch", "small_branch_cross", "small_branch_rail",
})
RAIL_MODELS = frozenset({"rail_readout", "small_branch_rail"})
DECODER_QUERY_MODELS = frozenset({"decoder_query", "query_only"})

reset_diagnostics = previous.reset_diagnostics
optimizer_parameter_groups = previous.optimizer_parameter_groups


class AnalogyReadout(AnalogyChange):
    """Retain D's protected rails; change only their external readout."""

    def __init__(
        self, original: AnalogyChange, *, unit_gain: bool, rail_readout: bool,
    ) -> None:
        d_model = original.router.in_features
        super().__init__(d_model)
        self.load_state_dict(original.state_dict())
        if unit_gain:
            del self.gain_logit
        if rail_readout:
            self.rail_beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, prepared: CanonicalInput) -> Tensor:
        logits = self.router(prepared.token_input).unsqueeze(-2)
        logits = logits + self.local_router(prepared.local_input)
        routes = logits.argmax(dim=-1)
        magnitude = (0.1 * self.magnitude(prepared.local_input).tanh()).exp()
        positive = prepared.positive.gather(-1, self.permutations[routes])
        negative = prepared.negative.gather(
            -1, self.conjugate_permutations[routes]
        )
        if self.training and torch.is_grad_enabled():
            matrix = torch.einsum(
                "...k,kij->...ij", logits.softmax(dim=-1),
                self.permutation_matrices,
            )
            soft_positive = torch.einsum(
                "...ij,...j->...i", matrix, prepared.positive.detach()
            )
            soft_negative = torch.einsum(
                "...ij,...j->...i", matrix.flip((-2, -1)),
                prepared.negative.detach(),
            )
            positive = positive + (soft_positive - soft_positive.detach())
            negative = negative + (soft_negative - soft_negative.detach())
        positive_change = positive * magnitude - prepared.positive
        negative_change = negative * magnitude - prepared.negative
        eligible = prepared.eligible.unsqueeze(-1)
        positive_change = torch.where(eligible, positive_change, 0.0)
        negative_change = torch.where(eligible, negative_change, 0.0)
        positive_change = positive_change.gather(-1, prepared.inverse_local)
        negative_change = negative_change.gather(
            -1, 3 - prepared.inverse_local
        )
        restored = (positive_change - negative_change).flatten(-2).gather(
            -1, prepared.inverse_global
        )
        if hasattr(self, "rail_beta"):
            summed = (positive_change + negative_change).flatten(-2).gather(
                -1, prepared.inverse_global
            )
            restored = restored + self.rail_beta.tanh() * summed
        if hasattr(self, "gain_logit"):
            restored = self.gain_logit.sigmoid() * restored
        if self.diagnostics_enabled:
            routed = prepared.eligible & (routes != 0)
            scaled = prepared.eligible & (magnitude.squeeze(-1) != 1)
            self._record(
                prepared.real.sum(), prepared.eligible.sum(), routed.sum(),
                scaled.sum(), (routed | scaled).sum(),
            )
        return restored


class SpecializedProjection(AnalogyProjection):
    """Reuse D's projections while choosing where analogy contributes."""

    def __init__(
        self, original: AnalogyProjection, *, small_branch: bool = False,
        rail_readout: bool = False, query_only: bool = False,
    ) -> None:
        QKVProjection.__init__(self)
        self.shared = original.shared
        self.roles = original.roles
        self.preparation = original.preparation
        active_roles = ("query",) if query_only else ROLE_NAMES
        self.changes = nn.ModuleDict({
            name: AnalogyReadout(
                original.changes[name], unit_gain=small_branch,
                rail_readout=rail_readout,
            )
            for name in active_roles
        })
        self.low_rank = original.low_rank
        self.small_branch = small_branch
        if self.low_rank:
            self.down = original.down
            self.up = original.up
        if small_branch and not self.low_rank:
            raise ValueError(
                "specialized branches require existing small matrices"
            )

    def _project(
        self, values: Tensor, mask: Tensor, role_names: tuple[str, ...]
    ) -> tuple[Tensor, ...]:
        has_analogy = any(name in self.changes for name in role_names)
        prepared = self.preparation(values, mask) if has_analogy else None
        needs_ordinary = self.small_branch or any(
            name not in self.changes for name in role_names
        )
        shared = self.shared(values) if needs_ordinary else None
        outputs = []
        for name in role_names:
            conditioned = values
            if name in self.changes:
                conditioned = values + self.changes[name](prepared).to(
                    values.dtype
                )
            if self.small_branch or name not in self.changes:
                projected = shared
            else:
                projected = self.shared(conditioned)
            output = self.roles[name](projected, mask)
            if self.low_rank:
                output = output + self.up[name](self.down[name](conditioned))
            outputs.append(output)
        return tuple(outputs)


def models_for_corpus(corpus_name: str) -> tuple[str, ...]:
    """The two Multi30k directions and COGS are the entire batch."""
    if corpus_name not in CORPUS_NAMES:
        raise ValueError(f"unsupported specialization corpus {corpus_name!r}")
    return MODEL_NAMES


def model_details(model_name: str) -> dict:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"unknown analogy specialization {model_name!r}")
    small_branch = model_name in SMALL_BRANCH_MODELS
    rail_readout = model_name in RAIL_MODELS
    details = previous.model_details("compact_qkv")
    details.update({
        "name": model_name,
        "label": MODEL_LABELS[model_name],
        "parent_model": "analogy_combined_v4/compact_qkv",
        "analogy_removed_reference": (
            "analogy_combined_v4/compact_qkv_no_analogy"
        ),
        "initialization": (
            "retain model D backbone and ordinary branch weights"
        ),
        "encoder_shared_projection_input": (
            "original unchanged attention input" if small_branch
            else "role-conditioned analogy input for active roles"
        ),
        "encoder_residual_input": (
            "full restored analogy view" if small_branch
            else "role-conditioned analogy input for active roles"
        ),
        "readout": (
            "full analogy view only in existing small role branches"
            if small_branch
            else "D's gained correction before shared projection"
        ),
        "correction_gain_initial": None if small_branch else 0.1,
        "global_passes": (
            "one per active input stream, shared by its active roles"
        ),
        "rail_readout": (
            "signed change + tanh(beta) * summed rail change, after restoring "
            "original coordinates and before any scalar correction gain"
            if rail_readout else "signed rail change only"
        ),
        "rail_beta_shape": (
            "one per original feature and role" if rail_readout else None
        ),
        "rail_beta_initial": 0.0 if rail_readout else None,
        "decoder_self_analogy_roles": (
            ["query"] if model_name in DECODER_QUERY_MODELS else []
        ),
    })
    if model_name == "small_branch_cross":
        details.update({
            "sites": "encoder and decoder cross-attention",
            "cross_shared_projection_input": (
                "original unchanged attention input"
            ),
            "decoder_residual_input": (
                "full restored analogy view in cross branches"
            ),
        })
    elif model_name in DECODER_QUERY_MODELS:
        details["sites"] = "encoder and decoder self-attention query"
    if model_name == "query_only":
        details["roles"] = ["query"]
        details["encoder_key_value_input"] = (
            "original unchanged attention input"
        )
    return details


def construct(
    corpus_name: str, model_name: str, full_shape: ModelConfig,
) -> Seq2SeqTransformer:
    models_for_corpus(corpus_name)
    model_details(model_name)
    model = previous.construct(corpus_name, "compact_qkv", full_shape)
    with torch.random.fork_rng(devices=[]):
        if model_name != "decoder_query":
            for layer in model.encoder:
                layer.self_attention.projection = SpecializedProjection(
                    layer.self_attention.projection,
                    small_branch=model_name in SMALL_BRANCH_MODELS,
                    rail_readout=model_name in RAIL_MODELS,
                    query_only=model_name == "query_only",
                )
        if model_name == "small_branch_cross":
            for layer in model.decoder:
                original = layer.cross_attention.projection
                projection = AnalogyProjection(original.original)
                projection.low_rank = True
                projection.down = original.down
                projection.up = original.up
                layer.cross_attention.projection = SpecializedProjection(
                    projection, small_branch=True,
                )
        if model_name in DECODER_QUERY_MODELS:
            for layer in model.decoder:
                original = AnalogyProjection(layer.self_attention.projection)
                layer.self_attention.projection = SpecializedProjection(
                    original, query_only=True,
                )
    if count_parameters(model) >= baseline_parameters(full_shape):
        raise ValueError(
            "the specialized model must stay below full Transformer size"
        )
    return model


def collect_diagnostics(
    model: nn.Module, reset: bool = False, disable: bool = True,
) -> dict:
    result = previous.collect_diagnostics(model, reset=reset, disable=disable)
    rail_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, AnalogyReadout) and hasattr(module, "rail_beta"):
            values = module.rail_beta.detach().tanh().float()
            rail_weights[name] = {
                "mean": float(values.mean().cpu()),
                "mean_absolute": float(values.abs().mean().cpu()),
                "minimum": float(values.min().cpu()),
                "maximum": float(values.max().cpu()),
            }
    result["rail_readout_weights"] = rail_weights
    return result
