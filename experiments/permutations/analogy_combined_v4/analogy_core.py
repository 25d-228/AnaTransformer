"""Shared projections guided by input-routed positive numerical analogies.

Only the hard D4 operation and common positive scale inside each accepted
positive rail carry the preservation claim. Routing, sorting, signed rail
differences, residual gains, linear projections, and mixers are outside it.
No analogy power is learned or computed.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ana.nn.permutations.grouping import D4_PERMUTATIONS
from ana.nn.projection import QKVProjection, SharedQKV

POSITIVE_EPSILON = 1e-6
ROLE_NAMES = ("query", "key", "value")


class _DiagnosticModule(nn.Module):
    diagnostic_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        super().__init__()
        self.diagnostics_enabled = False
        self.register_buffer(
            "diagnostic_counts",
            torch.zeros(len(self.diagnostic_names), dtype=torch.float64),
            persistent=False,
        )

    def _record(self, *values: Tensor) -> None:
        if self.diagnostics_enabled:
            with torch.no_grad():
                self.diagnostic_counts.add_(torch.stack(values).to(torch.float64))


class DynamicBenes(_DiagnosticModule):
    """Token-local hard switches, with sigmoid straight-through gradients."""

    diagnostic_names = ("global_switches", "global_swaps")

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < 4 or d_model & (d_model - 1):
            raise ValueError("Beneš requires a power-of-two feature width")
        strides = tuple(1 << bit for bit in range(d_model.bit_length() - 1))
        self.strides = strides + strides[-2::-1]
        index = torch.arange(len(self.strides) * (d_model // 2))
        jitter = ((index * 37) % 101).float() / 50.0 - 1.0
        self.bias = nn.Parameter(
            (0.02 * jitter).reshape(len(self.strides), d_model // 2)
        )
        self.context = nn.Parameter(
            torch.tensor([0.05, -0.05]).repeat(len(self.strides), 1)
        )

    def forward(self, values: Tensor, pad_mask: Tensor) -> tuple[Tensor, Tensor]:
        shape = values.shape
        positions = torch.arange(shape[-1], device=values.device).expand(shape)
        indices = positions
        real = pad_mask.bool()[..., None, None]
        for stage, stride in enumerate(self.strides):
            pairs = values.reshape(*shape[:-1], -1, 2, stride)
            left, right = pairs.unbind(dim=-2)
            rms = (0.5 * (left.square() + right.square()) + 1e-6).sqrt()
            logits = self.bias[stage].reshape(-1, stride) + (
                self.context[stage, 0] * left + self.context[stage, 1] * right
            ) / rms
            swaps = logits >= 0
            flipped = pairs.flip(-2)
            hard = torch.where(swaps.unsqueeze(-2), flipped, pairs)
            if self.training and torch.is_grad_enabled():
                probability = logits.sigmoid().unsqueeze(-2)
                hard = hard + (probability - probability.detach()) * (
                    flipped - pairs
                ).detach()
            values = hard.reshape(shape)
            paired_indices = indices.reshape(pairs.shape)
            indices = torch.where(
                swaps.unsqueeze(-2), paired_indices.flip(-2), paired_indices
            ).reshape(shape)
            if self.diagnostics_enabled:
                self._record(
                    real.expand_as(swaps).sum(), (swaps & real).sum()
                )
        inverse = torch.empty_like(indices).scatter_(-1, indices, positions)
        return values, inverse


def valid_canonical_quartets(values: Tensor) -> Tensor:
    """Allow the middle tie; reject either outer tie and nonfinite values."""
    a, b, c, d = values.unbind(dim=-1)
    return (
        torch.isfinite(values).all(dim=-1)
        & (a > 0)
        & (a < b)
        & (b <= c)
        & (c < d)
    )


@dataclass
class CanonicalInput:
    positive: Tensor
    negative: Tensor
    inverse_local: Tensor
    inverse_global: Tensor
    local_input: Tensor
    token_input: Tensor
    eligible: Tensor
    real: Tensor


class CanonicalPreparation(nn.Module):
    """Share the expensive hard routing and sorting across all three roles."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.shuffle = DynamicBenes(d_model)

    def forward(self, values: Tensor, pad_mask: Tensor) -> CanonicalInput:
        base = values.float()
        shuffled, inverse_global = self.shuffle(base, pad_mask)
        shape = shuffled.shape
        positive = (F.softplus(shuffled) + POSITIVE_EPSILON).reshape(
            *shape[:-1], -1, 4
        )
        negative = (F.softplus(-shuffled) + POSITIVE_EPSILON).reshape_as(positive)
        positive, sorting = positive.sort(dim=-1)
        negative = negative.gather(-1, sorting.flip(-1))
        inverse_local = torch.empty_like(sorting).scatter_(
            -1, sorting, torch.arange(4, device=sorting.device).expand_as(sorting)
        )
        real = pad_mask.bool().unsqueeze(-1).expand(positive.shape[:-1])
        eligible = (
            valid_canonical_quartets(positive)
            & valid_canonical_quartets(negative)
            & real
        )
        canonical_signed = positive - negative.flip(-1)
        rms = (canonical_signed.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        return CanonicalInput(
            positive=positive,
            negative=negative,
            inverse_local=inverse_local,
            inverse_global=inverse_global,
            local_input=canonical_signed / rms,
            token_input=F.layer_norm(base, (shape[-1],)),
            eligible=eligible,
            real=real,
        )


class AnalogyChange(_DiagnosticModule):
    """One role's preserving positive operation and external signed readout."""

    diagnostic_names = (
        "quartets_seen", "eligible_quartets", "nonidentity_routes",
        "scaled_quartets", "changed_quartets",
    )

    def __init__(self, d_model: int, *, mixer: bool = False) -> None:
        super().__init__()
        permutations = torch.tensor(D4_PERMUTATIONS)
        conjugates = 3 - permutations.flip(-1)
        self.register_buffer("permutations", permutations)
        self.register_buffer("conjugate_permutations", conjugates)
        self.register_buffer(
            "permutation_matrices", F.one_hot(permutations, num_classes=4).float()
        )
        self.router = nn.Linear(d_model, 8)
        self.local_router = nn.Linear(4, 8, bias=False)
        self.magnitude = nn.Linear(4, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        with torch.no_grad():
            self.router.bias[0] = 0.5
        nn.init.zeros_(self.local_router.weight)
        nn.init.zeros_(self.magnitude.weight)
        nn.init.zeros_(self.magnitude.bias)
        if mixer:
            self.mixer = nn.Parameter(0.1 * torch.eye(4).repeat(d_model // 4, 1, 1))
        else:
            self.gain_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

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
                "...k,kij->...ij", logits.softmax(dim=-1), self.permutation_matrices
            )
            soft_positive = torch.einsum(
                "...ij,...j->...i", matrix, prepared.positive.detach()
            )
            soft_negative = torch.einsum(
                "...ij,...j->...i",
                matrix.flip((-2, -1)), prepared.negative.detach(),
            )
            positive = positive + (soft_positive - soft_positive.detach())
            negative = negative + (soft_negative - soft_negative.detach())
        positive_change = positive * magnitude - prepared.positive
        negative_change = negative * magnitude - prepared.negative
        eligible = prepared.eligible.unsqueeze(-1)
        positive_change = torch.where(eligible, positive_change, 0.0)
        negative_change = torch.where(eligible, negative_change, 0.0)
        positive_change = positive_change.gather(-1, prepared.inverse_local)
        negative_change = negative_change.gather(-1, 3 - prepared.inverse_local)
        signed_change = positive_change - negative_change
        if hasattr(self, "mixer"):
            signed_change = torch.einsum("gij,...gj->...gi", self.mixer, signed_change)
        else:
            signed_change = self.gain_logit.sigmoid() * signed_change
        restored = signed_change.flatten(-2).gather(-1, prepared.inverse_global)
        if self.diagnostics_enabled:
            routed = prepared.eligible & (routes != 0)
            scaled = prepared.eligible & (magnitude.squeeze(-1) != 1)
            self._record(
                prepared.real.sum(), prepared.eligible.sum(), routed.sum(),
                scaled.sum(), (routed | scaled).sum(),
            )
        return restored


class AnalogyProjection(QKVProjection):
    """Three input transformations reuse one dense attention projection."""

    def __init__(
        self, original: SharedQKV, *, low_rank: bool = False,
        mixer: bool = False, clean_residual_input: bool = False,
    ) -> None:
        super().__init__()
        d_model = original.shared.in_features
        self.shared = original.shared
        self.roles = nn.ModuleDict({
            name: getattr(original, f"{name}_role") for name in ROLE_NAMES
        })
        self.preparation = CanonicalPreparation(d_model)
        self.changes = nn.ModuleDict({
            name: AnalogyChange(d_model, mixer=mixer) for name in ROLE_NAMES
        })
        self.post_projection = mixer
        self.low_rank = low_rank
        self.clean_residual_input = clean_residual_input
        if low_rank:
            rank = d_model // 8
            self.down = nn.ModuleDict({
                name: nn.Linear(d_model, rank, bias=False) for name in ROLE_NAMES
            })
            self.up = nn.ModuleDict({
                name: nn.Linear(rank, d_model, bias=False) for name in ROLE_NAMES
            })
            for projection in self.up.values():
                nn.init.zeros_(projection.weight)

    def _project(
        self, values: Tensor, mask: Tensor, role_names: tuple[str, ...]
    ) -> tuple[Tensor, ...]:
        base = self.shared(values) if self.post_projection else values
        prepared = self.preparation(base, mask)
        outputs = []
        for name in role_names:
            conditioned = base + self.changes[name](prepared).to(base.dtype)
            projected = (
                conditioned if self.post_projection else self.shared(conditioned)
            )
            output = self.roles[name](projected, mask)
            if self.low_rank:
                residual_input = (
                    values if self.clean_residual_input else conditioned
                )
                output = output + self.up[name](self.down[name](residual_input))
            outputs.append(output)
        return tuple(outputs)

    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor:
        return self._project(query_input, query_mask, ("query",))[0]

    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
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


class IndependentCrossProjection(QKVProjection):
    """Relax cross-attention sharing while retaining its initial function."""

    def __init__(self, original: SharedQKV, *, separate_value: bool) -> None:
        super().__init__()
        self.query = copy.deepcopy(original.shared)
        self.key = original.shared
        self.value = copy.deepcopy(original.shared) if separate_value else self.key
        self.query_role = original.query_role
        self.key_role = original.key_role
        self.value_role = original.value_role

    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor:
        return self.query_role(self.query(query_input), query_mask)

    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
        key_base = self.key(kv_input)
        value_base = key_base if self.value is self.key else self.value(kv_input)
        return self.key_role(key_base, kv_mask), self.value_role(value_base, kv_mask)

    def forward(
        self, query_input: Tensor, kv_input: Tensor,
        query_mask: Tensor, kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key, value = self.key_value(kv_input, kv_mask)
        return self.query_only(query_input, query_mask), key, value


def optimizer_parameter_groups(model: nn.Module) -> list[dict]:
    """Use the original schedule, with routing controllers slowed to 0.1."""
    routing_ids = set()
    for module in model.modules():
        if isinstance(module, DynamicBenes):
            routing_ids.update(id(parameter) for parameter in module.parameters())
        elif isinstance(module, AnalogyChange):
            for controller in (module.router, module.local_router, module.magnitude):
                routing_ids.update(
                    id(parameter) for parameter in controller.parameters()
                )
    ordinary, routing = [], []
    for parameter in model.parameters():
        (routing if id(parameter) in routing_ids else ordinary).append(parameter)
    return [
        {"params": ordinary, "lr_scale": 1.0, "group_name": "backbone_and_readout"},
        {"params": routing, "lr_scale": 0.1, "group_name": "routing_controllers"},
    ]


def reset_diagnostics(model: nn.Module, enabled: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, _DiagnosticModule):
            module.diagnostic_counts.zero_()
            module.diagnostics_enabled = enabled


def collect_diagnostics(
    model: nn.Module, reset: bool = False, disable: bool = True
) -> dict:
    totals: dict[str, int] = {}
    by_module, correction_gains, mixer_norms = {}, {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, _DiagnosticModule):
            continue
        values = module.diagnostic_counts.detach().cpu().tolist()
        counts = dict(zip(module.diagnostic_names, (int(value) for value in values)))
        by_module[name] = counts
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        if isinstance(module, AnalogyChange):
            if hasattr(module, "gain_logit"):
                correction_gains[name] = float(
                    module.gain_logit.detach().sigmoid().cpu()
                )
            if hasattr(module, "mixer"):
                mixer_norms[name] = float(
                    module.mixer.detach().square().sum(dim=(-2, -1)).sqrt().mean().cpu()
                )
        if reset:
            module.diagnostic_counts.zero_()
        if disable:
            module.diagnostics_enabled = False
    fractions = {}
    for numerator, denominator, label in (
        ("eligible_quartets", "quartets_seen", "eligible_fraction"),
        ("nonidentity_routes", "eligible_quartets", "routed_fraction_of_eligible"),
        ("changed_quartets", "eligible_quartets", "changed_fraction_of_eligible"),
        ("global_swaps", "global_switches", "global_swap_fraction"),
    ):
        count = totals.get(denominator, 0)
        fractions[label] = totals.get(numerator, 0) / count if count else None
    return {
        "counts": totals,
        "fractions": fractions,
        "by_module": by_module,
        "correction_gains": correction_gains,
        "mixer_mean_frobenius_norms": mixer_norms,
    }
