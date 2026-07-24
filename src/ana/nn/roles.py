"""The per-role operators: how one shared projection becomes a query, a key or a value.

This is the axis the experiment varies. Kowsher et al. use a diagonal, which can only
rescale each channel on its own. We test a doubly-stochastic mixing over groups of four,
wrapped around that same diagonal, so that turning the mixing off recovers their model
exactly rather than approximately.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ana.nn.grouping import (
    GROUP_SIZE,
    N_PERMUTATIONS,
    Grouping,
    PerGroupFeatureGrouping,
    permutation_matrices,
)

MAGNITUDE_CLAMP = 5.0

# The exponent the powered mixer routes, and the floor held under |z| before it is raised to
# that exponent. The derivative of |z|**q is q*|z|**(q-1), which grows without bound as |z|
# approaches zero for q < 1, so |z| is clamped from below.
POWER_RANGE = (0.25, 4.0)
POWER_FLOOR = 1e-4


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def signed_power(z: Tensor, exponent: Tensor) -> Tensor:
    """sign(z) * |z| ** exponent above POWER_FLOOR, and linear in |z| below it.

    Defined for negative z, where z ** exponent is not, and odd in z.

    Below POWER_FLOOR the power is replaced by the line through the origin and
    (POWER_FLOOR, POWER_FLOOR ** exponent). The two branches meet there, and the derivative
    below it is POWER_FLOOR ** (exponent - 1) rather than |z| ** (exponent - 1), which grows
    without bound as |z| approaches zero for exponent < 1.

    At exponent = 1 both branches return |z|, so signed_power(z, 1) is z for every z.

    THE TWO BRANCHES ARE ONE EXPRESSION, AND WRITING THEM AS TWO COSTS MEMORY.

    Let `base = max(|z|, FLOOR)`. Above the floor `base` is `|z|`, so `|z|/base` is exactly one
    and `base**q * (|z|/base)` is `|z|**q`. Below it `base` is `FLOOR`, so the same expression is
    `FLOOR**q * |z|/FLOOR`, which is `|z| * FLOOR**(q-1)` -- the linear branch. One expression
    covers both.

    The obvious spelling -- compute the power, compute the line, and `torch.where` between them --
    holds THREE extra tensors of the size of `z` for the backward pass: the line, the mask, and
    the selection. This runs twice per role and at three roles a site, so on a six-layer 512-wide
    model those three tensors are several gigabytes, and it was enough to exhaust a 24 GB card.
    """
    size = z.abs()
    base = size.clamp(min=POWER_FLOOR)
    return torch.sign(z) * base.pow(exponent) * (size / base)


class RoleTransform(nn.Module, ABC):
    """Maps the shared projection into one attention role."""

    @staticmethod
    def extra_parameters(d_model: int, grouping: Grouping | None) -> int:
        """Parameters this transform holds beyond the `DiagonalRescale` it replaces.

        `registry.shared_parameters` counts a model without building one, so the count is
        declared by the transform that holds the parameters. `tests/test_params.py` checks it
        against a built model.
        """
        return 0

    @abstractmethod
    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor: ...


class DiagonalRescale(RoleTransform):
    """Kowsher et al. (2024): multiply by a learned per-channel vector, initialised to ones.

    The three roles start identical, so at initialisation the query, key and value streams
    are the same tensor and the attention scores are symmetric. Training pulls the three
    vectors apart, but only along the diagonal, so the score matrix stays a symmetric
    bilinear form up to that diagonal metric.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        return z * self.scale


class D4Mixing(RoleTransform):
    """A routed blend over the eight permutations of D4, applied to groups of four.

    For each group the router emits eight weights, and their softmax picks a convex
    combination of the eight permutation matrices. That combination is doubly stochastic:
    a strictly richer operator than a diagonal, but still cheap, since the matrix is 4x4.
    Unlike the diagonal it need not be symmetric across roles, so the attention scores it
    produces are not forced to be symmetric either.

    The block is wrapped around the same diagonal Kowsher uses, and blended back through a gate.
    Send the gate to zero and what remains is exactly `DiagonalRescale`, which makes the two
    models nested rather than merely comparable.

    Router, magnitude, gate and residual are one design and are varied as one. What the grid
    does separate -- the axis, the routing granularity, the exponent, the coverage -- is set out
    under "Scope of the grid" in the README.

        role = scale * ( z + sigmoid(alpha) * ( magnitude * mix(z) - z ) )

    At initialisation the router is uniform, the magnitude is one and the gate is about
    0.12, so the block starts as a light smoother over each group and learns from there.

    THE MAGNITUDE IS NOT A NEUTRAL RESCALING, AND WHAT IT COSTS DEPENDS ON THE GROUPING.

    It exists because a doubly stochastic matrix has spectral norm one: mixing can only flatten
    a group, never sharpen it, and at initialisation the uniform blend of the eight permutations
    is exactly the all-quarters matrix, which replaces each group by its own mean. Something has
    to be free to put the scale back.

    But it is read from the same readout as the router, so its granularity follows the
    grouping's, and that decides whether it confounds the symmetry argument this block is
    motivated by. Under SEQUENCE and FEATURE the magnitude varies from token to token, and an
    input-dependent per-token scalar breaks A_ij = A_ji on its own -- measured on the real
    modules, the magnitude alone breaks it harder than the whole block does. Under
    FEATURE_PER_GROUP it is constant across tokens, so within a sentence it is a diagonal, and a
    diagonal provably cannot break that symmetry -- which is Kowsher's problem in the first
    place. There, and only there, every asymmetry the block produces comes from the 4x4s.
    """

    def __init__(self, d_model: int, grouping: Grouping, gate_init: float = -2.0) -> None:
        super().__init__()
        if d_model % GROUP_SIZE:
            raise ValueError(f"d_model {d_model} is not divisible by the group size {GROUP_SIZE}")

        # How wide one routing decision is read from is the grouping's business, not this
        # module's. A grouping that summarises a whole token asks for `d_model`; one that reads
        # a group's own four members asks for four, and the router is a hundred times smaller.
        width = grouping.readout_width(d_model)

        self.grouping = grouping
        self.router = nn.Linear(width, N_PERMUTATIONS)
        self.magnitude = nn.Linear(width, 1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.scale = nn.Parameter(torch.ones(d_model))
        self.register_buffer("permutations", permutation_matrices(), persistent=False)
        self.reset_role_parameters()

    @staticmethod
    def extra_parameters(d_model: int, grouping: Grouping | None) -> int:
        """Router (8w + 8), magnitude (w + 1) and gate (1), where w is the readout width.

        The diagonal (d) cancels against the `DiagonalRescale` this replaces.
        """
        width = grouping.readout_width(d_model)
        return 9 * width + 10

    def reset_role_parameters(self) -> None:
        """Uniform over the eight forms, unit magnitude, diagonal of ones.

        The model runs a Xavier pass over every linear layer after construction. It calls
        this afterwards so these choices survive: a router left at Xavier is not uniform,
        and a magnitude left at Xavier does not start at one.
        """
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        nn.init.zeros_(self.magnitude.weight)
        nn.init.constant_(self.magnitude.bias, _inverse_softplus(1.0))
        nn.init.ones_(self.scale)

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        readout = self.grouping.router_readout(z, pad_mask)

        weights = F.softmax(self.router(readout), dim=-1)
        matrix = torch.einsum("bnc,cji->bnji", weights, self.permutations)

        logits = self.magnitude(readout).clamp(-MAGNITUDE_CLAMP, MAGNITUDE_CLAMP)
        magnitude = F.softplus(logits)

        mixed = self.grouping.mix(z, matrix, magnitude, pad_mask)
        gated = z + torch.sigmoid(self.gate) * (mixed - z)
        return gated * self.scale


class D4MixingPowered(D4Mixing):
    """`D4Mixing` with a routed exponent applied to each group of four channels.

    Adds one `nn.Linear(GROUP_SIZE, 1)`. It reads a group's four channels within one token and
    emits an exponent q, so q varies with the token as well as with the group. The permutation
    blend and the magnitude come from the grouping, which for `PerGroupFeatureGrouping` emits
    one of each per group per sentence.

    Each group is divided by its largest term, raised to q through `signed_power`, mixed by the
    permutation blend, raised to 1/q, multiplied by the term it was divided by, and multiplied
    by the magnitude. Dividing by the largest term holds every value inside [-1, 1]; a doubly
    stochastic matrix returns an average of its inputs, so the mixed values stay inside [-1, 1]
    as well, and the exponent is applied to bounded values in both directions.

        group -> largest = max|group|
              -> normalised = group / largest
              -> q = softplus(exponent(normalised)), clamped to POWER_RANGE
              -> bent = signed_power(normalised, q)
              -> blended = matrix @ bent
              -> unbent = signed_power(blended, 1/q)
              -> magnitude * largest * unbent

    The magnitude is applied after the exponent is undone rather than before it, which keeps
    the value entering signed_power(.., 1/q) inside [-1, 1].

    At q = 1, `signed_power` returns its input and the division by `largest` cancels against
    the multiplication by it, so the block computes what `D4Mixing` computes. The exponent
    router is initialised to emit q = 1: its weight is zeroed and its bias is the inverse
    softplus of one, as the magnitude's is.

    It requires `PerGroupFeatureGrouping`, whose groups are four channels within a token and
    whose magnitude carries one value per group per sentence.
    """

    def __init__(self, d_model: int, grouping: Grouping, gate_init: float = -2.0) -> None:
        if not isinstance(grouping, PerGroupFeatureGrouping):
            raise ValueError(
                f"{type(self).__name__} reshapes z into groups of {GROUP_SIZE} channels and "
                f"broadcasts one magnitude per group over the tokens, which is the layout "
                f"PerGroupFeatureGrouping produces; it was given {type(grouping).__name__}"
            )

        super().__init__(d_model, grouping, gate_init)
        self.exponent = nn.Linear(GROUP_SIZE, 1)
        self.reset_role_parameters()

    @staticmethod
    def extra_parameters(d_model: int, grouping: Grouping | None) -> int:
        """What `D4Mixing` holds, plus the exponent router (GROUP_SIZE + 1)."""
        return D4Mixing.extra_parameters(d_model, grouping) + GROUP_SIZE + 1

    def reset_role_parameters(self) -> None:
        """As `D4Mixing`, and an exponent router that emits one.

        `D4Mixing.__init__` calls this before the exponent router exists, so its absence is
        the state during construction rather than an error.
        """
        super().reset_role_parameters()

        exponent = getattr(self, "exponent", None)
        if exponent is not None:
            nn.init.zeros_(exponent.weight)
            nn.init.constant_(exponent.bias, _inverse_softplus(1.0))

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        readout = self.grouping.router_readout(z, pad_mask)

        weights = F.softmax(self.router(readout), dim=-1)
        matrix = torch.einsum("bnc,cji->bnji", weights, self.permutations)

        logits = self.magnitude(readout).clamp(-MAGNITUDE_CLAMP, MAGNITUDE_CLAMP)
        magnitude = F.softplus(logits)

        batch, length, d_model = z.shape
        groups = d_model // GROUP_SIZE

        members = z.view(batch, length, groups, GROUP_SIZE)
        largest = members.abs().amax(dim=-1, keepdim=True).clamp(min=POWER_FLOOR)
        normalised = members / largest

        exponent = F.softplus(self.exponent(normalised)).clamp(*POWER_RANGE)

        bent = signed_power(normalised, exponent).view(batch, length, d_model)

        # The magnitude is applied below, after the exponent is undone, so the grouping is
        # asked for the permutation blend alone.
        blended = self.grouping.mix(bent, matrix, torch.ones_like(magnitude), pad_mask)

        unbent = signed_power(blended.view(batch, length, groups, GROUP_SIZE), 1.0 / exponent)
        restored = unbent * largest * magnitude.view(batch, 1, groups, 1)

        mixed = restored.reshape(batch, length, d_model)
        gated = z + torch.sigmoid(self.gate) * (mixed - z)
        return gated * self.scale
