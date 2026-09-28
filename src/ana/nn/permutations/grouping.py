"""What gets split into groups of four, what each routing decision is read from, and whether
that is safe where the decoder runs.

Three groupings are offered, and they differ along two axes rather than one.

WHICH AXIS IS CUT INTO FOURS. Sequence grouping takes four consecutive tokens. Both feature
groupings take four consecutive channels inside a single token.

WHICH AXIS INDEXES THE ROUTING DECISION. Sequence grouping decides once per group of four
tokens, from that group's own four members, and broadcasts the matrix it produces across
every channel. Per-group feature grouping is its mirror on the other axis: once per group of
four channels, from that group's own four members, broadcast across every token. Plain
feature grouping does neither — it cuts the feature axis but indexes its decision by the
SEQUENCE axis, one matrix per token, broadcast across that token's groups.

Those are two questions, not one, and keeping them apart is what lets one comparison vary the
axis alone and another vary the routing granularity alone. A grouping that collapses them
confounds the two, and the confound is invisible in the parameter count.

`moves_information_between_positions` decides where a grouping may run, and it is True for two
of the three — but for different reasons, and neither reason is the mixing. Sequence grouping
permutes positions outright. Per-group feature grouping never moves a channel out of its token
and is still not positionwise, because its ROUTING pools over the sentence.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch
from torch import Tensor

GROUP_SIZE = 4
N_PERMUTATIONS = 8
Permutation = tuple[int, int, int, int]
PermutationFamily = tuple[Permutation, ...]

# The eight permutations that Lepage gives for the equivalent forms of an analogy
# a:b::c:d. Read each row as a gather: output slot j takes input slot row[j].
# They are closed under composition and form the dihedral group of order eight,
# which is checked below at import time.
D4_PERMUTATIONS: PermutationFamily = (
    (0, 1, 2, 3),  # a:b::c:d   identity
    (0, 2, 1, 3),  # a:c::b:d   exchange the means
    (3, 2, 1, 0),  # d:c::b:a
    (2, 0, 3, 1),  # c:a::d:b
    (1, 0, 3, 2),  # b:a::d:c   invert both ratios
    (1, 3, 0, 2),  # b:d::a:c
    (2, 3, 0, 1),  # c:d::a:b   exchange the ratios
    (3, 1, 2, 0),  # d:b::c:a
)

# The complete symmetric group on four positions, in the fixed lexicographic order used by the
# three-seed S4 screen.
S4_PERMUTATIONS = tuple(itertools.permutations((0, 1, 2, 3)))

# The four permutations shared by D4 and every preregistered matched control family.
# Keep this order fixed: issue #8 reports conditional core distributions in this order.
V4_CORE: PermutationFamily = (
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
)

PERM_CONTROL_A: PermutationFamily = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)

PERM_CONTROL_B: PermutationFamily = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
)

PERM_CONTROL_C: PermutationFamily = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (1, 2, 3, 0),
    (2, 1, 0, 3),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
)

PERMUTATION_FAMILIES: dict[str, PermutationFamily] = {
    "ana_d4_enc": D4_PERMUTATIONS,
    "perm_ctrl_a_enc": PERM_CONTROL_A,
    "perm_ctrl_b_enc": PERM_CONTROL_B,
    "perm_ctrl_c_enc": PERM_CONTROL_C,
}


def _is_closed_group(perms: PermutationFamily) -> bool:
    members = set(perms)
    for p in perms:
        for q in perms:
            if tuple(p[q[i]] for i in range(GROUP_SIZE)) not in members:
                return False
    return len(members) == N_PERMUTATIONS


assert _is_closed_group(D4_PERMUTATIONS), "the eight forms are not a closed group"


def permutation_matrices(permutations: PermutationFamily = D4_PERMUTATIONS) -> Tensor:
    """Permutation tuples as (members, 4, 4) matrices, so a blend is one matmul.

    A convex combination of permutation matrices is doubly stochastic: every row and
    every column sums to one. That is the whole reason a softmax over these eight is
    a meaningful operator and not just a shuffle.
    """
    if not permutations:
        raise ValueError("a permutation collection must not be empty")
    matrices = torch.zeros(len(permutations), GROUP_SIZE, GROUP_SIZE)
    for c, perm in enumerate(permutations):
        if tuple(sorted(perm)) != tuple(range(GROUP_SIZE)):
            raise ValueError(f"family member {perm!r} is not a permutation of four positions")
        for j, i in enumerate(perm):
            matrices[c, j, i] = 1.0
    return matrices


class Grouping(ABC):
    """Splits a representation into groups of four and applies a mixing matrix to each.

    `moves_information_between_positions` decides where the grouping may be used. An
    operator that mixes across query positions cannot run anywhere the decoder produces
    queries: in decoder self-attention it would let a position read from later ones, and
    during generation there is only one token in hand, so groups of four do not exist and
    the model would meet an operator at test time that it never saw in training.
    """

    moves_information_between_positions: bool

    def readout_width(self, d_model: int) -> int:
        """How wide the vector one routing decision is read from.

        The router and the magnitude are both built against it, so it decides what they cost.
        It is `d_model` where a decision summarises a whole token — a router of 8d + 8
        parameters — and it is the group size where a decision is read from the group's own
        four members, which is a router of forty parameters however wide the model is.
        """
        return d_model

    @abstractmethod
    def router_readout(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """What the router and the magnitude read.

        Returns (batch, n_decisions, readout_width). One 4x4 is produced per decision, and
        `mix` decides which parts of `z` that decision then governs.
        """

    @abstractmethod
    def mix(self, z: Tensor, matrix: Tensor, magnitude: Tensor, pad_mask: Tensor) -> Tensor:
        """Apply one 4x4 mixing matrix per decision. Returns the same shape as `z`."""


class SequenceGrouping(Grouping):
    """Four consecutive tokens form a group. One routing decision per group.

    Groups are cut inside each sentence's own length. A group that would contain any
    padding is left untouched, which keeps two properties that are easy to lose:
    padding is never permuted into a real token slot, and a sentence's representation
    does not depend on which other sentences share its batch.
    """

    moves_information_between_positions = True

    def _group_is_real(self, pad_mask: Tensor, n_groups: int) -> Tensor:
        """(batch, n_groups) bool: True where all four positions are real tokens."""
        head = pad_mask[:, : n_groups * GROUP_SIZE]
        return head.view(pad_mask.size(0), n_groups, GROUP_SIZE).all(dim=-1)

    def router_readout(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        n_groups = z.size(1) // GROUP_SIZE
        if n_groups == 0:
            return z.new_zeros(z.size(0), 0, z.size(2))
        head = z[:, : n_groups * GROUP_SIZE]
        return head.view(z.size(0), n_groups, GROUP_SIZE, z.size(2)).mean(dim=2)

    def mix(self, z: Tensor, matrix: Tensor, magnitude: Tensor, pad_mask: Tensor) -> Tensor:
        batch, length, d_model = z.shape
        n_groups = length // GROUP_SIZE
        if n_groups == 0:
            return z

        cut = n_groups * GROUP_SIZE
        head = z[:, :cut].view(batch, n_groups, GROUP_SIZE, d_model)

        mixed = torch.einsum("bnji,bnid->bnjd", matrix, head)
        mixed = mixed * magnitude.view(batch, n_groups, 1, 1)

        keep = self._group_is_real(pad_mask, n_groups).view(batch, n_groups, 1, 1)
        head = torch.where(keep, mixed, head)

        return torch.cat([head.reshape(batch, cut, d_model), z[:, cut:]], dim=1)


class FeatureGrouping(Grouping):
    """Four consecutive channels inside one token form a group. One decision per TOKEN.

    It cuts the feature axis but indexes its decision by the sequence axis: the router reads
    the token's whole vector and emits ONE 4x4, which is then applied to all `d_model / 4`
    channel groups of that token. So a group does not choose its own rewriting — the token
    chooses one on behalf of all of them. `PerGroupFeatureGrouping` is the same cut with the
    other choice, and the two together isolate that difference.

    Nothing crosses a token boundary and nothing outside the token is read, so this is the one
    grouping that is positionwise, and therefore the only one safe at every attention site —
    including a single generation step where only one token is available.
    """

    moves_information_between_positions = False

    def router_readout(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        return z

    def mix(self, z: Tensor, matrix: Tensor, magnitude: Tensor, pad_mask: Tensor) -> Tensor:
        batch, length, d_model = z.shape
        groups = d_model // GROUP_SIZE

        chunks = z.view(batch, length, groups, GROUP_SIZE)
        mixed = torch.einsum("blji,blgi->blgj", matrix, chunks)
        mixed = mixed * magnitude.view(batch, length, 1, 1)

        return mixed.reshape(batch, length, d_model)


class PerGroupFeatureGrouping(Grouping):
    """Four consecutive channels form a group. One decision per GROUP, shared by every token.

    The mirror of `SequenceGrouping` on the other axis. Sequence grouping routes each group of
    four tokens from that group's own four members and broadcasts the matrix across every
    channel; this routes each group of four channels from its own four members and broadcasts
    the matrix across every token. It gives `d_model / 4` matrices per sentence, each applied
    `length` times, where sequence grouping gives `length / 4`, each applied `d_model` times.

    So the query, key and value streams become `Z @ M` with `M` block-diagonal, doubly
    stochastic, and chosen once per sentence — a structured, input-conditioned recovery of the
    projections that sharing removed, rather than a projection that varies from token to token.

    The router reads four numbers rather than `d_model` of them, so it holds forty parameters
    instead of eight thousand. That is not a saving for its own sake: it is what lets a decision
    be made per group at all. A router emitting a matrix for each of 128 groups from the whole
    token would need `Linear(512, 1024)`, which costs about as much as the projection sharing
    deleted, and the model would come out larger than the baseline it is supposed to shrink.

    IT IS NOT POSITIONWISE, AND THE MIXING IS NOT THE REASON.

    A channel never leaves its token here. The ROUTING is what reaches outside: it pools over
    the sentence, so a token's output depends on every other token, and that is enough. At a
    decoder site the pool would run over positions that have not been produced yet, and during
    generation it would run over the single token in hand rather than the sentence the model
    was trained on. Encoder self-attention only — the same restriction sequence grouping has,
    arrived at by a different route.
    """

    moves_information_between_positions = True

    def readout_width(self, d_model: int) -> int:
        return GROUP_SIZE

    def router_readout(self, z: Tensor, pad_mask: Tensor) -> Tensor:
        """Each group's four channels, averaged over the sentence's REAL tokens.

        Averaging over the padding as well would make a sentence's representation depend on how
        long its batch-mates happen to be, which is the hazard `SequenceGrouping` avoids by
        leaving padded groups alone. It is the same hazard; it arrives here through the pool.
        """
        batch, _, d_model = z.shape

        real = pad_mask.unsqueeze(-1).to(z.dtype)
        counted = real.sum(dim=1).clamp(min=1.0)
        pooled = (z * real).sum(dim=1) / counted

        return pooled.view(batch, d_model // GROUP_SIZE, GROUP_SIZE)

    def mix(self, z: Tensor, matrix: Tensor, magnitude: Tensor, pad_mask: Tensor) -> Tensor:
        batch, length, d_model = z.shape
        groups = d_model // GROUP_SIZE

        chunks = z.view(batch, length, groups, GROUP_SIZE)

        # `matrix` carries a group index and no length index: group g's 4x4 reaches every
        # token, which is what makes this the mirror of sequence grouping rather than a third
        # thing. The same holds of the magnitude.
        mixed = torch.einsum("bgji,blgi->blgj", matrix, chunks)
        mixed = mixed * magnitude.view(batch, 1, groups, 1)

        return mixed.reshape(batch, length, d_model)


SEQUENCE = SequenceGrouping()
FEATURE = FeatureGrouping()
FEATURE_PER_GROUP = PerGroupFeatureGrouping()
