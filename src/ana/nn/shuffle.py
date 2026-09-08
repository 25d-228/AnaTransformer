"""Learned global feature permutations before the existing local D4 role mixers."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ana.nn.grouping import FEATURE
from ana.nn.projection import SharedQKV
from ana.nn.roles import D4MixingWithoutMagnitude

ShuffleTopology = Literal["benes", "butterfly"]


def _strides(d_model: int, topology: ShuffleTopology) -> tuple[int, ...]:
    if d_model < 4 or d_model & (d_model - 1):
        raise ValueError("global shuffles require a power-of-two d_model of at least four")
    ascending = tuple(1 << bit for bit in range(d_model.bit_length() - 1))
    if topology == "benes":
        return ascending + ascending[-2::-1]
    if topology == "butterfly":
        return ascending
    raise ValueError(f"unknown shuffle topology {topology!r}; choose 'benes' or 'butterfly'")


class BinarySwapPermutation(nn.Module):
    """A static permutation learned through independent binary keep/swap switches.

    Beneš wiring covers every permutation with ``2 log2(d) - 1`` stages of ``d/2``
    switches. The cheaper ascending butterfly uses ``log2(d)`` stages and covers
    only a subset. Each stage pairs channels whose indices differ in one bit.

    Both training and evaluation apply hard switches. Training supplies an approximate
    sigmoid gradient to each switch; it never applies a soft feature mixture forward.
    Evaluation composes the switches into one index vector and gathers the features.
    Neither path constructs a dense d-by-d matrix.
    """

    def __init__(
        self,
        d_model: int,
        topology: ShuffleTopology = "benes",
        logit_init: float = -0.25,
    ) -> None:
        super().__init__()
        self.strides = _strides(d_model, topology)
        self.d_model = d_model
        self.topology = topology
        self.logit_init = float(logit_init)
        self.logits = nn.Parameter(torch.empty(len(self.strides), d_model // 2))
        self.reset_parameters()

    @staticmethod
    def switch_count(d_model: int, topology: ShuffleTopology = "benes") -> int:
        return len(_strides(d_model, topology)) * (d_model // 2)

    def reset_parameters(self) -> None:
        """Default identity, with deterministic jitter to separate mirrored stages."""
        # Unequal distances from zero discourage mirrored Beneš stages from crossing
        # their thresholds together and undoing each other's swaps. No RNG is consumed.
        with torch.no_grad():
            index = torch.arange(self.logits.numel(), device=self.logits.device)
            jitter = ((index * 37) % 101).to(dtype=self.logits.dtype) / 50.0 - 1.0
            values = self.logit_init + 0.2 * abs(self.logit_init) * jitter
            self.logits.copy_(values.reshape_as(self.logits))

    @torch.no_grad()
    def permutation_indices(self) -> Tensor:
        """Return the input channel taken by each output channel, on the module device."""
        indices = torch.arange(self.d_model, device=self.logits.device)
        for stage, stride in enumerate(self.strides):
            pairs = indices.reshape(-1, 2, stride)
            swap = self.logits[stage].reshape(-1, 1, stride) >= 0
            indices = torch.where(swap, pairs.flip(-2), pairs).reshape(self.d_model)
        return indices

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.d_model:
            raise ValueError(f"expected {self.d_model} feature channels, got {z.shape[-1]}")
        if not self.training:
            return z.index_select(-1, self.permutation_indices())

        original_shape = z.shape
        for stage, stride in enumerate(self.strides):
            pairs = z.reshape(*original_shape[:-1], -1, 2, stride)
            partners = pairs.flip(-2)
            logits = self.logits[stage].reshape(-1, 1, stride)
            hard = torch.where(logits >= 0, partners, pairs)
            soft = torch.sigmoid(logits).to(dtype=z.dtype)
            # Zero forward, sigmoid gradient backward. Detaching the feature difference
            # leaves the input gradient exactly that of the hard permutation.
            correction = (soft - soft.detach()) * (partners - pairs).detach()
            z = (hard + correction).reshape(original_shape)
        return z


class ShuffledD4QKV(SharedQKV):
    """Shared projection, separate K/V global shuffles, then local D4 role mixing.

    Query anchors the feature order. Each key/value shuffle is static across tokens,
    whereas all three local D4 routers retain their token-dependent behavior.
    """

    def __init__(
        self,
        d_model: int,
        topology: ShuffleTopology = "benes",
        gate_init: float = -2.0,
        logit_init: float = -0.25,
    ) -> None:
        # Preserve SharedQKV's construction and registration order so an identity
        # shuffle starts from exactly the same initialized D4 model for a given seed.
        super().__init__(
            d_model,
            lambda width: D4MixingWithoutMagnitude(width, FEATURE, gate_init=gate_init),
        )
        self.key_shuffle = BinarySwapPermutation(d_model, topology, logit_init)
        self.value_shuffle = BinarySwapPermutation(d_model, topology, logit_init)

    @staticmethod
    def extra_parameters(d_model: int, topology: ShuffleTopology = "benes") -> int:
        """Count beyond the shared projection and three existing role diagonals."""
        return 3 * D4MixingWithoutMagnitude.extra_parameters(d_model, FEATURE) + (
            2 * BinarySwapPermutation.switch_count(d_model, topology)
        )

    def reset_role_parameters(self) -> None:
        # The model resets child D4 roles separately; only reset our new switches here.
        self.key_shuffle.reset_parameters()
        self.value_shuffle.reset_parameters()

    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
        z = self.shared(kv_input)
        return (
            self.key_role(self.key_shuffle(z), kv_mask),
            self.value_role(self.value_shuffle(z), kv_mask),
        )

    def forward(
        self,
        query_input: Tensor,
        kv_input: Tensor,
        query_mask: Tensor,
        kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        z_query = self.shared(query_input)
        z_kv = z_query if kv_input is query_input else self.shared(kv_input)
        return (
            self.query_role(z_query, query_mask),
            self.key_role(self.key_shuffle(z_kv), kv_mask),
            self.value_role(self.value_shuffle(z_kv), kv_mask),
        )
