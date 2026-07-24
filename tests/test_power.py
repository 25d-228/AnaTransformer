"""The exponent `ana_feat_2_enc` applies to a group before it is mixed, and undoes after.

Three things are checked. `signed_power` handles negative values, which `z ** q` does not, and
it composes. The block reduces to `ana_feat_1_enc` when the exponent is one, which is what the
exponent router is initialised to emit, so the two models start from the same function. And the
exponent is read from a group's four channels within one token, so it varies with the token
while the permutation blend, which the grouping emits once per group per sentence, does not.
"""

from __future__ import annotations

import pytest
import torch

from ana.config import ModelConfig
from ana.nn.grouping import FEATURE, FEATURE_PER_GROUP, GROUP_SIZE, SEQUENCE
from ana.nn.roles import POWER_FLOOR, D4Mixing, D4MixingPowered, signed_power
from ana.registry import build_model, count_parameters

VOCAB = 64
CONFIG = ModelConfig(vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, dropout=0.0)


def test_signed_power_is_defined_where_a_plain_power_is_not() -> None:
    """|z| is raised to the exponent and the sign is put back, so z may be negative."""
    z = torch.tensor([-8.0, -1.0, 0.5, 8.0])
    q = torch.full_like(z, 1.0 / 3.0)

    cubed_root = signed_power(z, q)

    torch.testing.assert_close(cubed_root, torch.tensor([-2.0, -1.0, 0.5 ** (1 / 3), 2.0]))
    assert torch.isfinite(cubed_root).all()


def test_signed_power_is_undone_by_its_reciprocal() -> None:
    """Above the floor the power acts, and raising to 1/q returns what was raised to q.

    The round trip needs |z| ** q to stay above POWER_FLOOR as well, so it is checked away
    from zero: at the largest exponent in POWER_RANGE that is |z| > POWER_FLOOR ** (1/4), or
    0.1. Inside the block the values entering signed_power are a group divided by its largest
    term, so they reach 1 at the largest member and approach zero at the smallest.
    """
    torch.manual_seed(0)
    z = torch.randn(256)
    z = z[z.abs() > 0.15]

    for exponent in (0.25, 0.5, 1.0, 2.0, 4.0):
        q = torch.full_like(z, exponent)
        there_and_back = signed_power(signed_power(z, q), 1.0 / q)
        torch.testing.assert_close(there_and_back, z, atol=1e-4, rtol=1e-4)


def test_the_folded_form_is_the_two_branch_form() -> None:
    """`signed_power` is written as one expression, and it used to be written as two.

    The obvious spelling computes the power, computes the line, and selects between them with
    `torch.where`. That holds three extra tensors of the size of z for the backward pass -- the
    line, the mask, and the selection -- and the block runs it twice per role at three roles a
    site. On IWSLT's six-layer, 512-wide model that was several gigabytes, and it exhausted a
    24 GB card.

    The two are the same function, which is why the memory could be given back for nothing. This
    checks it rather than asserting it, and it checks it BELOW the floor as well, where the
    equality is a fact about real numbers and not about the code.
    """
    torch.manual_seed(0)
    z = torch.cat(
        [
            torch.randn(256) * 3.0,
            torch.zeros(1),
            torch.tensor([POWER_FLOOR, -POWER_FLOOR]),
            torch.linspace(-POWER_FLOOR, POWER_FLOOR, 65),  # straddling the floor
            torch.tensor([1e-9, -1e-12]),
        ]
    )

    def two_branch(z: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        size = z.abs()
        powered = size.clamp(min=POWER_FLOOR).pow(exponent)
        linear = size * POWER_FLOOR ** (exponent - 1.0)
        return torch.sign(z) * torch.where(size > POWER_FLOOR, powered, linear)

    for exponent in (0.25, 0.5, 1.0, 2.0, 4.0):
        q = torch.full_like(z, exponent)
        torch.testing.assert_close(signed_power(z, q), two_branch(z, q), atol=1e-7, rtol=1e-6)


def test_an_exponent_of_one_is_the_identity_everywhere() -> None:
    """Including below POWER_FLOOR, where the linear branch acts. This is what nests the two
    models: at an exponent of one the block must compute what the block it extends computes."""
    torch.manual_seed(0)
    z = torch.cat(
        [
            torch.randn(64) * 3.0,
            torch.zeros(1),
            torch.tensor([POWER_FLOOR, -POWER_FLOOR, POWER_FLOOR / 100, -1e-9]),
        ]
    )

    torch.testing.assert_close(signed_power(z, torch.ones_like(z)), z)


def test_at_initialisation_the_block_is_the_one_it_extends() -> None:
    """The exponent router emits one, so the block computes what `D4Mixing` computes.

    The normalisation by the largest term cancels against the multiplication by it, and
    `signed_power(., 1)` returns its input, so the two differ only where |z| falls under
    POWER_FLOOR.
    """
    torch.manual_seed(0)
    d_model = 32

    plain = D4Mixing(d_model, FEATURE_PER_GROUP).eval()
    powered = D4MixingPowered(d_model, FEATURE_PER_GROUP).eval()

    # The routers and the diagonal are reset to the same values by reset_role_parameters; the
    # exponent router is what powered holds in addition, and it emits one.
    powered.load_state_dict(plain.state_dict(), strict=False)

    z = torch.randn(2, 5, d_model)
    mask = torch.ones(2, 5, dtype=torch.long)

    with torch.no_grad():
        torch.testing.assert_close(powered(z, mask), plain(z, mask), atol=1e-5, rtol=1e-5)


def _routing_held_constant(role: D4MixingPowered) -> None:
    """Zero the permutation router and the magnitude router, so only the exponent reads z.

    Both then emit what their biases hold, whatever z is: a uniform blend of the eight forms,
    and a magnitude of one.
    """
    torch.nn.init.zeros_(role.router.weight)
    torch.nn.init.zeros_(role.magnitude.weight)


def test_scaling_the_input_scales_the_output_by_the_same_factor() -> None:
    """The exponent is read from a group divided by its largest term, so it does not move when
    the group is scaled, and the block is homogeneous of degree one in z."""
    torch.manual_seed(0)
    d_model = 32
    role = D4MixingPowered(d_model, FEATURE_PER_GROUP).eval()
    _routing_held_constant(role)
    torch.nn.init.normal_(role.exponent.weight, std=1.0)  # an exponent that actually varies

    z = torch.randn(2, 5, d_model)
    mask = torch.ones(2, 5, dtype=torch.long)

    with torch.no_grad():
        once = role(z, mask)
        scaled = role(z * 4.0, mask)

    torch.testing.assert_close(scaled, once * 4.0, atol=1e-4, rtol=1e-4)


def test_a_nonzero_exponent_router_changes_what_the_block_computes() -> None:
    """The exponent router starts at zero weight, emitting one. Moving it moves the output."""
    torch.manual_seed(0)
    d_model = 32

    at_one = D4MixingPowered(d_model, FEATURE_PER_GROUP).eval()
    routed = D4MixingPowered(d_model, FEATURE_PER_GROUP).eval()
    routed.load_state_dict(at_one.state_dict())
    torch.nn.init.normal_(routed.exponent.weight, std=1.0)

    z = torch.randn(2, 5, d_model)
    mask = torch.ones(2, 5, dtype=torch.long)

    with torch.no_grad():
        assert not torch.allclose(routed(z, mask), at_one(z, mask), atol=1e-4)


@pytest.mark.parametrize("grouping", [SEQUENCE, FEATURE])
def test_the_block_is_refused_on_a_grouping_it_does_not_reshape(grouping) -> None:
    with pytest.raises(ValueError, match="PerGroupFeatureGrouping"):
        D4MixingPowered(32, grouping)


def test_the_model_holds_one_linear_more_per_role_than_the_one_it_extends() -> None:
    """GROUP_SIZE + 1 parameters per role per encoder layer, at three roles a layer."""
    plain = count_parameters(build_model("ana_feat_1_enc", CONFIG))
    powered = count_parameters(build_model("ana_feat_2_enc", CONFIG))

    added = 3 * CONFIG.n_encoder_layers * (GROUP_SIZE + 1)
    assert powered - plain == added
