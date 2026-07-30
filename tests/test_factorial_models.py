"""The magnitude-only and D4-only cells in the preregistered Multi30k screen."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ana.config import ENCODER_ONLY, ModelConfig
from ana.nn.grouping import FEATURE
from ana.nn.roles import D4Mixing, D4MixingWithoutMagnitude, DynamicMagnification
from ana.registry import PILOT_MODELS, REGISTRY, build_model

CONFIG = ModelConfig(
    vocab_size=64,
    d_model=32,
    n_heads=4,
    d_ff=64,
    n_encoder_layers=2,
    n_decoder_layers=2,
    dropout=0.0,
)


@pytest.mark.parametrize(
    ("name", "role_type"),
    [
        ("ana_mag_enc", DynamicMagnification),
        ("ana_d4_enc", D4MixingWithoutMagnitude),
    ],
)
def test_factorial_models_are_registered_encoder_only(name, role_type) -> None:
    spec = REGISTRY[name]
    assert spec.mixer is role_type
    assert spec.grouping is FEATURE
    assert spec.sites == ENCODER_ONLY
    assert build_model(name, CONFIG).config == CONFIG
    assert name not in PILOT_MODELS


@pytest.mark.parametrize("name", ["ana_mag_enc", "ana_d4_enc"])
def test_factorial_models_have_finite_forward_and_backward(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, CONFIG)

    source = torch.randint(4, CONFIG.vocab_size, (3, 7))
    source_mask = torch.ones_like(source)
    labels = torch.randint(4, CONFIG.vocab_size, (3, 6))

    loss, scores = model(source, source_mask, labels)
    loss.backward()

    assert scores.shape == (3, 6, CONFIG.vocab_size)
    assert torch.isfinite(loss)
    assert torch.isfinite(scores).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_ablation_models_do_not_hold_the_component_they_remove() -> None:
    magnitude_only = dict(build_model("ana_mag_enc", CONFIG).named_parameters())
    d4_only = dict(build_model("ana_d4_enc", CONFIG).named_parameters())

    assert any(".magnitude." in name for name in magnitude_only)
    assert all(".router." not in name for name in magnitude_only)

    assert any(".router." in name for name in d4_only)
    assert all(".magnitude." not in name for name in d4_only)


@pytest.mark.parametrize("role_type", [DynamicMagnification, D4MixingWithoutMagnitude])
def test_factorial_transforms_do_not_cross_token_positions(role_type) -> None:
    torch.manual_seed(0)
    role = role_type(CONFIG.d_model, FEATURE).eval()
    learned = role.magnitude if isinstance(role, DynamicMagnification) else role.router
    torch.nn.init.normal_(learned.weight)

    z = torch.randn(2, 6, CONFIG.d_model)
    mask = torch.ones(2, 6, dtype=torch.long)
    changed = z.clone()
    changed[:, 3] += torch.randn_like(changed[:, 3])

    with torch.no_grad():
        before = role(z, mask)
        after = role(changed, mask)

    kept = torch.tensor([0, 1, 2, 4, 5])
    torch.testing.assert_close(before[:, kept], after[:, kept])
    assert not torch.equal(before[:, 3], after[:, 3])


def test_ablation_components_keep_the_combined_models_initialisation() -> None:
    full = D4Mixing(CONFIG.d_model, FEATURE)
    magnitude_only = DynamicMagnification(CONFIG.d_model, FEATURE)
    d4_only = D4MixingWithoutMagnitude(CONFIG.d_model, FEATURE)

    torch.testing.assert_close(magnitude_only.magnitude.weight, full.magnitude.weight)
    torch.testing.assert_close(magnitude_only.magnitude.bias, full.magnitude.bias)
    torch.testing.assert_close(d4_only.router.weight, full.router.weight)
    torch.testing.assert_close(d4_only.router.bias, full.router.bias)
    torch.testing.assert_close(magnitude_only.gate, full.gate)
    torch.testing.assert_close(d4_only.gate, full.gate)
    torch.testing.assert_close(magnitude_only.scale, full.scale)
    torch.testing.assert_close(d4_only.scale, full.scale)


def test_combined_transform_still_computes_the_original_formula() -> None:
    torch.manual_seed(0)
    role = D4Mixing(CONFIG.d_model, FEATURE)
    torch.nn.init.normal_(role.router.weight)
    torch.nn.init.normal_(role.magnitude.weight)
    torch.nn.init.normal_(role.scale)
    role.gate.data.fill_(0.3)

    z = torch.randn(2, 5, CONFIG.d_model)
    mask = torch.ones(2, 5, dtype=torch.long)
    readout = FEATURE.router_readout(z, mask)
    weights = F.softmax(role.router(readout), dim=-1)
    matrix = torch.einsum("bnc,cji->bnji", weights, role.permutations)
    magnitude = F.softplus(role.magnitude(readout).clamp(-5.0, 5.0))
    mixed = FEATURE.mix(z, matrix, magnitude, mask)
    expected = role.scale * (z + torch.sigmoid(role.gate) * (mixed - z))

    torch.testing.assert_close(role(z, mask), expected)
