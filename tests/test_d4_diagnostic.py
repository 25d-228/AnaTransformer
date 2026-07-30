"""Inference interventions, router statistics, and diagnostic artifact guards."""

from __future__ import annotations

import copy
import json
import math
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from ana.config import ModelConfig
from ana.d4_diagnostic import (
    CONDITIONS,
    MODELS,
    ROLES,
    SEEDS,
    _ModuleAccumulator,
    _RoleDifferentiationAccumulator,
    build_artifact,
    labeled_encoder_d4_roles,
    markdown_report,
    require_reproduction,
    validate_artifact,
)
from ana.nn.grouping import FEATURE, N_PERMUTATIONS, permutation_matrices
from ana.nn.roles import (
    D4Mixing,
    D4MixingWithoutMagnitude,
    temporary_d4_intervention,
)
from ana.registry import build_model

CONFIG = ModelConfig(
    vocab_size=32,
    d_model=16,
    n_heads=4,
    d_ff=32,
    n_encoder_layers=4,
    n_decoder_layers=2,
    dropout=0.0,
)


def _deterministic_role(role_type):
    role = role_type(4, FEATURE).eval()
    with torch.no_grad():
        role.router.weight.copy_(
            torch.arange(N_PERMUTATIONS * 4, dtype=torch.float32).view(N_PERMUTATIONS, 4) / 50
        )
        role.router.bias.copy_(torch.linspace(-0.4, 0.3, N_PERMUTATIONS))
        role.gate.fill_(0.4)
        role.scale.copy_(torch.tensor([0.8, 0.9, 1.1, 1.2]))
        if isinstance(role, D4Mixing):
            role.magnitude.weight.copy_(torch.tensor([[0.2, -0.1, 0.3, 0.1]]))
            role.magnitude.bias.fill_(0.25)
    return role


def _expected(role, z, mask, condition):
    logits = role.router(z)
    learned = F.softmax(logits, dim=-1)
    if condition == "uniform_router":
        probabilities = torch.full_like(learned, 1 / N_PERMUTATIONS)
    elif condition == "identity_router":
        probabilities = torch.zeros_like(learned)
        probabilities[..., 0] = 1
    elif condition == "hard_argmax":
        probabilities = F.one_hot(learned.argmax(dim=-1), N_PERMUTATIONS).to(learned.dtype)
    else:
        probabilities = learned

    matrix = torch.einsum("blc,cji->blji", probabilities, permutation_matrices())
    if isinstance(role, D4Mixing) and condition != "magnitude_one":
        magnitude = F.softplus(role.magnitude(z).clamp(-5, 5))
    else:
        magnitude = torch.ones_like(z[..., :1])
    mixed = role.grouping.mix(z, matrix, magnitude, mask)
    gate = 0.0 if condition == "gate_zero" else float(torch.sigmoid(role.gate).detach())
    return role.scale * (z + gate * (mixed - z))


@pytest.mark.parametrize(
    ("role_type", "conditions"),
    [
        (
            D4MixingWithoutMagnitude,
            ("original", "gate_zero", "uniform_router", "identity_router", "hard_argmax"),
        ),
        (
            D4Mixing,
            (
                "original",
                "gate_zero",
                "uniform_router",
                "identity_router",
                "hard_argmax",
                "magnitude_one",
            ),
        ),
    ],
)
def test_every_intervention_implements_its_formula(role_type, conditions) -> None:
    role = _deterministic_role(role_type)
    z = torch.tensor(
        [
            [
                [0.2, -0.5, 1.0, 0.7],
                [-0.3, 0.4, 0.8, -0.2],
            ]
        ]
    )
    mask = torch.ones(1, 2, dtype=torch.long)

    for condition in conditions:
        with temporary_d4_intervention(role, condition):
            observed = role(z, mask)
        torch.testing.assert_close(observed, _expected(role, z, mask, condition))


def test_hard_argmax_is_an_exact_one_hot_d4_selection() -> None:
    role = _deterministic_role(D4Mixing)
    readout = torch.randn(2, 3, 4)
    with temporary_d4_intervention(role, "hard_argmax"):
        probabilities = role.route_probabilities(readout)
    assert torch.equal(probabilities.sum(dim=-1), torch.ones(2, 3))
    assert set(probabilities.unique().tolist()) <= {0.0, 1.0}

    matrices = torch.einsum("blc,cji->blji", probabilities, permutation_matrices())
    for matrix in matrices.reshape(-1, 4, 4):
        assert any(torch.equal(matrix, candidate) for candidate in permutation_matrices())


def test_temporary_overrides_restore_default_output_and_state_after_exit() -> None:
    role = _deterministic_role(D4Mixing)
    z = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.long)
    before_state = {name: value.clone() for name, value in role.state_dict().items()}
    before_output = role(z, mask)

    with temporary_d4_intervention(role, "uniform_router"):
        assert role._d4_intervention == "uniform_router"
        assert not torch.equal(role(z, mask), before_output)

    assert role._d4_intervention == "original"
    torch.testing.assert_close(role(z, mask), before_output)
    assert before_state.keys() == role.state_dict().keys()
    for name, value in role.state_dict().items():
        assert torch.equal(value, before_state[name])


def test_temporary_overrides_restore_after_an_exception() -> None:
    role = _deterministic_role(D4Mixing)
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        temporary_d4_intervention(role, "identity_router"),
    ):
        raise RuntimeError("deliberate")
    assert role._d4_intervention == "original"

    with (
        pytest.raises(ValueError, match="learned magnitude"),
        temporary_d4_intervention(
            _deterministic_role(D4MixingWithoutMagnitude),
            "magnitude_one",
        ),
    ):
        pass


def test_padding_positions_are_excluded_from_router_and_role_statistics() -> None:
    permutations = permutation_matrices()
    module = _ModuleAccumulator(permutations, has_magnitude=True)
    probabilities = torch.stack(
        [
            torch.full((N_PERMUTATIONS,), 1 / N_PERMUTATIONS),
            F.one_hot(torch.tensor(0), N_PERMUTATIONS).float(),
        ]
    ).view(1, 2, N_PERMUTATIONS)
    magnitude = torch.tensor([[[2.0], [99.0]]])
    mask = torch.tensor([[1, 0]])
    module.add(probabilities, mask, magnitude)
    summary = module.finish(gate_strength=0.25)

    assert summary["real_token_count"] == 1
    assert summary["normalized_router_entropy"] == pytest.approx(1.0)
    assert summary["maximum_route_probability"] == pytest.approx(1 / N_PERMUTATIONS)
    assert summary["identity_probability"] == pytest.approx(1 / N_PERMUTATIONS)
    assert summary["magnitude"]["mean"] == pytest.approx(2.0)

    roles = _RoleDifferentiationAccumulator()
    routed = {
        "Q": probabilities.clone(),
        "K": probabilities.clone(),
        "V": probabilities.clone(),
    }
    routed["K"][0, 1] = F.one_hot(torch.tensor(1), N_PERMUTATIONS).float()
    routed["V"][0, 1] = F.one_hot(torch.tensor(2), N_PERMUTATIONS).float()
    roles.add(routed, mask)
    assert roles.finish() == pytest.approx(0.0)


@pytest.mark.parametrize("model_name", MODELS)
def test_module_labels_cover_each_encoder_layer_and_role_once(model_name: str) -> None:
    model = build_model(model_name, CONFIG)
    labels = [(layer, role) for layer, role, _ in labeled_encoder_d4_roles(model)]
    assert labels == list(product(range(1, 5), ROLES))
    assert len(labels) == len(set(labels)) == 12


def _module_rows(model: str) -> list[dict]:
    rows = []
    for layer, role in product(range(1, 5), ROLES):
        rows.append(
            {
                "layer": layer,
                "role": role,
                "real_token_count": 10,
                "gate_strength": 0.2,
                "normalized_router_entropy": 0.8,
                "maximum_route_probability": 0.3,
                "effective_number_of_routes": 6.0,
                "identity_probability": 0.1,
                "mean_route_distribution": [1 / N_PERMUTATIONS] * N_PERMUTATIONS,
                "nearest_d4_distance": 0.4,
                "token_conditioned_routing_variation": 0.02,
                "qkv_role_differentiation": 0.03,
                "magnitude": (
                    {
                        "mean": 1.1,
                        "sample_standard_deviation": 0.2,
                        "p05": 0.8,
                        "p50": 1.0,
                        "p95": 1.4,
                    }
                    if model == "ana_feat_enc"
                    else None
                ),
            }
        )
    return rows


def _checkpoints() -> list[dict]:
    checkpoints = []
    for model in MODELS:
        for seed in SEEDS:
            original = 40 + (seed - 42) / 10
            interventions = []
            for index, condition in enumerate(CONDITIONS[model]):
                score = original if condition == "original" else original - index / 10
                interventions.append(
                    {
                        "condition": condition,
                        "development_bleu": score,
                        "difference_from_original": score - original,
                    }
                )
            checkpoints.append(
                {
                    "model": model,
                    "seed": seed,
                    "selected_step": 20_000,
                    "stored_development_bleu": original,
                    "original_reproduction_bleu": original,
                    "original_reproduction_difference": 0.0,
                    "modules": _module_rows(model),
                    "interventions": interventions,
                }
            )
    return checkpoints


def test_artifact_validation_requires_exact_checkpoints_and_conditions() -> None:
    artifact = build_artifact(_checkpoints(), "analysis123", "source123")
    validate_artifact(artifact)

    missing = copy.deepcopy(artifact)
    missing["checkpoints"].pop()
    with pytest.raises(ValueError, match="exactly the six"):
        validate_artifact(missing)

    unexpected = copy.deepcopy(artifact)
    unexpected["checkpoints"][0]["interventions"].append(
        {
            "condition": "magnitude_one",
            "development_bleu": 40.0,
            "difference_from_original": 0.0,
        }
    )
    with pytest.raises(ValueError, match="conditions"):
        validate_artifact(unexpected)


def test_reproduction_guard_rejects_a_mismatched_score() -> None:
    assert require_reproduction(40.0, 40.05) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="beyond"):
        require_reproduction(40.0, 40.051)


@pytest.mark.parametrize(
    ("stored", "observed"),
    [
        (math.nan, 40.0),
        (40.0, math.nan),
        (math.inf, 40.0),
        (40.0, -math.inf),
    ],
)
def test_reproduction_guard_rejects_non_finite_scores(stored: float, observed: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        require_reproduction(stored, observed)


def test_artifact_validation_rejects_non_finite_reported_values() -> None:
    artifact = build_artifact(_checkpoints(), "analysis123", "source123")

    non_finite_intervention = copy.deepcopy(artifact)
    non_finite_intervention["checkpoints"][0]["interventions"][0]["development_bleu"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_artifact(non_finite_intervention)

    non_finite_statistic = copy.deepcopy(artifact)
    non_finite_statistic["checkpoints"][0]["modules"][0]["gate_strength"] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_artifact(non_finite_statistic)


def test_json_and_markdown_generation_are_deterministic() -> None:
    first = build_artifact(_checkpoints(), "analysis123", "source123")
    second = build_artifact(_checkpoints(), "analysis123", "source123")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert markdown_report(first) == markdown_report(second)
    assert first["decode_count"] == 33
    assert "No p-values" in markdown_report(first)
    assert math.isfinite(first["aggregates"]["modules_across_seeds"][0]["gate_strength"])
