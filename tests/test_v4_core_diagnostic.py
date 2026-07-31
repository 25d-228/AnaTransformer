"""Tuple-derived V4 subset interventions, statistics, and artifact guards."""

from __future__ import annotations

import copy
import math
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from ana.nn.grouping import (
    FEATURE,
    PERM_CONTROL_A,
    PERM_CONTROL_B,
    V4_CORE,
    permutation_matrices,
)
from ana.nn.roles import (
    FAMILY_SUBSET_INTERVENTIONS,
    D4MixingWithoutMagnitude,
    family_subset_masks,
    temporary_family_subset_intervention,
)
from ana.permutation_family import FAMILIES, FAMILY_MODELS, SEEDS
from ana.v4_core_diagnostic import (
    CONDITIONS,
    CoreUsageAccumulator,
    _state_dict_sha256,
    build_artifact,
    centroid_metadata,
    core_definition,
    expected_new_intervention_decode_count,
    family_partitions,
    markdown_report,
    partition_from_buffer,
    require_complete_reproduction_phase,
    validate_artifact,
    validate_family_partitions,
)


def _role(family=PERM_CONTROL_B) -> D4MixingWithoutMagnitude:
    role = D4MixingWithoutMagnitude(4, FEATURE, permutations=family).eval()
    with torch.no_grad():
        role.router.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4) / 25)
        role.router.bias.copy_(torch.tensor([0.4, -0.3, 0.2, -0.1, 0.1, 0.7, -0.5, 0.3]))
        role.gate.fill_(0.2)
        role.scale.copy_(torch.tensor([0.8, 0.9, 1.1, 1.2]))
    return role


def test_v4_core_is_defined_once_hashed_and_partitions_every_family() -> None:
    assert V4_CORE == (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    )
    assert core_definition()["sha256"] == (
        "3cc4ea231875cb1d9579cff27efb1056fe3b79e7db3ca9bb328f76d9a59d8249"
    )
    validate_family_partitions()
    assert len(family_partitions()) == 4
    for model in FAMILY_MODELS:
        assert set(V4_CORE) <= set(FAMILIES[model])
        assert len(set(FAMILIES[model]) - set(V4_CORE)) == 4


def test_masks_and_indices_are_derived_from_the_live_buffer_order() -> None:
    reordered = (
        PERM_CONTROL_A[4],
        PERM_CONTROL_A[7],
        PERM_CONTROL_A[2],
        PERM_CONTROL_A[0],
        PERM_CONTROL_A[6],
        PERM_CONTROL_A[3],
        PERM_CONTROL_A[1],
        PERM_CONTROL_A[5],
    )
    matrices = permutation_matrices(reordered)
    core, noncore = family_subset_masks(matrices)
    assert core.nonzero().flatten().tolist() == [1, 3, 5, 7]
    assert noncore.nonzero().flatten().tolist() == [0, 2, 4, 6]
    partition = partition_from_buffer(matrices)
    assert partition["core_router_indices_in_v4_order"] == [3, 5, 7, 1]
    assert partition["noncore_router_indices_in_family_order"] == [0, 2, 4, 6]


@pytest.mark.parametrize("condition", FAMILY_SUBSET_INTERVENTIONS)
def test_every_subset_intervention_implements_its_formula(condition: str) -> None:
    role = _role()
    readout = torch.tensor([[[0.2, -0.5, 0.7, 1.0], [-0.3, 0.4, 0.1, -0.2]]])
    logits = role.router(readout)
    core, noncore = family_subset_masks(role.permutations)
    active = core if condition.startswith("core_") else noncore

    if condition == "core_uniform":
        expected = active.to(logits.dtype).view(1, 1, 8).expand_as(logits) / 4
    else:
        masked = logits.masked_fill(~active, -torch.inf)
        if condition.endswith("_hard"):
            expected = F.one_hot(masked.argmax(dim=-1), 8).to(logits.dtype)
        else:
            expected = F.softmax(masked, dim=-1)

    with temporary_family_subset_intervention(role, condition):
        observed = role.route_probabilities(readout)
    torch.testing.assert_close(observed, expected)
    assert bool(torch.isfinite(observed).all())
    torch.testing.assert_close(observed.sum(dim=-1), torch.ones(1, 2))
    assert torch.equal(observed[..., ~active], torch.zeros_like(observed[..., ~active]))


def test_subset_intervention_restores_default_output_and_state_after_exceptions() -> None:
    role = _role()
    z = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.long)
    before_output = role(z, mask)
    before_state = {name: value.clone() for name, value in role.state_dict().items()}
    before_hash = _state_dict_sha256(role)

    with temporary_family_subset_intervention(role, "core_soft"):
        assert role._d4_intervention == "core_soft"
        assert not torch.equal(role(z, mask), before_output)
    assert role._d4_intervention == "original"
    torch.testing.assert_close(role(z, mask), before_output)

    with (
        pytest.raises(RuntimeError, match="deliberate"),
        temporary_family_subset_intervention(role, "noncore_hard"),
    ):
        raise RuntimeError("deliberate")
    assert role._d4_intervention == "original"
    for name, value in role.state_dict().items():
        assert torch.equal(value, before_state[name])
    assert _state_dict_sha256(role) == before_hash


def test_core_usage_excludes_padding_and_handles_zero_subset_mass() -> None:
    accumulator = CoreUsageAccumulator(permutation_matrices(PERM_CONTROL_B))
    probabilities = torch.zeros(1, 3, 8)
    probabilities[0, 0] = 1 / 8
    probabilities[0, 1, 0] = 1  # padding: must not affect any statistic
    probabilities[0, 2, 1] = 1  # a real token with exactly zero core mass
    mask = torch.tensor([[1, 0, 1]])
    accumulator.add(probabilities, mask)
    row = accumulator.finish()

    assert row["real_token_count"] == 2
    assert row["mean_core_probability_mass"] == pytest.approx(0.25)
    assert row["mean_noncore_probability_mass"] == pytest.approx(0.75)
    assert row["all_family_argmax_core_frequency"] == pytest.approx(0.5)
    assert row["core_conditioned_token_count"] == 1
    assert row["noncore_conditioned_token_count"] == 2
    assert row["normalized_conditional_core_entropy"] == pytest.approx(1.0)
    assert math.isfinite(row["normalized_conditional_noncore_entropy"])
    assert sum(row["core_route_distribution_v4_order"]) == pytest.approx(1.0)
    assert sum(row["noncore_route_distribution_family_order"]) == pytest.approx(1.0)


def test_centroids_record_the_initial_function_difference() -> None:
    rows = {row["model"]: row for row in centroid_metadata()}
    d4 = rows["ana_d4_enc"]
    assert d4["frobenius_distance_from_d4_all_quarters"] == pytest.approx(0.0)
    assert d4["uniform_router_centroid"] == [[0.25] * 4 for _ in range(4)]
    assert d4["singular_values"] == [1.0, 0.0, 0.0, 0.0]
    assert rows["perm_ctrl_a_enc"]["singular_values"] == [
        1.0,
        0.353553390593,
        0.0,
        0.0,
    ]
    for model in FAMILY_MODELS[1:]:
        assert rows[model]["frobenius_distance_from_d4_all_quarters"] > 0
        assert len(rows[model]["singular_values"]) == 4
        assert len(rows[model]["initial_effective_residual_matrix_before_diagonal_scale"]) == 4


def _existing_modules() -> list[dict]:
    return [
        {
            "layer": layer,
            "role": role,
            "real_token_count": 10,
            "mean_route_distribution": [1 / 8] * 8,
        }
        for layer, role in product(range(1, 5), ("Q", "K", "V"))
    ]


def _core_usage(model: str) -> list[dict]:
    partition = next(row for row in family_partitions() if row["model"] == model)
    return [
        {
            "layer": layer,
            "role": role,
            "real_token_count": 10,
            "mean_core_probability_mass": 0.6,
            "core_probability_mass_p05": 0.2,
            "core_probability_mass_p50": 0.6,
            "core_probability_mass_p95": 0.9,
            "mean_noncore_probability_mass": 0.4,
            "all_family_argmax_core_frequency": 0.65,
            "core_route_distribution_v4_order": [0.1, 0.2, 0.3, 0.4],
            "noncore_router_order": copy.deepcopy(partition["noncore_router_order"]),
            "noncore_route_distribution_family_order": [0.4, 0.3, 0.2, 0.1],
            "normalized_conditional_core_entropy": 0.5,
            "normalized_conditional_noncore_entropy": 0.4,
            "core_conditioned_token_count": 10,
            "noncore_conditioned_token_count": 10,
        }
        for layer, role in product(range(1, 5), ("Q", "K", "V"))
    ]


def _checkpoints() -> list[dict]:
    rows = []
    for model, seed in product(FAMILY_MODELS, SEEDS):
        original = 40 + (seed - 42) / 10
        source_study = (
            "multi30k_factorial_v1" if model == "ana_d4_enc" else "multi30k_permutation_family_v1"
        )
        source_commit = "d4source" if model == "ana_d4_enc" else "familysource"
        rows.append(
            {
                "model": model,
                "seed": seed,
                "source_checkpoint": {
                    "source_study": source_study,
                    "source_experiment_commit": source_commit,
                    "source_analysis_study": "source-analysis",
                    "source_analysis_commit": "analysis",
                    "selected_step": 20_000,
                    "model_config": {"vocab_size": 32},
                    "seeded_before_model_init": True,
                    "smoke": False,
                },
                "reproduction": {
                    "selected_step": 20_000,
                    "stored_development_bleu": original,
                    "reproduced_development_bleu": original,
                    "difference": 0.0,
                },
                "reused_references": [
                    {
                        "origin": "reused reference",
                        "condition": "original",
                        "development_bleu": original,
                    },
                    {
                        "origin": "reused reference",
                        "condition": "all_family_hard",
                        "development_bleu": original - 0.05,
                    },
                ],
                "existing_router_statistics": _existing_modules(),
                "core_usage": _core_usage(model),
                "new_interventions": [
                    {
                        "origin": "new intervention",
                        "condition": condition,
                        "development_bleu": original - 0.1 - index / 100,
                        "difference_from_original": -0.1 - index / 100,
                    }
                    for index, condition in enumerate(CONDITIONS)
                ],
                "state_dict_sha256": {"before": "a" * 64, "after": "a" * 64},
            }
        )
    return rows


def _source_artifacts() -> dict:
    return {
        "d4_diagnostic": {
            "path": "results/d4.json",
            "study_id": "multi30k_d4_checkpoint_diagnostic_v1",
            "analysis_git_commit": "d4analysis",
            "source_experiment_commit": "d4source",
        },
        "permutation_family": {
            "path": "results/family.json",
            "study_id": "multi30k_permutation_family_v1",
            "analysis_git_commit": "familyanalysis",
            "experiment_git_commit": "familysource",
        },
    }


def test_artifact_is_complete_finite_provenanced_and_deterministic() -> None:
    artifact = build_artifact(_checkpoints(), _source_artifacts(), "current-analysis")
    validate_artifact(artifact)
    assert artifact["execution_guard"]["original_reproduction_decode_count"] == 12
    assert artifact["execution_guard"]["new_intervention_decode_count"] == 60
    assert artifact["execution_guard"]["all_family_hard_decode_count"] == 0
    assert artifact["test_split_evaluated"] is False
    assert expected_new_intervention_decode_count() == 60
    assert markdown_report(artifact) == markdown_report(copy.deepcopy(artifact))

    missing = copy.deepcopy(artifact)
    missing["checkpoints"].pop()
    with pytest.raises(ValueError, match="exactly the 12"):
        validate_artifact(missing)

    missing_condition = copy.deepcopy(artifact)
    missing_condition["checkpoints"][0]["new_interventions"].pop()
    with pytest.raises(ValueError, match="subset interventions"):
        validate_artifact(missing_condition)

    non_finite = copy.deepcopy(artifact)
    non_finite["checkpoints"][0]["new_interventions"][0]["development_bleu"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_artifact(non_finite)

    stale = copy.deepcopy(artifact)
    stale["checkpoints"][0]["source_checkpoint"]["source_experiment_commit"] = "stale"
    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_artifact(stale)

    failed_reproduction = copy.deepcopy(artifact)
    failed_reproduction["checkpoints"][0]["reproduction"]["difference"] = 0.06
    with pytest.raises(ValueError, match="reproduction guard"):
        validate_artifact(failed_reproduction)

    changed_state = copy.deepcopy(artifact)
    changed_state["checkpoints"][0]["state_dict_sha256"]["after"] = "b" * 64
    with pytest.raises(ValueError, match="state dictionary changed"):
        validate_artifact(changed_state)


def test_all_12_original_guards_must_complete_and_pass_before_interventions() -> None:
    with pytest.raises(ValueError, match="all 12"):
        require_complete_reproduction_phase([{}] * 11, [])
    with pytest.raises(ValueError, match="evaluation is blocked"):
        require_complete_reproduction_phase([{}] * 12, ["one mismatch"])
    require_complete_reproduction_phase([{}] * 12, [])
