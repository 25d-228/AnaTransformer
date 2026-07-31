"""Fixed-family models, development-only execution, and study artifact guards."""

from __future__ import annotations

import copy
import json
import math
from itertools import combinations, product

import pytest
import torch

from ana.config import ENCODER_ONLY, ModelConfig, Selection, TrainConfig
from ana.data.corpus import SyntheticCorpus
from ana.experiment import run_cell
from ana.nn.grouping import (
    D4_PERMUTATIONS,
    FEATURE,
    PERM_CONTROL_A,
    PERM_CONTROL_B,
    PERM_CONTROL_C,
    permutation_matrices,
)
from ana.nn.roles import D4MixingWithoutMagnitude, temporary_d4_intervention
from ana.permutation_family import (
    CONTROL_MODELS,
    FAMILIES,
    FAMILY_MODELS,
    SEEDS,
    SOURCE_DIAGNOSTIC_STUDY,
    SOURCE_FACTORIAL_STUDY,
    STUDY_ID,
    build_artifact,
    family_definitions,
    family_properties,
    initialization_hashes,
    load_reused_references,
    markdown_report,
    validate_artifact,
    validate_preregistered_families,
)
from ana.registry import PILOT_MODELS, REGISTRY, build_model
from ana.trainer import TrainOutcome, set_seed

CONFIG = ModelConfig(
    vocab_size=64,
    d_model=32,
    n_heads=4,
    d_ff=64,
    n_encoder_layers=4,
    n_decoder_layers=2,
    dropout=0.0,
)

EXPECTED_FAMILIES = {
    "ana_d4_enc": (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (3, 2, 1, 0),
        (2, 0, 3, 1),
        (1, 0, 3, 2),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
        (3, 1, 2, 0),
    ),
    "perm_ctrl_a_enc": (
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 3, 2, 1),
        (1, 0, 3, 2),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ),
    "perm_ctrl_b_enc": (
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 3, 2, 1),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    ),
    "perm_ctrl_c_enc": (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (1, 2, 3, 0),
        (2, 1, 0, 3),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    ),
}


def test_all_four_families_have_the_exact_preregistered_order_and_properties() -> None:
    assert FAMILIES == EXPECTED_FAMILIES
    assert EXPECTED_FAMILIES["ana_d4_enc"] == D4_PERMUTATIONS
    assert EXPECTED_FAMILIES["perm_ctrl_a_enc"] == PERM_CONTROL_A
    assert EXPECTED_FAMILIES["perm_ctrl_b_enc"] == PERM_CONTROL_B
    assert EXPECTED_FAMILIES["perm_ctrl_c_enc"] == PERM_CONTROL_C
    validate_preregistered_families()

    expected_cycles = {
        "identity": 1,
        "transposition": 2,
        "double_transposition": 3,
        "three_cycle": 0,
        "four_cycle": 2,
    }
    for model in FAMILY_MODELS:
        properties = family_properties(model)
        assert properties["cycle_type_counts"] == expected_cycles
        assert properties["identity_count"] == 1
        assert properties["distinct_member_count"] == 8

    d4 = family_properties("ana_d4_enc")
    assert d4["closed_under_composition"] is True
    assert d4["hamming_distance_histogram"] == {"2": 8, "4": 20}
    for model in CONTROL_MODELS:
        properties = family_properties(model)
        assert properties["closed_under_composition"] is False
        assert properties["generated_closure_size"] == 24
        assert properties["d4_intersection_size"] == 4
        assert properties["hamming_distance_histogram"] == {"2": 8, "3": 4, "4": 16}

    d4_core = set(D4_PERMUTATIONS)
    non_core = [
        len((set(FAMILIES[left]) & set(FAMILIES[right])) - d4_core)
        for left, right in combinations(CONTROL_MODELS, 2)
    ]
    assert non_core == [2, 1, 1]


@pytest.mark.parametrize("model_name", CONTROL_MODELS)
def test_control_models_are_encoder_only_nonpilot_fixed_family_models(model_name: str) -> None:
    spec = REGISTRY[model_name]
    assert spec.mixer is D4MixingWithoutMagnitude
    assert spec.grouping is FEATURE
    assert spec.sites == ENCODER_ONLY
    assert spec.permutations == FAMILIES[model_name]
    assert model_name not in PILOT_MODELS

    model = build_model(model_name, CONFIG)
    assert model.config == CONFIG
    assert all("permutations" not in name for name in model.state_dict())
    buffers = [value for name, value in model.named_buffers() if name.endswith("permutations")]
    assert len(buffers) == 12
    assert all(
        torch.equal(buffer, permutation_matrices(FAMILIES[model_name])) for buffer in buffers
    )


def test_family_models_share_trainable_structure_and_same_seed_initialization_hashes() -> None:
    structures = {}
    for model_name in FAMILY_MODELS:
        model = build_model(model_name, CONFIG)
        structures[model_name] = [
            (name, tuple(parameter.shape)) for name, parameter in model.named_parameters()
        ]
    assert all(structure == structures["ana_d4_enc"] for structure in structures.values())

    hashes = initialization_hashes(CONFIG)
    assert [row["seed"] for row in hashes] == list(SEEDS)
    assert all(len(set(row["models"].values())) == 1 for row in hashes)
    assert len({row["common_sha256"] for row in hashes}) == len(SEEDS)


def test_d4_default_state_and_outputs_remain_strictly_compatible() -> None:
    set_seed(42)
    original = build_model("ana_d4_enc", CONFIG).eval()
    checkpoint = copy.deepcopy(original.state_dict())

    set_seed(99)
    restored = build_model("ana_d4_enc", CONFIG).eval()
    restored.load_state_dict(checkpoint, strict=True)

    source = torch.randint(4, CONFIG.vocab_size, (2, 7))
    mask = torch.ones_like(source)
    with torch.no_grad():
        expected = original.encode(source, mask)
        observed = restored.encode(source, mask)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    for module in restored.modules():
        if isinstance(module, D4MixingWithoutMagnitude):
            assert torch.equal(module.permutations, permutation_matrices())


@pytest.mark.parametrize("model_name", FAMILY_MODELS)
def test_hard_argmax_selects_an_exact_member_of_the_active_family(model_name: str) -> None:
    model = build_model(model_name, CONFIG)
    role = next(
        module for module in model.modules() if isinstance(module, D4MixingWithoutMagnitude)
    )
    with torch.no_grad():
        role.router.weight.zero_()
        role.router.bias.copy_(torch.arange(8, dtype=role.router.bias.dtype))
    readout = torch.randn(2, 3, CONFIG.d_model)
    with temporary_d4_intervention(role, "hard_argmax"):
        weights = role.route_probabilities(readout)
    matrices = torch.einsum("blc,cji->blji", weights, role.permutations)
    assert torch.equal(matrices, role.permutations[7].expand_as(matrices))
    assert torch.equal(role.permutations, permutation_matrices(FAMILIES[model_name]))


def test_development_only_run_never_calls_corpus_wide_or_test_loader(
    tmp_path,
    monkeypatch,
) -> None:
    loaded = []

    class NarrowSynthetic(SyntheticCorpus):
        def load(self):
            raise AssertionError("development-only execution must not call corpus.load()")

        def load_split(self, split):
            loaded.append(split)
            if split == "test":
                raise AssertionError("development-only execution must not load test")
            return self._make(self.n_train if split == "train" else self.n_eval, 1)

        def tokenizer_path(self, smoke=False):
            return str(tmp_path / "tokenizer.model")

    def fake_train(model, train_examples, dev_examples, config, device, on_eval=None):
        return TrainOutcome(
            steps_run=0,
            best_dev_loss=0.0,
            best_step=0,
            scored_step=0,
            selection=Selection.BEST_DEV_LOSS,
            seconds=0.0,
        )

    def fake_score(model, corpus, examples, tokenizer, device, batch_size, beam_size):
        return 0.0, [""] * len(examples)

    monkeypatch.setattr("ana.experiment.build_corpus", lambda name: NarrowSynthetic())
    monkeypatch.setattr("ana.experiment.train", fake_train)
    monkeypatch.setattr("ana.experiment.score_split", fake_score)

    record = run_cell(
        "perm_ctrl_a_enc",
        "synthetic",
        TrainConfig(seed=42),
        output_dir=str(tmp_path / "runs"),
        smoke=True,
        device=torch.device("cpu"),
        study_id=STUDY_ID,
        score_dev=True,
        evaluated_splits=("dev",),
    )
    assert loaded == ["train", "dev"]
    assert set(record["scores"]) == {"dev"}
    assert record["manifest"]["evaluated_splits"] == ["dev"]


def _modules(value: float = 0.2) -> list[dict]:
    return [
        {
            "layer": layer,
            "role": role,
            "real_token_count": 10,
            "gate_strength": value,
            "normalized_router_entropy": 0.8,
            "maximum_route_probability": 0.3,
            "effective_number_of_routes": 6.0,
            "identity_probability": 0.1,
            "mean_route_distribution": [1 / 8] * 8,
            "nearest_family_distance": 0.4,
            "token_conditioned_routing_variation": 0.02,
            "qkv_role_differentiation": 0.03,
            "magnitude": None,
        }
        for layer, role in product(range(1, 5), ("Q", "K", "V"))
    ]


def _preflight() -> dict:
    hashes = []
    for seed in SEEDS:
        value = f"{seed:064x}"
        hashes.append(
            {
                "seed": seed,
                "common_sha256": value,
                "models": {model: value for model in FAMILY_MODELS},
            }
        )
    return {
        "study_id": STUDY_ID,
        "implementation_git_commit": "implementation123",
        "source_factorial_study": SOURCE_FACTORIAL_STUDY,
        "source_experiment_commit": "source123",
        "reproduction_tolerance_bleu": 0.05,
        "d4_checkpoint_reproductions": [
            {
                "model": model,
                "seed": seed,
                "stored_development_bleu": 40.0,
                "reproduced_development_bleu": 40.0,
                "difference": 0.0,
            }
            for model, seed in product(("ana_d4_enc", "ana_feat_enc"), SEEDS)
        ],
        "initialization_hashes": hashes,
    }


def _new_runs() -> list[dict]:
    rows = []
    for model_index, model in enumerate(CONTROL_MODELS):
        for seed in SEEDS:
            soft = 39.0 + (seed - 42) / 10 - model_index / 10
            hard = soft - 0.05
            rows.append(
                {
                    "origin": "new run",
                    "model": model,
                    "seed": seed,
                    "selected_step": 20_000,
                    "stored_development_bleu": soft,
                    "original_reproduction_difference": 0.0,
                    "parameters": 1_000,
                    "wall_clock_seconds": 1_400.0,
                    "run_manifest": {
                        "study_id": STUDY_ID,
                        "evaluated_splits": ["dev"],
                        "git_commit": "implementation123",
                    },
                    "conditions": [
                        {
                            "condition": "original",
                            "development_bleu": soft,
                            "difference_from_original": 0.0,
                        },
                        {
                            "condition": "hard_argmax",
                            "development_bleu": hard,
                            "difference_from_original": hard - soft,
                        },
                    ],
                    "modules": _modules(),
                }
            )
    return rows


def _references() -> list[dict]:
    rows = []
    for model in ("baseline_matched", "shared_qkv", "ana_d4_enc"):
        for seed in SEEDS:
            score = {
                "baseline_matched": 40.0,
                "shared_qkv": 38.5,
                "ana_d4_enc": 39.5,
            }[model] + (seed - 42) / 10
            row = {
                "origin": "reused reference",
                "model": model,
                "seed": seed,
                "development_bleu": score,
                "parameters": 1_000,
                "wall_clock_seconds": 1_400.0,
                "source_study": SOURCE_FACTORIAL_STUDY,
                "source_experiment_commit": "source123",
                "conditions": [
                    {
                        "condition": "original",
                        "development_bleu": score,
                        "difference_from_original": 0.0,
                    }
                ],
                "modules": None,
            }
            if model == "ana_d4_enc":
                row["conditions"].append(
                    {
                        "condition": "hard_argmax",
                        "development_bleu": score - 0.08,
                        "difference_from_original": -0.08,
                    }
                )
                row["modules"] = _modules()
            rows.append(row)
    return rows


def _artifact() -> dict:
    source_artifacts = {
        "factorial": {
            "path": "results/factorial.json",
            "study_id": SOURCE_FACTORIAL_STUDY,
            "source_experiment_commit": "source123",
        },
        "d4_diagnostic": {
            "path": "results/diagnostic.json",
            "study_id": SOURCE_DIAGNOSTIC_STUDY,
            "analysis_git_commit": "diagnostic123",
            "source_experiment_commit": "source123",
        },
    }
    return build_artifact(
        _new_runs(),
        _references(),
        source_artifacts,
        _preflight(),
        "analysis123",
        "implementation123",
    )


def test_reused_reference_loader_validates_the_committed_source_artifacts() -> None:
    references, provenance = load_reused_references(
        "results/multi30k_factorial_v1.json",
        "results/multi30k_d4_checkpoint_diagnostic_v1.json",
    )
    assert len(references) == 9
    assert {(row["model"], row["seed"]) for row in references} == set(
        product(("baseline_matched", "shared_qkv", "ana_d4_enc"), SEEDS)
    )
    assert provenance["factorial"]["study_id"] == SOURCE_FACTORIAL_STUDY
    assert provenance["d4_diagnostic"]["study_id"] == SOURCE_DIAGNOSTIC_STUDY
    assert all(row["origin"] == "reused reference" for row in references)


def test_artifact_validation_is_complete_finite_provenanced_and_deterministic() -> None:
    artifact = _artifact()
    validate_artifact(artifact)
    assert artifact["family_definitions"] == family_definitions()
    assert artifact["test_split_evaluated"] is False
    assert markdown_report(artifact) == markdown_report(copy.deepcopy(artifact))
    assert json.dumps(artifact, sort_keys=True) == json.dumps(_artifact(), sort_keys=True)

    missing_cell = copy.deepcopy(artifact)
    missing_cell["new_runs"].pop()
    with pytest.raises(ValueError, match="exactly nine"):
        validate_artifact(missing_cell)

    missing_condition = copy.deepcopy(artifact)
    missing_condition["new_runs"][0]["conditions"].pop()
    with pytest.raises(ValueError, match="conditions"):
        validate_artifact(missing_condition)

    non_finite = copy.deepcopy(artifact)
    non_finite["new_runs"][0]["conditions"][0]["development_bleu"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_artifact(non_finite)

    stale = copy.deepcopy(artifact)
    stale["source_artifacts"]["d4_diagnostic"]["source_experiment_commit"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        validate_artifact(stale)

    failed_reproduction = copy.deepcopy(artifact)
    failed_reproduction["new_runs"][0]["original_reproduction_difference"] = 0.051
    with pytest.raises(ValueError, match="reproduction"):
        validate_artifact(failed_reproduction)
