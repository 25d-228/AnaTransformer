"""Essential checks for the three-seed full-S4 screen."""

from __future__ import annotations

import itertools

import pytest
import torch

from ana.config import ENCODER_ONLY, ModelConfig, Selection, TrainConfig
from ana.data.corpus import SyntheticCorpus
from ana.experiment import run_cell
from ana.nn.grouping import FEATURE, S4_PERMUTATIONS, permutation_matrices
from ana.nn.roles import D4MixingWithoutMagnitude, S4MixingWithoutMagnitude
from ana.registry import (
    PILOT_MODELS,
    REGISTRY,
    build_model,
    count_parameters,
    model_parameters,
)
from ana.s4_screen import MODEL, STUDY_ID
from ana.trainer import TrainOutcome

SMALL_CONFIG = ModelConfig(
    vocab_size=64,
    d_model=32,
    n_heads=4,
    d_ff=64,
    n_encoder_layers=2,
    n_decoder_layers=2,
    dropout=0.0,
)
MULTI30K_CONFIG = ModelConfig(
    vocab_size=10_000,
    d_model=128,
    n_heads=4,
    d_ff=256,
    n_encoder_layers=4,
    n_decoder_layers=4,
    dropout=0.3,
)


def test_s4_family_is_the_exact_24_member_lexicographic_family() -> None:
    assert tuple(itertools.permutations((0, 1, 2, 3))) == S4_PERMUTATIONS
    assert len(S4_PERMUTATIONS) == len(set(S4_PERMUTATIONS)) == 24
    assert all(tuple(sorted(permutation)) == (0, 1, 2, 3) for permutation in S4_PERMUTATIONS)


def test_s4_model_has_24_route_encoder_only_roles_and_no_magnitude() -> None:
    spec = REGISTRY[MODEL]
    assert spec.mixer is S4MixingWithoutMagnitude
    assert spec.grouping is FEATURE
    assert spec.sites == ENCODER_ONLY
    assert MODEL not in PILOT_MODELS

    model = build_model(MODEL, SMALL_CONFIG)
    roles = [module for module in model.modules() if isinstance(module, S4MixingWithoutMagnitude)]
    assert len(roles) == 3 * SMALL_CONFIG.n_encoder_layers
    expected = permutation_matrices(S4_PERMUTATIONS)
    for role in roles:
        assert role.router.out_features == 24
        assert tuple(role.permutations.shape) == (24, 4, 4)
        assert torch.equal(role.permutations, expected)
        assert not hasattr(role, "magnitude")
    assert all("permutations" not in name for name in model.state_dict())


def test_s4_declared_and_built_parameter_counts_match_the_expected_budget() -> None:
    s4 = build_model(MODEL, MULTI30K_CONFIG)
    matched = build_model("baseline_matched", MULTI30K_CONFIG)
    assert count_parameters(s4) == model_parameters(MODEL, MULTI30K_CONFIG) == 2_251_052
    assert count_parameters(matched) == 2_249_936
    assert (count_parameters(s4) - count_parameters(matched)) / count_parameters(matched) == (
        pytest.approx(0.000496, abs=1e-6)
    )


@pytest.mark.parametrize("name", ["ana_d4_enc", "perm_ctrl_b_enc"])
def test_existing_eight_route_model_still_completes_forward_and_backward(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, SMALL_CONFIG)
    roles = [
        module for module in model.modules() if isinstance(module, D4MixingWithoutMagnitude)
    ]
    assert roles and all(role.router.out_features == 8 for role in roles)

    source = torch.randint(4, SMALL_CONFIG.vocab_size, (2, 7))
    source_mask = torch.ones_like(source)
    labels = torch.randint(4, SMALL_CONFIG.vocab_size, (2, 6))
    loss, scores = model(source, source_mask, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores).all()


def test_s4_development_only_path_loads_train_and_dev_but_not_test(
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
        MODEL,
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
