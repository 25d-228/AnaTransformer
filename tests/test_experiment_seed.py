"""The experiment seed controls model initialisation, not only training-time randomness."""

from __future__ import annotations

import torch

from ana.config import Selection, TrainConfig
from ana.experiment import run_cell
from ana.trainer import TrainOutcome


def _run_and_capture(
    tmp_path,
    monkeypatch,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    initial: dict[str, torch.Tensor] = {}

    def capture_train(model, train_examples, dev_examples, config, device, on_eval=None):
        initial.update({name: value.detach().clone() for name, value in model.state_dict().items()})
        return TrainOutcome(
            steps_run=0,
            best_dev_loss=0.0,
            best_step=0,
            scored_step=0,
            selection=Selection.BEST_DEV_LOSS,
            seconds=0.0,
        )

    def score(model, corpus, examples, tokenizer, device, batch_size, beam_size):
        return 0.0, [""] * len(examples)

    monkeypatch.setattr("ana.experiment.train", capture_train)
    monkeypatch.setattr("ana.experiment.score_split", score)

    record = run_cell(
        "ana_feat_enc",
        "synthetic",
        TrainConfig(seed=seed),
        output_dir=str(tmp_path / f"seed{seed}"),
        smoke=True,
        device=torch.device("cpu"),
        study_id="seed_test",
        score_dev=True,
    )
    return initial, record


def test_run_cell_seeds_model_initialisation_and_marks_the_record(tmp_path, monkeypatch) -> None:
    first, record = _run_and_capture(tmp_path / "first", monkeypatch, seed=42)
    second, _ = _run_and_capture(tmp_path / "second", monkeypatch, seed=42)
    different, _ = _run_and_capture(tmp_path / "different", monkeypatch, seed=43)

    assert first.keys() == second.keys() == different.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], different[name]) for name in first)
    assert record["manifest"]["seeded_before_model_init"] is True
    assert record["manifest"]["study_id"] == "seed_test"
    assert record["manifest"]["score_dev"] is True
    assert record["manifest"]["evaluated_splits"] == ["dev", "test"]
    assert set(record["scores"]) == {"dev", "test"}
