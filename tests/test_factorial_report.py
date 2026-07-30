"""The factorial report is complete, paired by seed, and uses the preregistered contrasts."""

from __future__ import annotations

import pytest

from ana.factorial import (
    CORPUS,
    MODELS,
    SEEDS,
    STUDY_ID,
    compact_record,
    markdown_report,
    validate_records,
)

DEV = {
    "baseline_matched": 13.0,
    "shared_qkv": 10.0,
    "ana_mag_enc": 12.0,
    "ana_d4_enc": 11.0,
    "ana_feat_enc": 14.0,
}


def _records() -> list[dict]:
    records = []
    for model in MODELS:
        for offset, seed in enumerate(SEEDS):
            records.append(
                {
                    "model": model,
                    "corpus": CORPUS,
                    "scores": {"dev": DEV[model] + offset, "test": DEV[model] + offset + 0.5},
                    "parameters": 100,
                    "saved_fraction": 0.2,
                    "best_dev_loss": 2.5,
                    "scored_step": 20_000,
                    "seconds": 3600,
                    "manifest": {
                        "seed": seed,
                        "seeded_before_model_init": True,
                        "study_id": STUDY_ID,
                        "git_commit": "abc1234",
                        "train_config": {"seed": seed, "learning_rate": 0.005},
                        "model_config": {"d_model": 128},
                        "smoke": False,
                        "score_dev": True,
                    },
                }
            )
    return records


def test_compact_record_preserves_all_fifteen_configs() -> None:
    compact = compact_record(_records())

    assert compact["study_id"] == STUDY_ID
    assert len(compact["runs"]) == 15
    assert all(run["seeded_before_model_init"] is True for run in compact["runs"])
    assert all("training_configuration" in run for run in compact["runs"])
    assert all("model_configuration" in run for run in compact["runs"])


def test_markdown_report_contains_paired_and_factorial_results() -> None:
    report = markdown_report(_records())

    assert "Magnitude alone changes development BLEU by +2.00" in report
    assert "D4 mixing without the magnifier changes development BLEU by +1.00" in report
    assert "| **mean** | **+2.50** | **+1.50** | **+1.00**" in report
    assert "positive on 3/3 seeds" in report
    assert "No p-values" in report


def test_validation_refuses_stale_or_partial_records() -> None:
    stale = _records()
    stale[0]["manifest"]["seeded_before_model_init"] = False
    with pytest.raises(ValueError, match="pre-construction"):
        validate_records(stale)

    with pytest.raises(ValueError, match="missing"):
        validate_records(_records()[:-1])
