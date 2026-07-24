"""The learning-rate schedule must stay the one the recipe was tuned for.

Linear warmup followed by inverse-square-root decay is exactly the original transformer's
schedule when the peak is derived from the width of the model. That equivalence is the reason
this codebase can take a recipe from a model trained under the other formulation. If someone
later 'simplifies' the schedule, the recipe silently stops meaning what it meant, and nothing
about the resulting numbers would look wrong.
"""

from __future__ import annotations

import pytest

from ana.config import TrainConfig, noam_peak

D_MODEL = 512
WARMUP = 2_000


def noam(step: int, d_model: int = D_MODEL, warmup: int = WARMUP) -> float:
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


@pytest.mark.parametrize("step", [1, 10, 500, 1_999, 2_000, 2_001, 10_000, 30_000])
def test_the_schedule_is_noam(step: int) -> None:
    config = TrainConfig(warmup_steps=WARMUP).resolved(D_MODEL)
    assert config.learning_rate_at(step) == pytest.approx(noam(step), rel=1e-12)


def test_the_peak_is_derived_from_the_width() -> None:
    assert noam_peak(D_MODEL, WARMUP) == pytest.approx(9.882118e-4, rel=1e-6)
    assert TrainConfig(warmup_steps=WARMUP).resolved(D_MODEL).learning_rate == noam_peak(
        D_MODEL, WARMUP
    )


def test_an_explicit_learning_rate_is_left_alone() -> None:
    """A search sets the rate by hand. Resolving must not overwrite it."""
    config = TrainConfig(learning_rate=3e-4).resolved(D_MODEL)
    assert config.learning_rate == 3e-4


def test_training_without_resolving_is_refused() -> None:
    with pytest.raises(ValueError, match="resolved"):
        TrainConfig().learning_rate_at(100)
