"""The matched-budget invariant. This is the experiment, so it is the one test that must hold.

If the models being compared are not the same size, then any difference between their scores
could be a difference in capacity, and the whole comparison says nothing.

What is NOT pinned here is a parameter count. Every corpus is trained in the architecture its
published number was reported under, and those architectures differ — COGS is two layers wide
of 512, IWSLT is six of 512, Multi30k is four of 128 — so QKV sharing removes a different
number of parameters on each. The count is not the invariant. The PROPORTION is what the study
reports and the matched budget is what it depends on, so those are what is tested.
"""

from __future__ import annotations

import pytest

from ana.data.corpora import CORPORA, build_corpus
from ana.registry import (
    REGISTRY,
    baseline_parameters,
    build_model,
    config_for,
    count_parameters,
    model_parameters,
)

# What is left over once the baseline is matched is the difference among the shared models
# themselves: a router at every site costs more than a router at the encoder only, and
# `shared_qkv` has no routers at all. It is small, and stating it is more useful than pretending
# it is zero. The matched baseline is sized against the LARGEST of them, so our own models never
# hold a capacity advantage over the ordinary transformer that controls them.
MAX_SPREAD = 0.02

# The real corpora, with the vocabulary each tokenizer settles on. `synthetic` is excluded: it
# has no published number and exists only to prove the pipeline runs.
CORPORA_UNDER_TEST = [
    ("cogs", 871),
    ("multi30k", 10_000),
    ("iwslt14", 10_000),
]


def sizes(corpus_name: str, vocab_size: int) -> dict[str, int]:
    config = build_corpus(corpus_name).model_config(vocab_size)
    return {name: count_parameters(build_model(name, config)) for name in REGISTRY}


@pytest.mark.parametrize(("corpus", "vocab_size"), CORPORA_UNDER_TEST)
def test_every_model_but_the_baseline_is_the_same_size(corpus, vocab_size):
    counts = sizes(corpus, vocab_size)
    matched = {n: c for n, c in counts.items() if n != "baseline"}

    smallest, largest = min(matched.values()), max(matched.values())
    spread = (largest - smallest) / smallest

    assert spread <= MAX_SPREAD, (
        f"{corpus}: the models being compared differ by {spread:.2%}, "
        f"which is capacity and not architecture. sizes: "
        + ", ".join(f"{n}={c / 1e6:.3f}M" for n, c in sorted(matched.items()))
    )


@pytest.mark.parametrize(("corpus", "vocab_size"), CORPORA_UNDER_TEST)
def test_the_control_is_the_size_of_the_models_it_controls(corpus, vocab_size):
    """`baseline_matched` answers "would an ordinary transformer of this size do as well?".

    It is solved against `shared_qkv`, which is the size the comparison is set at. It cannot land
    exactly there: the routers put the `ana_*` models slightly above `shared_qkv`, and the match
    is quantised besides, since `d_model` has to stay a multiple of `n_heads`. What has to hold
    is that the residual stays small enough that a win against the control is architecture rather
    than capacity, and that it is a number we can quote rather than one we hope about.
    """
    counts = sizes(corpus, vocab_size)
    control = counts["baseline_matched"]

    for name, count in counts.items():
        if name == "baseline":
            continue
        slack = abs(count - control) / control
        assert slack <= MAX_SPREAD, (
            f"{corpus}: {name} differs from baseline_matched by {slack:.2%} "
            f"({count / 1e6:.3f}M against {control / 1e6:.3f}M), so a difference between them "
            f"could be capacity rather than architecture"
        )


@pytest.mark.parametrize(("corpus", "vocab_size"), CORPORA_UNDER_TEST)
def test_sharing_saves_a_real_share_of_the_model(corpus, vocab_size):
    """The saving is what the method is for, so it has to be worth having on every corpus."""
    counts = sizes(corpus, vocab_size)
    saved = (counts["baseline"] - counts["shared_qkv"]) / counts["baseline"]

    assert 0.10 <= saved <= 0.50, (
        f"{corpus}: sharing removes {saved:.1%} of the baseline, which is outside the range "
        f"this study is about"
    )


@pytest.mark.parametrize(("corpus", "vocab_size"), CORPORA_UNDER_TEST)
def test_the_baseline_is_larger_than_what_it_is_compared_against(corpus, vocab_size):
    counts = sizes(corpus, vocab_size)
    for name, count in counts.items():
        if name == "baseline":
            continue
        assert count < counts["baseline"], f"{corpus}: {name} is not smaller than the baseline"


@pytest.mark.parametrize(("corpus", "vocab_size"), CORPORA_UNDER_TEST)
def test_the_arithmetic_agrees_with_the_models_it_describes(corpus, vocab_size):
    """`matched_config` solves for a width by counting parameters without building anything.

    If that arithmetic drifted from what the model actually holds, the matched baseline would be
    solved against the wrong target and every "same size" claim in the study would be false
    while every test that only compared arithmetic to arithmetic went on passing.
    """
    config = build_corpus(corpus).model_config(vocab_size)
    built = sizes(corpus, vocab_size)

    for name in REGISTRY:
        assert model_parameters(name, config) == built[name], (
            f"{corpus}: the arithmetic says {name} has {model_parameters(name, config):,} "
            f"parameters and the built model has {built[name]:,}"
        )

    shape = config_for("baseline_matched", config)
    assert baseline_parameters(shape) == built["baseline_matched"]


def test_every_corpus_declares_what_a_correct_baseline_reaches():
    """A corpus with no published anchor cannot verify anything, and must say so rather than
    quietly gate on nothing. `synthetic` is the only one allowed to have no calibration."""
    for name in CORPORA:
        corpus = build_corpus(name)
        if name == "synthetic":
            assert corpus.calibration is None
            continue

        assert corpus.calibration is not None, f"{name} has no calibration"
        assert corpus.calibration.low < corpus.calibration.high
        assert corpus.calibration.reference.strip(), f"{name} does not say where its range is from"

        # The gate must not be checked on the split the study is measuring. Tuning towards the
        # outcome is not tuning, it is fitting.
        assert corpus.calibration.split not in corpus.outcome_splits, (
            f"{name} gates on {corpus.calibration.split}, which is one of its outcome splits"
        )
