"""A metric's per-sentence statistics must reproduce its score exactly, not approximately.

The bootstrap resamples the test set a thousand times. It used to rescore each resample from
the text, which re-tokenises every sentence a thousand times over: on IWSLT's 6,750 sentences
that was about thirty seconds a model, and a single report ran for the better part of an hour --
for a report that has to be regenerated every time a cell lands.

BLEU is a function of per-sentence counts that do not depend on which resample a sentence lands
in, so they are extracted once and a resample becomes a sum. Forty times faster.

That is only sound if the two agree. If they drifted, the tables would be wrong by an amount
nobody could see, and the fast path would have quietly changed the study's numbers rather than
its runtime. So the equality is checked, on the corpora as they are actually scored, rather than
asserted in a comment.
"""

from __future__ import annotations

import random

import pytest

from ana.metrics import Bleu, ExactMatch, Metric

HYPOTHESES = [
    "the cat sat on the mat",
    "a dog barked loudly at the postman",
    "",
    "she said , that is not what i meant .",
    "the quick brown fox jumps over the lazy dog",
    "one",
]
REFERENCES = [
    "the cat sat on a mat",
    "a dog barked at the postman",
    "nothing was said",
    "she said that is not what i meant",
    "the quick brown fox jumped over the lazy dog",
    "one",
]

METRICS: list[Metric] = [Bleu(), Bleu(tokenize="none"), ExactMatch()]


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_the_statistics_reproduce_the_score(metric: Metric) -> None:
    direct = metric.score(HYPOTHESES, REFERENCES)
    from_statistics = metric.from_statistics(metric.statistics(HYPOTHESES, REFERENCES))

    assert from_statistics == pytest.approx(direct, abs=1e-9), (
        f"{metric.name}: scoring the text gives {direct}, and summing the per-sentence "
        f"statistics gives {from_statistics}. The bootstrap uses the second, so the tables "
        f"would not be the numbers the report claims."
    )


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_the_statistics_reproduce_the_score_of_a_RESAMPLE(metric: Metric) -> None:
    """The case the bootstrap actually exercises: a selection, with repeats and omissions.

    Agreeing on the whole test set is not enough. The bootstrap only ever scores subsets drawn
    with replacement, and a statistic that was secretly a per-corpus quantity -- a length ratio,
    say, or a smoothing constant -- would agree on the whole and disagree on every draw.
    """
    statistics = metric.statistics(HYPOTHESES, REFERENCES)
    rng = random.Random(0)

    for _ in range(25):
        picks = [rng.randrange(len(REFERENCES)) for _ in range(len(REFERENCES))]
        direct = metric.score([HYPOTHESES[i] for i in picks], [REFERENCES[i] for i in picks])
        resampled = metric.from_statistics([statistics[i] for i in picks])

        assert resampled == pytest.approx(direct, abs=1e-9), (
            f"{metric.name}: on the resample {picks} the text gives {direct} and the "
            f"statistics give {resampled}"
        )
