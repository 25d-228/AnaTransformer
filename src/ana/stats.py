"""Two sources of variation, kept under different names because they are different quantities.

THE TEST SET. `bootstrap_score` puts an interval on one model's score, and `paired_bootstrap`
puts one on the difference between two, by resampling the test set with replacement. This is
what the translation corpora use, and it is what their papers use: fairseq's IWSLT figure and
the Multi30k numbers are single training runs.

THE TRAINING RUN. `across_seed_test` compares models that were each trained several times. This
is what COGS uses, over five seeds, which is what Kim and Linzen (2020) and Csordas et al.
(2021) both report over.

Every interval this module produces carries `what_varies` with it, and `ana report` prints that
word beside the number, so the two are never read as one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ana.metrics import Metric


@dataclass(frozen=True)
class Interval:
    """One model's score, with the range a different test set would have moved it into."""

    score: float
    low: float
    high: float
    n: int
    what_varies: str

    @property
    def half_width(self) -> float:
        """The +/- to print beside the score. Half the width of the 95 percent interval."""
        return (self.high - self.low) / 2


@dataclass(frozen=True)
class Comparison:
    difference: float
    low: float
    high: float
    p_value: float
    n: int
    what_varies: str

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def bootstrap_score(
    system: list[str],
    references: list[str],
    metric: Metric,
    resamples: int = 1_000,
    seed: int = 12_345,
) -> Interval:
    """A bootstrap interval on one model's score, by resampling the test set with replacement.

    Draw another test set of the same size from the same distribution, and the score lands in
    this range 95 times in 100. It is what gives the translation corpora a `+/-`.

    A resample is a selection from the metric's per-sentence statistics, not a rescoring of the
    text. Rescoring means re-tokenising every sentence a thousand times over; the statistics do
    not depend on which resample a sentence lands in, so they are extracted once. Same score, to
    the last decimal, in a fortieth of the time.
    """
    if len(system) != len(references):
        raise ValueError("the system and the references must have the same length")

    statistics = metric.statistics(system, references)
    observed = metric.from_statistics(statistics)

    rng = random.Random(seed)
    size = len(references)
    scores = []
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        scores.append(metric.from_statistics([statistics[i] for i in picks]))

    scores.sort()
    return Interval(
        score=observed,
        low=scores[int(0.025 * resamples)],
        high=scores[int(0.975 * resamples) - 1],
        n=size,
        what_varies="the test set",
    )


def paired_bootstrap(
    system: list[str],
    other: list[str],
    references: list[str],
    metric: Metric,
    resamples: int = 1_000,
    seed: int = 12_345,
) -> Comparison:
    """Koehn's paired bootstrap over the test set. Both systems see the same resample.

    Seeing the SAME resample is the point. The two systems share whatever made a draw of the test
    set easy or hard, and taking the difference on that draw cancels it -- which is why a
    difference of half a point can be significant when each score's own interval is two points
    wide. Scoring them on independent resamples would throw exactly that away.

    Resampling the metric's per-sentence statistics rather than the text, as `bootstrap_score`
    does, and for the same reason.
    """
    if not (len(system) == len(other) == len(references)):
        raise ValueError("the two systems and the references must have the same length")

    ours = metric.statistics(system, references)
    theirs = metric.statistics(other, references)
    observed = metric.from_statistics(ours) - metric.from_statistics(theirs)

    rng = random.Random(seed)
    size = len(references)
    differences = []
    wins_for_other = 0

    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        delta = metric.from_statistics([ours[i] for i in picks]) - metric.from_statistics(
            [theirs[i] for i in picks]
        )
        differences.append(delta)
        if (delta <= 0) == (observed > 0):
            wins_for_other += 1

    differences.sort()
    low = differences[int(0.025 * resamples)]
    high = differences[int(0.975 * resamples) - 1]

    return Comparison(
        difference=observed,
        low=low,
        high=high,
        p_value=wins_for_other / resamples,
        n=size,
        what_varies="the test set",
    )


def across_seed_test(system: list[float], other: list[float]) -> Comparison:
    """A paired test over the seeds. Each entry is one model's score from one training run.

    The interval it returns is on the DIFFERENCE, which is why it can be tight while each
    model's own spread across seeds is wide: the seeds are shared, so whatever made a seed good
    or bad for both models cancels.
    """
    if len(system) != len(other):
        raise ValueError("both models must have been trained on the same seeds")
    n = len(system)
    if n < 2:
        raise ValueError("a difference across seeds needs at least two seeds")

    deltas = [a - b for a, b in zip(system, other, strict=True)]
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    standard_error = math.sqrt(variance / n) if variance > 0 else 0.0

    if standard_error == 0.0:
        p_value = 0.0 if mean != 0 else 1.0
        half_width = 0.0
    else:
        t = mean / standard_error
        p_value = _two_sided_p(abs(t), n - 1)
        half_width = _t_critical(n - 1) * standard_error

    return Comparison(
        difference=mean,
        low=mean - half_width,
        high=mean + half_width,
        p_value=p_value,
        n=n,
        what_varies="the training run",
    )


def _two_sided_p(t: float, degrees_of_freedom: int) -> float:
    """Student's t, through its relationship to the incomplete beta function."""
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    return _regularised_beta(x, degrees_of_freedom / 2, 0.5)


def _t_critical(degrees_of_freedom: int, level: float = 0.95) -> float:
    """Bisection on the p-value. Small tables invite typos; this cannot drift."""
    target = 1.0 - level
    low, high = 0.0, 100.0
    for _ in range(80):
        middle = (low + high) / 2
        if _two_sided_p(middle, degrees_of_freedom) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _regularised_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - log_beta)

    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(x, a, b) / a
    return (
        1
        - math.exp(b * math.log(1 - x) + a * math.log(x) - log_beta)
        * _beta_continued_fraction(1 - x, b, a)
        / b
    )


def _beta_continued_fraction(x: float, a: float, b: float, iterations: int = 200) -> float:
    tiny = 1e-300
    epsilon = 3e-14

    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d

    for m in range(1, iterations + 1):
        even = m * (b - m) * x / ((a - 1.0 + 2 * m) * (a + 2 * m))
        d = 1.0 + even * d
        c = 1.0 + even / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        result *= d * c

        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 1.0 + 2 * m))
        d = 1.0 + odd * d
        c = 1.0 + odd / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        step = d * c
        result *= step

        if abs(step - 1.0) < epsilon:
            break

    return result
