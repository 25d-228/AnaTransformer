"""How a corpus is scored.

The metric belongs to the corpus rather than to the trainer. That is what lets a task which is
not translation join the study without any part of the training or reporting code learning
about it.

A metric also knows how to be BOOTSTRAPPED, and that is not the same as knowing how to score.

A bootstrap resamples the test set a thousand times. Scoring each resample from its text means
re-tokenising every sentence a thousand times over, and on IWSLT's 6,750 sentences that took
about thirty seconds a model -- so a single report ran for the better part of an hour, and it
has to be regenerated every time a cell lands.

But BLEU is a function of per-sentence counts that do not depend on which resample a sentence
lands in. Extract them once, and a resample is a sum. It is 41 times faster and the score is
identical to the last decimal, because it is the same arithmetic in a different order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import sacrebleu
from sacrebleu.metrics import BLEU


class Metric(ABC):
    name: str

    @abstractmethod
    def score(self, hypotheses: list[str], references: list[str]) -> float:
        """A single number, higher is better."""

    @abstractmethod
    def statistics(self, hypotheses: list[str], references: list[str]) -> list[Any]:
        """One sufficient statistic per sentence, in order.

        Whatever `score` needs from a sentence, and nothing that depends on the other sentences.
        A resample is then a selection from this list rather than a rescoring of the text.
        """

    @abstractmethod
    def from_statistics(self, statistics: list[Any]) -> float:
        """The score of a set of sentences, from their statistics. Must equal `score` exactly."""


class Bleu(Metric):
    """Corpus BLEU through sacreBLEU, with the target language's tokenizer."""

    def __init__(self, tokenize: str = "13a") -> None:
        self.tokenize = tokenize
        self.name = f"BLEU/{tokenize}"
        self._bleu = BLEU(tokenize=tokenize)

    def score(self, hypotheses: list[str], references: list[str]) -> float:
        return sacrebleu.corpus_bleu(hypotheses, [references], tokenize=self.tokenize).score

    def statistics(self, hypotheses: list[str], references: list[str]) -> list[Any]:
        """sacreBLEU's own per-sentence counts: the n-gram matches, totals and lengths.

        These are exactly what the corpus score is aggregated from, so summing a selection of
        them gives the score of that selection -- not an approximation of it.
        """
        return self._bleu._extract_corpus_statistics(hypotheses, [references])

    def from_statistics(self, statistics: list[Any]) -> float:
        return self._bleu._aggregate_and_compute(statistics).score


class ExactMatch(Metric):
    """The share of outputs that match the reference exactly, once whitespace is normalised."""

    name = "exact match"

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join((text or "").split())

    def score(self, hypotheses: list[str], references: list[str]) -> float:
        return self.from_statistics(self.statistics(hypotheses, references))

    def statistics(self, hypotheses: list[str], references: list[str]) -> list[Any]:
        return [
            1.0 if self._normalise(h) == self._normalise(r) else 0.0
            for h, r in zip(hypotheses, references, strict=True)
        ]

    def from_statistics(self, statistics: list[Any]) -> float:
        return 100.0 * sum(statistics) / max(len(statistics), 1)
