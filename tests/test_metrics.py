"""The metric has to match the text it is scoring, and only the text can say whether it does.

IWSLT was scored with sacreBLEU's `none` tokenizer, on the belief that the corpus arrived
already Moses-tokenised -- which is what fairseq's `prepare-iwslt14.sh` produces, and what the
class docstring asserted. The copy that actually ships is not tokenised at all.

Nothing failed. `none` splits on whitespace, so `wind,` became one token that matched nothing
instead of `wind` plus `,`, and the score came out 4.5 BLEU low: 28.92 against 33.44 for the
same model on the same outputs. The baseline missed its published range by five points and
looked like a broken model.

So the check is on the corpus, not on the comment. A metric that assumes something about its
input and is wrong does not raise -- it reports a plausible number, and the model takes the
blame.
"""

from __future__ import annotations

import pytest

from ana.data.corpora import CORPORA, build_corpus
from ana.metrics import Bleu

ENDS_A_SENTENCE = (",", ".", "?", "!", ";", ":")

TRANSLATION = ["multi30k", "iwslt14"]


def looks_pre_tokenised(lines: list[str]) -> bool:
    """Pre-tokenised text separates punctuation with a space: `wind ,` rather than `wind,`."""
    glued = sum(
        1
        for line in lines
        if any(word.endswith(ENDS_A_SENTENCE) and len(word) > 1 for word in line.split())
    )
    return glued < 0.1 * len(lines)


@pytest.mark.parametrize("name", TRANSLATION)
def test_the_bleu_tokenizer_matches_the_text_it_scores(name: str) -> None:
    corpus = build_corpus(name)
    metric = corpus.metric
    assert isinstance(metric, Bleu)

    references = [example.target for example in corpus.load()["test"]]
    pre_tokenised = looks_pre_tokenised(references)

    if metric.tokenize == "none":
        assert pre_tokenised, (
            f"{name} is scored with tokenize='none', which splits on whitespace and assumes the "
            f"text is already tokenised. It is not: punctuation is glued to the word before it. "
            f"Scoring it this way costs several BLEU and makes a correct model look broken."
        )
    else:
        assert not pre_tokenised, (
            f"{name} is pre-tokenised, so running BLEU's own tokenizer over it splits on "
            f"boundaries that are already split, and the number is comparable with nothing. "
            f"Use tokenize='none'."
        )


def test_every_corpus_has_a_metric() -> None:
    for name in CORPORA:
        assert build_corpus(name).metric.name
