"""The three tasks, each in the configuration its published number was reported in.

Two are translation, drawn from different corpora so that a result is not a property of one
collection of text. They are not so much small and large as easy and hard: Multi30k is 29,000
short image captions in a narrow domain, and IWSLT is 160,239 talk transcripts with five times
the data. The third maps a sentence to a logical form and is scored by exact match, which is
the point of it: if the finding only holds for translation, that is worth knowing.

Every corpus reads plain files from disk — they ship with the repository — so a run needs no
network, and the text a model sees cannot change underneath it between one run and the next.

WHY EACH CORPUS CARRIES ITS OWN ARCHITECTURE

There is no single transformer that reproduces all three published numbers. Multi30k has 29,000
pairs and a full-size model overfits it: Wu et al. (2021) measure transformer-base at 38.3 BLEU
and a 2.6M-parameter model at 41.0. IWSLT wants six layers, a narrow feed-forward and dropout
0.3. COGS wants two layers and no label smoothing at all. Forcing one shape across all three
would leave every baseline short of its published figure, and a baseline that reproduces
nothing cannot tell a broken mask from a real result.

So the architecture is a property of the corpus, and the parameters that QKV sharing removes
are therefore a different number on each — which is why the invariant this study pins down is
the PROPORTION removed, not a count. See `tests/test_params.py`.

To add a dataset: subclass `Corpus`, give it an `Architecture`, a `Recipe` and a `Calibration`,
and register it in `CORPORA` below.
"""

from __future__ import annotations

import csv
import os

from ana.config import Architecture, Schedule, Selection
from ana.data.corpus import Corpus, Example, SyntheticCorpus
from ana.metrics import ExactMatch, Metric
from ana.recipes import Calibration, Recipe


class LineAligned(Corpus):
    """Two files per split, one line of one language against one line of the other."""

    folder: str
    source_language: str = "en"
    target_language: str = "de"

    def _read(self, split: str) -> list[Example]:
        base = os.path.join(self.data_dir(), self.folder, split)
        with open(f"{base}.{self.source_language}", encoding="utf-8") as handle:
            source = [line.strip() for line in handle]
        with open(f"{base}.{self.target_language}", encoding="utf-8") as handle:
            target = [line.strip() for line in handle]

        return [
            Example(s, t)
            for s, t in zip(source, target, strict=True)
            if s and t and s != "__NULL__" and t != "__NULL__"
        ]

    def load(self) -> dict[str, list[Example]]:
        return {split: self._read(split) for split in ("train", "dev", "test")}


class Multi30k(LineAligned):
    """Image captions: 29,000 pairs, English into German. The smallest of the three.

    A full-size transformer overfits this corpus, and the published numbers say so plainly.
    Wu et al. (2021, arXiv 2105.14462) train the same data three ways and report, on test2016
    with beam 5: transformer-base (6 layers, 512 wide, 49.1M parameters) 38.33 BLEU, a smaller
    model 39.68, and a 2.6M-parameter model — four layers, 128 wide, a 256-wide feed-forward —
    **41.02**. The small model wins by 2.7 BLEU, and that is the configuration used here.

    Dropout is 0.3. Every published Multi30k recipe uses 0.3, because 29,000 pairs overfit at
    the 0.1 that large-corpus recipes use.
    """

    name = "multi30k"
    folder = "multi30k"
    vocab_size = 10_000
    max_source_length = 80
    max_target_length = 80
    max_decode_length = 80

    architecture = Architecture(
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_encoder_layers=4,
        n_decoder_layers=4,
        dropout=0.3,
        label_smoothing=0.1,
    )

    recipe = Recipe(
        max_steps=20_000,
        batch_size=256,
        warmup_steps=2_000,
        learning_rate=5e-3,
        schedule=Schedule.INVERSE_SQRT,
        selection=Selection.BEST_DEV_LOSS,
        source=(
            "Wu et al. 2021 (arXiv 2105.14462): transformer_tiny, dropout 0.3, joint BPE, "
            "inverse-sqrt from 5e-3 with 2,000 warmup steps. Their batch is set in tokens and "
            "ours in sentences, so the step count is approximate and `ana tune` confirms the "
            "rate against development loss."
        ),
    )

    calibration = Calibration(
        split="test",
        low=35.0,
        high=42.0,
        reference=(
            "Wu et al. (2021) report 41.02 BLEU on test2016 for this configuration with beam "
            "5, and 38.33 for a full-size transformer. Anything below 35 is not a tuning "
            "problem: it means the tokenizer, the mask or the decoder is wrong."
        ),
    )


class Iwslt14(LineAligned):
    """Talk transcripts: 160,239 pairs, German into English.

    This runs the opposite way from the other two, and on purpose. German into English is the
    direction the whole field reports for this corpus, and it gives us something to check the
    code against: a pipeline with a broken mask still produces a plausible BLEU, and without a
    published number to compare against, nobody would ever notice.

    The configuration is fairseq's `transformer_iwslt_de_en`, which Wu et al. (2019, ICLR,
    arXiv 1901.10430) report at **34.4 BLEU** with beam search — six layers, 512 wide, a
    1024-wide feed-forward, four heads, dropout 0.3, and a joint 10k BPE vocabulary.

    Two of those are not the large-corpus defaults, and both matter. Dropout 0.3, not 0.1:
    160,000 pairs overfit at 0.1, and every published figure at or above 34 uses 0.3. And a
    10,000-piece joint vocabulary, not 32,000: a 32k vocabulary over 160k sentences leaves most
    of its pieces seen a handful of times.

    IT IS SCORED WITH THE ORDINARY BLEU TOKENIZER, BECAUSE THE TEXT IS NOT PRE-TOKENISED.

    This corpus was scored with `tokenize="none"` on the belief that the text arrived already
    Moses-tokenised, which is what fairseq's `prepare-iwslt14.sh` produces. The copy on the hub
    is not: 6,716 of its 6,750 test references have punctuation glued to the word before it.
    Scoring that with `none` splits on whitespace alone, so `wind,` is one token that matches
    nothing rather than `wind` plus `,` -- and it cost 4.5 BLEU.

    The same run scores 28.92 under `none` and 33.44 under `13a`. The first fails the gate; the
    second lands where a correct implementation should, about a point under the published 34.4,
    which is roughly what beam 4 and checkpoint averaging are worth and we do neither.

    The lesson is not about a flag. A metric that assumes something about its input and is
    wrong does not raise: it reports a plausible number, and the model takes the blame.
    """

    name = "iwslt14"
    folder = "iwslt14"
    source_language = "de"
    target_language = "en"
    vocab_size = 10_000

    architecture = Architecture(
        d_model=512,
        n_heads=4,
        d_ff=1024,
        n_encoder_layers=6,
        n_decoder_layers=6,
        dropout=0.3,
        label_smoothing=0.1,
    )

    recipe = Recipe(
        max_steps=50_000,
        batch_size=160,
        warmup_steps=4_000,
        learning_rate=5e-4,
        schedule=Schedule.INVERSE_SQRT,
        selection=Selection.BEST_DEV_LOSS,
        weight_decay=1e-4,
        source=(
            "fairseq `transformer_iwslt_de_en`: Adam (0.9, 0.98), inverse-sqrt from 5e-4 with "
            "4,000 warmup steps, weight decay 1e-4, label smoothing 0.1. Their batch is 4,096 "
            "tokens; ours is 160 sentences, which is about the same, and `ana tune` confirms "
            "the rate against development loss."
        ),
    )

    calibration = Calibration(
        split="test",
        low=31.0,
        high=36.0,
        reference=(
            "Wu et al. (2019, ICLR) report 34.4 BLEU for this exact configuration with beam "
            "search; it is one of the most reproduced numbers in translation. Greedy decoding "
            "costs about 1.1 BLEU (33.00 against 34.11, ENGINE, ACL 2020), which is why this "
            "codebase decodes with a beam. Anything below 31 is a bug, not a learning rate."
        ),
    )

    # The default, 13a. The text is not pre-tokenised, whatever the fairseq pipeline would have
    # produced -- see the class docstring, and `tests/test_metrics.py`, which checks the corpus
    # rather than trusting a comment about it.


class Cogs(Corpus):
    """Kim and Linzen (2020): a sentence in, a logical form out, scored by exact match.

    The generalization split puts familiar words in grammatical positions they never held
    during training, so it is the split the study is actually about. The in-distribution test
    set is scored alongside it, because a model that has not learned the task at all fails
    both, and it is worth being able to tell those two failures apart. The gate is on the
    in-distribution split for exactly that reason: it says the model learned the task, and says
    nothing about the answer.

    THE TRAINING PROTOCOL IS THE RESULT HERE, AND IT IS NOT THE ORIGINAL ONE.

    Kim and Linzen report 96 percent in-distribution and 35 percent on the generalization
    split, and their recipe selects the checkpoint with the lowest development loss. On COGS
    that is the wrong thing to do. Csordas et al. (2021, arXiv 2108.12284) show that
    development loss and generalization accuracy are decorrelated on this corpus — accuracy
    keeps climbing while the loss climbs with it — so selecting on loss throws away the model
    you want. In Kim and Linzen's own codebase, removing the loss-based early stopping and
    changing nothing else takes 35 percent to 65. Adding a fixed learning rate and dropping
    label smoothing takes it to 81.

    It also makes the result reproducible. The original recipe is violently seed-sensitive:
    Csordas et al. report that changing the seed from 1 to 2 in the official repository drops
    final accuracy to 2.5 percent. Their fixed-budget recipe has a spread of +/- 0.00 across
    five seeds. This study runs five seeds on COGS, as both papers do, so that difference
    decides whether the corpus can support a comparison at all.

    Hence: a constant learning rate, 50,000 steps, no label smoothing, and the FINAL weights
    are scored rather than the best-development-loss ones. The rule is applied identically to
    all eight models, so the comparison between them is untouched by it.

    Tokenisation is word-level, not BPE. Every published COGS number is reported over
    whitespace tokens, and the whole vocabulary is under 900 types.
    """

    name = "cogs"
    vocab_size = 1_000
    tokenizer_type = "word"
    character_coverage = 1.0
    max_source_length = 96
    # Logical forms run long: up to 153 words in training and about 480 on the generalization
    # split. Truncating them would score the truncation instead of the model.
    max_target_length = 512
    max_decode_length = 512
    max_positions = 640

    architecture = Architecture(
        d_model=512,
        n_heads=8,
        d_ff=512,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.1,
        label_smoothing=0.0,
    )

    recipe = Recipe(
        max_steps=50_000,
        batch_size=128,
        warmup_steps=0,
        learning_rate=1e-4,
        schedule=Schedule.CONSTANT,
        selection=Selection.FINAL,
        adam_betas=(0.9, 0.999),
        source=(
            "Csordas et al. 2021 (arXiv 2108.12284): a constant 1e-4, 50,000 steps, no early "
            "stopping, no label smoothing. Reaches 81 percent on the generalization split "
            "against 35 for the original recipe, with a spread across seeds of +/- 0.00."
        ),
    )

    calibration = Calibration(
        split="test",
        low=95.0,
        high=99.9,
        reference=(
            "Kim and Linzen (2020) report 96 percent in-distribution accuracy; Csordas et al. "
            "(2021) report 100. The generalization split is NOT gated on: it is the outcome "
            "this study measures, and a target on it would be a target on the answer. For "
            "context only, a correct implementation should land near 80 percent there."
        ),
    )

    outcome_splits = ("gen",)

    @property
    def metric(self) -> Metric:
        return ExactMatch()

    def _read(self, filename: str) -> list[Example]:
        path = os.path.join(self.data_dir(), "cogs", filename)
        with open(path, encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            return [Example(row[0], row[1]) for row in reader if len(row) >= 2]

    def load(self) -> dict[str, list[Example]]:
        return {
            "train": self._read("train.tsv"),
            "dev": self._read("dev.tsv"),
            "test": self._read("test.tsv"),
            "gen": self._read("gen.tsv"),
        }


CORPORA: dict[str, type[Corpus]] = {
    "multi30k": Multi30k,
    "iwslt14": Iwslt14,
    "cogs": Cogs,
    # Needs no data on disk, so the whole pipeline can be exercised end to end before any
    # corpus is read. Reversing a short sequence is learnable but not trivially so. It has no
    # published number, so it has no calibration: it proves the code runs, and claims nothing.
    "synthetic": SyntheticCorpus,
}


def build_corpus(name: str) -> Corpus:
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; choose from {', '.join(CORPORA)}")
    return CORPORA[name]()
