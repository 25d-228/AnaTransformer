"""What a dataset has to provide, and how examples become batches.

A corpus is the single extension point of this study. It owns five things, and nothing outside
it needs to change when a new dataset is added:

    the text          `load()`, returning splits by name
    the vocabulary    `vocab_size`, `tokenizer_type`, the length limits
    the ARCHITECTURE  the model shape the literature reports its number in
    the RECIPE        the published training settings, which `ana tune` then confirms
    the CALIBRATION   what a correct baseline must score, and where that figure comes from

The architecture and the recipe live here, rather than in one global default, because the
configuration that reaches the published number is different for every corpus. COGS is a
two-layer model with a 512-wide feed-forward trained at a fixed learning rate; IWSLT is six
layers with 1024 and an inverse-square-root schedule; Multi30k is a 128-wide model that a
full-size transformer simply overfits. Holding one shape across all three would reproduce
nothing, and a study whose baseline reproduces nothing cannot tell a broken mask from a real
result.

To add a dataset: write one `Corpus` subclass here, give it those five things, and register it
in `CORPORA`. Nothing else in the codebase knows the names of the datasets.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from ana.config import Architecture, ModelConfig
from ana.metrics import Bleu, Metric
from ana.recipes import Calibration, Recipe


@dataclass(frozen=True)
class Example:
    source: str
    target: str


@dataclass
class Batch:
    source_ids: Tensor
    source_mask: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            self.source_ids.to(device),
            self.source_mask.to(device),
            self.labels.to(device),
        )

    def __len__(self) -> int:
        return self.source_ids.size(0)


class Corpus(ABC):
    """One task. Subclasses supply the text, the shape, the recipe, and the target."""

    name: str

    # The vocabulary. `tokenizer_type` is "bpe" for the translation corpora and "word" where
    # the published numbers are reported over whitespace tokens, as on COGS.
    vocab_size: int = 32_000
    tokenizer_type: str = "bpe"
    character_coverage: float = 0.9995

    max_source_length: int = 96
    max_target_length: int = 96
    # How far generation may run. It is separate from the training length because a task can
    # have longer outputs at test time than in training, and cutting generation short there
    # would score the truncation rather than the model.
    max_decode_length: int = 96
    max_positions: int = 256

    # The shape the literature reports this corpus's number in, and the settings it uses.
    architecture: Architecture = Architecture()
    recipe: Recipe = Recipe()

    # What a correct baseline must score. None means the corpus has no published anchor, so a
    # run on it can describe but cannot verify — the smoke corpus is the only such case.
    calibration: Calibration | None = None

    # Splits printed beside the gated one for context, and never gated on. On COGS this is the
    # generalization split: it is the outcome the study measures, so tuning towards it would be
    # fitting, and a target on it would be a target on the answer.
    outcome_splits: tuple[str, ...] = ()

    @abstractmethod
    def load(self) -> dict[str, list[Example]]:
        """Returns splits named `train`, `dev` and `test`, and any others it wants scored."""

    def load_split(self, split: str) -> list[Example]:
        """Load one named split.

        Corpus-wide training preparation still uses `load()`. Evaluation-only diagnostics use
        this narrower interface so a development-only study need not read the test set.
        """
        splits = self.load()
        if split not in splits:
            raise ValueError(f"{self.name} has no split {split!r}")
        return splits[split]

    @property
    def metric(self) -> Metric:
        return Bleu()

    def model_config(self, vocab_size: int) -> ModelConfig:
        """The corpus's architecture, with the vocabulary the tokenizer actually produced."""
        return ModelConfig(
            vocab_size=vocab_size,
            max_positions=self.max_positions,
            d_model=self.architecture.d_model,
            n_heads=self.architecture.n_heads,
            d_ff=self.architecture.d_ff,
            n_encoder_layers=self.architecture.n_encoder_layers,
            n_decoder_layers=self.architecture.n_decoder_layers,
            dropout=self.architecture.dropout,
            label_smoothing=self.architecture.label_smoothing,
        )

    def data_dir(self) -> str:
        """Where the corpora live. The repository carries them, so a fresh clone just works.

        Looked for in order: `$ANA_DATA`, the `data` directory beside the source tree, and
        `data` under the working directory. Set `$ANA_DATA` on a machine that keeps its
        corpora somewhere else.
        """
        override = os.environ.get("ANA_DATA")
        if override:
            return override

        here = os.path.abspath(__file__)  # <repo>/src/ana/data/corpus.py
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
        for candidate in (os.path.join(repo, "data"), os.path.join(os.getcwd(), "data")):
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            "cannot find the corpora. They ship with the repository, in data/. "
            "Set $ANA_DATA if they live somewhere else."
        )

    def tokenizer_path(self, smoke: bool = False) -> str:
        """Where the vocabulary is cached. A smoke run gets its own, and this is not cosmetic.

        A smoke run trains on a couple of hundred examples. If it wrote its vocabulary to the
        path a real run reads, every later run on that machine would quietly tokenise the whole
        corpus with a vocabulary built from two hundred sentences -- and it would not fail, it
        would train, score, and report a number that looked entirely reasonable.
        """
        suffix = ".smoke" if smoke else ""
        return os.path.join(self.data_dir(), f"{self.name}{suffix}.model")


def collate(
    examples: list[tuple[list[int], list[int]]],
    pad_id: int,
    ignore_index: int = -100,
) -> Batch:
    """Right-pad a list of tokenised pairs. Padding in the labels becomes the ignore index."""
    source_len = max(len(s) for s, _ in examples)
    target_len = max(len(t) for _, t in examples)

    source_ids = torch.full((len(examples), source_len), pad_id, dtype=torch.long)
    source_mask = torch.zeros((len(examples), source_len), dtype=torch.long)
    labels = torch.full((len(examples), target_len), ignore_index, dtype=torch.long)

    for row, (source, target) in enumerate(examples):
        source_ids[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        source_mask[row, : len(source)] = 1
        labels[row, : len(target)] = torch.tensor(target, dtype=torch.long)

    return Batch(source_ids, source_mask, labels)


class SyntheticCorpus(Corpus):
    """A toy task used by the smoke test and by anything that needs to run without data.

    The target is the source reversed, so a working model can actually learn it, and a broken
    one cannot fake it. It has no published number, so it has no calibration: it proves the
    pipeline runs, and claims nothing.
    """

    name = "synthetic"
    vocab_size = 64
    max_source_length = 12
    max_target_length = 12
    max_positions = 32

    architecture = Architecture(
        d_model=64, n_heads=4, d_ff=128, n_encoder_layers=2, n_decoder_layers=2
    )
    recipe = Recipe(max_steps=200, batch_size=32, warmup_steps=50)

    def __init__(self, n_train: int = 256, n_eval: int = 32) -> None:
        self.n_train = n_train
        self.n_eval = n_eval

    def _make(self, count: int, seed: int) -> list[Example]:
        rng = torch.Generator().manual_seed(seed)
        words = [f"w{i}" for i in range(20)]
        examples = []
        for _ in range(count):
            length = int(torch.randint(3, 9, (1,), generator=rng))
            picked = [
                words[int(torch.randint(0, len(words), (1,), generator=rng))] for _ in range(length)
            ]
            examples.append(Example(" ".join(picked), " ".join(reversed(picked))))
        return examples

    def load(self) -> dict[str, list[Example]]:
        return {
            "train": self._make(self.n_train, 1),
            "dev": self._make(self.n_eval, 2),
            "test": self._make(self.n_eval, 3),
        }
