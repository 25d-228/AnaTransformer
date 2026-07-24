"""The configuration objects, and the enum that names an attention site.

Everything downstream reads these. There is no second source of defaults: a corpus supplies
the architecture and the recipe, the command line may override fields on them, and the
resolved objects are written into every run's manifest, so a run can be reproduced from its
own output.

`ModelConfig` is the shape of the network and `TrainConfig` is how it is trained. Both are
per-corpus, because the published configuration that reaches the published number is
per-corpus: COGS is a two-layer model with a 512-wide feed-forward, IWSLT is six layers with
1024, and Multi30k is a tiny 128-wide model. A single shape for all three would reach the
published number on none of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


def noam_peak(d_model: int, warmup_steps: int) -> float:
    """The peak of the original transformer's schedule, derived rather than written down.

        lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    Linear warmup followed by inverse-square-root decay is exactly that schedule when the peak
    is set to d_model^-0.5 * warmup^-0.5 — in both phases, not approximately. It is what the
    translation recipes use, and it is what `ana tune` scales when it searches.
    """
    return d_model**-0.5 * warmup_steps**-0.5


class Schedule(str, Enum):
    """How the learning rate moves.

    INVERSE_SQRT is the transformer schedule: linear warmup to a peak, then decay. It is what
    both translation recipes use.

    CONSTANT holds the rate fixed for the whole run. It is what the COGS recipe uses, and it
    is not a simplification: Csordas et al. (2021) reach 81 percent on the generalization
    split with a fixed 1e-4, against 35 percent for the original schedule, and the difference
    is reproducible.
    """

    INVERSE_SQRT = "inverse_sqrt"
    CONSTANT = "constant"


class Selection(str, Enum):
    """Which weights get scored when training ends.

    BEST_DEV_LOSS keeps the checkpoint with the lowest development loss. It is the ordinary
    choice and it is what the translation corpora use.

    FINAL scores the weights the run ended on and never looks at the development loss. COGS
    needs this. On COGS the development loss and the generalization accuracy are decorrelated
    — accuracy keeps climbing while the loss climbs with it — so selecting on loss actively
    throws away the model you want. Removing loss-based selection, changing nothing else,
    takes the published generalization figure from 35 percent to 65 (Csordas et al. 2021),
    and it also collapses the spread across seeds from about six points to under one.

    The rule is part of the recipe, so it is fixed within a corpus and free across corpora.
    Every model in a comparison is selected the same way, which is what the comparison needs.
    """

    BEST_DEV_LOSS = "best_dev_loss"
    FINAL = "final"


class Site(Enum):
    """Where an attention module sits in the encoder-decoder.

    The distinction that matters is whether the query stream is fully available at once.
    Encoder self-attention sees the whole source sentence. Decoder self-attention is masked so
    a position cannot read later ones, and during generation it is fed one token at a time.
    Cross-attention takes its queries from that same decoder stream, so it inherits the
    restriction even though its keys and values come from the finished encoder output.
    """

    ENCODER_SELF = "encoder_self"
    DECODER_SELF = "decoder_self"
    CROSS = "cross"

    @property
    def query_stream_is_complete(self) -> bool:
        """True when every query position is present at once, in training and in generation.

        Only encoder self-attention satisfies this. Any operator that moves information
        between query positions may run here and nowhere else.
        """
        return self is Site.ENCODER_SELF


ALL_SITES = frozenset(Site)
ENCODER_ONLY = frozenset({Site.ENCODER_SELF})


@dataclass(frozen=True)
class Architecture:
    """The shape of the published model for one corpus, before a vocabulary is known.

    A corpus owns one of these. It is the configuration under which the literature reports its
    number, so it is the configuration in which our baseline has to reproduce that number.
    """

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.1

    @property
    def n_attention_sites(self) -> int:
        """Encoder self, decoder self, and cross. Sharing saves at every one of them."""
        return self.n_encoder_layers + 2 * self.n_decoder_layers


@dataclass(frozen=True)
class ModelConfig:
    """An `Architecture` with a vocabulary attached. This is what a model is built from."""

    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.1
    max_positions: int = 256

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_attention_sites(self) -> int:
        return self.n_encoder_layers + 2 * self.n_decoder_layers

    @property
    def shared_qkv_saving(self) -> int:
        """Parameters that collapsing Q, K and V into one projection removes from this model.

        Three d x d projections with biases become one, plus three diagonals of length d:

            per site:  2 * (d^2 + d) - 3d

        It is a property of the architecture, not a figure carried over from anyone's paper,
        and it is a different number for every corpus because every corpus has a different
        shape. What is comparable across corpora is the PROPORTION of the model it removes,
        which is what `ana report` prints and what `tests/test_params.py` pins down.
        """
        d = self.d_model
        return self.n_attention_sites * (2 * (d * d + d) - 3 * d)


@dataclass(frozen=True)
class TrainConfig:
    """The recipe. Identical across every model in a comparison, by construction.

    Training is counted in optimiser steps rather than epochs. Counting in epochs ties the
    amount of optimisation to the size of the corpus, which turns every cross-corpus
    comparison into a comparison of training budgets as well.
    """

    max_steps: int = 30_000
    batch_size: int = 128
    learning_rate: float | None = None
    warmup_steps: int = 2_000
    schedule: Schedule = Schedule.INVERSE_SQRT
    selection: Selection = Selection.BEST_DEV_LOSS
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-9

    eval_every: int = 1_000
    seed: int = 42
    # A beam multiplies the decoding batch by its width, and the key-value cache grows with the
    # length of the output. On COGS, whose logical forms run to hundreds of tokens, 64 rows at
    # beam 5 is over a gigabyte of cache per decoder layer.
    decode_batch_size: int = 32
    beam_size: int = 5

    def resolved(self, d_model: int) -> TrainConfig:
        """Fill in the learning rate, if it was left to be derived from the model's width."""
        if self.learning_rate is not None:
            return self
        return replace(self, learning_rate=noam_peak(d_model, max(self.warmup_steps, 1)))

    def learning_rate_at(self, step: int) -> float:
        if self.learning_rate is None:
            raise ValueError(
                "resolved(d_model) has not been called; there is no peak to decay from"
            )
        if self.schedule is Schedule.CONSTANT:
            return self.learning_rate

        step = max(step, 1)
        if self.warmup_steps <= 0:
            return self.learning_rate
        if step <= self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        return self.learning_rate * (self.warmup_steps / step) ** 0.5


@dataclass(frozen=True)
class RunConfig:
    """One cell of the grid: a model, a corpus, a seed."""

    model: str
    corpus: str
    model_config: ModelConfig
    train_config: TrainConfig
    output_dir: str = "runs"
    smoke: bool = False
    extra: dict = field(default_factory=dict)
