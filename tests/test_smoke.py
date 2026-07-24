"""Every model builds, trains and decodes. Runs in seconds, on a task with a right answer.

The synthetic task is to reverse a short sequence. A model that is wired up correctly
learns it quickly; one that is not cannot fake it, which is more than a forward-pass check
would tell us.
"""

from __future__ import annotations

import pytest
import torch

from ana.config import ModelConfig, TrainConfig
from ana.data.corpus import SyntheticCorpus, collate
from ana.decoding import greedy_decode
from ana.registry import REGISTRY, build_model
from ana.trainer import train

VOCAB = 40
CONFIG = ModelConfig(
    vocab_size=VOCAB,
    d_model=32,
    n_heads=4,
    d_ff=64,
    n_encoder_layers=2,
    n_decoder_layers=2,
    dropout=0.0,
    max_positions=32,
)


def toy_examples(count: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Source is a random sequence of ids; target is the same sequence reversed."""
    rng = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(count):
        length = int(torch.randint(4, 9, (1,), generator=rng))
        body = torch.randint(4, VOCAB, (length,), generator=rng).tolist()
        source = [CONFIG.bos_id, *body, CONFIG.eos_id]
        target = [*reversed(body), CONFIG.eos_id]
        pairs.append((source, target))
    return pairs


@pytest.mark.parametrize("name", list(REGISTRY))
def test_every_model_trains_and_decodes(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, CONFIG)

    config = TrainConfig(max_steps=10, batch_size=8, warmup_steps=5, eval_every=10, seed=0)
    outcome = train(model, toy_examples(64, 1), toy_examples(16, 2), config, torch.device("cpu"))

    assert outcome.steps_run == 10
    assert outcome.best_dev_loss == pytest.approx(outcome.best_dev_loss)  # not NaN

    batch = collate(toy_examples(4, 3), CONFIG.pad_id)
    generated = greedy_decode(model, batch.source_ids, batch.source_mask, max_length=12)
    assert generated.size(0) == 4
    assert generated.dtype == torch.long


def test_the_loss_goes_down() -> None:
    """One model, trained long enough to check the loop actually optimises anything."""
    torch.manual_seed(0)
    model = build_model("ana_feat_all", CONFIG)

    train_set, dev_set = toy_examples(256, 1), toy_examples(32, 2)
    config = TrainConfig(max_steps=1, batch_size=16, warmup_steps=1, eval_every=1, seed=0)
    start = train(model, train_set, dev_set, config, torch.device("cpu")).best_dev_loss

    config = TrainConfig(max_steps=150, batch_size=16, warmup_steps=20, eval_every=50, seed=0)
    end = train(model, train_set, dev_set, config, torch.device("cpu")).best_dev_loss

    assert end < start, f"development loss did not fall: {start:.3f} -> {end:.3f}"


def test_the_synthetic_corpus_loads() -> None:
    splits = SyntheticCorpus(n_train=16, n_eval=4).load()
    assert set(splits) == {"train", "dev", "test"}
    assert splits["train"][0].target.split() == list(reversed(splits["train"][0].source.split()))
