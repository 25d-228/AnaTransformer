"""The beam decoder, which is what every published number this study calibrates against uses.

Two things have to hold. A beam of one has to be greedy exactly — otherwise the two paths are
different decoders and the greedy tests below say nothing about the one that actually runs. And
a wider beam has to find sequences the greedy path does not, or it is doing no work at all.

A beam search that quietly loses track of its backpointers does not raise. It returns fluent,
plausible text and a BLEU a point or two low, which looks exactly like a model that needed more
training.
"""

from __future__ import annotations

import pytest
import torch

from ana.config import ModelConfig
from ana.decoding import beam_decode, greedy_decode
from ana.registry import REGISTRY, build_model

VOCAB = 48
CONFIG = ModelConfig(vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, dropout=0.0)


def _sources() -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged sources, so the padding mask has real work to do."""
    source = torch.randint(4, VOCAB, (3, 10))
    source_mask = torch.ones_like(source)
    source_mask[1, 7:] = 0
    source_mask[2, 4:] = 0
    return source * source_mask, source_mask


@pytest.mark.parametrize("name", list(REGISTRY))
def test_a_beam_of_one_is_greedy(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, CONFIG).eval()
    source, source_mask = _sources()

    greedy = greedy_decode(model, source, source_mask, max_length=16)
    beam = beam_decode(model, source, source_mask, max_length=16, beam_size=1)

    assert torch.equal(greedy, beam), (
        f"{name}: beam_size=1 did not reproduce the greedy path.\n"
        f"greedy: {greedy.tolist()}\nbeam:   {beam.tolist()}"
    )


@pytest.mark.parametrize("name", list(REGISTRY))
def test_a_beam_scores_at_least_as_well_as_greedy(name: str) -> None:
    """The beam is searching for the highest-scoring sequence, so it must not find a worse one.

    Scored on the model's own length-normalised log-probability, which is the quantity the beam
    is maximising. A beam that returned a sequence the greedy path beats on its own objective
    would be reordering its backpointers wrongly -- the classic failure, and an invisible one.
    """
    torch.manual_seed(0)
    model = build_model(name, CONFIG).eval()
    source, source_mask = _sources()

    greedy = greedy_decode(model, source, source_mask, max_length=16)
    beam = beam_decode(model, source, source_mask, max_length=16, beam_size=4)

    for row in range(source.size(0)):
        one_source = source[row : row + 1]
        one_mask = source_mask[row : row + 1]

        found = _log_probability(model, one_source, one_mask, beam[row])
        settled_for = _log_probability(model, one_source, one_mask, greedy[row])

        assert found >= settled_for - 1e-4, (
            f"{name}: row {row}: the beam returned a sequence scoring {found:.4f} when the "
            f"greedy path found one scoring {settled_for:.4f}. A beam cannot do worse than "
            f"greedy on the objective it is maximising, so its backpointers are wrong."
        )


@torch.no_grad()
def _log_probability(model, source, source_mask, tokens: torch.Tensor) -> float:
    """Length-normalised log-probability the model assigns to `tokens`, ignoring padding."""
    config = model.config
    kept = tokens[tokens != config.pad_id]
    if kept.numel() == 0:
        return 0.0

    labels = kept.unsqueeze(0)
    target = torch.cat([labels.new_full((1, 1), config.bos_id), labels[:, :-1]], dim=1)
    mask = torch.ones_like(target)

    scores = model.logits(source, source_mask, target, mask)
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    chosen = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return float(chosen.sum() / kept.numel())
