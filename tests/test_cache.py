"""The cached decoder must produce exactly what the uncached one produces.

Caching keys and values is only sound if every operator in the decoder is positionwise. That
holds here by construction — mixing over four consecutive tokens is refused at any decoder
site — but "by construction" is an argument, and a wrong cache does not raise, it just quietly
scores a different model than the one that was trained. So the two paths are run against each
other, on every model in the registry.
"""

from __future__ import annotations

import pytest
import torch

from ana.config import ModelConfig
from ana.decoding import greedy_decode, greedy_decode_uncached
from ana.registry import REGISTRY, build_model

VOCAB = 48
CONFIG = ModelConfig(vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, dropout=0.0)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_the_cache_changes_nothing(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, CONFIG).eval()

    # Ragged sources, so the padding mask has real work to do.
    source = torch.randint(4, VOCAB, (3, 10))
    source_mask = torch.ones_like(source)
    source_mask[1, 7:] = 0
    source_mask[2, 4:] = 0
    source = source * source_mask

    cached = greedy_decode(model, source, source_mask, max_length=16)
    uncached = greedy_decode_uncached(model, source, source_mask, max_length=16)

    assert cached.shape == uncached.shape, f"{name}: the two paths stopped at different lengths"
    assert torch.equal(cached, uncached), (
        f"{name}: the cached decoder produced different tokens from the uncached one.\n"
        f"cached:   {cached.tolist()}\nuncached: {uncached.tolist()}"
    )
