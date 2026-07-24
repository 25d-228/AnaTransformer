"""Two properties that would silently invalidate a run if they failed.

The first is that the decoder cannot read from positions it has not produced yet. An
operator that mixes four consecutive tokens breaks this if it is allowed near the decoder,
and the failure is quiet: the model trains, the loss drops, and the score is meaningless.

The second is that an operator must depend only on the sentence it is applied to, and not on
which other sentences happen to share its batch. Two of the three groupings can lose this, by
different routes, and both are checked here. Sequence grouping loses it by cutting groups over
a padded batch. Per-group feature grouping never touches another position, but it POOLS over
the sentence to route, so it loses it by averaging padding into that pool. The symptom is the
same either way: a model scores differently at a batch size of 64 than it did at 192, and
nothing raises.
"""

from __future__ import annotations

import pytest
import torch

from ana.config import ALL_SITES, ModelConfig
from ana.nn.grouping import FEATURE_PER_GROUP, GROUP_SIZE, SEQUENCE
from ana.nn.roles import D4Mixing
from ana.registry import REGISTRY, ModelSpec, build_model

VOCAB = 64
CONFIG = ModelConfig(vocab_size=VOCAB, d_model=32, n_heads=4, d_ff=64, dropout=0.0)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_the_decoder_cannot_read_the_future(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name, CONFIG).eval()

    source = torch.randint(4, VOCAB, (2, 9))
    source_mask = torch.ones_like(source)
    target = torch.randint(4, VOCAB, (2, 11))
    target_mask = torch.ones_like(target)

    cut = 6
    disturbed = target.clone()
    disturbed[:, cut:] = torch.randint(4, VOCAB, (2, 11 - cut))

    with torch.no_grad():
        before = model.logits(source, source_mask, target, target_mask)
        after = model.logits(source, source_mask, disturbed, target_mask)

    torch.testing.assert_close(
        before[:, :cut],
        after[:, :cut],
        msg=f"{name}: changing target positions from {cut} onwards moved earlier outputs",
    )


def test_sequence_mixing_is_refused_where_the_decoder_produces_queries() -> None:
    with pytest.raises(ValueError, match="later ones"):
        ModelSpec(
            name="illegal",
            purpose="mixes tokens at a causal site",
            shared=True,
            mixer=D4Mixing,
            grouping=SEQUENCE,
            sites=ALL_SITES,
        )


def test_sequence_mixing_ignores_the_rest_of_the_batch() -> None:
    """The same sentence must come out the same whoever it is batched with."""
    torch.manual_seed(0)
    d_model = 32
    role = D4Mixing(d_model, SEQUENCE).eval()
    torch.nn.init.normal_(role.router.weight, std=0.5)  # a router that actually routes

    alone_length = 7  # not a multiple of four: exercises the ragged tail
    sentence = torch.randn(1, alone_length, d_model)
    alone_mask = torch.ones(1, alone_length, dtype=torch.long)

    padded_length = 15
    batched = torch.randn(3, padded_length, d_model)
    batched[0, :alone_length] = sentence[0]
    batched_mask = torch.zeros(3, padded_length, dtype=torch.long)
    batched_mask[0, :alone_length] = 1
    batched_mask[1, :padded_length] = 1
    batched_mask[2, :12] = 1

    with torch.no_grad():
        on_its_own = role(sentence, alone_mask)
        in_a_batch = role(batched, batched_mask)

    torch.testing.assert_close(
        on_its_own[0],
        in_a_batch[0, :alone_length],
        msg="a sentence's representation changed when it was put in a different batch",
    )


def test_the_same_matrix_reaches_every_token() -> None:
    """Per-group feature grouping decides once per channel GROUP, not once per token.

    That is the whole of the design: it is what makes it the mirror of sequence grouping, which
    decides once per token-group and broadcasts across every channel. Get the einsum wrong --
    give `matrix` a length index it should not have -- and the model still trains, still decodes,
    and is quietly a different model from the one the comparison is about.
    """
    batch, length, d_model = 1, 5, 16
    groups = d_model // GROUP_SIZE

    z = torch.randn(batch, length, d_model)
    matrix = torch.rand(batch, groups, GROUP_SIZE, GROUP_SIZE)
    magnitude = torch.ones(batch, groups, 1)
    mask = torch.ones(batch, length, dtype=torch.long)

    mixed = FEATURE_PER_GROUP.mix(z, matrix, magnitude, mask)

    for token in range(length):
        for group in range(groups):
            channels = slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)
            torch.testing.assert_close(
                mixed[0, token, channels],
                matrix[0, group] @ z[0, token, channels],
                msg=f"group {group}'s matrix did not reach token {token} unchanged",
            )


def test_the_sentence_pool_ignores_the_rest_of_the_batch() -> None:
    """The routing pools over the sentence, so the pool must see the real tokens and no others.

    The mixing here never leaves a token, so this is the only way the batch could reach in --
    and it is easy to miss for exactly that reason.
    """
    torch.manual_seed(0)
    d_model = 32
    role = D4Mixing(d_model, FEATURE_PER_GROUP).eval()
    torch.nn.init.normal_(role.router.weight, std=0.5)  # a router that actually routes
    torch.nn.init.normal_(role.magnitude.weight, std=0.5)

    alone_length = 7
    sentence = torch.randn(1, alone_length, d_model)
    alone_mask = torch.ones(1, alone_length, dtype=torch.long)

    padded_length = 15
    batched = torch.randn(3, padded_length, d_model)
    batched[0, :alone_length] = sentence[0]
    batched_mask = torch.zeros(3, padded_length, dtype=torch.long)
    batched_mask[0, :alone_length] = 1
    batched_mask[1, :padded_length] = 1
    batched_mask[2, :12] = 1

    with torch.no_grad():
        on_its_own = role(sentence, alone_mask)
        in_a_batch = role(batched, batched_mask)

    torch.testing.assert_close(
        on_its_own[0],
        in_a_batch[0, :alone_length],
        msg="the routing pool read padding: a sentence's representation moved with its batch",
    )
