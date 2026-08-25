"""Generation: greedy, and beam search.

Every published number this study calibrates against is decoded with a beam — fairseq's IWSLT
figure with beam 4 to 5, the Multi30k figures with beam 5. Greedy costs about 1.1 BLEU (33.00
against 34.11, ENGINE, ACL 2020), which is a large enough gap to move a baseline out of its
published range and make a correct pipeline look broken. So the study decodes with a beam, and
`beam_size=1` falls through to the greedy path, which keeps the two exactly consistent.

Generation keeps the keys and values of the positions it has already produced, so a step costs
the same whether it is the third or the three-hundredth. Rerunning the whole prefix at every
step, which is the obvious way to write this, makes generation quadratic in the length of the
output and needs an attention matrix of (batch, heads, length, length), which on the longer
COGS forms runs to hundreds of megabytes per layer and exhausts the card.

Caching is only sound because every operator that reaches the decoder is positionwise: a
token's key and value depend on that token alone, so a key computed at step three is still the
right key at step three hundred. Mixing four consecutive tokens is not positionwise, and the
model refuses to place it at any decoder site. `greedy_decode_uncached` keeps the naive path so
the two can be checked against each other, which is what tests/test_cache.py does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ana.model import Seq2SeqTransformer


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    source_ids: Tensor,
    source_mask: Tensor,
    max_length: int,
) -> Tensor:
    """Returns the generated ids, without the start symbol, padded after the end symbol."""
    model.eval()
    config = model.config
    batch = source_ids.size(0)
    device = source_ids.device

    memory = model.encode(source_ids, source_mask)
    cache = model.new_cache()

    token = torch.full((batch, 1), config.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    produced: list[Tensor] = []

    limit = min(max_length, config.max_positions - 1)
    for position in range(limit):
        scores = model.decode_step(token, memory, source_mask, cache, position)
        chosen = scores.argmax(dim=-1)

        # A row that has emitted the end symbol must stop contributing. Without this it goes
        # on emitting ordinary tokens until every row in the batch is done, and that trailing
        # text survives detokenisation and destroys an exact-match score.
        chosen = chosen.masked_fill(finished, config.pad_id)
        produced.append(chosen)

        finished |= chosen == config.eos_id
        if bool(finished.all()):
            break
        token = chosen.unsqueeze(1)

    return torch.stack(produced, dim=1)


@torch.no_grad()
def beam_decode(
    model: Seq2SeqTransformer,
    source_ids: Tensor,
    source_mask: Tensor,
    max_length: int,
    beam_size: int = 5,
    length_penalty: float = 1.0,
) -> Tensor:
    """Beam search with a length-normalised score. Returns the best hypothesis per row.

    The beams of one sentence are laid out contiguously, so the working batch is
    `batch * beam_size` rows and every cached key and value is reordered by the backpointers at
    each step — a beam that survives carries its own history with it.

    Scores are normalised by length before the winner is picked, which is what a length penalty
    of 1.0 means. Without it a beam is rewarded for stopping early, because every extra token
    adds a negative log-probability, and translations come out systematically short.
    """
    if beam_size <= 1:
        return greedy_decode(model, source_ids, source_mask, max_length)

    model.eval()
    config = model.config
    device = source_ids.device
    rows, width = source_ids.size(0), beam_size
    vocab = config.vocab_size

    memory = model.encode(source_ids, source_mask)
    length, dim = memory.size(1), memory.size(2)
    memory = memory.unsqueeze(1).expand(rows, width, length, dim).reshape(rows * width, length, dim)
    mask = source_mask.unsqueeze(1).expand(rows, width, length).reshape(rows * width, length)

    cache = model.new_cache()
    token = torch.full((rows * width, 1), config.bos_id, dtype=torch.long, device=device)

    # Only the first beam of each row is alive at the start. If they all began at zero the top-k
    # would return the same token `width` times over and every beam would be a copy of the others.
    scores = torch.full((rows, width), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(rows * width)

    finished = torch.zeros(rows * width, dtype=torch.bool, device=device)
    sequences = torch.zeros(rows * width, 0, dtype=torch.long, device=device)
    offset = torch.arange(rows, device=device).unsqueeze(1) * width

    limit = min(max_length, config.max_positions - 1)
    for position in range(limit):
        logits = model.decode_step(token, memory, mask, cache, position)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # A finished beam is frozen, not dropped: it stays in the running with the score it
        # already has. Its only legal continuation is padding, at no cost. Dropping it instead
        # would let a worse but unfinished beam take its place in the top-k.
        log_probs = log_probs.masked_fill(finished.unsqueeze(1), float("-inf"))
        log_probs[finished, config.pad_id] = 0.0

        total = (scores.unsqueeze(1) + log_probs).view(rows, width * vocab)
        best, index = total.topk(width, dim=-1)

        source_beam = (offset + index // vocab).view(-1)
        next_token = (index % vocab).view(-1)

        sequences = torch.cat([sequences[source_beam], next_token.unsqueeze(1)], dim=1)
        finished = finished[source_beam] | (next_token == config.eos_id)
        scores = best.view(-1)

        # The surviving beams are not the ones that held these cache slots a step ago, so the
        # keys and values have to travel with them.
        for self_cache, cross_cache in cache:
            for held in (self_cache, cross_cache):
                if held.key is not None:
                    held.key = held.key[source_beam]
                if held.value is not None:
                    held.value = held.value[source_beam]

        if bool(finished.all()):
            break
        token = next_token.unsqueeze(1)

    counts = (sequences != config.pad_id).sum(dim=1).clamp(min=1).to(scores.dtype)
    normalised = (scores / counts**length_penalty).view(rows, width)
    winner = (offset.squeeze(1) + normalised.argmax(dim=-1)).view(-1)
    return sequences[winner]


@torch.no_grad()
def greedy_decode_uncached(
    model: Seq2SeqTransformer,
    source_ids: Tensor,
    source_mask: Tensor,
    max_length: int,
) -> Tensor:
    """The same thing, rerunning the whole prefix each step. Kept to check the cache."""
    model.eval()
    config = model.config
    batch = source_ids.size(0)
    device = source_ids.device

    memory = model.encode(source_ids, source_mask)
    generated = torch.full((batch, 1), config.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    limit = min(max_length, config.max_positions - 1)
    for _ in range(limit):
        target_mask = torch.ones_like(generated)
        hidden = model.decode(generated, memory, source_mask, target_mask)
        scores = model.embedding.project(hidden[:, -1])

        chosen = scores.argmax(dim=-1).masked_fill(finished, config.pad_id)
        generated = torch.cat([generated, chosen.unsqueeze(1)], dim=1)

        finished |= chosen == config.eos_id
        if bool(finished.all()):
            break

    return generated[:, 1:]
