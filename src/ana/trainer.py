"""The training loop.

Training is counted in optimiser steps. Counting in epochs would tie the amount of
optimisation to the size of the corpus, so a comparison across corpora would also be a
comparison of training budgets, and the two would be impossible to tell apart afterwards.

WHICH WEIGHTS GET SCORED IS PART OF THE RECIPE, NOT A DETAIL OF THE LOOP.

Under `Selection.BEST_DEV_LOSS` the model kept is the one with the lowest development loss.
That is the ordinary choice and it is what the translation corpora use.

Under `Selection.FINAL` the weights the run ended on are scored, and the development loss is
recorded but never acted on. COGS needs this: there, development loss and generalization
accuracy move in the same direction rather than opposite ones, so keeping the lowest-loss
checkpoint discards the model that generalizes best. Csordas et al. (2021) measure the cost at
thirty points of accuracy, and the seed-to-seed spread it introduces is larger than any
difference this study is trying to detect.

Either way the rule is the same for every model in a comparison, so it cannot favour one.
There is no separate rule for picking a checkpoint off disk later: what was scored is what was
selected.
"""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from ana.config import Selection, TrainConfig
from ana.data.corpus import Batch, collate
from ana.model import Seq2SeqTransformer


@dataclass
class TrainOutcome:
    steps_run: int
    best_dev_loss: float
    best_step: int
    # The step whose weights were actually scored. Under BEST_DEV_LOSS it is `best_step`; under
    # FINAL it is the last step, whatever the development loss was doing by then. Recording both
    # is what lets a reader of the manifest see which rule a run was under.
    scored_step: int
    selection: Selection
    seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_stream(
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    pad_id: int,
    seed: int,
) -> Iterator[Batch]:
    """Shuffle, cut into batches, repeat. Reshuffled with a fresh order on every pass."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            picked = [examples[i] for i in order[start : start + batch_size]]
            yield collate(picked, pad_id)


def fixed_batches(
    examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    pad_id: int,
) -> list[Batch]:
    return [
        collate(examples[start : start + batch_size], pad_id)
        for start in range(0, len(examples), batch_size)
    ]


@torch.no_grad()
def dev_loss(model: Seq2SeqTransformer, batches: list[Batch], device: torch.device) -> float:
    model.eval()
    total, counted = 0.0, 0
    for batch in batches:
        batch = batch.to(device)
        loss, _ = model(batch.source_ids, batch.source_mask, batch.labels)
        total += float(loss) * len(batch)
        counted += len(batch)
    model.train()
    return total / max(counted, 1)


def train(
    model: Seq2SeqTransformer,
    train_examples: list[tuple[list[int], list[int]]],
    dev_examples: list[tuple[list[int], list[int]]],
    config: TrainConfig,
    device: torch.device,
    on_eval=None,
) -> TrainOutcome:
    # The learning rate is derived from the width of the model unless it was set by hand, and
    # the trainer is the first place that knows both. Resolving here means no caller can start
    # a run with a peak that does not belong to the model it is training.
    config = config.resolved(model.config.d_model)

    set_seed(config.seed)
    model.to(device).train()

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    pad_id = model.config.pad_id
    stream = batch_stream(train_examples, config.batch_size, pad_id, config.seed)
    dev_batches = fixed_batches(dev_examples, config.batch_size, pad_id)

    # The development loss is tracked under both rules, because it is what `ana tune` chooses a
    # learning rate on and what tells us a run was still improving when its budget ran out. It
    # is only ACTED on -- by keeping a checkpoint -- under BEST_DEV_LOSS.
    keep_best = config.selection is Selection.BEST_DEV_LOSS

    best_loss, best_step = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict()) if keep_best else None
    started = time.monotonic()

    for step in range(1, config.max_steps + 1):
        for group in optimiser.param_groups:
            group["lr"] = config.learning_rate_at(step)

        batch = next(stream).to(device)
        loss, _ = model(batch.source_ids, batch.source_mask, batch.labels)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimiser.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            current = dev_loss(model, dev_batches, device)
            if current < best_loss:
                best_loss, best_step = current, step
                if keep_best:
                    best_state = copy.deepcopy(model.state_dict())
            if on_eval is not None:
                on_eval(step, float(loss.detach()), current, best_loss)

    if keep_best and best_state is not None:
        model.load_state_dict(best_state)

    return TrainOutcome(
        steps_run=config.max_steps,
        best_dev_loss=best_loss,
        best_step=best_step,
        scored_step=best_step if keep_best else config.max_steps,
        selection=config.selection,
        seconds=time.monotonic() - started,
    )
