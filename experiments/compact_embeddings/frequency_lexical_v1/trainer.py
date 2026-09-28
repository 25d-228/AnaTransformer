"""Task-local training with two-view, fixed-power prediction consistency.

Two independent dropout passes keep gradients. Their existing supervised CE
losses are averaged, then an exact all-vocabulary-pairs power-analogy penalty
is added. The architecture and one-pass development/inference paths do not
change. This is a separate training objective, not an inference-time module.

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
import errno
import json
import os
import random
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ana.config import Selection, TrainConfig
from ana.data.corpus import Batch, collate
from ana.model import Seq2SeqTransformer

CONSISTENCY_POWER = 0.5
CONSISTENCY_WEIGHT = 1.0
PROBABILITY_FLOOR_MASS = 1e-6


def consistency_loss_config() -> dict:
    """Fixed task objective, included in checkpoint compatibility and manifests."""
    return {
        "name": "all_pairs_power_dropout_consistency",
        "power": CONSISTENCY_POWER,
        "weight": CONSISTENCY_WEIGHT,
        "probability_floor_mass": PROBABILITY_FLOOR_MASS,
        "dropout_views": 2,
        "both_views_receive_gradients": True,
        "supervised_loss": "mean of two existing label-smoothed cross-entropies",
        "consistency_reduction": (
            "sum over centered vocabulary defects, mean over valid target positions"
        ),
        "target_ignore_index": -100,
    }


def power_consistency(logits1: Tensor, logits2: Tensor, labels: Tensor) -> Tensor:
    """Exact all-pairs p=0.5 defect without a vocabulary-by-vocabulary tensor.

    For two dropout distributions Q1,Q2, delta_i=sqrt(Q1_i)-sqrt(Q2_i).
    Sum_i(delta_i-mean(delta))^2 equals (1/V) times the sum over i<j of
    (sqrt(Q1_i)+sqrt(Q2_j)-sqrt(Q1_j)-sqrt(Q2_i))^2. The tiny uniform
    mixture makes all four probability terms positive. Only label positions
    used by the ordinary supervised CE contribute; target padding is -100.
    """
    if logits1.shape != logits2.shape or logits1.ndim < 2 or logits1.shape[-1] < 1:
        raise ValueError("dropout logits must have the same nonempty vocabulary shape")
    if labels.shape != logits1.shape[:-1]:
        raise ValueError("labels must align with every predicted target position")
    if not logits1.is_floating_point() or not logits2.is_floating_point():
        raise ValueError("dropout logits must be floating point")
    dtype = torch.promote_types(logits1.dtype, logits2.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    vocabulary = logits1.shape[-1]
    floor = PROBABILITY_FLOOR_MASS / vocabulary
    # softmax's dtype avoids retaining separate full float32 copies of low-
    # precision logits. Do not mutate its output: backward needs that tensor.
    root1 = (
        torch.softmax(logits1, dim=-1, dtype=dtype)
        .mul(1.0 - PROBABILITY_FLOOR_MASS)
        .add(floor)
        .sqrt()
    )
    root2 = (
        torch.softmax(logits2, dim=-1, dtype=dtype)
        .mul(1.0 - PROBABILITY_FLOOR_MASS)
        .add(floor)
        .sqrt()
    )
    delta = root1 - root2
    centered = delta - delta.mean(dim=-1, keepdim=True)
    per_token = centered.square().sum(dim=-1)
    valid = labels.ne(-100).to(dtype=dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


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
    *,
    skip_batches: int = 0,
) -> Iterator[Batch]:
    """Replay only shuffled indices when resuming; never collate skipped batches."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            if skip_batches:
                skip_batches -= 1
                continue
            picked = [examples[i] for i in order[start : start + batch_size]]
            yield collate(picked, pad_id)


def _atomic_checkpoint_save(state: dict, path: Path) -> None:
    """Keep the previous checkpoint intact until the new one is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                # Some shared filesystems do not support directory fsync. The file
                # itself was still flushed before the atomic rename.
                if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _plain(value):
    """Keep enum subclasses out of the trusted, weights-only-loadable payload."""
    return json.loads(json.dumps(value))


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
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_context: dict | None = None,
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
    dev_batches = fixed_batches(dev_examples, config.batch_size, pad_id)

    # The development loss is tracked under both rules, because it is what `ana tune` chooses a
    # learning rate on and what tells us a run was still improving when its budget ran out. It
    # is only ACTED on -- by keeping a checkpoint -- under BEST_DEV_LOSS.
    keep_best = config.selection is Selection.BEST_DEV_LOSS

    best_loss, best_step = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict()) if keep_best else None
    completed_steps, previous_seconds = 0, 0.0
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    compatibility = None
    if path is not None:
        if len(train_examples) < config.batch_size:
            raise ValueError("training data must contain at least one complete batch")
        compatibility = {
            "format_version": 1,
            "train_config": _plain(asdict(config)),
            "model_config": _plain(asdict(model.config)),
            "model_schema": [
                (name, tuple(value.shape), str(value.dtype))
                for name, value in model.state_dict().items()
            ],
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "checkpoint_context": _plain(checkpoint_context),
            "loss_configuration": consistency_loss_config(),
            "device_type": device.type,
        }
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for key, expected in compatibility.items():
                if saved.get(key) != expected:
                    raise ValueError(f"resume checkpoint mismatch: {key}")
            completed_steps = saved["step"]
            if not isinstance(completed_steps, int) or not 0 < completed_steps <= config.max_steps:
                raise ValueError("invalid completed step in resume checkpoint")
            if completed_steps != config.max_steps and completed_steps % config.eval_every:
                raise ValueError("resume checkpoint is not at an evaluation boundary")
            if not 0 < saved["best_step"] <= completed_steps:
                raise ValueError("invalid best step in resume checkpoint")
            if keep_best != (saved["best_state"] is not None):
                raise ValueError("resume checkpoint has the wrong weight-selection state")
            model.load_state_dict(saved["model_state"])
            optimiser.load_state_dict(saved["optimizer_state"])
            best_loss, best_step = saved["best_dev_loss"], saved["best_step"]
            best_state = saved["best_state"]
            if best_state is not None:
                best_state = {name: value.to(device) for name, value in best_state.items()}
            previous_seconds = saved["seconds"]
            random.setstate(saved["python_rng_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
            del saved

    stream = batch_stream(
        train_examples, config.batch_size, pad_id, config.seed, skip_batches=completed_steps
    )
    started = time.monotonic()

    for step in range(completed_steps + 1, config.max_steps + 1):
        for group in optimiser.param_groups:
            group["lr"] = config.learning_rate_at(step)

        batch = next(stream).to(device)
        loss1, logits1 = model(batch.source_ids, batch.source_mask, batch.labels)
        loss2, logits2 = model(batch.source_ids, batch.source_mask, batch.labels)
        supervised = 0.5 * (loss1 + loss2)
        consistency = power_consistency(logits1, logits2, batch.labels)
        loss = supervised + CONSISTENCY_WEIGHT * consistency
        # Keep their gradients through the scalar losses, but do not retain
        # unnecessary full-vocabulary output references through evaluation.
        del loss1, loss2, logits1, logits2

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimiser.step()
        # Only scalar device tensors survive the step. Detaching here releases
        # graph references before development scoring/checkpointing without a
        # host synchronization on ordinary, unlogged updates.
        loss = loss.detach()
        supervised = supervised.detach()
        consistency = consistency.detach()

        if step - completed_steps in (1, 100):
            early_loss = float(loss.detach())
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "phase": "early_training",
                        "step": step,
                        "train_loss": early_loss,
                        "train_ce": float(supervised.detach()),
                        "train_consistency": float(consistency.detach()),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": (step - completed_steps) / max(elapsed, 1e-9),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        if step % config.eval_every == 0 or step == config.max_steps:
            print(
                json.dumps(
                    {
                        "phase": "training_components",
                        "step": step,
                        "train_loss": float(loss.detach()),
                        "train_ce": float(supervised.detach()),
                        "train_consistency": float(consistency.detach()),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            current = dev_loss(model, dev_batches, device)
            if current < best_loss:
                best_loss, best_step = current, step
                if keep_best:
                    best_state = copy.deepcopy(model.state_dict())
            if on_eval is not None:
                on_eval(step, float(loss.detach()), current, best_loss)
            if path is not None:
                _atomic_checkpoint_save(
                    {
                        **compatibility,
                        "step": step,
                        "seconds": previous_seconds + time.monotonic() - started,
                        "last_dev_loss": current,
                        "best_dev_loss": best_loss,
                        "best_step": best_step,
                        # These are the current training weights, not selected dev weights.
                        "model_state": model.state_dict(),
                        "optimizer_state": optimiser.state_dict(),
                        "best_state": best_state,
                        "python_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": (
                            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                        ),
                    },
                    path,
                )

    if keep_best and best_state is not None:
        model.load_state_dict(best_state)

    return TrainOutcome(
        steps_run=config.max_steps,
        best_dev_loss=best_loss,
        best_step=best_step,
        scored_step=best_step if keep_best else config.max_steps,
        selection=config.selection,
        seconds=previous_seconds + time.monotonic() - started,
    )
