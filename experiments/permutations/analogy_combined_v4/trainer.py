"""Single-pass cross-entropy training with resumable corpus recipes."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from models import collect_diagnostics, optimizer_parameter_groups, reset_diagnostics
from torch import nn

from ana.config import Selection, TrainConfig
from ana.data.corpus import Batch, collate
from ana.trainer import fixed_batches, set_seed


@dataclass
class TrainOutcome:
    steps_run: int
    best_dev_loss: float
    best_step: int
    scored_step: int
    selection: Selection
    seconds: float
    micro_batch_size: int
    diagnostics_final: dict | None = None
    diagnostics_scored: dict | None = None
    diagnostics_history: list[dict] = field(default_factory=list)


def plain(value):
    return json.loads(json.dumps(value))


def _atomic_checkpoint_save(state: dict, path: Path) -> None:
    """Retain the previous checkpoint until its replacement is fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def batch_stream(examples, batch_size, pad_id, seed, skip_batches=0):
    """Replay shuffled indices without loading batches already checkpointed."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            if skip_batches:
                skip_batches -= 1
                continue
            yield collate([examples[i] for i in order[start:start + batch_size]], pad_id)


def batch_loss(model, batch, micro_batch_size, *, backward=False):
    """One dropout pass per example and one update per original-size batch."""
    valid_total = batch.labels.ne(-100).sum().clamp_min(1)
    total = None
    for start in range(0, len(batch), micro_batch_size):
        stop = start + micro_batch_size
        part = Batch(
            batch.source_ids[start:stop], batch.source_mask[start:stop], batch.labels[start:stop]
        )
        loss, logits = model(part.source_ids, part.source_mask, part.labels)
        del logits
        weight = part.labels.ne(-100).sum().to(loss.dtype) / valid_total
        weighted = loss * weight
        if backward:
            weighted.backward()
        detached = weighted.detach()
        total = detached if total is None else total + detached
        del loss, weighted, part
    return total


@torch.no_grad()
def dev_loss(model, batches, device, micro_batch_size):
    """Keep the original dev-loss aggregation and collect real-token diagnostics."""
    model.eval()
    reset_diagnostics(model, enabled=True)
    total, counted = 0.0, 0
    try:
        for batch in batches:
            loss = batch_loss(model, batch.to(device), micro_batch_size)
            total += float(loss) * len(batch)
            counted += len(batch)
        diagnostics = collect_diagnostics(model, disable=True)
    finally:
        reset_diagnostics(model, enabled=False)
        model.train()
    return total / max(counted, 1), diagnostics


def train(
    model,
    train_examples,
    dev_examples,
    config: TrainConfig,
    device,
    on_eval=None,
    *,
    checkpoint_path=None,
    checkpoint_context=None,
    micro_batch_size=None,
) -> TrainOutcome:
    config = config.resolved(model.config.d_model)
    micro_batch_size = min(config.batch_size, micro_batch_size or config.batch_size)
    if micro_batch_size < 1 or len(train_examples) < config.batch_size:
        raise ValueError("Need positive microbatches and at least one full training batch.")
    set_seed(config.seed)
    model.to(device).train()
    reset_diagnostics(model, enabled=False)
    parameter_groups = optimizer_parameter_groups(model)
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    group_metadata = [
        {
            "group_name": group.get("group_name"),
            "lr_scale": float(group["lr_scale"]),
            "parameter_names": [parameter_names[id(value)] for value in group["params"]],
        }
        for group in parameter_groups
    ]
    optimiser = torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    dev_batches = fixed_batches(dev_examples, config.batch_size, model.config.pad_id)
    keep_best = config.selection is Selection.BEST_DEV_LOSS
    best_loss, best_step, best_state = float("inf"), 0, None
    best_diagnostics = last_diagnostics = None
    history = []
    completed_steps, previous_seconds = 0, 0.0
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    compatibility = {
        "format_version": 1,
        "train_config": plain(asdict(config)),
        "model_config": plain(asdict(model.config)),
        "checkpoint_context": plain(checkpoint_context),
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "micro_batch_size": micro_batch_size,
        "objective": "ordinary_single_pass_cross_entropy",
        "optimizer_groups": group_metadata,
    }
    if path is not None and path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        for key, value in compatibility.items():
            if saved.get(key) != value:
                raise ValueError(f"Resume checkpoint mismatch: {key}")
        completed_steps = saved["step"]
        if not 0 < completed_steps <= config.max_steps:
            raise ValueError("Invalid completed training step.")
        model.load_state_dict(saved["model_state"])
        optimiser.load_state_dict(saved["optimizer_state"])
        best_loss, best_step = saved["best_dev_loss"], saved["best_step"]
        best_state = saved["best_state"]
        previous_seconds = saved["seconds"]
        last_diagnostics = saved["diagnostics_last"]
        best_diagnostics = saved["diagnostics_best"]
        history = saved["diagnostics_history"]
        random.setstate(saved["python_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        del saved
        if best_state is not None:
            _atomic_checkpoint_save({
                "state_dict": best_state, "step": best_step,
                "best_dev_loss": best_loss, "context": plain(checkpoint_context),
                "diagnostics": best_diagnostics, "diagnostic_only": not keep_best,
            }, path.with_name("best.pt"))
        print(f"Resumed step {completed_steps}.", flush=True)
    stream = batch_stream(
        train_examples, config.batch_size, model.config.pad_id, config.seed, completed_steps
    )
    started = time.monotonic()
    for step in range(completed_steps + 1, config.max_steps + 1):
        for group in optimiser.param_groups:
            group["lr"] = config.learning_rate_at(step) * group["lr_scale"]
        batch = next(stream).to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, micro_batch_size, backward=True)
        nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm, error_if_nonfinite=True
        )
        optimiser.step()
        if step - completed_steps in (1, 100) or step % 200 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({
                "phase": "training_progress", "step": step, "train_loss": float(loss),
                "elapsed_seconds": elapsed,
                "steps_per_second": (step - completed_steps) / max(elapsed, 1e-9),
            }, allow_nan=False), flush=True)
        if step % config.eval_every == 0 or step == config.max_steps:
            current, diagnostics = dev_loss(model, dev_batches, device, micro_batch_size)
            last_diagnostics = {"step": step, "scope": "development_real_tokens", **diagnostics}
            history.append(last_diagnostics)
            if current < best_loss:
                best_loss, best_step = current, step
                best_diagnostics = last_diagnostics
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                if path is not None:
                    _atomic_checkpoint_save({
                        "state_dict": best_state, "step": best_step,
                        "best_dev_loss": best_loss,
                        "context": plain(checkpoint_context),
                        "diagnostics": best_diagnostics,
                        "diagnostic_only": not keep_best,
                    }, path.with_name("best.pt"))
            if on_eval is not None:
                on_eval(step, float(loss), current, best_loss)
            print(json.dumps({
                "phase": "analogy_diagnostics", **last_diagnostics,
            }, allow_nan=False), flush=True)
            if path is not None:
                _atomic_checkpoint_save({
                    **compatibility,
                    "step": step,
                    "seconds": previous_seconds + time.monotonic() - started,
                    "best_dev_loss": best_loss, "best_step": best_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimiser.state_dict(), "best_state": best_state,
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                    ),
                    "diagnostics_last": last_diagnostics,
                    "diagnostics_best": best_diagnostics, "diagnostics_history": history,
                }, path)
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    return TrainOutcome(
        steps_run=config.max_steps,
        best_dev_loss=best_loss,
        best_step=best_step,
        scored_step=best_step if keep_best else config.max_steps,
        selection=config.selection,
        seconds=previous_seconds + time.monotonic() - started,
        micro_batch_size=micro_batch_size,
        diagnostics_final=last_diagnostics,
        diagnostics_scored=best_diagnostics if keep_best else last_diagnostics,
        diagnostics_history=history,
    )
