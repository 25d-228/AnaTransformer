"""Two-view CE control and prediction-guided adaptive-power consistency.

The model and inference path are unchanged. Adaptive p is a training-only
scalar: two discarded AdamW updates on training batch A are compared by their
ordinary prediction loss on disjoint training batch B. No development or test
examples enter that choice. Only real updates advance the model or optimizer.
"""

from __future__ import annotations

import copy
import errno
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from ana.config import Selection, TrainConfig
from ana.data.corpus import Batch, collate
from ana.model import Seq2SeqTransformer

CONSISTENCY_POWER = 0.5
CONSISTENCY_WEIGHT = 1.0
PROBABILITY_FLOOR_MASS = 1e-6
POWER_MIN = 0.25
POWER_MAX = 0.75
PROBE_EVERY = 200
PROBE_OFFSET = 0.05
POWER_MOVE = 0.01
PROBE_CE_DEADBAND = 1e-7
PROBE_RNG_OFFSET = 91073


def consistency_loss_config(adaptive: bool = True) -> dict:
    """Objective and finite-difference policy recorded with every checkpoint."""
    common = {
        "name": (
            "all_pairs_prediction_guided_adaptive_power_consistency"
            if adaptive
            else "two_dropout_views_supervised_ce_only"
        ),
        "dropout_views": 2,
        "both_views_receive_gradients": True,
        "supervised_loss": "mean of two existing label-smoothed cross-entropies",
        "target_ignore_index": -100,
        "weight": CONSISTENCY_WEIGHT if adaptive else 0.0,
    }
    if adaptive:
        common.update(
            {
                "initial_power": CONSISTENCY_POWER,
                "power_bounds": [POWER_MIN, POWER_MAX],
                "probability_floor_mass": PROBABILITY_FLOOR_MASS,
                "probability_power_difference_scale": "V^(p-1/2)/(2*p)",
                "normalization_scope": (
                    "equal derivative at uniform probabilities, not equal global loss scale"
                ),
                "consistency_reduction": (
                    "sum over centered vocabulary defects, mean over valid target positions"
                ),
                "adaptation": {
                    "name": "bounded_sign_finite_difference_of_prediction_loss",
                    "interval_steps": PROBE_EVERY,
                    "start": "first interval boundary strictly after optimizer warmup",
                    "candidate_offset": PROBE_OFFSET,
                    "maximum_power_move": POWER_MOVE,
                    "absolute_ce_deadband": PROBE_CE_DEADBAND,
                    "probe_batch_size": (
                        "16 when maximum training target length exceeds 128, otherwise 32; capped"
                        " at half the training set"
                    ),
                    "sampling": (
                        "independent persistent RNG, disjoint training-only A and B sampled"
                        " without replacement per probe"
                    ),
                    "probe_rng_seed_offset": PROBE_RNG_OFFSET,
                    "candidate_update": (
                        "one exact clipped AdamW update at current learning rate, current moments,"
                        " common dropout RNG"
                    ),
                    "criterion": "ordinary one-pass label-smoothed CE on batch B, dropout off",
                    "restore": (
                        "model, optimizer, gradients, mode flags, Python/Torch/device RNG; discard"
                        " both hypothetical updates"
                    ),
                    "test_or_development_feedback": False,
                },
            }
        )
    return common


def power_consistency(
    logits1: Tensor,
    logits2: Tensor,
    labels: Tensor,
    *,
    power: float = CONSISTENCY_POWER,
) -> Tensor:
    """Exact centered all-pairs defect; p=.5 equals the existing square-root loss.

    Delta_i = V**(p-.5)/(2*p) * (Q1_i**p-Q2_i**p). Its centered square
    sum is the scaled sum of all four-term vocabulary-pair analogy defects,
    computed without allocating a vocabulary-by-vocabulary tensor.
    """
    if logits1.shape != logits2.shape or logits1.ndim < 2 or logits1.shape[-1] < 1:
        raise ValueError("dropout logits must have the same nonempty vocabulary shape")
    if labels.shape != logits1.shape[:-1]:
        raise ValueError("labels must align with every predicted target position")
    if not logits1.is_floating_point() or not logits2.is_floating_point():
        raise ValueError("dropout logits must be floating point")
    if isinstance(power, bool) or not math.isfinite(power) or not POWER_MIN <= power <= POWER_MAX:
        raise ValueError(f"power must be in [{POWER_MIN}, {POWER_MAX}]")
    dtype = torch.promote_types(logits1.dtype, logits2.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    vocabulary = logits1.shape[-1]
    floor = PROBABILITY_FLOOR_MASS / vocabulary
    prob1 = (
        torch.softmax(logits1, dim=-1, dtype=dtype).mul(1.0 - PROBABILITY_FLOOR_MASS).add(floor)
    )
    prob2 = (
        torch.softmax(logits2, dim=-1, dtype=dtype).mul(1.0 - PROBABILITY_FLOOR_MASS).add(floor)
    )
    # Keep the original sqrt operation exactly at the existing reference power.
    if power == 0.5:
        delta = prob1.sqrt() - prob2.sqrt()
    else:
        scale = vocabulary ** (power - 0.5) / (2.0 * power)
        delta = scale * (prob1.pow(power) - prob2.pow(power))
    # sqrt/pow retain only what backward needs; avoid holding extra probability
    # references while materializing centered defects on small-memory cards.
    del prob1, prob2
    centered = delta - delta.mean(dim=-1, keepdim=True)
    per_token = centered.square().sum(dim=-1)
    valid = labels.ne(-100).to(dtype=dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


@dataclass
class TrainOutcome:
    steps_run: int
    best_dev_loss: float
    best_step: int
    scored_step: int
    selection: Selection
    seconds: float
    power_final: float | None = None
    power_probe_seconds: float = 0.0
    power_history: list[dict] = field(default_factory=list)
    micro_batch_size: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_stream(
    examples, batch_size: int, pad_id: int, seed: int, *, skip_batches: int = 0
) -> Iterator[Batch]:
    rng = random.Random(seed)
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            if skip_batches:
                skip_batches -= 1
                continue
            yield collate([examples[i] for i in order[start : start + batch_size]], pad_id)


def _atomic_checkpoint_save(state: dict, path: Path) -> None:
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
                if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _plain(value):
    return json.loads(json.dumps(value))


def fixed_batches(examples, batch_size: int, pad_id: int) -> list[Batch]:
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


def _two_view_loss(model, batch: Batch, power: float, adaptive: bool):
    loss1, logits1 = model(batch.source_ids, batch.source_mask, batch.labels)
    loss2, logits2 = model(batch.source_ids, batch.source_mask, batch.labels)
    supervised = 0.5 * (loss1 + loss2)
    consistency = (
        power_consistency(logits1, logits2, batch.labels, power=power)
        if adaptive
        else supervised.new_zeros(())
    )
    return supervised + CONSISTENCY_WEIGHT * consistency, supervised, consistency


def _backward_two_view_batch(
    model, batch: Batch, power: float, adaptive: bool, micro_batch_size: int
):
    """Accumulate one original-batch gradient, weighted by valid target tokens.

    Both underlying losses are token means, not means over sentences. Keep the
    original batch's padding shape and normalize each contribution by the same
    full-batch token count. The caller clips and steps only after all parts.
    """
    if micro_batch_size >= len(batch):
        loss, supervised, consistency = _two_view_loss(model, batch, power, adaptive)
        loss.backward()
        return loss.detach(), supervised.detach(), consistency.detach()

    valid_total = batch.labels.ne(-100).sum().clamp_min(1)
    totals = None
    for start in range(0, len(batch), micro_batch_size):
        stop = start + micro_batch_size
        part = Batch(
            batch.source_ids[start:stop], batch.source_mask[start:stop], batch.labels[start:stop]
        )
        loss, supervised, consistency = _two_view_loss(model, part, power, adaptive)
        weight = part.labels.ne(-100).sum().to(dtype=loss.dtype) / valid_total
        (loss * weight).backward()
        values = tuple(
            component.detach() * weight for component in (loss, supervised, consistency)
        )
        totals = (
            values
            if totals is None
            else tuple(total + value for total, value in zip(totals, values, strict=False))
        )
        del loss, supervised, consistency, part
    return totals


def _cpu_snapshot(value):
    """Temporary state lives on CPU, not in a second full GPU model."""
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _adapt_power(
    model,
    optimiser,
    update_batch: Batch,
    score_batch: Batch,
    power: float,
    config: TrainConfig,
    device: torch.device,
) -> tuple[float, dict]:
    """Compare p-neighbors from identical state, then restore that state exactly."""
    started = time.monotonic()
    candidates = [max(POWER_MIN, power - PROBE_OFFSET), min(POWER_MAX, power + PROBE_OFFSET)]
    model_state = _cpu_snapshot(model.state_dict())
    optimizer_state = _cpu_snapshot(optimiser.state_dict())
    parameters = list(model.parameters())
    gradients = [
        None if parameter.grad is None else _cpu_snapshot(parameter.grad)
        for parameter in parameters
    ]
    modes = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    def restore():
        model.load_state_dict(model_state)
        # On CPU, load_state_dict can reuse incoming tensors. Clone the saved
        # snapshot so each hypothetical optimizer step starts at the same moments.
        optimiser.load_state_dict(copy.deepcopy(optimizer_state))
        for parameter, gradient in zip(parameters, gradients, strict=False):
            parameter.grad = (
                None if gradient is None else gradient.to(device=parameter.device, copy=True)
            )
        for module, mode in modes:
            module.training = mode
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)

    scores = []
    try:
        for candidate in candidates:
            restore()
            model.train()
            optimiser.zero_grad(set_to_none=True)
            loss, _, _ = _two_view_loss(model, update_batch, candidate, True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimiser.step()
            del loss
            optimiser.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                predictive_loss, logits = model(
                    score_batch.source_ids, score_batch.source_mask, score_batch.labels
                )
                scores.append(float(predictive_loss))
                del predictive_loss, logits
    finally:
        restore()

    if not all(math.isfinite(score) for score in scores):
        raise FloatingPointError("non-finite prediction loss in adaptive-power probe")
    difference = scores[1] - scores[0]
    gradient = difference / (candidates[1] - candidates[0])
    # Sign finite-difference descent prevents one unusually large loss difference
    # from moving p more than .01. Ties at float32 resolution leave p unchanged.
    movement = (
        0.0 if abs(difference) <= PROBE_CE_DEADBAND else -math.copysign(POWER_MOVE, gradient)
    )
    next_power = min(POWER_MAX, max(POWER_MIN, round(power + movement, 12)))
    return next_power, {
        "power_before": power,
        "power_after": next_power,
        "candidate_powers": candidates,
        "candidate_prediction_ce": scores,
        "finite_difference_gradient": gradient,
        "probe_update_examples": len(update_batch),
        "probe_score_examples": len(score_batch),
        "seconds": time.monotonic() - started,
    }


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
    adaptive: bool = True,
) -> TrainOutcome:
    config = config.resolved(model.config.d_model)
    requested_micro_batch_size = int(os.environ.get("ANA_MICRO_BATCH_SIZE", "0"))
    if requested_micro_batch_size < 0:
        raise ValueError("ANA_MICRO_BATCH_SIZE must be zero (full batch) or positive")
    micro_batch_size = min(config.batch_size, requested_micro_batch_size or config.batch_size)
    if len(train_examples) < config.batch_size:
        raise ValueError("training data must contain at least one complete batch")
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
    keep_best = config.selection is Selection.BEST_DEV_LOSS
    best_loss, best_step = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict()) if keep_best else None
    completed_steps, previous_seconds = 0, 0.0
    power = CONSISTENCY_POWER
    probe_rng = random.Random(config.seed + PROBE_RNG_OFFSET)
    probe_batch_size = min(
        16 if max(len(target) for _, target in train_examples) > 128 else 32,
        len(train_examples) // 2,
    )
    if adaptive and probe_batch_size < 1:
        raise ValueError("adaptive training needs at least two training examples")
    power_history = []
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    compatibility = None
    print(
        json.dumps(
            {
                "phase": "training_execution",
                "effective_batch_size": config.batch_size,
                "micro_batch_size": micro_batch_size,
                "accumulation_weight": (
                    "valid target token count / full batch valid target token count"
                ),
            }
        ),
        flush=True,
    )
    if path is not None:
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
            "loss_configuration": consistency_loss_config(adaptive),
            "device_type": device.type,
            "probe_batch_size": probe_batch_size if adaptive else 0,
            "micro_batch_size": micro_batch_size,
        }
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for key, expected in compatibility.items():
                # Earlier full-batch checkpoints predate this execution option.
                actual = (
                    saved.get(key, config.batch_size)
                    if key == "micro_batch_size"
                    else saved.get(key)
                )
                if actual != expected:
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
            power = saved["adaptive_power_state"]["power"]
            if not math.isfinite(power) or not POWER_MIN <= power <= POWER_MAX:
                raise ValueError("invalid adaptive power in resume checkpoint")
            probe_rng.setstate(saved["adaptive_power_state"]["probe_rng_state"])
            power_history = saved["adaptive_power_state"]["history"]
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
        power_used = power
        optimiser.zero_grad(set_to_none=True)
        loss, supervised, consistency = _backward_two_view_batch(
            model, batch, power, adaptive, micro_batch_size
        )
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimiser.step()

        if adaptive and step > config.warmup_steps and step % PROBE_EVERY == 0:
            optimiser.zero_grad(set_to_none=True)
            picked = probe_rng.sample(range(len(train_examples)), 2 * probe_batch_size)
            update_batch = collate(
                [train_examples[index] for index in picked[:probe_batch_size]], pad_id
            ).to(device)
            score_batch = collate(
                [train_examples[index] for index in picked[probe_batch_size:]], pad_id
            ).to(device)
            power, probe = _adapt_power(
                model, optimiser, update_batch, score_batch, power, config, device
            )
            probe["step"] = step
            power_history.append(probe)
            print(
                json.dumps({"phase": "adaptive_power_probe", **probe}, allow_nan=False), flush=True
            )
            del update_batch, score_batch

        components = {
            "step": step,
            "train_power": power_used if adaptive else None,
            "next_power": power if adaptive else None,
        }
        if (
            step - completed_steps in (1, 100)
            or step % config.eval_every == 0
            or step == config.max_steps
        ):
            components.update(
                {
                    "train_loss": float(loss),
                    "train_ce": float(supervised),
                    "train_consistency": float(consistency),
                }
            )
        if step - completed_steps in (1, 100):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "phase": "early_training",
                        **components,
                        "elapsed_seconds": elapsed,
                        "steps_per_second": (step - completed_steps) / max(elapsed, 1e-9),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        if step % config.eval_every == 0 or step == config.max_steps:
            print(
                json.dumps({"phase": "training_components", **components}, allow_nan=False),
                flush=True,
            )
            current = dev_loss(model, dev_batches, device)
            if current < best_loss:
                best_loss, best_step = current, step
                if keep_best:
                    best_state = copy.deepcopy(model.state_dict())
            if on_eval is not None:
                on_eval(step, float(loss), current, best_loss)
            if path is not None:
                _atomic_checkpoint_save(
                    {
                        **compatibility,
                        "step": step,
                        "seconds": previous_seconds + time.monotonic() - started,
                        "last_dev_loss": current,
                        "best_dev_loss": best_loss,
                        "best_step": best_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimiser.state_dict(),
                        "best_state": best_state,
                        "python_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": (
                            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                        ),
                        "adaptive_power_state": {
                            "power": power,
                            "probe_rng_state": probe_rng.getstate(),
                            "history": power_history,
                        },
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
        power_final=power if adaptive else None,
        power_probe_seconds=sum(probe["seconds"] for probe in power_history),
        power_history=power_history,
        micro_batch_size=micro_batch_size,
    )
