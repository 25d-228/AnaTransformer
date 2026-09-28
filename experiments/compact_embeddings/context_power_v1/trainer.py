"""Two-view training with prediction-trained, context-dependent powers.

The compact backbone and dataset recipes stay unchanged. A small prediction
component learns powers through ordinary supervised CE. The analogy exponent
is the detached mean of the two dropout-view powers, so its head cannot reduce
the penalty merely by changing that exponent. No trial-update controller runs.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from models import model_family, power_forward, uses_analogy
from torch import Tensor, nn

from ana.config import Selection, TrainConfig
from ana.data.corpus import Batch, collate
from ana.model import Seq2SeqTransformer

CONSISTENCY_WEIGHT = 1.0
PROBABILITY_FLOOR_MASS = 1e-6
BALANCE_DENOMINATOR_FLOOR = 1e-12


def consistency_loss_config(model_name: str) -> dict:
    """Describe both the prediction objective and the optional analogy term."""
    family = model_family(model_name)
    analogy = uses_analogy(model_name)
    common = {
        "name": f"context_power_{family}_two_view_training",
        "model_name": model_name,
        "dropout_views": 2,
        "both_views_receive_gradients": True,
        "supervised_loss": "mean of two existing label-smoothed cross-entropies",
        "target_ignore_index": -100,
        "consistency_weight": CONSISTENCY_WEIGHT if analogy else 0.0,
        "power_training": "ordinary CE through the prediction path, not trial updates",
        "analogy_power": "detached mean of the two view powers; one exponent per quartet",
        "analogy_inputs": "pre-power prediction quantities, without a power-head path",
        "power_summary_scope": "valid development target positions, dropout off",
        "power_summary_interval": "original evaluation interval",
    }
    if family == "prediction":
        common.update(
            {
                "initial_power": 0.5,
                "power_bounds": [0.25, 0.75],
                "powers_per_target_position": 1,
                "prediction_logit_scale": "p / 0.5",
                "probability_floor_mass": PROBABILITY_FLOOR_MASS,
                "probability_power_difference_scale": "V^(p-0.5)/(2*p)",
                "consistency_reduction": (
                    "sum of squared centered vocabulary defects, mean over valid target positions"
                ),
                "batch_magnitude_balancing": {
                    "reference_power": 0.5,
                    "rule": "raw * detach(reference / max(detach(raw), denominator_floor))",
                    "denominator_floor": BALANCE_DENOMINATOR_FLOOR,
                    "reference_predictions": "same raw probabilities and valid-token reduction",
                    "reference_gradient": "none",
                    "reference_power_behavior": (
                        "all p=.5 bypasses scaling and uses sqrt exactly"
                    ),
                    "zero_behavior": "finite zero loss and gradients",
                    "tiny_loss_behavior": "exact magnitude matching relaxed below the floor",
                    "granularity": "one scalar per loss call, including each microbatch",
                }
                if analogy
                else None,
            }
        )
    else:
        fixed = model_name == "context_power_features_fixed"
        common.update(
            {
                "initial_power": 1.0,
                "power_bounds": [1.0, 1.0] if fixed else [0.5, 1.5],
                "power_training": "fixed p=1" if fixed else common["power_training"],
                "powers_per_target_position": "one per pair of final decoder features",
                "positive_inputs": "softplus of pre-transform decoder features plus epsilon",
                "prediction_transform": "h + u^p - u",
                "analogy_defect": "((u1_a^p-u2_a^p)-(u1_b^p-u2_b^p))/p",
                "consistency_reduction": (
                    "mean squared defect over pairs and valid target positions"
                ),
                "batch_magnitude_balancing": None,
            }
        )
    return common


def _valid_mean(values: Tensor, labels: Tensor) -> Tensor:
    valid = labels.ne(-100).to(dtype=values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def prediction_analogy(
    logits1: Tensor,
    logits2: Tensor,
    power1: Tensor,
    power2: Tensor,
    labels: Tensor,
) -> Tensor:
    """Exact all-vocabulary-pair defects with one shared power per position.

    The detached balancing scalar is shared across the whole loss-call batch,
    not separately chosen for each position. Gradient microbatches therefore
    have their own scalar and need not match full-batch gradient weighting.
    """
    if logits1.shape != logits2.shape or logits1.ndim != 3 or logits1.shape[-1] < 1:
        raise ValueError("prediction logits need matching [batch, time, vocabulary] shapes")
    if labels.shape != logits1.shape[:-1]:
        raise ValueError("labels must align with target positions")
    expected_power = (*labels.shape, 1)
    if power1.shape != expected_power or power2.shape != expected_power:
        raise ValueError("prediction powers need one value per target position")
    dtype = torch.promote_types(logits1.dtype, logits2.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    power = (0.5 * (power1.detach() + power2.detach())).to(dtype=dtype)
    vocabulary = logits1.shape[-1]
    floor = PROBABILITY_FLOOR_MASS / vocabulary
    probability1 = logits1.softmax(-1, dtype=dtype).mul(1.0 - PROBABILITY_FLOOR_MASS).add(floor)
    probability2 = logits2.softmax(-1, dtype=dtype).mul(1.0 - PROBABILITY_FLOOR_MASS).add(floor)
    # This also preserves the original fixed-p=.5 numerical operation at
    # initialization, without keeping sqrt and pow graphs simultaneously.
    if bool(torch.all(power == 0.5)):
        delta = probability1.sqrt() - probability2.sqrt()
        centered = delta - delta.mean(-1, keepdim=True)
        return _valid_mean(centered.square().sum(-1), labels)
    with torch.no_grad():
        reference_delta = probability1.sqrt() - probability2.sqrt()
        reference_centered = reference_delta - reference_delta.mean(-1, keepdim=True)
        reference = _valid_mean(reference_centered.square().sum(-1), labels)
    del reference_delta, reference_centered
    scale = vocabulary ** (power - 0.5) / (2.0 * power)
    delta = scale * (probability1.pow(power) - probability2.pow(power))
    del probability1, probability2
    centered = delta - delta.mean(-1, keepdim=True)
    raw = _valid_mean(centered.square().sum(-1), labels)
    balance = (reference / raw.detach().clamp_min(BALANCE_DENOMINATOR_FLOOR)).detach()
    return raw * balance


def feature_analogy(
    positive1: Tensor,
    positive2: Tensor,
    power1: Tensor,
    power2: Tensor,
    labels: Tensor,
) -> Tensor:
    """Compare each feature pair across two views using exactly four positives."""
    if (
        positive1.shape != positive2.shape
        or positive1.ndim != 4
        or positive1.shape[-1] != 2
        or positive1.shape[-2] < 1
    ):
        raise ValueError("positive features need matching [batch, time, pairs, 2] shapes")
    if labels.shape != positive1.shape[:2]:
        raise ValueError("labels must align with target positions")
    if power1.shape != positive1.shape[:-1] or power2.shape != power1.shape:
        raise ValueError("feature powers need one value per pair and target position")
    dtype = torch.promote_types(positive1.dtype, positive2.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    power = (0.5 * (power1.detach() + power2.detach())).to(dtype=dtype)
    difference = positive1.to(dtype=dtype).pow(power.unsqueeze(-1)) - positive2.to(
        dtype=dtype
    ).pow(power.unsqueeze(-1))
    defect = (difference[..., 0] - difference[..., 1]) / power
    return _valid_mean(defect.square().mean(-1), labels)


@dataclass
class TrainOutcome:
    steps_run: int
    best_dev_loss: float
    best_step: int
    scored_step: int
    selection: Selection
    seconds: float
    micro_batch_size: int | None = None
    power_final: dict | None = None
    power_scored: dict | None = None
    power_history: list[dict] = field(default_factory=list)


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
def dev_loss(model, batches: list[Batch], device: torch.device, *, with_power=False):
    """Keep the original CE aggregation and summarize actual context powers."""
    model.eval()
    total, counted = 0.0, 0
    power_count, target_count = 0, 0
    power_sum, power_squares = 0.0, 0.0
    power_min, power_max = float("inf"), float("-inf")
    for batch in batches:
        batch = batch.to(device)
        loss, details = power_forward(model, batch)
        total += float(loss) * len(batch)
        counted += len(batch)
        if with_power:
            valid = batch.labels.ne(-100)
            values = details["power"][valid].double()
            if values.numel():
                power_count += values.numel()
                target_count += int(valid.sum())
                power_sum += float(values.sum())
                power_squares += float(values.square().sum())
                power_min = min(power_min, float(values.min()))
                power_max = max(power_max, float(values.max()))
        del loss, details
    model.train()
    mean_loss = total / max(counted, 1)
    if not with_power:
        return mean_loss
    if power_count == 0:
        raise ValueError("power summaries need valid development target positions")
    mean = power_sum / power_count
    return mean_loss, {
        "min": power_min,
        "max": power_max,
        "mean": mean,
        "std": max(0.0, power_squares / power_count - mean**2) ** 0.5,
        "power_values": power_count,
        "valid_target_positions": target_count,
        "scope": "development_target_positions_dropout_off",
    }


def _two_view_loss(model, batch: Batch, model_name: str):
    loss1, details1 = power_forward(model, batch)
    loss2, details2 = power_forward(model, batch)
    supervised = 0.5 * (loss1 + loss2)
    if not uses_analogy(model_name):
        consistency = supervised.new_zeros(())
    elif model_family(model_name) == "prediction":
        consistency = prediction_analogy(
            details1["raw_logits"],
            details2["raw_logits"],
            details1["power"],
            details2["power"],
            batch.labels,
        )
    else:
        consistency = feature_analogy(
            details1["positive"],
            details2["positive"],
            details1["power"],
            details2["power"],
            batch.labels,
        )
    return supervised + CONSISTENCY_WEIGHT * consistency, supervised, consistency


def _backward_two_view_batch(model, batch: Batch, model_name: str, micro_batch_size: int):
    """Keep valid-token weighting and one optimizer step per original batch."""
    if micro_batch_size >= len(batch):
        loss, supervised, consistency = _two_view_loss(model, batch, model_name)
        loss.backward()
        return loss.detach(), supervised.detach(), consistency.detach()
    valid_total = batch.labels.ne(-100).sum().clamp_min(1)
    totals = None
    for start in range(0, len(batch), micro_batch_size):
        stop = start + micro_batch_size
        part = Batch(
            batch.source_ids[start:stop], batch.source_mask[start:stop], batch.labels[start:stop]
        )
        loss, supervised, consistency = _two_view_loss(model, part, model_name)
        weight = part.labels.ne(-100).sum().to(dtype=loss.dtype) / valid_total
        (loss * weight).backward()
        values = tuple(
            component.detach() * weight for component in (loss, supervised, consistency)
        )
        totals = (
            values
            if totals is None
            else tuple(total + value for total, value in zip(totals, values, strict=True))
        )
        del loss, supervised, consistency, part
    return totals


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
    model_name: str,
) -> TrainOutcome:
    config = config.resolved(model.config.d_model)
    objective_config = consistency_loss_config(model_name)
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
    best_power_summary = last_power_summary = None
    power_history = []
    completed_steps, previous_seconds = 0, 0.0
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
                "prediction_analogy_balancing": ("one detached ratio per microbatch loss call"),
                "power_learning": objective_config["power_training"],
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
            "loss_configuration": objective_config,
            "device_type": device.type,
            "micro_batch_size": micro_batch_size,
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
            last_power_summary = saved["power_summary_state"]["last"]
            best_power_summary = saved["power_summary_state"]["best"]
            power_history = saved["power_summary_state"]["history"]
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
        optimiser.zero_grad(set_to_none=True)
        loss, supervised, consistency = _backward_two_view_batch(
            model, batch, model_name, micro_batch_size
        )
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimiser.step()
        if step - completed_steps in (1, 100) or step % 200 == 0 or step == config.max_steps:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "phase": "training_progress",
                        "step": step,
                        "train_loss": float(loss),
                        "train_ce": float(supervised),
                        "train_consistency": float(consistency),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": (step - completed_steps) / max(elapsed, 1e-9),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        if step % config.eval_every == 0 or step == config.max_steps:
            current, power_summary = dev_loss(model, dev_batches, device, with_power=True)
            last_power_summary = {"step": step, **power_summary}
            power_history.append(last_power_summary)
            print(
                json.dumps(
                    {"phase": "context_power_summary", **last_power_summary}, allow_nan=False
                ),
                flush=True,
            )
            if current < best_loss:
                best_loss, best_step = current, step
                best_power_summary = last_power_summary.copy()
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
                        "power_summary_state": {
                            "last": last_power_summary,
                            "best": best_power_summary,
                            "history": power_history,
                        },
                    },
                    path,
                )
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    scored_step = best_step if keep_best else config.max_steps
    scored_power_summary = best_power_summary if keep_best else last_power_summary
    return TrainOutcome(
        steps_run=config.max_steps,
        best_dev_loss=best_loss,
        best_step=best_step,
        scored_step=scored_step,
        selection=config.selection,
        seconds=previous_seconds + time.monotonic() - started,
        micro_batch_size=micro_batch_size,
        power_final={**last_power_summary, "model_state": "final_training_weights"},
        power_scored={**scored_power_summary, "model_state": "scored_weights"},
        power_history=power_history,
    )
