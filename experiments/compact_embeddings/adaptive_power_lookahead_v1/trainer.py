"""Balanced adaptive power with short lookahead and a current-power comparison.

The model, balanced analogy loss, and real training updates are unchanged.
Every candidate takes three discarded AdamW updates and is scored on three
disjoint training batches. A candidate must improve over current p by the
declared margin and win on at least two scoring batches. Otherwise p stays.
No development or test examples enter adaptation. Trial state is restored.
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
BALANCE_DENOMINATOR_FLOOR = 1e-12
POWER_MIN = 0.25
POWER_MAX = 0.75
PROBE_EVERY = 200
PROBE_OFFSET = 0.05
POWER_MOVE = 0.01
PROBE_CE_DEADBAND = 1e-7
PROBE_RNG_OFFSET = 91073
PROBE_UPDATE_STEPS = 3
PROBE_SCORE_BATCHES = 3
PROBE_RELATIVE_IMPROVEMENT = 1e-4
PROBE_MIN_BATCH_WINS = 2


def consistency_loss_config(adaptive: bool = True) -> dict:
    """Objective and finite-difference policy recorded with every checkpoint."""
    common = {
        "name": (
            "all_pairs_balanced_lookahead_adaptive_power_consistency"
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
                    "detached batch-magnitude matching to p=.5, not gradient-norm matching"
                ),
                "batch_magnitude_balancing": {
                    "reference_power": 0.5,
                    "rule": "raw * detach(reference / max(detach(raw), denominator_floor))",
                    "denominator_floor": BALANCE_DENOMINATOR_FLOOR,
                    "reference_predictions": "same two dropout passes as the raw power loss",
                    "reference_reduction": "same centered defects and valid target token mean",
                    "reference_gradient": "none",
                    "reference_power_behavior": "p=.5 bypasses scaling exactly",
                    "zero_behavior": "raw=reference=0 gives finite zero loss and gradient",
                    "tiny_loss_behavior": "magnitude matching is relaxed below denominator floor",
                    "granularity": (
                        "one scalar per loss call; microbatches are balanced separately"
                    ),
                },
                "consistency_reduction": (
                    "sum over centered vocabulary defects, mean over valid target positions"
                ),
                "adaptation": {
                    "name": "bounded_three_step_prediction_lookahead",
                    "interval_steps": PROBE_EVERY,
                    "start": "first interval boundary strictly after optimizer warmup",
                    "candidate_offset": PROBE_OFFSET,
                    "maximum_power_move": POWER_MOVE,
                    "absolute_ce_improvement_floor": PROBE_CE_DEADBAND,
                    "relative_ce_improvement": PROBE_RELATIVE_IMPROVEMENT,
                    "minimum_scoring_batch_wins": PROBE_MIN_BATCH_WINS,
                    "candidate_powers": "unique clipped [p-offset, current p, p+offset]",
                    "hypothetical_update_steps": PROBE_UPDATE_STEPS,
                    "scoring_batches": PROBE_SCORE_BATCHES,
                    "hold_rule": "hold current p unless another candidate clears both conditions",
                    "probe_batch_size": (
                        "16 when maximum training target length exceeds 128, otherwise 32; capped"
                        " at one sixth of the training set"
                    ),
                    "sampling": (
                        "independent persistent RNG, six mutually disjoint training-only batches"
                        " sampled without replacement per probe; three update and three score"
                    ),
                    "probe_rng_seed_offset": PROBE_RNG_OFFSET,
                    "candidate_update": (
                        "three sequential clipped AdamW updates at current learning rate and"
                        " current moments; identical update batches and initial dropout RNG"
                        " for every candidate"
                    ),
                    "criterion": (
                        "mean ordinary one-pass label-smoothed CE over three identical scoring"
                        " batches for every candidate, dropout off"
                    ),
                    "restore": (
                        "model, optimizer, gradients, mode flags, Python/Torch/device RNG; discard"
                        " all hypothetical updates"
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
    """Balance the all-pairs defect to the same-batch square-root reference.

    Raw Delta_i = V**(p-.5)/(2*p) * (Q1_i**p-Q2_i**p). The centered square
    sum remains the exact scaled all-pairs four-number analogy defect. A
    detached reference/raw scalar matches its value to the p=.5 defect without
    replacing its gradients with the reference gradients. No vocabulary-pair
    tensor is allocated. At p=.5, bypass balancing for exactly the old result.

    Clamp only the detached denominator at 1e-12: zero defects and all-padding
    batches stay finite; below that floor, exact magnitude matching is relaxed.
    Each loss call uses its own scalar, including each gradient microbatch.
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
    reference_loss = None
    if power != 0.5:
        # Reuse probabilities without an extra forward pass or reference graph.
        # Release reference-sized temporaries before constructing the raw defect.
        with torch.no_grad():
            reference_delta = prob1.sqrt() - prob2.sqrt()
            reference_centered = reference_delta - reference_delta.mean(dim=-1, keepdim=True)
            reference_per_token = reference_centered.square().sum(dim=-1)
            reference_valid = labels.ne(-100).to(dtype=dtype)
            reference_loss = (
                reference_per_token * reference_valid
            ).sum() / reference_valid.sum().clamp_min(1.0)
        del reference_delta, reference_centered, reference_per_token, reference_valid
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
    raw_loss = (per_token * valid).sum() / valid.sum().clamp_min(1.0)
    if reference_loss is None:
        return raw_loss
    scale = (reference_loss / raw_loss.detach().clamp_min(BALANCE_DENOMINATOR_FLOOR)).detach()
    return raw_loss * scale


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
    Balancing is per microbatch: its gradient is not generally identical to
    using one shared ratio over the whole batch. Full batches avoid that change.
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


def _choose_power(power: float, candidates: list[float], batch_scores: list[list[float]]):
    """Choose a candidate only when mean improvement and batch wins both agree."""
    if len(candidates) != len(batch_scores) or power not in candidates:
        raise ValueError("candidate scores must include the current power")
    if any(len(scores) != PROBE_SCORE_BATCHES for scores in batch_scores):
        raise ValueError("each candidate needs three scoring-batch losses")
    if not all(math.isfinite(score) for scores in batch_scores for score in scores):
        raise FloatingPointError("non-finite prediction loss in adaptive-power probe")
    means = [sum(scores) / PROBE_SCORE_BATCHES for scores in batch_scores]
    current_index = candidates.index(power)
    current_mean = means[current_index]
    required_gain = max(PROBE_CE_DEADBAND, abs(current_mean) * PROBE_RELATIVE_IMPROVEMENT)
    wins = [
        sum(
            candidate_score < current_score
            for candidate_score, current_score in zip(
                scores, batch_scores[current_index], strict=True
            )
        )
        for scores in batch_scores
    ]
    gain_eligible = [
        index
        for index, candidate in enumerate(candidates)
        if candidate != power and current_mean - means[index] >= required_gain
    ]
    eligible = [index for index in gain_eligible if wins[index] >= PROBE_MIN_BATCH_WINS]
    if eligible:
        selected_index = min(
            eligible,
            key=lambda index: (means[index], abs(candidates[index] - power), candidates[index]),
        )
        selected = candidates[selected_index]
        movement = math.copysign(min(POWER_MOVE, abs(selected - power)), selected - power)
        next_power = min(POWER_MAX, max(POWER_MIN, round(power + movement, 12)))
        reason = "candidate_improves_mean_and_wins_batches"
    else:
        selected_index = current_index
        next_power = power
        if min(means) >= current_mean:
            reason = "current_best_or_tied"
        elif not gain_eligible:
            reason = "mean_improvement_below_threshold"
        else:
            reason = "insufficient_scoring_batch_wins"
    return next_power, {
        "candidate_prediction_ce": means,
        "candidate_batch_prediction_ce": batch_scores,
        "current_prediction_ce": current_mean,
        "required_ce_improvement": required_gain,
        "candidate_scoring_batch_wins": wins,
        "eligible_candidate_powers": [candidates[index] for index in eligible],
        "selected_candidate_power": candidates[selected_index],
        "decision_reason": reason,
        "power_movement": round(next_power - power, 12),
    }


def _adapt_power(
    model,
    optimiser,
    update_batches: list[Batch],
    score_batches: list[Batch],
    power: float,
    config: TrainConfig,
    device: torch.device,
) -> tuple[float, dict]:
    """Compare three-step candidates from identical state and discard all trials."""
    if len(update_batches) != PROBE_UPDATE_STEPS or len(score_batches) != PROBE_SCORE_BATCHES:
        raise ValueError("lookahead needs three update and three scoring batches")
    started = time.monotonic()
    candidates = list(
        dict.fromkeys(
            (
                max(POWER_MIN, round(power - PROBE_OFFSET, 12)),
                power,
                min(POWER_MAX, round(power + PROBE_OFFSET, 12)),
            )
        )
    )
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
        # CPU load_state_dict may reuse tensors; preserve the pristine moments.
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

    batch_scores = []
    try:
        for candidate in candidates:
            restore()
            model.train()
            for update_batch in update_batches:
                optimiser.zero_grad(set_to_none=True)
                loss, _, _ = _two_view_loss(model, update_batch, candidate, True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimiser.step()
                del loss
            optimiser.zero_grad(set_to_none=True)
            model.eval()
            scores = []
            with torch.no_grad():
                for score_batch in score_batches:
                    predictive_loss, logits = model(
                        score_batch.source_ids, score_batch.source_mask, score_batch.labels
                    )
                    scores.append(float(predictive_loss))
                    del predictive_loss, logits
            batch_scores.append(scores)
    finally:
        restore()

    next_power, decision = _choose_power(power, candidates, batch_scores)
    return next_power, {
        "power_before": power,
        "power_after": next_power,
        "candidate_powers": candidates,
        **decision,
        "probe_update_steps_per_candidate": PROBE_UPDATE_STEPS,
        "probe_scoring_batches_per_candidate": PROBE_SCORE_BATCHES,
        "probe_update_examples_per_batch": [len(batch) for batch in update_batches],
        "probe_score_examples_per_batch": [len(batch) for batch in score_batches],
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
        len(train_examples) // (PROBE_UPDATE_STEPS + PROBE_SCORE_BATCHES),
    )
    if adaptive and probe_batch_size < 1:
        raise ValueError("adaptive lookahead needs at least six training examples")
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
            probe_count = PROBE_UPDATE_STEPS + PROBE_SCORE_BATCHES
            picked = probe_rng.sample(range(len(train_examples)), probe_count * probe_batch_size)
            probe_batches = [
                collate(
                    [train_examples[index] for index in picked[start : start + probe_batch_size]],
                    pad_id,
                ).to(device)
                for start in range(0, len(picked), probe_batch_size)
            ]
            power, probe = _adapt_power(
                model,
                optimiser,
                probe_batches[:PROBE_UPDATE_STEPS],
                probe_batches[PROBE_UPDATE_STEPS:],
                power,
                config,
                device,
            )
            probe["step"] = step
            power_history.append(probe)
            print(
                json.dumps({"phase": "adaptive_power_probe", **probe}, allow_nan=False), flush=True
            )
            del probe_batches

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
