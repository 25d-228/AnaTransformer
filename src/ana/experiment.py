"""One cell of the grid, from a corpus name and a model name to a results file.

Everything that decides what a run does is written into its manifest: the model, the shape it
was built at, the recipe, the selection rule, and the step actually reached. A record of the
model but not of the recipe cannot reproduce its own run, and a run that cannot be reproduced
is not evidence of anything.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, replace

import torch

from ana.config import TrainConfig
from ana.data.corpora import build_corpus
from ana.data.corpus import Corpus, Example
from ana.data.tokenizer import Tokenizer, train_or_load
from ana.decoding import beam_decode
from ana.registry import (
    REGISTRY,
    baseline_parameters,
    build_model,
    config_for,
    count_parameters,
)
from ana.trainer import fixed_batches, set_seed, train

SMOKE_STEPS = 30
SMOKE_EXAMPLES = 200


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def encode_split(
    examples: list[Example], tokenizer: Tokenizer, corpus: Corpus
) -> list[tuple[list[int], list[int]]]:
    return [
        (
            tokenizer.encode(e.source, corpus.max_source_length),
            tokenizer.encode_target(e.target, corpus.max_target_length),
        )
        for e in examples
    ]


def prepare(
    corpus: Corpus,
    smoke: bool,
    evaluated_splits: tuple[str, ...] | None = None,
) -> tuple[Tokenizer, dict[str, list[Example]]]:
    if evaluated_splits is None:
        splits = corpus.load()
    else:
        if not evaluated_splits or "train" in evaluated_splits:
            raise ValueError("evaluated_splits must name one or more non-training splits")
        required = tuple(dict.fromkeys(("train", "dev", *evaluated_splits)))
        splits = {split: corpus.load_split(split) for split in required}
    if smoke:
        splits = {name: rows[:SMOKE_EXAMPLES] for name, rows in splits.items()}

    text = [e.source for e in splits["train"]] + [e.target for e in splits["train"]]
    tokenizer = train_or_load(
        corpus.tokenizer_path(smoke),
        text,
        corpus.vocab_size,
        corpus.character_coverage,
        corpus.tokenizer_type,
    )
    return tokenizer, splits


def score_split(
    model,
    corpus: Corpus,
    examples: list[Example],
    tokenizer: Tokenizer,
    device: torch.device,
    batch_size: int,
    beam_size: int,
) -> tuple[float, list[str]]:
    """Decode the split and score it.

    Sentences are grouped by length before decoding and put back in their original order
    afterwards. Generation stops when every row in a batch has finished, so a batch that mixes a
    five-token output with a five-hundred-token one pays for the long one on every row. Sorting
    first cuts that waste, and it changes nothing about what is produced: as far as the decoder
    is concerned the model sees one sentence at a time.
    """
    encoded = encode_split(examples, tokenizer, corpus)

    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    by_length = [encoded[i] for i in order]
    batches = fixed_batches(by_length, batch_size, model.config.pad_id)

    generated_in_order: list[str] = []
    for batch in batches:
        generated = beam_decode(
            model,
            batch.source_ids.to(device),
            batch.source_mask.to(device),
            corpus.max_decode_length,
            beam_size,
        )
        generated_in_order.extend(tokenizer.decode(row.tolist()).strip() for row in generated)

    hypotheses = [""] * len(examples)
    for position, original in enumerate(order):
        hypotheses[original] = generated_in_order[position]

    references = [e.target for e in examples]
    return corpus.metric.score(hypotheses, references), hypotheses


def run_cell(
    model_name: str,
    corpus_name: str,
    train_config: TrainConfig,
    output_dir: str = "runs",
    smoke: bool = False,
    device: torch.device | None = None,
    study_id: str | None = None,
    score_dev: bool = False,
    evaluated_splits: tuple[str, ...] | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The declared seed owns the initial parameters as well as the training stream. Keep the
    # second reset in `train()`: model construction consumes a different amount of randomness
    # across architectures, while training must begin from the same RNG state for every model.
    set_seed(train_config.seed)

    corpus = build_corpus(corpus_name)
    tokenizer, splits = prepare(corpus, smoke, evaluated_splits)

    if smoke:
        train_config = replace(
            train_config, max_steps=SMOKE_STEPS, eval_every=SMOKE_STEPS, batch_size=8, beam_size=2
        )

    # The corpus owns the architecture: it is the shape the literature reports its number in.
    model_config = corpus.model_config(len(tokenizer))
    if smoke:
        model_config = replace(
            model_config, d_model=32, n_heads=4, d_ff=64, n_encoder_layers=2, n_decoder_layers=2
        )

    model = build_model(model_name, model_config)
    shape = config_for(model_name, model_config)  # the matched baseline is built narrower
    parameters = count_parameters(model)

    train_config = train_config.resolved(model_config.d_model)

    started = time.monotonic()
    tag = f"{corpus_name}/{model_name}/seed{train_config.seed}"
    print(
        f"{tag}  {parameters / 1e6:.2f}M parameters, d_model {shape.d_model}, "
        f"{train_config.max_steps:,} steps, scoring the "
        f"{'final' if train_config.selection.value == 'final' else 'best-dev-loss'} weights",
        flush=True,
    )

    def report(step: int, loss: float, dev: float, best: float) -> None:
        """Print the development loss as it falls, so a run can be watched rather than waited on."""
        elapsed = time.monotonic() - started
        rate = step / max(elapsed, 1e-9)
        remaining = (train_config.max_steps - step) / max(rate, 1e-9) / 60
        marker = "  <- best" if dev <= best else ""
        print(
            f"{tag}  step {step:>6,}/{train_config.max_steps:,}  "
            f"train {loss:6.3f}  dev {dev:6.3f}  best {best:6.3f}  "
            f"{rate:5.1f} steps/s  {remaining:5.1f} min left{marker}",
            flush=True,
        )

    outcome = train(
        model,
        encode_split(splits["train"], tokenizer, corpus),
        encode_split(splits["dev"], tokenizer, corpus),
        train_config,
        device,
        on_eval=report,
    )
    print(f"{tag}  trained in {outcome.seconds / 60:.1f} min; decoding", flush=True)

    scores: dict[str, float] = {}
    hypotheses: dict[str, list[str]] = {}
    for split in splits:
        if evaluated_splits is not None:
            should_score = split in evaluated_splits
        else:
            should_score = split != "train" and (split != "dev" or score_dev)
        if not should_score:
            continue
        value, generated = score_split(
            model,
            corpus,
            splits[split],
            tokenizer,
            device,
            train_config.decode_batch_size,
            train_config.beam_size,
        )
        scores[split] = value
        hypotheses[split] = generated

    # The baseline is the model the literature has a number for, so it is the only one that can
    # be checked. A score far below the published range means the code is wrong, not the recipe,
    # and without something to compare against a wrong number looks like a plausible one.
    calibration = corpus.calibration
    if calibration is not None and model_name == "baseline" and not smoke:
        observed = scores.get(calibration.split)
        if observed is not None:
            print(
                f"{tag}  {calibration.split} {observed:.2f}  {calibration.verdict(observed)}",
                flush=True,
            )

    cell = f"{corpus_name}_{model_name}_seed{train_config.seed}"
    folder = os.path.join(output_dir, cell)
    os.makedirs(folder, exist_ok=True)

    # The weights that were scored, not the ones the run ended on. Without them the routers are
    # destroyed, and which of the eight permutations a trained model actually selects is the one
    # question the whole construction exists to raise. Keeping them also means a split can be
    # re-decoded -- with a wider beam, say -- without retraining anything.
    if not smoke:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model": model_name,
                "corpus": corpus_name,
                "model_config": asdict(shape),
                "step": outcome.scored_step,
                "dev_loss": outcome.best_dev_loss,
            },
            os.path.join(folder, "weights.pt"),
        )

    for split, generated in hypotheses.items():
        with open(os.path.join(folder, f"hypotheses.{split}.txt"), "w", encoding="utf-8") as h:
            h.write("\n".join(generated) + "\n")
        with open(os.path.join(folder, f"references.{split}.txt"), "w", encoding="utf-8") as h:
            h.write("\n".join(e.target for e in splits[split]) + "\n")

    # The denominator is an ordinary transformer at the SAME shape this run was built from, not
    # at the corpus's declared architecture. Those differ under --smoke, and a saving measured
    # against a model nobody trained is not a saving.
    reference = baseline_parameters(model_config)
    record = {
        "model": model_name,
        "corpus": corpus_name,
        "metric": corpus.metric.name,
        "scores": scores,
        "parameters": parameters,
        # What this model saves against the ordinary transformer for THIS corpus, as a share of
        # it. The count is a different number on every corpus because every corpus has a
        # different shape; the proportion is the thing that compares.
        "saved_fraction": (reference - parameters) / reference,
        "baseline_parameters": reference,
        "best_dev_loss": outcome.best_dev_loss,
        "best_step": outcome.best_step,
        "scored_step": outcome.scored_step,
        "selection": outcome.selection.value,
        "steps_run": outcome.steps_run,
        "seconds": round(outcome.seconds, 1),
        "manifest": {
            "seed": train_config.seed,
            "seeded_before_model_init": True,
            "study_id": study_id,
            "git_commit": git_commit(),
            "train_config": asdict(train_config),
            "model_config": asdict(shape),
            "smoke": smoke,
            "score_dev": score_dev,
            "evaluated_splits": list(scores),
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }

    with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    return record


def all_models() -> list[str]:
    return list(REGISTRY)
