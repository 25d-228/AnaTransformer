"""The command line. `ana --help` shows the study in the order it is meant to be run.

    ana tune   --corpus cogs      confirm the learning rate for that corpus, and record it
    ana run    --corpus cogs      train every model, at every seed, on that one recipe
    ana report                    read the runs and print the comparison

`ana train` runs a single cell. That is what `run` calls underneath, and what you want when one
cell needs repeating by hand.

WHAT `tune` IS FOR, AND WHAT IT IS NOT

The recipe comes from the paper. Each corpus ships the one its published number was reported
under, so `tune` is not a search — it is a CHECK, and it asks one question:

    does the BASELINE, trained on the published recipe, reach the number the field reports?

If it does, that recipe stands, and the check cost one run. There is nothing to optimise: a
better learning rate than the paper's would not make the baseline more faithful, it would make
it less.

If it does NOT, the check has found something, and only then is a search worth running:

    ana tune --corpus iwslt14 --scales 0.5,1,2

If some other rate reaches the number, it is used and flagged loudly as a deviation from the
paper — our batch is counted in sentences and theirs in tokens, so our pipeline can genuinely
need a different rate, and the write-up has to say so. If no rate reaches it, no recipe is
written: the fault is the model, the data or the step budget, and searching harder will not
find it.

It never picks the rate with the lowest development loss. On COGS every rate drives the loss to
about 0.001 — the model has solved the in-distribution set, and what is left is noise. A rule
that chased it would widen the search forever and land on a rate no paper reports. It is the
same decorrelation that makes development loss the wrong way to choose a CHECKPOINT on that
corpus, appearing one level up.

Only the baseline is checked. The rate is therefore not chosen to suit anything in the study —
not `shared_qkv`, which is prior work, and not `ana_*`, which is ours. Every architecture is
trained on it unchanged. No model in the comparison has a vote in the recipe it is judged under.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import replace

import torch

from ana import recipes
from ana.config import TrainConfig
from ana.d4_diagnostic import STUDY_ID as D4_DIAGNOSTIC_STUDY_ID
from ana.d4_diagnostic import run_diagnostic
from ana.data.corpora import CORPORA, build_corpus
from ana.experiment import run_cell
from ana.factorial import CORPUS as FACTORIAL_CORPUS
from ana.factorial import MODELS as FACTORIAL_MODELS
from ana.factorial import SEEDS as FACTORIAL_SEEDS
from ana.factorial import STUDY_ID as FACTORIAL_STUDY_ID
from ana.permutation_family import (
    CONTROL_MODELS as PERMUTATION_FAMILY_MODELS,
)
from ana.permutation_family import SEEDS as PERMUTATION_FAMILY_SEEDS
from ana.permutation_family import STUDY_ID as PERMUTATION_FAMILY_STUDY_ID
from ana.permutation_family import run_analysis as run_permutation_family_analysis
from ana.permutation_family import run_compatibility_preflight
from ana.registry import PILOT_MODELS, REGISTRY
from ana.s4_screen import CORPUS as S4_CORPUS
from ana.s4_screen import MODEL as S4_MODEL
from ana.s4_screen import SEEDS as S4_SEEDS
from ana.s4_screen import STUDY_ID as S4_STUDY_ID
from ana.s4_screen import write_results as write_s4_results
from ana.stats import across_seed_test, bootstrap_score, paired_bootstrap
from ana.v4_core_diagnostic import STUDY_ID as V4_CORE_DIAGNOSTIC_STUDY_ID
from ana.v4_core_diagnostic import run_diagnostic as run_v4_core_diagnostic

# One run, at the rate the paper used. The recipe comes from the paper, so there is nothing to
# search for -- there is only something to CHECK, and one run checks it.
#
# A wider net is a fallback, not a default. If the published rate does not reach the published
# number, then and only then is it worth asking whether the fault is the learning rate:
#
#     ana tune --corpus iwslt14 --scales 0.5,1,2
#
# Searching by default was waste, and the runs proved it. On COGS the published rate gave the
# best generalization of the three (77.87 against 76.00 and 72.82) and on Multi30k the best BLEU
# (40.46 against 40.09 and 39.97). Two thirds of that compute bought nothing, because the answer
# was written in the paper.
SEARCH_SCALES = (1.0,)
SEARCH_MODEL = "baseline"

# Five seeds on COGS, which is what Kim and Linzen (2020) and Csordas et al. (2021) both report
# their COGS numbers over. One run on each translation corpus, which is what their papers report.
# The +/- follows from that: a spread across seeds where there are seeds, a bootstrap over the
# test set where there is one run.
DEFAULT_SEEDS = {"cogs": "42,43,44,45,46", "multi30k": "42", "iwslt14": "42", "synthetic": "42"}


def _recipe(corpus: str) -> recipes.Recipe:
    """The confirmed recipe if `tune` has written one, else the corpus's published one."""
    return recipes.load(corpus, build_corpus(corpus).recipe)


def _train_config(recipe: recipes.Recipe, seed: int) -> TrainConfig:
    return TrainConfig(
        max_steps=recipe.max_steps,
        batch_size=recipe.batch_size,
        learning_rate=recipe.learning_rate,
        warmup_steps=recipe.warmup_steps,
        schedule=recipe.schedule,
        selection=recipe.selection,
        weight_decay=recipe.weight_decay,
        adam_betas=recipe.adam_betas,
        seed=seed,
    )


def _train(args: argparse.Namespace) -> None:
    models = list(REGISTRY) if args.model == "all" else args.model.split(",")

    recipe = _recipe(args.corpus)
    if not recipe.verified and not args.smoke:
        print(f"NOTE: {args.corpus} has no confirmed recipe. {recipe.source}", flush=True)

    config = _train_config(recipe, args.seed)
    overrides = {
        "max_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})

    # A smoke run trains a 32-wide model for thirty steps. It must not be able to leave a
    # results file where `ana report` reads them, so it goes somewhere else unless the caller
    # asked for a specific directory. `_report` refuses to read them either -- one guard would
    # do, but this is the kind of mistake that is invisible until it is in a table.
    out = os.path.join(args.out, "smoke") if args.smoke and args.out == "runs" else args.out
    evaluated_splits = (
        tuple(split.strip() for split in args.evaluated_splits.split(",") if split.strip())
        if args.evaluated_splits
        else None
    )

    for name in models:
        record = run_cell(
            name,
            args.corpus,
            config,
            out,
            smoke=args.smoke,
            study_id=args.study_id,
            score_dev=args.score_dev,
            evaluated_splits=evaluated_splits,
        )
        scores = "  ".join(f"{k} {v:.2f}" for k, v in record["scores"].items())
        print(
            f"{record['corpus']:10} {record['model']:18} seed {record['manifest']['seed']}  "
            f"{record['parameters'] / 1e6:6.2f}M  "
            f"({record['saved_fraction']:+.1%} vs baseline)  "
            f"dev {record['best_dev_loss']:.3f}  "
            f"{record['metric']}: {scores}"
        )


def _shard(jobs: list[str], gpus: int | list[int], tag: str, dry_run: bool) -> None:
    """One run per card at a time.

    Runs are independent, so there is nothing to coordinate: split the list, give each card its
    own share, and let them go. A cell whose results file already exists is skipped before it
    gets here, so a machine that dies part way through can be restarted without losing what it
    finished.
    """
    if not jobs:
        print("nothing to do; every cell already has a results file")
        return

    os.makedirs("logs", exist_ok=True)
    gpu_ids = list(range(gpus)) if isinstance(gpus, int) else gpus
    for position, gpu in enumerate(gpu_ids):
        shard = jobs[position :: len(gpu_ids)]
        if not shard:
            continue
        script = f"logs/{tag}_gpu{gpu}.sh"
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/bash\n")
            for job in shard:
                handle.write(f"CUDA_VISIBLE_DEVICES={gpu} {job}\n")
        os.chmod(script, 0o755)
        print(f"  gpu {gpu}: {len(shard)} runs")

        if not dry_run:
            subprocess.Popen(
                f"nohup bash {script} > logs/{tag}_gpu{gpu}.out 2>&1 &",
                shell=True,
                start_new_session=True,
            )

    if dry_run:
        print(f"\ndry run. the scripts are in logs/{tag}_gpu*.sh and nothing was started.")
    else:
        print(f"\nstarted. watch with:  tail -f logs/{tag}_gpu*.out")


def _scales(args: argparse.Namespace) -> tuple[float, ...]:
    return tuple(float(s) for s in args.scales.split(",")) if args.scales else SEARCH_SCALES


def _passes(record: dict, calibration: recipes.Calibration) -> bool:
    """Did this run put the baseline on the number the field reports for the corpus?

    This, and not the development loss, is what decides whether a learning rate is usable. The
    gate is checked on a split that is not the study's outcome -- on COGS, the in-distribution
    test set rather than the generalization one -- so passing it says the model learned the task
    and says nothing about the answer.
    """
    observed = record["scores"].get(calibration.split)
    return observed is not None and calibration.passed(observed)


def _tune(args: argparse.Namespace) -> None:
    if args.report:
        _tune_report(args)
        return

    corpus = build_corpus(args.corpus)
    published = corpus.recipe
    peak = published.learning_rate
    if peak is None:
        raise SystemExit(f"{args.corpus} ships no published learning rate to search around")

    scales = _scales(args)
    jobs = []
    for scale in scales:
        out = os.path.join(args.out, args.corpus, f"{SEARCH_MODEL}_lr{scale:g}x")
        done = os.path.join(out, f"{args.corpus}_{SEARCH_MODEL}_seed42", "results.json")
        if os.path.exists(done):
            continue
        jobs.append(
            f"{args.python} -m ana.cli.__main__ train --model {SEARCH_MODEL} "
            f"--corpus {args.corpus} --seed 42 --learning-rate {scale * peak:.8e} --out {out}"
        )

    print(
        f"{args.corpus}: {len(scales)} learning rates on `{SEARCH_MODEL}` "
        f"= {len(scales)} runs, {len(jobs)} still to do\n"
        f"the published rate is {peak:.3e}; the search tries "
        f"{', '.join(f'{s:g}x' for s in scales)} of it.\n"
        f"everything else is held at the published recipe: {published.source}\n"
    )

    _shard(jobs, args.gpus, f"tune_{args.corpus}", args.dry_run)
    if jobs and not args.dry_run:
        print(f"then:  ana tune --corpus {args.corpus} --report")


def _tune_report(args: argparse.Namespace) -> None:
    corpus = build_corpus(args.corpus)
    published = corpus.recipe

    rows = []
    for path in sorted(glob.glob(os.path.join(args.out, args.corpus, "*", "*", "results.json"))):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        _, scale = os.path.basename(os.path.dirname(os.path.dirname(path))).rsplit("_lr", 1)
        rows.append((float(scale.rstrip("x")), record))

    if not rows:
        print(f"no search under {args.out}/{args.corpus}. Run `ana tune --corpus {args.corpus}`.")
        return

    # The search runs in the background, so asking for the report too early is the ordinary
    # mistake, not an exotic one. A recipe chosen from half a search is worse than no recipe: it
    # looks like a result, and every model in the study is then trained underneath it.
    wanted = len(_scales(args))
    if len(rows) < wanted:
        done = ", ".join(f"{s:g}x" for s, _ in sorted(rows))
        print(
            f"\n{args.corpus}: {len(rows)} of {wanted} runs have finished. Waiting.\n"
            f"  done: {done}\n"
            f"  watch:  tail -f logs/tune_{args.corpus}_gpu*.out\n"
        )
        return

    checking = "checking" if len(rows) == 1 else "searching"
    print(f"\n{args.corpus}   {checking} `{SEARCH_MODEL}` only\n")
    print(f"  {'lr':>10}{'x':>6}{'dev loss':>11}{'best step':>11}{'of':>8}   scores")

    calibration = corpus.calibration
    for scale, record in sorted(rows):
        scores = "  ".join(f"{k} {v:.2f}" for k, v in record["scores"].items())
        starved = record["best_step"] >= 0.9 * record["steps_run"]
        note = "  <- still improving at the end; the step budget is too small" if starved else ""
        gate = ""
        if calibration is not None and calibration.split in record["scores"]:
            gate = "  PASS" if _passes(record, calibration) else "  FAIL"
        print(
            f"  {record['manifest']['train_config']['learning_rate']:>10.2e}{scale:>5g}x"
            f"{record['best_dev_loss']:>11.4f}{record['best_step']:>11,}"
            f"{record['steps_run']:>8,}   {scores}{gate}{note}"
        )

    # WHICH RATE TO USE.
    #
    # The published rate is the one the paper reports its number under, so a faithful
    # reproduction uses it, and the search exists to catch the case where OUR pipeline needs a
    # different one -- our batch is counted in sentences and theirs in tokens, so that is
    # possible. It catches that by whether the BASELINE reaches its published range, not by
    # which run squeezed out the lowest development loss.
    #
    # Selecting on development loss alone is not safe. On COGS every rate drives it to about
    # 0.001: the model has solved the in-distribution set and what is left is noise. A rule that
    # chased those differences would widen the search forever and end up on a rate no paper
    # reports. It is the same decorrelation that makes loss the wrong way to pick a CHECKPOINT
    # on this corpus, one level up.
    if calibration is None:
        scale, record = min(rows, key=lambda row: row[1]["best_dev_loss"])
        reason = f"no published anchor, so the lowest development loss wins: {scale:g}x"
    else:
        passing = [(s, r) for s, r in sorted(rows) if _passes(r, calibration)]
        if not passing:
            best = max(rows, key=lambda row: row[1]["scores"].get(calibration.split, 0.0))
            observed = best[1]["scores"][calibration.split]
            verdict = calibration.verdict(observed)
            print(f"\n  best baseline {calibration.split} {observed:.2f}: {verdict}")
            print(f"\n  {calibration.reference}")
            print("\n  No recipe is written.")

            # This is the only situation in which a SEARCH is worth its compute. Until the
            # published recipe has failed, there is nothing to look for -- the answer was in the
            # paper. Once it has failed, the question is whether the learning rate is to blame,
            # and three runs answer that.
            if len(rows) == 1:
                print(
                    "\n  The published recipe did not reach the published number. That is worth"
                    "\n  one search -- if some other rate reaches it, the fault is the rate; if"
                    "\n  none does, the fault is the model, the data or the step budget:"
                    f"\n    ana tune --corpus {args.corpus} --scales 0.5,1,2"
                )
            else:
                print(
                    "\n  No rate in the search reaches it either, so the learning rate is not the"
                    "\n  problem. An ordinary transformer in the published configuration should"
                    "\n  reach this number. Look at the model, the data and the step budget."
                )
            return

        published_run = next((r for s, r in rows if s == 1.0), None)
        if published_run is not None and _passes(published_run, calibration):
            scale, record = 1.0, published_run
            reason = (
                f"the published rate reaches the published range "
                f"({record['scores'][calibration.split]:.2f}), so it stands. Confirmed, not "
                f"replaced."
            )
        else:
            scale, record = min(passing, key=lambda row: row[1]["best_dev_loss"])
            reason = (
                f"the published rate does NOT reach the published range, so it is replaced by "
                f"{scale:g}x, the best of the rates that do. THIS IS A DEVIATION FROM THE PAPER "
                f"and must be reported as one."
            )
            print(
                f"\n  WARNING: the published rate did not reach the published range. Using"
                f"\n  {scale:g}x instead. Our pipeline differs from theirs somewhere, and the"
                f"\n  write-up has to say so."
            )

    print(f"\n  {reason}")
    if calibration is not None:
        observed = record["scores"][calibration.split]
        print(f"  baseline {calibration.split} {observed:.2f}: {calibration.verdict(observed)}")

    scales = sorted({s for s, _ in rows})
    recipe = replace(
        published,
        learning_rate=scale * published.learning_rate,
        verified=True,
        source=(
            f"{args.corpus}: searched {len(rows)} rates "
            f"({', '.join(f'{s:g}x' for s in scales)} the published one) on `{SEARCH_MODEL}` "
            f"alone. {reason} Every architecture in the grid is trained on this recipe "
            f"unchanged; none of them had a vote in it. "
            f"Published starting point: {published.source}"
        ),
    )
    path = recipes.save(
        args.corpus,
        recipe,
        [
            {
                "scale": s,
                "dev": r["best_dev_loss"],
                "scores": r["scores"],
                "chosen": s == scale,
            }
            for s, r in sorted(rows)
        ],
    )
    print(f"\n  written to {path}. Commit it: a recipe is a result, and results are kept.")
    print(f"  next:  ana run --corpus {args.corpus}")


def _run(args: argparse.Namespace) -> None:
    recipe = _recipe(args.corpus)
    if not recipe.verified and not args.force:
        print(
            f"{args.corpus} has no confirmed recipe, so every model would be trained on a rate"
            f"\nthat nothing has checked, and an undertrained baseline flatters every model"
            f"\nbeneath it without leaving a trace in the numbers. Run `ana tune --corpus"
            f"\n{args.corpus}` first, or pass --force to go anyway."
        )
        return

    seeds = [int(s) for s in (args.seeds or DEFAULT_SEEDS.get(args.corpus, "42")).split(",")]
    jobs = []
    for seed in seeds:
        for model in PILOT_MODELS:
            if os.path.exists(
                os.path.join(args.out, f"{args.corpus}_{model}_seed{seed}", "results.json")
            ):
                continue
            jobs.append(
                f"{args.python} -m ana.cli.__main__ train --model {model} "
                f"--corpus {args.corpus} --seed {seed} --out {args.out}"
            )

    total = len(seeds) * len(PILOT_MODELS)
    print(
        f"{args.corpus}: {len(PILOT_MODELS)} models x {len(seeds)} seed(s) {seeds} = {total} runs, "
        f"{len(jobs)} still to do\n"
    )
    _shard(jobs, args.gpus, f"run_{args.corpus}", args.dry_run)
    if jobs and not args.dry_run:
        print("then:  ana report")


def _factorial(args: argparse.Namespace) -> None:
    """Launch only the fifteen preregistered cells, under an isolated study namespace."""
    recipe = _recipe(FACTORIAL_CORPUS)
    if not recipe.verified:
        raise SystemExit(
            f"{FACTORIAL_CORPUS} has no confirmed recipe; the factorial screen cannot start"
        )

    jobs = []
    for seed in FACTORIAL_SEEDS:
        for model in FACTORIAL_MODELS:
            done = os.path.join(args.out, f"{FACTORIAL_CORPUS}_{model}_seed{seed}", "results.json")
            if os.path.exists(done):
                continue
            jobs.append(
                f"{args.python} -m ana.cli.__main__ train --model {model} "
                f"--corpus {FACTORIAL_CORPUS} --seed {seed} --out {args.out} "
                f"--study-id {FACTORIAL_STUDY_ID} --score-dev"
            )

    gpu_ids = (
        [int(value) for value in args.gpu_ids.split(",")]
        if args.gpu_ids
        else list(range(args.gpus))
    )
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise SystemExit("--gpu-ids must be a non-empty comma-separated list of unique integers")

    total = len(FACTORIAL_MODELS) * len(FACTORIAL_SEEDS)
    print(
        f"{FACTORIAL_STUDY_ID}: {len(FACTORIAL_MODELS)} models x "
        f"{len(FACTORIAL_SEEDS)} seeds = {total} runs, {len(jobs)} still to do\n"
    )
    _shard(jobs, gpu_ids, FACTORIAL_STUDY_ID, args.dry_run)


def _d4_diagnostic(args: argparse.Namespace) -> None:
    """Analyze trained factorial checkpoints; this command never enters the training path."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested {device}, but CUDA is unavailable")
    artifact = run_diagnostic(
        args.run_dir,
        args.json,
        args.markdown,
        device,
    )
    print(
        f"{artifact['study_id']}: wrote {artifact['decode_count']} development decodes to "
        f"{args.json} and {args.markdown}",
        flush=True,
    )


def _permutation_family(args: argparse.Namespace) -> None:
    """Preflight, launch, or analyze the nine-cell fixed-family screen."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit(f"requested {device}, but CUDA is unavailable")

    if args.analyze:
        artifact = run_permutation_family_analysis(
            args.out,
            args.factorial_artifact,
            args.diagnostic_artifact,
            args.json,
            args.markdown,
            device,
        )
        print(
            f"{artifact['study_id']}: wrote nine new cells and reused references to "
            f"{args.json} and {args.markdown}",
            flush=True,
        )
        return

    recipe = _recipe(FACTORIAL_CORPUS)
    if not recipe.verified:
        raise SystemExit(
            f"{FACTORIAL_CORPUS} has no confirmed recipe; the family screen cannot start"
        )
    gpu_ids = (
        [int(value) for value in args.gpu_ids.split(",")]
        if args.gpu_ids
        else list(range(args.gpus))
    )
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise SystemExit("--gpu-ids must be a non-empty comma-separated list of unique integers")

    jobs = []
    for seed in PERMUTATION_FAMILY_SEEDS:
        for model in PERMUTATION_FAMILY_MODELS:
            done = os.path.join(args.out, f"{FACTORIAL_CORPUS}_{model}_seed{seed}", "results.json")
            if os.path.exists(done):
                continue
            jobs.append(
                f"{args.python} -m ana.cli.__main__ train --model {model} "
                f"--corpus {FACTORIAL_CORPUS} --seed {seed} --out {args.out} "
                f"--study-id {PERMUTATION_FAMILY_STUDY_ID} --score-dev "
                f"--evaluated-splits dev"
            )

    total = len(PERMUTATION_FAMILY_MODELS) * len(PERMUTATION_FAMILY_SEEDS)
    print(
        f"{PERMUTATION_FAMILY_STUDY_ID}: {len(PERMUTATION_FAMILY_MODELS)} controls x "
        f"{len(PERMUTATION_FAMILY_SEEDS)} seeds = {total} new runs, "
        f"{len(jobs)} still to do\n"
    )
    if args.dry_run:
        print("dry run: compatibility preflight is deferred because no training will start")
    elif jobs:
        run_compatibility_preflight(args.source_run_dir, args.out, device)
    _shard(jobs, gpu_ids, PERMUTATION_FAMILY_STUDY_ID, args.dry_run)
    if jobs and not args.dry_run:
        print(
            "then rerun with --analyze after all nine results files exist",
            flush=True,
        )


def _s4_screen(args: argparse.Namespace) -> None:
    """Launch or summarize exactly three development-only full-S4 cells."""
    if args.analyze:
        artifact = write_s4_results(
            args.out,
            args.reference_artifact,
            args.json,
            args.markdown,
        )
        print(
            f"{artifact['study_id']}: wrote three development-only cells to "
            f"{args.json} and {args.markdown}",
            flush=True,
        )
        return

    recipe = _recipe(S4_CORPUS)
    if not recipe.verified:
        raise SystemExit(f"{S4_CORPUS} has no confirmed recipe; the S4 screen cannot start")
    gpu_ids = (
        [int(value) for value in args.gpu_ids.split(",")]
        if args.gpu_ids
        else list(range(args.gpus))
    )
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise SystemExit("--gpu-ids must be a non-empty comma-separated list of unique integers")

    jobs = []
    for seed in S4_SEEDS:
        done = os.path.join(args.out, f"{S4_CORPUS}_{S4_MODEL}_seed{seed}", "results.json")
        if os.path.exists(done):
            continue
        jobs.append(
            f"{args.python} -m ana.cli.__main__ train --model {S4_MODEL} "
            f"--corpus {S4_CORPUS} --seed {seed} --out {args.out} "
            f"--study-id {S4_STUDY_ID} --score-dev --evaluated-splits dev"
        )

    print(f"{S4_STUDY_ID}: one model x three seeds = 3 runs, {len(jobs)} still to do\n")
    _shard(jobs, gpu_ids, S4_STUDY_ID, args.dry_run)
    if jobs and not args.dry_run:
        print("then rerun with --analyze after all three results files exist", flush=True)


def _v4_core_diagnostic(args: argparse.Namespace) -> None:
    """Analyze the common V4 core in 12 existing checkpoints without training."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested {device}, but CUDA is unavailable")
    artifact = run_v4_core_diagnostic(
        args.d4_run_dir,
        args.family_run_dir,
        args.d4_artifact,
        args.family_artifact,
        args.json,
        args.markdown,
        device,
    )
    print(
        f"{artifact['study_id']}: reproduced 12 checkpoints and wrote 60 new development-only "
        f"interventions to {args.json} and {args.markdown}",
        flush=True,
    )


def _report(args: argparse.Namespace) -> None:
    records, smoked = [], 0
    for path in sorted(glob.glob(os.path.join(args.out, "*", "results.json"))):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        # A smoke run is a 32-wide model trained for thirty steps. It is not a result, and a
        # table that quietly included one would look entirely ordinary.
        if record["manifest"].get("smoke"):
            smoked += 1
            continue
        records.append(record)

    if smoked:
        print(f"ignoring {smoked} smoke run(s) under {args.out}; they are not results.")

    if not records:
        print(f"no runs under {args.out}. Start with `ana tune --corpus cogs`.")
        return

    by_corpus: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        by_corpus[record["corpus"]][record["model"]][record["manifest"]["seed"]] = record

    for corpus, models in sorted(by_corpus.items()):
        every = [run for runs in models.values() for run in runs.values()]
        splits = sorted({split for run in every for split in run["scores"]})
        seeds = sorted({seed for runs in models.values() for seed in runs})

        # WHERE THE ± COMES FROM, AND WHY IT MUST BE ONE THING PER TABLE.
        #
        # A corpus that is run over several seeds gets its ± from the SEEDS: it is the spread you
        # would see if you trained again, which is the question a comparison of architectures is
        # asking. A corpus that is run once -- which is what the papers for the translation
        # corpora do -- gets its ± from a BOOTSTRAP over the test set instead, which is a real
        # interval but a different one.
        #
        # The choice is a property of the CORPUS, never of the model. Deciding it per model would
        # put a seed spread and a bootstrap band in the same column, under the same ± sign, with
        # nothing to tell them apart -- and mid-grid, when some models have more seeds than
        # others, that is exactly what happens.
        seeded = len(DEFAULT_SEEDS.get(corpus, "42").split(",")) > 1

        print(f"\n{corpus}   seeds {seeds}   metric {every[0]['metric']}")
        if seeded:
            print("  ± is the spread across training runs (seeds).")
        else:
            print("  One training run per model, as the papers for this corpus report.")
            print("  ± is a bootstrap over the test set.")

        header = f"\n  {'model':18}{'params':>9}{'saved':>8}"
        print(header + "".join(f"{s:>15}" for s in splits))
        for name in REGISTRY:
            if name not in models:
                continue
            runs = models[name]
            first = next(iter(runs.values()))
            row = f"  {name:18}{first['parameters'] / 1e6:8.2f}M{first['saved_fraction']:>7.1%}"
            for split in splits:
                values = [r["scores"][split] for r in runs.values() if split in r["scores"]]
                if not values:
                    row += f"{'-':>15}"
                elif len(values) > 1:
                    row += f"{sum(values) / len(values):>10.2f}±{_sd(values):4.1f}"
                elif seeded:
                    # A seeded corpus mid-grid. One run is not a spread, and a bootstrap band
                    # here would be a different quantity wearing the same ± sign.
                    row += f"{values[0]:>13.2f}  "
                else:
                    band = _interval(corpus, name, split, seeds[0], args.out)
                    row += (
                        f"{values[0]:>10.2f}±{band:4.1f}"
                        if band is not None
                        else f"{values[0]:>15.2f}"
                    )
            print(row)

        if seeded and any(len(runs) < len(seeds) for runs in models.values()):
            print("\n  (a row without a ± has not finished all its seeds yet)")

        for against in args.against:
            if against not in models:
                continue
            if seeded:
                _across_seed_block(models, splits, seeds, against)
            else:
                _bootstrap_block(corpus, models, splits, seeds[0], against, args.out)


def _across_seed_block(models, splits, seeds, against) -> None:
    print(f"\n  against {against}, paired across seeds. The variation is the training run.\n")

    printed = 0
    for split in splits:
        for name in REGISTRY:
            if name not in models or name == against:
                continue

            # The test is paired, so the two models must be compared on the SAME seeds. Taking
            # the seeds each of them happens to have finished and checking only that the counts
            # match would pair seed 43 of one against seed 44 of the other -- and that goes
            # wrong precisely when a machine has died part way through a grid, which is the case
            # the restart behaviour makes ordinary.
            shared = sorted(set(models[name]) & set(models[against]))
            shared = [s for s in shared if split in models[name][s]["scores"]]
            if len(shared) < 2:
                continue

            ours = [models[name][s]["scores"][split] for s in shared]
            base = [models[against][s]["scores"][split] for s in shared]
            result = across_seed_test(ours, base)
            mark = "*" if result.significant else " "
            note = "" if len(shared) == len(seeds) else f"  (seeds {shared})"
            print(
                f"    {split:8} {name:18} {result.difference:+7.2f} "
                f"[{result.low:+.2f}, {result.high:+.2f}]  p={result.p_value:.3f} {mark}{note}"
            )
            printed += 1

    print(f"\n  {printed} comparisons, over {len(seeds)} seeds, uncorrected for multiplicity.")


def _bootstrap_block(corpus, models, splits, seed, against, out) -> None:
    """With one seed there is no across-seed test, so say what a bootstrap can and cannot do.

    Resampling the test set answers: if we had drawn a different test set from the same
    distribution, how much would this difference move? It says nothing about what a second
    training run would do, and on a single seed that is the question that actually matters.
    """
    metric = build_corpus(corpus).metric

    def hypotheses(model: str, split: str) -> list[str] | None:
        path = os.path.join(out, f"{corpus}_{model}_seed{seed}", f"hypotheses.{split}.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()

    def references(split: str) -> list[str] | None:
        path = os.path.join(out, f"{corpus}_{against}_seed{seed}", f"references.{split}.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()

    printed, counted = False, 0
    for split in splits:
        refs = references(split)
        base = hypotheses(against, split)
        if refs is None or base is None:
            continue
        for name in REGISTRY:
            if name not in models or name == against:
                continue
            ours = hypotheses(name, split)
            if ours is None or len(ours) != len(refs):
                continue
            if not printed:
                print(
                    f"\n  against {against}, paired bootstrap. Both models are scored on the "
                    f"same\n  resample of the test set, so the variation they share cancels.\n"
                )
                printed = True
            result = paired_bootstrap(ours, base, refs, metric)
            mark = "*" if result.significant else " "
            print(
                f"    {split:8} {name:18} {result.difference:+7.2f} "
                f"[{result.low:+.2f}, {result.high:+.2f}]  p={result.p_value:.3f} {mark}"
            )
            counted += 1

    if counted:
        print(
            f"\n  {counted} comparisons, uncorrected for multiplicity. The variation is the"
            f"\n  test set."
        )


def _lines(path: str) -> list[str] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read().splitlines()


def _interval(corpus: str, model: str, split: str, seed: int, out: str) -> float | None:
    """The ± to print beside a single-run score: a bootstrap over the test set.

    A model trained once still has a stated uncertainty -- it is just uncertainty about the test
    set rather than about the training run. Resampling the test set with replacement gives the
    range a different test set of the same size would have put the score in, and that is a real
    interval, reported as such. It is what the translation corpora carry in place of seeds.
    """
    cell = os.path.join(out, f"{corpus}_{model}_seed{seed}")
    system = _lines(os.path.join(cell, f"hypotheses.{split}.txt"))
    references = _lines(os.path.join(cell, f"references.{split}.txt"))
    if system is None or references is None or len(system) != len(references):
        return None

    return bootstrap_score(system, references, build_corpus(corpus).metric).half_width


def _sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ana",
        description=(
            "Does a routed doubly-stochastic per-role operator recover the accuracy that "
            "sharing the query, key and value projections gives up?"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "the study, in order:\n\n"
            "  ana tune --corpus cogs             check the published recipe       (1 run)\n"
            "  ana tune --corpus cogs --report    check the baseline, write the recipe\n"
            "  ana run  --corpus cogs             all 8 pilot models, all 5 seeds (40 runs)\n"
            "  ana report                         the comparison\n\n"
            "repeat tune and run for multi30k and iwslt14. Nothing needs the network, and a\n"
            "machine that dies part way through can be restarted: finished cells are skipped.\n\n"
            "each corpus carries the architecture and recipe its published number was reported\n"
            "under, so `baseline` has something real to be checked against. `tune` will not\n"
            "write a recipe until it reaches that number, because an undertrained baseline\n"
            "flatters every model beneath it and leaves no trace in the final table."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tune = sub.add_parser("tune", help="[1] confirm the learning rate for a corpus, and record it")
    tune.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    tune.add_argument("--report", action="store_true", help="read the search and write the recipe")
    default_scales = ",".join(f"{s:g}" for s in SEARCH_SCALES)
    tune.add_argument(
        "--scales",
        default=None,
        help=f"which multiples of the published rate to try (default {default_scales})",
    )
    tune.add_argument("--gpus", type=int, default=4)
    tune.add_argument("--out", default="runs/tune")
    tune.add_argument("--python", default="python3")
    tune.add_argument("--dry-run", action="store_true", help="write the scripts, start nothing")
    tune.set_defaults(handler=_tune)

    run = sub.add_parser("run", help="[2] train every model on a corpus, at every seed")
    run.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    run.add_argument("--seeds", default=None, help="default: five seeds on cogs, one elsewhere")
    run.add_argument("--gpus", type=int, default=4)
    run.add_argument("--out", default="runs")
    run.add_argument("--python", default="python3")
    run.add_argument("--dry-run", action="store_true", help="write the scripts, start nothing")
    run.add_argument("--force", action="store_true", help="run without a confirmed recipe")
    run.set_defaults(handler=_run)

    factorial = sub.add_parser(
        "factorial",
        help="run the preregistered 15-cell Multi30k magnitude x D4 screen",
    )
    factorial.add_argument("--gpus", type=int, default=1)
    factorial.add_argument(
        "--gpu-ids",
        default=None,
        help="specific visible GPU indices, comma-separated; overrides --gpus",
    )
    factorial.add_argument("--out", default=f"runs/{FACTORIAL_STUDY_ID}")
    factorial.add_argument("--python", default="python3")
    factorial.add_argument(
        "--dry-run", action="store_true", help="write the scripts, start nothing"
    )
    factorial.set_defaults(handler=_factorial)

    diagnostic = sub.add_parser(
        "diagnose-d4",
        help="analyze the six trained Multi30k D4 factorial checkpoints without training",
    )
    diagnostic.add_argument(
        "--run-dir",
        default=f"runs/{FACTORIAL_STUDY_ID}",
        help="private factorial run directory containing weights.pt and results.json siblings",
    )
    diagnostic.add_argument(
        "--json",
        default=f"results/{D4_DIAGNOSTIC_STUDY_ID}.json",
        help="compact JSON artifact path",
    )
    diagnostic.add_argument(
        "--markdown",
        default=f"results/{D4_DIAGNOSTIC_STUDY_ID}.md",
        help="generated Markdown report path",
    )
    diagnostic.add_argument("--device", default="cuda:0")
    diagnostic.set_defaults(handler=_d4_diagnostic)

    family = sub.add_parser(
        "permutation-family",
        help="run or analyze the nine-cell Multi30k fixed-permutation-family screen",
    )
    family.add_argument("--gpus", type=int, default=1)
    family.add_argument(
        "--gpu-ids",
        default=None,
        help="specific visible GPU indices, comma-separated; overrides --gpus",
    )
    family.add_argument("--out", default=f"runs/{PERMUTATION_FAMILY_STUDY_ID}")
    family.add_argument("--python", default="python3")
    family.add_argument(
        "--source-run-dir",
        default=f"runs/{FACTORIAL_STUDY_ID}",
        help="private factorial checkpoint directory used by the mandatory preflight",
    )
    family.add_argument(
        "--factorial-artifact",
        default=f"results/{FACTORIAL_STUDY_ID}.json",
    )
    family.add_argument(
        "--diagnostic-artifact",
        default=f"results/{D4_DIAGNOSTIC_STUDY_ID}.json",
    )
    family.add_argument(
        "--json",
        default=f"results/{PERMUTATION_FAMILY_STUDY_ID}.json",
    )
    family.add_argument(
        "--markdown",
        default=f"results/{PERMUTATION_FAMILY_STUDY_ID}.md",
    )
    family.add_argument("--device", default="cuda:0")
    family.add_argument("--analyze", action="store_true")
    family.add_argument("--dry-run", action="store_true", help="write scripts, start nothing")
    family.set_defaults(handler=_permutation_family)

    s4 = sub.add_parser(
        "s4-screen",
        help="run or summarize the three-cell development-only full-S4 screen",
    )
    s4.add_argument("--gpus", type=int, default=1)
    s4.add_argument(
        "--gpu-ids",
        default=None,
        help="specific visible GPU indices, comma-separated; overrides --gpus",
    )
    s4.add_argument("--out", default=f"runs/{S4_STUDY_ID}")
    s4.add_argument("--python", default="python3")
    s4.add_argument(
        "--reference-artifact",
        default=f"results/{PERMUTATION_FAMILY_STUDY_ID}.json",
    )
    s4.add_argument("--json", default=f"results/{S4_STUDY_ID}.json")
    s4.add_argument("--markdown", default=f"results/{S4_STUDY_ID}.md")
    s4.add_argument("--analyze", action="store_true")
    s4.add_argument("--dry-run", action="store_true", help="write scripts, start nothing")
    s4.set_defaults(handler=_s4_screen)

    core = sub.add_parser(
        "diagnose-v4-core",
        help="run the inference-only 12-checkpoint shared-V4-core diagnostic",
    )
    core.add_argument(
        "--d4-run-dir",
        default=f"runs/{FACTORIAL_STUDY_ID}",
        help="private factorial directory containing the three D4-only checkpoints",
    )
    core.add_argument(
        "--family-run-dir",
        default=f"runs/{PERMUTATION_FAMILY_STUDY_ID}",
        help="private permutation-family directory containing the nine control checkpoints",
    )
    core.add_argument(
        "--d4-artifact",
        default=f"results/{D4_DIAGNOSTIC_STUDY_ID}.json",
    )
    core.add_argument(
        "--family-artifact",
        default=f"results/{PERMUTATION_FAMILY_STUDY_ID}.json",
    )
    core.add_argument(
        "--json",
        default=f"results/{V4_CORE_DIAGNOSTIC_STUDY_ID}.json",
    )
    core.add_argument(
        "--markdown",
        default=f"results/{V4_CORE_DIAGNOSTIC_STUDY_ID}.md",
    )
    core.add_argument("--device", default="cuda:0")
    core.set_defaults(handler=_v4_core_diagnostic)

    report = sub.add_parser("report", help="[3] read the runs and print the comparison")
    report.add_argument("--out", default="runs")
    report.add_argument(
        "--against",
        nargs="+",
        default=["baseline", "shared_qkv", "baseline_matched"],
        help="the models to compare against. All three matter: `baseline` is the full-size "
        "accuracy being traded away, `shared_qkv` is the prior work being improved on, and "
        "`baseline_matched` decides whether beating shared_qkv means anything",
    )
    report.set_defaults(handler=_report)

    train = sub.add_parser("train", help="one model, one corpus, one seed. `run` calls this")
    train.add_argument("--model", default="all", help=f"{', '.join(REGISTRY)}, or all")
    train.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--steps", type=int, default=None, help="overrides the recipe")
    train.add_argument("--batch-size", type=int, default=None, help="overrides the recipe")
    train.add_argument("--learning-rate", type=float, default=None, help="overrides the recipe")
    train.add_argument("--warmup", type=int, default=None, help="overrides the recipe")
    train.add_argument("--out", default="runs")
    train.add_argument("--smoke", action="store_true", help="tiny model, 30 steps")
    train.add_argument(
        "--study-id", default=None, help="provenance namespace written to the record"
    )
    train.add_argument(
        "--score-dev",
        action="store_true",
        help="decode and score the development split as well as outcome splits",
    )
    train.add_argument(
        "--evaluated-splits",
        default=None,
        help="comma-separated explicit scoring splits; also limits data loading to train/dev plus "
        "these splits",
    )
    train.set_defaults(handler=_train)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
