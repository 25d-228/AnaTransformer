"""Compact development-only result for the three-seed full-S4 screen."""

from __future__ import annotations

import json
import os
import statistics

STUDY_ID = "multi30k_s4_v1"
SOURCE_STUDY_ID = "multi30k_permutation_family_v1"
CORPUS = "multi30k"
MODEL = "s4_enc"
SEEDS = (42, 43, 44)
REFERENCE_MODELS = ("shared_qkv", "ana_d4_enc", "perm_ctrl_b_enc", "baseline_matched")

MEAN_TARGET = 39.90
MINIMUM_MEAN_GAIN_OVER_B = 0.20
BASELINE_AROUND_ONE_GAP = -0.80
MAXIMUM_RUNTIME_RATIO = 1.50


def _seed(record: dict) -> int:
    return int(record["manifest"]["seed"])


def load_records(run_dir: str) -> list[dict]:
    """Load exactly the three preregistered S4 cells."""
    records = []
    missing = []
    for seed in SEEDS:
        path = os.path.join(run_dir, f"{CORPUS}_{MODEL}_seed{seed}", "results.json")
        if not os.path.exists(path):
            missing.append(f"seed{seed}")
            continue
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    if missing:
        raise ValueError(f"{STUDY_ID} is incomplete; missing {', '.join(missing)}")
    _validate_records(records)
    return records


def _validate_records(records: list[dict]) -> None:
    """Reject mixed, partial, smoke, or non-development-only runs."""
    if len(records) != len(SEEDS):
        raise ValueError(f"{STUDY_ID} requires exactly three records")

    observed = set()
    commits = set()
    recipes = set()
    model_configs = set()
    for record in records:
        manifest = record.get("manifest", {})
        seed = _seed(record)
        if record.get("model") != MODEL or record.get("corpus") != CORPUS or seed not in SEEDS:
            raise ValueError(f"unexpected S4 screen cell {record.get('model')}/seed{seed}")
        if seed in observed:
            raise ValueError(f"duplicate S4 screen seed {seed}")
        observed.add(seed)
        if manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"seed{seed} is not marked as study {STUDY_ID}")
        if manifest.get("seeded_before_model_init") is not True or manifest.get("smoke"):
            raise ValueError(f"seed{seed} is not a full, pre-construction-seeded run")
        if manifest.get("score_dev") is not True:
            raise ValueError(f"seed{seed} did not request development scoring")
        if manifest.get("evaluated_splits") != ["dev"] or set(record.get("scores", {})) != {"dev"}:
            raise ValueError(f"seed{seed} is not development-only")

        train_config = manifest.get("train_config", {})
        if train_config.get("seed") != seed or train_config.get("max_steps") != 20_000:
            raise ValueError(f"seed{seed} does not use the frozen 20,000-step recipe")
        recipes.add(
            json.dumps(
                {key: value for key, value in train_config.items() if key != "seed"},
                sort_keys=True,
            )
        )
        model_configs.add(json.dumps(manifest.get("model_config"), sort_keys=True))
        commits.add(manifest.get("git_commit"))

    if observed != set(SEEDS):
        raise ValueError(f"expected seeds {SEEDS}, found {sorted(observed)}")
    if len(recipes) != 1 or len(model_configs) != 1:
        raise ValueError("S4 cells do not share one frozen training and model configuration")
    if len(commits) != 1 or None in commits:
        raise ValueError(f"S4 cells do not share one recorded Git commit: {commits}")


def _original_development_bleu(row: dict) -> float:
    conditions = {
        condition["condition"]: condition["development_bleu"]
        for condition in row.get("conditions", [])
    }
    if "original" not in conditions:
        raise ValueError(
            f"{row.get('model')}/seed{row.get('seed')} lacks the original development BLEU"
        )
    return float(conditions["original"])


def load_references(path: str) -> dict[str, dict[int, dict]]:
    """Read the four reference models from the committed family-screen artifact."""
    with open(path, encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("study_id") != SOURCE_STUDY_ID:
        raise ValueError(f"{path} is not the committed {SOURCE_STUDY_ID} artifact")

    rows = [*source.get("reused_references", []), *source.get("new_runs", [])]
    references: dict[str, dict[int, dict]] = {model: {} for model in REFERENCE_MODELS}
    for row in rows:
        model = row.get("model")
        seed = row.get("seed")
        if model not in references or seed not in SEEDS:
            continue
        if seed in references[model]:
            raise ValueError(f"duplicate reference {model}/seed{seed}")
        references[model][seed] = {
            "development_bleu": _original_development_bleu(row),
            "parameters": int(row["parameters"]),
            "wall_clock_seconds": float(row["wall_clock_seconds"]),
        }

    missing = [
        f"{model}/seed{seed}"
        for model in REFERENCE_MODELS
        for seed in SEEDS
        if seed not in references[model]
    ]
    if missing:
        raise ValueError(f"source artifact lacks {', '.join(missing)}")
    return references


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _sd(values: list[float]) -> float:
    return statistics.stdev(values)


def build_artifact(records: list[dict], reference_path: str) -> dict:
    """Build the compact preregistered result and its paired descriptive comparisons."""
    _validate_records(records)
    references = load_references(reference_path)
    ordered = sorted(records, key=_seed)

    runs = []
    for record in ordered:
        seed = _seed(record)
        score = float(record["scores"]["dev"])
        manifest = record["manifest"]
        runs.append(
            {
                "seed": seed,
                "development_bleu": score,
                "selected_step": int(record["scored_step"]),
                "development_loss": float(record["best_dev_loss"]),
                "parameters": int(record["parameters"]),
                "wall_clock_seconds": float(record["seconds"]),
                "run_commit": manifest["git_commit"],
                "training_configuration": manifest["train_config"],
                "model_configuration": manifest["model_config"],
                "paired_differences_bleu": {
                    model: score - references[model][seed]["development_bleu"]
                    for model in REFERENCE_MODELS
                },
            }
        )

    s4_scores = [row["development_bleu"] for row in runs]
    s4_times = [row["wall_clock_seconds"] for row in runs]
    reference_summary = {}
    comparisons = {}
    for model in REFERENCE_MODELS:
        scores = [references[model][seed]["development_bleu"] for seed in SEEDS]
        times = [references[model][seed]["wall_clock_seconds"] for seed in SEEDS]
        paired = [
            row["development_bleu"] - references[model][row["seed"]]["development_bleu"]
            for row in runs
        ]
        reference_summary[model] = {
            "development_bleu_by_seed": {
                str(seed): references[model][seed]["development_bleu"] for seed in SEEDS
            },
            "mean_development_bleu": _mean(scores),
            "sample_standard_deviation_development_bleu": _sd(scores),
            "parameters": references[model][SEEDS[0]]["parameters"],
            "mean_wall_clock_seconds": _mean(times),
        }
        comparisons[model] = {
            "paired_differences_bleu": paired,
            "mean_difference_bleu": _mean(paired),
        }

    mean_score = _mean(s4_scores)
    mean_time = _mean(s4_times)
    b_comparison = comparisons["perm_ctrl_b_enc"]
    b_wins = sum(value > 0 for value in b_comparison["paired_differences_bleu"])
    seeds_meeting_target = sum(value >= MEAN_TARGET for value in s4_scores)
    runtime_ratio = mean_time / reference_summary["perm_ctrl_b_enc"]["mean_wall_clock_seconds"]
    baseline_gap = comparisons["baseline_matched"]["mean_difference_bleu"]

    continue_checks = {
        "mean_development_bleu_at_least_39_90": mean_score >= MEAN_TARGET,
        "beats_perm_ctrl_b_on_at_least_two_seeds": b_wins >= 2,
        "at_least_two_seeds_reach_39_90": seeds_meeting_target >= 2,
        "runtime_at_most_1_5x_perm_ctrl_b": runtime_ratio <= MAXIMUM_RUNTIME_RATIO,
    }
    stop_checks = {
        "mean_gain_over_perm_ctrl_b_below_0_20": (
            b_comparison["mean_difference_bleu"] < MINIMUM_MEAN_GAIN_OVER_B
        ),
        "gain_over_perm_ctrl_b_on_at_most_one_seed": b_wins <= 1,
        "mean_gap_to_baseline_matched_at_most_minus_0_80": (
            baseline_gap <= BASELINE_AROUND_ONE_GAP
        ),
    }
    promising = all(continue_checks.values()) and not any(stop_checks.values())

    return {
        "study_id": STUDY_ID,
        "corpus": CORPUS,
        "model": MODEL,
        "seeds": list(SEEDS),
        "evaluated_splits": ["dev"],
        "test_split_evaluated": False,
        "source_artifact": reference_path,
        "runs": runs,
        "summary": {
            "mean_development_bleu": mean_score,
            "sample_standard_deviation_development_bleu": _sd(s4_scores),
            "parameters": runs[0]["parameters"],
            "mean_wall_clock_seconds": mean_time,
        },
        "references": reference_summary,
        "comparisons": comparisons,
        "decision": {
            "outcome": (
                "continue_this_direction" if promising else "stop_fixed_permutation_expansion"
            ),
            "promising": promising,
            "continue_checks": continue_checks,
            "stop_checks": stop_checks,
            "perm_ctrl_b_seed_wins": b_wins,
            "seeds_reaching_39_90": seeds_meeting_target,
            "runtime_ratio_vs_perm_ctrl_b": runtime_ratio,
        },
    }


def markdown_report(artifact: dict) -> str:
    """Render one concise table plus the preregistered decision."""
    references = artifact["references"]
    lines = [
        f"# {STUDY_ID}",
        "",
        "Three-seed, development-only Multi30k screen of a soft router over all 24 "
        "permutations of four channels. Differences are `s4_enc` minus the named reference.",
        "",
        "| seed | S4 dev BLEU | shared_qkv (Δ) | ana_d4_enc (Δ) | "
        "perm_ctrl_b_enc (Δ) | baseline_matched (Δ) | step | dev loss | params | hours |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in artifact["runs"]:
        seed = run["seed"]
        cells = []
        for model in REFERENCE_MODELS:
            score = references[model]["development_bleu_by_seed"][str(seed)]
            difference = run["paired_differences_bleu"][model]
            cells.append(f"{score:.2f} ({difference:+.2f})")
        lines.append(
            f"| {seed} | {run['development_bleu']:.2f} | {' | '.join(cells)} | "
            f"{run['selected_step']:,} | {run['development_loss']:.4f} | "
            f"{run['parameters']:,} | {run['wall_clock_seconds'] / 3600:.2f} |"
        )

    summary = artifact["summary"]
    mean_cells = []
    for model in REFERENCE_MODELS:
        reference = references[model]
        difference = artifact["comparisons"][model]["mean_difference_bleu"]
        mean_cells.append(f"{reference['mean_development_bleu']:.2f} ({difference:+.2f})")
    lines.append(
        f"| **mean ± sample SD** | **{summary['mean_development_bleu']:.2f} ± "
        f"{summary['sample_standard_deviation_development_bleu']:.2f}** | "
        f"{' | '.join(mean_cells)} | — | — | {summary['parameters']:,} | "
        f"{summary['mean_wall_clock_seconds'] / 3600:.2f} |"
    )

    decision = artifact["decision"]
    b_gain = artifact["comparisons"]["perm_ctrl_b_enc"]["mean_difference_bleu"]
    baseline_gap = artifact["comparisons"]["baseline_matched"]["mean_difference_bleu"]
    lines += [
        "",
        "## Decision",
        "",
        f"**{decision['outcome'].replace('_', ' ')}.** Mean gain over `perm_ctrl_b_enc` is "
        f"{b_gain:+.2f} BLEU ({decision['perm_ctrl_b_seed_wins']}/3 paired seed wins); "
        f"{decision['seeds_reaching_39_90']}/3 seeds reach 39.90. Mean gap to "
        f"`baseline_matched` is {baseline_gap:+.2f} BLEU, and runtime is "
        f"{decision['runtime_ratio_vs_perm_ctrl_b']:.2f}× the eight-route control.",
        "",
        "For the preregistered “not one exceptional seed” criterion, this compact screen uses "
        "at least two individual S4 seeds reaching 39.90. “Not grossly worse” uses at most "
        "1.5× the mean `perm_ctrl_b_enc` runtime. No test split, hard routing, router "
        "statistics, p-values, or bootstrap tests are included.",
        "",
    ]
    return "\n".join(lines)


def write_results(
    run_dir: str,
    reference_path: str,
    json_path: str,
    markdown_path: str,
) -> dict:
    """Load the three cells and write deterministic compact JSON and Markdown artifacts."""
    artifact = build_artifact(load_records(run_dir), reference_path)
    for path in (json_path, markdown_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report(artifact))
    return artifact
