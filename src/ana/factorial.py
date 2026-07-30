"""The preregistered Multi30k magnitude × D4 screening study."""

from __future__ import annotations

import json
import os
import statistics
from itertools import product

STUDY_ID = "multi30k_factorial_v1"
CORPUS = "multi30k"
MODELS = (
    "baseline_matched",
    "shared_qkv",
    "ana_mag_enc",
    "ana_d4_enc",
    "ana_feat_enc",
)
SEEDS = (42, 43, 44)
SPLITS = ("dev", "test")

FACTORIAL_CELLS = {
    "00": "shared_qkv",
    "10": "ana_mag_enc",
    "01": "ana_d4_enc",
    "11": "ana_feat_enc",
}


def load_records(run_dir: str) -> list[dict]:
    """Load exactly the preregistered cells from an isolated run directory."""
    records = []
    missing = []
    for model, seed in product(MODELS, SEEDS):
        path = os.path.join(run_dir, f"{CORPUS}_{model}_seed{seed}", "results.json")
        if not os.path.exists(path):
            missing.append(f"{model}/seed{seed}")
            continue
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))

    if missing:
        raise ValueError(f"{STUDY_ID} is incomplete; missing {', '.join(missing)}")
    validate_records(records)
    return records


def validate_records(records: list[dict]) -> None:
    """Refuse mixed, stale, partial, or silently reused experiment records."""
    expected = set(product(MODELS, SEEDS))
    observed: set[tuple[str, int]] = set()
    commits = set()
    recipes = set()

    for record in records:
        manifest = record.get("manifest", {})
        model = record.get("model")
        seed = manifest.get("seed")
        if model not in MODELS or seed not in SEEDS:
            raise ValueError(f"unexpected factorial cell {model}/seed{seed}")
        cell = (model, seed)
        if cell in observed:
            raise ValueError(f"duplicate factorial cell {model}/seed{seed}")
        observed.add(cell)

        if record.get("corpus") != CORPUS:
            raise ValueError(f"{model}/seed{seed} uses corpus {record.get('corpus')!r}")
        if manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"{model}/seed{seed} is not marked as study {STUDY_ID}")
        if manifest.get("seeded_before_model_init") is not True:
            raise ValueError(f"{model}/seed{seed} lacks pre-construction seeding provenance")
        if manifest.get("smoke"):
            raise ValueError(f"{model}/seed{seed} is a smoke run, not a full result")
        if manifest.get("score_dev") is not True:
            raise ValueError(f"{model}/seed{seed} did not request development-set scoring")
        if any(split not in record.get("scores", {}) for split in SPLITS):
            raise ValueError(f"{model}/seed{seed} lacks development or test BLEU")

        train_config = manifest.get("train_config", {})
        if train_config.get("seed") != seed:
            raise ValueError(f"{model}/seed{seed} has a mismatched training seed")
        recipe_without_seed = {key: value for key, value in train_config.items() if key != "seed"}
        recipes.add(json.dumps(recipe_without_seed, sort_keys=True))
        commits.add(manifest.get("git_commit"))

    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ValueError("; ".join(details))
    if len(commits) != 1 or None in commits:
        raise ValueError(f"factorial cells do not share one recorded Git commit: {commits}")
    if len(recipes) != 1:
        raise ValueError("factorial cells do not share one frozen training recipe")


def _ordered(records: list[dict]) -> list[dict]:
    model_order = {name: position for position, name in enumerate(MODELS)}
    return sorted(records, key=lambda record: (model_order[record["model"]], _seed(record)))


def _seed(record: dict) -> int:
    return int(record["manifest"]["seed"])


def compact_record(records: list[dict]) -> dict:
    """Keep the evidence needed for audit without checkpoints or hypothesis files."""
    validate_records(records)
    runs = []
    for record in _ordered(records):
        manifest = record["manifest"]
        runs.append(
            {
                "model": record["model"],
                "seed": _seed(record),
                "development_loss": record["best_dev_loss"],
                "selected_step": record["scored_step"],
                "development_bleu": record["scores"]["dev"],
                "test_bleu": record["scores"]["test"],
                "parameters": record["parameters"],
                "saved_fraction_vs_full_baseline": record["saved_fraction"],
                "wall_clock_seconds": record["seconds"],
                "training_configuration": manifest["train_config"],
                "model_configuration": manifest["model_config"],
                "git_commit": manifest["git_commit"],
                "seeded_before_model_init": manifest["seeded_before_model_init"],
            }
        )

    return {
        "study_id": STUDY_ID,
        "corpus": CORPUS,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "primary_signal": "development_bleu",
        "runs": runs,
    }


def _by_cell(records: list[dict]) -> dict[tuple[str, int], dict]:
    return {(record["model"], _seed(record)): record for record in records}


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _sd(values: list[float]) -> float:
    return statistics.stdev(values)


def _factorial_effects(
    cells: dict[tuple[str, int], dict], seed: int, split: str
) -> dict[str, float]:
    score = {cell: cells[(model, seed)]["scores"][split] for cell, model in FACTORIAL_CELLS.items()}
    return {
        "magnitude": ((score["10"] - score["00"]) + (score["11"] - score["01"])) / 2,
        "d4": ((score["01"] - score["00"]) + (score["11"] - score["10"])) / 2,
        "interaction": score["11"] - score["10"] - score["01"] + score["00"],
    }


def _difference(
    cells: dict[tuple[str, int], dict],
    model: str,
    reference: str,
    seed: int,
    split: str,
) -> float:
    return cells[(model, seed)]["scores"][split] - cells[(reference, seed)]["scores"][split]


def markdown_report(records: list[dict]) -> str:
    """Render individual runs, paired comparisons, and the preregistered contrasts."""
    validate_records(records)
    cells = _by_cell(records)
    lines = [
        f"# {STUDY_ID}",
        "",
        "Multi30k magnitude × routed-D4 factorial screen. Development BLEU is the primary "
        "screening signal; test BLEU is secondary descriptive evidence.",
        "",
        "## Individual runs",
        "",
        "| model | seed | dev loss | selected step | dev BLEU | test BLEU | "
        "params | saved | hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in _ordered(records):
        lines.append(
            f"| `{record['model']}` | {_seed(record)} | {record['best_dev_loss']:.4f} | "
            f"{record['scored_step']:,} | {record['scores']['dev']:.2f} | "
            f"{record['scores']['test']:.2f} | {record['parameters']:,} | "
            f"{record['saved_fraction']:.1%} | {record['seconds'] / 3600:.2f} |"
        )

    lines += [
        "",
        "## Mean and sample standard deviation",
        "",
        "| model | dev BLEU | test BLEU |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        dev = [cells[(model, seed)]["scores"]["dev"] for seed in SEEDS]
        test = [cells[(model, seed)]["scores"]["test"] for seed in SEEDS]
        lines.append(
            f"| `{model}` | {_mean(dev):.2f} ± {_sd(dev):.2f} | "
            f"{_mean(test):.2f} ± {_sd(test):.2f} |"
        )

    for reference in ("shared_qkv", "baseline_matched"):
        lines += [
            "",
            f"## Paired per-seed differences against `{reference}`",
            "",
            "| model | seed | dev BLEU Δ | test BLEU Δ |",
            "|---|---:|---:|---:|",
        ]
        for model in MODELS:
            if model == reference:
                continue
            for seed in SEEDS:
                dev = _difference(cells, model, reference, seed, "dev")
                test = _difference(cells, model, reference, seed, "test")
                lines.append(f"| `{model}` | {seed} | {dev:+.2f} | {test:+.2f} |")

    effects = {
        split: {seed: _factorial_effects(cells, seed, split) for seed in SEEDS} for split in SPLITS
    }
    lines += [
        "",
        "## Factorial contrasts",
        "",
        "| seed | dev magnitude | dev D4 | dev interaction | "
        "test magnitude | test D4 | test interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        dev = effects["dev"][seed]
        test = effects["test"][seed]
        lines.append(
            f"| {seed} | {dev['magnitude']:+.2f} | {dev['d4']:+.2f} | "
            f"{dev['interaction']:+.2f} | {test['magnitude']:+.2f} | "
            f"{test['d4']:+.2f} | {test['interaction']:+.2f} |"
        )
    dev_mean = {
        key: _mean([effects["dev"][seed][key] for seed in SEEDS])
        for key in effects["dev"][SEEDS[0]]
    }
    test_mean = {
        key: _mean([effects["test"][seed][key] for seed in SEEDS])
        for key in effects["test"][SEEDS[0]]
    }
    lines.append(
        f"| **mean** | **{dev_mean['magnitude']:+.2f}** | **{dev_mean['d4']:+.2f}** | "
        f"**{dev_mean['interaction']:+.2f}** | **{test_mean['magnitude']:+.2f}** | "
        f"**{test_mean['d4']:+.2f}** | **{test_mean['interaction']:+.2f}** |"
    )

    combined = [_difference(cells, "ana_feat_enc", "shared_qkv", seed, "dev") for seed in SEEDS]
    magnitude = [_difference(cells, "ana_mag_enc", "shared_qkv", seed, "dev") for seed in SEEDS]
    d4 = [_difference(cells, "ana_d4_enc", "shared_qkv", seed, "dev") for seed in SEEDS]
    best_single = [
        max(
            cells[("ana_mag_enc", seed)]["scores"]["dev"],
            cells[("ana_d4_enc", seed)]["scores"]["dev"],
        )
        for seed in SEEDS
    ]
    combined_over_single = [
        cells[("ana_feat_enc", seed)]["scores"]["dev"] - best
        for seed, best in zip(SEEDS, best_single, strict=True)
    ]
    combined_over_magnitude = [
        _difference(cells, "ana_feat_enc", "ana_mag_enc", seed, "dev") for seed in SEEDS
    ]
    combined_over_d4 = [
        _difference(cells, "ana_feat_enc", "ana_d4_enc", seed, "dev") for seed in SEEDS
    ]
    mean_combined = _mean(combined)
    mean_magnitude = _mean(magnitude)
    if abs(mean_combined) > 1e-12:
        magnitude_share = 100.0 * mean_magnitude / mean_combined
        magnitude_summary = (
            f"Magnitude alone reproduces {magnitude_share:.0f}% of the combined model's mean "
            f"development gain over `shared_qkv` ({mean_magnitude:+.2f} of {mean_combined:+.2f})."
        )
    else:
        magnitude_summary = (
            "The combined model has no mean development gain over `shared_qkv`; magnitude alone "
            f"changes development BLEU by {mean_magnitude:+.2f}."
        )
    best_shared_model = max(
        ("shared_qkv", "ana_mag_enc", "ana_d4_enc", "ana_feat_enc"),
        key=lambda model: _mean([cells[(model, seed)]["scores"]["dev"] for seed in SEEDS]),
    )
    best_shared_dev = _mean([cells[(best_shared_model, seed)]["scores"]["dev"] for seed in SEEDS])
    matched_dev = _mean([cells[("baseline_matched", seed)]["scores"]["dev"] for seed in SEEDS])

    lines += [
        "",
        "## Screening questions",
        "",
        f"- {magnitude_summary}",
        f"- D4 mixing without the magnifier changes development BLEU by {_mean(d4):+.2f} versus "
        f"`shared_qkv`; the paired difference is positive on "
        f"{sum(value > 0 for value in d4)}/{len(SEEDS)} seeds: "
        f"{', '.join(f'{value:+.2f}' for value in d4)}.",
        f"- The combined model averages {_mean(combined_over_magnitude):+.2f} development BLEU "
        f"versus magnitude alone and {_mean(combined_over_d4):+.2f} versus D4 alone. Against the "
        f"better single-component model on each seed, it averages "
        f"{_mean(combined_over_single):+.2f} and is positive on "
        f"{sum(value > 0 for value in combined_over_single)}/{len(SEEDS)} seeds.",
        f"- The combined gain over `shared_qkv` is positive on "
        f"{sum(value > 0 for value in combined)}/{len(SEEDS)} seeds: "
        f"{', '.join(f'{value:+.2f}' for value in combined)}.",
        f"- The best shared-QKV variant is `{best_shared_model}` at {best_shared_dev:.2f} mean "
        f"development BLEU, versus {matched_dev:.2f} for `baseline_matched` "
        f"({best_shared_dev - matched_dev:+.2f}).",
        "",
        "No p-values or bootstrap significance tests are reported for this screening study.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(run_dir: str, json_path: str, markdown_path: str) -> None:
    records = load_records(run_dir)
    compact = compact_record(records)
    report = markdown_report(records)

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(markdown_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(compact, handle, indent=2)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(report)
