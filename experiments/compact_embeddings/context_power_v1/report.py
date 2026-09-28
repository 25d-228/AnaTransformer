"""Context-conditioned power comparisons from saved predictions."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path(os.environ.get("ANA_SERVER", "/home/Yue_Ziran/workspace/ana-context-power-v1"))
NAS = Path(os.environ.get("ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-context-power-v1"))
ORDINARY = "embedding_linear"
TWO_PASS = "embedding_two_pass"
FIXED = "embedding_power_consistency"
LOOKAHEAD = "embedding_adaptive_power_lookahead"
PREDICTION = "context_power_prediction"
PREDICTION_ANALOGY = "context_power_prediction_analogy"
FEATURES = "context_power_features"
FEATURES_ANALOGY = "context_power_features_analogy"
FEATURES_FIXED = "context_power_features_fixed"
REFERENCES = (ORDINARY, TWO_PASS, FIXED, LOOKAHEAD)
NEW_MODELS = (PREDICTION, PREDICTION_ANALOGY, FEATURES, FEATURES_ANALOGY, FEATURES_FIXED)
MODELS = (*REFERENCES, *NEW_MODELS)
WITHIN_FAMILY = {
    "prediction_analogy_minus_prediction": (PREDICTION_ANALOGY, PREDICTION),
    "features_analogy_minus_features": (FEATURES_ANALOGY, FEATURES),
    "features_analogy_minus_fixed": (FEATURES_ANALOGY, FEATURES_FIXED),
}
BASE_PARAMETERS = {"multi30k": 2248512, "multi30k_enfr": 2248512, "cogs": 5689236}
SPLITS = {"multi30k": "test", "multi30k_enfr": "test", "cogs": "gen"}
TITLES = {
    "multi30k": "Multi30k English → German",
    "multi30k_enfr": "Multi30k English → French",
    "cogs": "COGS generalization",
}
LABELS = {
    ORDINARY: "Compact embeddings, ordinary training",
    TWO_PASS: "Same compact model, two dropout passes without analogy",
    FIXED: "Same compact model + analogy training, fixed p = 0.5",
    LOOKAHEAD: "Compact model + previous lookahead power training",
    PREDICTION: "A: context-dependent prediction power, without analogy",
    PREDICTION_ANALOGY: "A+: context-dependent prediction power + analogy",
    FEATURES: "B: context-dependent feature power, without analogy",
    FEATURES_ANALOGY: "B+: context-dependent feature power + analogy",
    FEATURES_FIXED: "B fixed: p = 1 feature analogy, without a power head",
}
NOTE = (
    "± is half the width of a 95% example-bootstrap interval (1,000 resamples). "
    "Exact endpoints are retained in JSON. COGS reports generalization only."
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def run_folder(task_root, corpus, model):
    subfolder = "reference_runs" if model in REFERENCES else "runs"
    return task_root / subfolder / f"{corpus}_{model}_seed42"


def predictions(root, split):
    return tuple(
        (root / f"{kind}.{split}.txt").read_text(encoding="utf-8").splitlines()
        for kind in ("hypotheses", "references")
    )


def expected_parameters(corpus, model):
    width = 400 if corpus == "cogs" else 128
    extra = 0
    if model in (PREDICTION, PREDICTION_ANALOGY):
        extra = width + 1
    elif model in (FEATURES, FEATURES_ANALOGY):
        extra = (width // 2) * (width + 1)
    return BASE_PARAMETERS[corpus] + extra


def check_record(record, corpus, model):
    if record.get("model") != model or record.get("corpus") != corpus:
        raise ValueError(f"Result belongs to another model or dataset: {corpus}/{model}")
    if record["parameters"] != expected_parameters(corpus, model):
        raise ValueError(f"Unexpected compact-model parameter count: {corpus}/{model}")
    config = record["manifest"]["train_config"]
    expected = (
        (50000, 128, 0.0001, 0, "constant", "final")
        if corpus == "cogs"
        else (20000, 256, 0.005, 2000, "inverse_sqrt", "best_dev_loss")
    )
    keys = ("max_steps", "batch_size", "learning_rate", "warmup_steps", "schedule", "selection")
    if (
        tuple(config[key] for key in keys) != expected
        or record["steps_run"] != expected[0]
        or not 0 < record["scored_step"] <= expected[0]
        or record.get("extra_steps", 0)
    ):
        raise ValueError(f"Completed result differs from the fixed {corpus} recipe: {model}")
    if corpus == "cogs" and record["scored_step"] != expected[0]:
        raise ValueError("COGS must score the final checkpoint.")
    if model == FIXED:
        context = record["manifest"]["context"]
        required = {
            "base_model": "embedding_linear",
            "parent_checkpoint_used": False,
            "fixed_power": 0.5,
            "learned_power": False,
            "power_scope": "training_loss_only",
            "consistency_coefficient": 1.0,
            "probability_uniform_floor_mass": 1e-6,
            "gradient_bearing_dropout_passes_per_update": 2,
        }
        if any(context.get(key) != value for key, value in required.items()):
            raise ValueError("Expected the unchanged two-gradient p=0.5 reference objective.")


def load_row(corpus, task_root, model, expected, cached=None):
    split = SPLITS[corpus.name]
    root = run_folder(task_root, corpus.name, model)
    result_path = root / "results.json"
    row = {
        "model": model,
        "status": "pending",
        "parameters": expected_parameters(corpus.name, model),
        "source_results": str(result_path),
        "existing_run_reused": model in REFERENCES,
    }
    provenance = root / "source_provenance.json"
    if provenance.is_file():
        row["source_provenance"] = read_json(provenance)
    record = read_json(result_path) if result_path.is_file() else None
    if record is None or split not in record.get("scores", {}):
        marker = root / "training_complete.json"
        partial = (
            record if record is not None else (read_json(marker) if marker.is_file() else None)
        )
        if partial is not None:
            row.update(
                parameters=partial.get("parameters"),
                status="decoding_pending",
                source_record=partial,
            )
        return row
    check_record(record, corpus.name, model)
    hypotheses, references = predictions(root, split)
    if not expected or references != expected or len(hypotheses) != len(expected):
        raise ValueError(f"Saved predictions differ from {corpus.name}/{split} rows: {model}")
    point = corpus.metric.score(hypotheses, references)
    if not math.isclose(point, record["scores"][split], rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Predictions and recorded score disagree: {model}")
    if (
        cached
        and cached.get("source_record") == record
        and cached.get("source_results") == str(result_path)
        and "interval" in cached
        and math.isclose(cached["interval"]["score"], point, rel_tol=0, abs_tol=1e-8)
    ):
        interval = cached["interval"]
    else:
        value = bootstrap_score(hypotheses, references, corpus.metric, resamples=1000, seed=12345)
        interval = {**asdict(value), "half_width": value.half_width}
    full = record.get("standard_transformer_parameters", record.get("baseline_parameters"))
    row.update(
        status="completed",
        parameters=record["parameters"],
        full_parameters=full,
        saved_percent=100 * (1 - record["parameters"] / full) if full else None,
        model_config=record["manifest"]["model_config"],
        training_steps=record["steps_run"],
        scored_step=record["scored_step"],
        training_seconds=record.get("seconds"),
        interval=interval,
        source_record=record,
        source_hypotheses=str(root / f"hypotheses.{split}.txt"),
        source_references=str(root / f"references.{split}.txt"),
    )
    if model == LOOKAHEAD or model in NEW_MODELS:
        row["final_power"] = record.get("power_final")
        row["scored_power"] = record.get("power_scored")
    return row


def compare(corpus, first, other, cached=None):
    sources = [first["source_results"], other["source_results"]]
    records = [first["source_record"], other["source_record"]]
    if (
        cached
        and cached.get("source_results") == sources
        and cached.get("source_records") == records
    ):
        return cached
    split = SPLITS[corpus.name]
    hypotheses, references = predictions(Path(sources[0]).parent, split)
    other_hypotheses, other_references = predictions(Path(sources[1]).parent, split)
    if references != other_references:
        raise ValueError("Paired comparison references differ.")
    value = paired_bootstrap(
        hypotheses, other_hypotheses, references, corpus.metric, resamples=1000, seed=12345
    )
    return {
        "first_model": first["model"],
        "second_model": other["model"],
        "direction": "first_model minus second_model",
        "difference": value.difference,
        "low": value.low,
        "high": value.high,
        "n": value.n,
        "what_varies": value.what_varies,
        "source_results": sources,
        "source_records": records,
    }


def comparison_cell(value):
    if value is None:
        return "Pending"
    return f"{value['difference']:+.2f} [{value['low']:+.2f}, {value['high']:+.2f}]"


def markdown(report):
    metric = "Exact match (%)" if report["corpus"] == "cogs" else "BLEU"
    lines = [
        f"# {TITLES[report['corpus']]}: context-conditioned power",
        "",
        (
            "Four completed reference rows are reused. Five new variants share the "
            "same compact backbone and two-dropout-pass training recipe. A learns "
            "one prediction power per position; B learns one power per feature pair "
            "and position. Their heads are active at inference and are learned "
            "through the supervised prediction loss. The fixed B control has no "
            "power head. A and B use different analogy penalties; coefficient 1 "
            "does not make their penalty strengths equivalent."
        ),
        "",
        f"| Model and training | Parameters | {metric} |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        row = report["models"][model]
        count = f"{row['parameters']:,}" if row["parameters"] is not None else "—"
        if row["status"] == "completed":
            value = row["interval"]
            result = f"{value['score']:.2f} ± {value['half_width']:.2f}"
        else:
            result = "Decoding pending" if row["status"] == "decoding_pending" else "Pending"
        lines.append(f"| {LABELS[model]} | {count} | {result} |")
    lines += ["", NOTE, ""]
    power_rows = []
    for model in NEW_MODELS:
        summary = report["models"][model].get("scored_power")
        if isinstance(summary, dict) and all(key in summary for key in ("mean", "min", "max")):
            power_rows.append(
                f"| {LABELS[model]} | {summary['mean']:.4f} | "
                f"{summary['min']:.4f} | {summary['max']:.4f} |"
            )
    if power_rows:
        lines += [
            "## Context-dependent p at the scored checkpoint",
            "",
            "These summaries use valid development positions with dropout off; "
            "they are not a single global p or test-set feedback.",
            "",
            "| Model | Mean p | Minimum p | Maximum p |",
            "|---|---:|---:|---:|",
            *power_rows,
            "",
        ]
    if report["within_family_comparisons"]:
        lines += [
            "## Matched comparisons",
            "",
            "| First model minus second model | Difference and 95% paired interval |",
            "|---|---:|",
        ]
        for value in report["within_family_comparisons"].values():
            first = LABELS[value["first_model"]]
            second = LABELS[value["second_model"]]
            lines.append(f"| {first} minus {second} | {comparison_cell(value)} |")
        lines.append("")
    if report["comparisons"]:
        lines += [
            "## New models minus completed references",
            "",
            "Each cell gives the difference and its 95% paired bootstrap interval.",
            "",
            "| New model | Ordinary | Two-pass, no analogy | Fixed p = 0.5 | Lookahead |",
            "|---|---:|---:|---:|---:|",
        ]
        for model, values in report["comparisons"].items():
            cells = " | ".join(comparison_cell(values.get(ref)) for ref in REFERENCES)
            lines.append(f"| {LABELS[model]} | {cells} |")
        lines.append("")
    lines.append(
        "Completed scores are computed from saved predictions. Exact interval "
        "endpoints, source records, model counts, and paired comparisons remain in "
        "JSON. Pending entries have no inferred scores. Each dataset retains its "
        "own base training recipe and checkpoint rule; COGS reports generalization only."
    )
    return "\n".join(lines) + "\n"


def report_corpus(name, task_root, reports):
    corpus = build_corpus(name)
    expected = [row.target for row in corpus.load_split(SPLITS[name])]
    previous_path = reports / f"{name}.json"
    previous = read_json(previous_path) if previous_path.is_file() else {}
    if not (
        previous.get("confidence") == 0.95
        and previous.get("resamples") == 1000
        and previous.get("bootstrap_seed") == 12345
    ):
        previous = {}
    models = {
        model: load_row(corpus, task_root, model, expected, previous.get("models", {}).get(model))
        for model in MODELS
    }
    completed = [row for row in models.values() if row["status"] == "completed"]
    if completed:
        first = completed[0]
        if any(row["model_config"] != first["model_config"] for row in completed[1:]):
            raise ValueError("All rows must use the same compact-backbone configuration.")
    comparisons = {}
    for model in NEW_MODELS:
        if models[model]["status"] != "completed":
            continue
        values = {}
        for reference in REFERENCES:
            if models[reference]["status"] == "completed":
                values[reference] = compare(
                    corpus,
                    models[model],
                    models[reference],
                    previous.get("comparisons", {}).get(model, {}).get(reference),
                )
        if values:
            comparisons[model] = values
    within_family = {}
    for name_pair, (first, second) in WITHIN_FAMILY.items():
        if models[first]["status"] == models[second]["status"] == "completed":
            within_family[name_pair] = compare(
                corpus,
                models[first],
                models[second],
                previous.get("within_family_comparisons", {}).get(name_pair),
            )
    report = {
        "study_id": "context_power_v1",
        "corpus": name,
        "split": SPLITS[name],
        "metric": corpus.metric.name,
        "confidence": 0.95,
        "resamples": 1000,
        "bootstrap_seed": 12345,
        "uncertainty_note": NOTE,
        "power_scope": {
            "context_heads": "prediction_and_inference",
            "analogy_penalties": "training_only",
            "reused_references": "unchanged_original_protocols",
        },
        "models": models,
        "comparisons": comparisons,
        "within_family_comparisons": within_family,
        "complete": all(row["status"] == "completed" for row in models.values()),
    }
    provenance_path = task_root / "reference_runs" / "provenance.json"
    if provenance_path.is_file():
        report["reference_source_provenance"] = read_json(provenance_path)
    atomic_write(previous_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write(reports / f"{name}.md", markdown(report))
    print(markdown(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--task-root", type=Path, default=NAS)
    parser.add_argument("--reports-dir", type=Path, default=SERVER / "reports")
    args = parser.parse_args()
    os.environ["ANA_DATA"] = str(args.task_root / "data")
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS if args.corpus == "all" else (args.corpus,):
        with (args.reports_dir / f"{name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            report_corpus(name, args.task_root, args.reports_dir)


if __name__ == "__main__":
    main()
