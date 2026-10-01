"""Bootstrap the saved predictions; include supplied historical control rows."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from models import MODEL_LABELS, models_for_corpus

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score

SERVER = Path(os.environ.get(
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-combined-v4"
))
NAS = Path(os.environ.get(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-combined-v4"
))
SPLITS = {
    "multi30k": "test",
    "multi30k_enfr": "test",
    "cogs": "gen",
    "iwslt14": "test",
}
TITLES = {
    "multi30k": "Multi30k English → German",
    "multi30k_enfr": "Multi30k English → French",
    "cogs": "COGS generalization",
    "iwslt14": "IWSLT14 German → English",
}
REFERENCE_LABELS = {
    "baseline": "Full Transformer, standard embeddings",
    "shared_qkv": "Shared QKV",
    "pre_crossq": "Design 2: before projection + separate cross-attention Q",
    "pre_lowrank": "Design 4: before projection + small role-specific mixing",
}
STUDY_ID = "analogy_combined_v4"
STUDY_TITLE = "combined analogy-preserving projections, batch 4"
TRAINING_NOTE = (
    "Original standard embeddings and corpus recipes; ordinary single-pass "
    "cross-entropy training. Routers use one tenth of the base learning rate. "
    "COGS still scores final weights; its best-dev checkpoint is diagnostic only."
)
NOTE = (
    "New-run ± is half the width of a 95% example-bootstrap interval "
    "(1,000 resamples). Reference rows are supplied existing results, not "
    "new runs. Their provenance is retained in JSON. COGS reports generalization only."
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_row(corpus, task_root, model, expected, cached=None):
    split = SPLITS[corpus.name]
    root = task_root / "runs" / f"{corpus.name}_{model}_seed42"
    result_path = root / "results.json"
    row = {
        "model": model, "status": "pending", "parameters": None,
        "source_results": str(result_path),
    }
    if not result_path.is_file():
        marker = root / "training_complete.json"
        if marker.is_file():
            row.update(status="decoding_pending", parameters=read_json(marker)["parameters"])
        return row
    record = read_json(result_path)
    if record.get("model") != model or record.get("corpus") != corpus.name:
        raise ValueError(f"Result belongs to another model or corpus: {result_path}")
    row["parameters"] = record["parameters"]
    if split not in record.get("scores", {}):
        row["status"] = "decoding_pending"
        return row
    expected_steps = 50_000 if corpus.name in ("cogs", "iwslt14") else 20_000
    if record["steps_run"] != expected_steps or not 0 < record["scored_step"] <= expected_steps:
        raise ValueError(f"Incomplete training record: {result_path}")
    if corpus.name == "cogs" and record["scored_step"] != expected_steps:
        raise ValueError("COGS must use final weights.")
    hypotheses, references = (
        (root / f"{kind}.{split}.txt").read_text(encoding="utf-8").splitlines()
        for kind in ("hypotheses", "references")
    )
    if not expected or references != expected or len(hypotheses) != len(expected):
        raise ValueError(f"Prediction rows differ from {corpus.name}/{split}: {model}")
    point = corpus.metric.score(hypotheses, references)
    if not math.isclose(point, record["scores"][split], rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Recorded score differs from predictions: {model}")
    stamp = {
        "results": result_path.stat().st_mtime_ns,
        "hypotheses": (root / f"hypotheses.{split}.txt").stat().st_mtime_ns,
        "references": (root / f"references.{split}.txt").stat().st_mtime_ns,
    }
    if (
        cached and cached.get("source_mtime_ns") == stamp
        and cached.get("source_results") == str(result_path)
        and cached.get("interval", {}).get("score") == point
    ):
        interval = cached["interval"]
    else:
        value = bootstrap_score(hypotheses, references, corpus.metric, resamples=1000, seed=12345)
        interval = {**asdict(value), "half_width": value.half_width}
    full = record["baseline_parameters"]
    row.update(
        status="completed", interval=interval, source_mtime_ns=stamp,
        baseline_parameters=full, saved_percent=100 * (1 - record["parameters"] / full),
        model_config=record["manifest"]["model_config"],
        train_config=record["manifest"]["train_config"],
        model_details=record["manifest"]["model_details"],
        training_steps=record["steps_run"], scored_step=record["scored_step"],
        training_seconds=record["seconds"],
        diagnostics_scored=record.get("diagnostics_scored"),
        diagnostics_final=record.get("diagnostics_final"),
        source_hypotheses=str(root / f"hypotheses.{split}.txt"),
        source_references=str(root / f"references.{split}.txt"),
    )
    return row


def markdown(report):
    metric = "Exact match (%)" if report["corpus"] == "cogs" else "BLEU"
    lines = [
        f"# {TITLES[report['corpus']]}: {STUDY_TITLE}", "",
        TRAINING_NOTE, "",
        f"| Model | Parameters | {metric} |", "|---|---:|---:|",
    ]
    for name, row in report["references"].items():
        parameters = f"{row['parameters']:,}" if row.get("parameters") is not None else "—"
        result = f"{row['score']:.2f} ± {row['half_width']:.2f}"
        label = row.get("label", REFERENCE_LABELS.get(name, name))
        lines.append(f"| {label} (existing) | {parameters} | {result} |")
    for name in models_for_corpus(report["corpus"]):
        row = report["models"][name]
        parameters = f"{row['parameters']:,}" if row["parameters"] is not None else "—"
        if row["status"] == "completed":
            value = row["interval"]
            result = f"{value['score']:.2f} ± {value['half_width']:.2f}"
        else:
            result = "Decoding pending" if row["status"] == "decoding_pending" else "Pending"
        lines.append(f"| {MODEL_LABELS[name]} | {parameters} | {result} |")
    lines += [
        "", NOTE, "",
        "Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. "
        "They are measured on development data with dropout off, not on test feedback.", "",
    ]
    return "\n".join(lines)


def report_corpus(name, task_root, reports, references_file):
    corpus = build_corpus(name)
    expected = [row.target for row in corpus.load_split(SPLITS[name])]
    shared_reports = task_root / "reports"
    shared_reports.mkdir(parents=True, exist_ok=True)
    with (shared_reports / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_path = shared_reports / f"{name}.json"
        previous = read_json(previous_path) if previous_path.is_file() else {}
        if previous.get("resamples") != 1000 or previous.get("bootstrap_seed") != 12345:
            previous = {}
        models = {
            model: load_row(
                corpus, task_root, model, expected, previous.get("models", {}).get(model)
            )
            for model in models_for_corpus(name)
        }
        supplied = read_json(references_file).get(name, {}) if references_file.is_file() else {}
        references = {
            key: supplied[key] for key in REFERENCE_LABELS
            if key in supplied and key not in models
        }
        report = {
            "study_id": STUDY_ID, "corpus": name, "split": SPLITS[name],
            "metric": corpus.metric.name, "confidence": 0.95,
            "resamples": 1000, "bootstrap_seed": 12345, "uncertainty_note": NOTE,
            "references": references, "models": models,
            "complete": all(row["status"] == "completed" for row in models.values()),
        }
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        rendered = markdown(report)
        atomic_write(previous_path, serialized)
        atomic_write(shared_reports / f"{name}.md", rendered)
        if reports.resolve() != shared_reports.resolve():
            atomic_write(reports / f"{name}.json", serialized)
            atomic_write(reports / f"{name}.md", rendered)
    print(rendered, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--task-root", type=Path, default=NAS)
    parser.add_argument("--reports-dir", type=Path, default=SERVER / "reports")
    parser.add_argument("--references-file", type=Path, default=SERVER / "reference_results.json")
    args = parser.parse_args()
    os.environ["ANA_DATA"] = str(args.task_root / "data")
    for name in SPLITS if args.corpus == "all" else (args.corpus,):
        report_corpus(name, args.task_root, args.reports_dir, args.references_file)


if __name__ == "__main__":
    main()
