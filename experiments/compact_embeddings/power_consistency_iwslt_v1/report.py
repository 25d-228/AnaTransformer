"""Report IWSLT14 compact models with and without power training, plus reported references."""

import argparse
import fcntl
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path("/home/Yue_Ziran/workspace/ana-power-consistency-iwslt-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-power-consistency-iwslt-v1")
REFERENCE_SNAPSHOT = Path(__file__).with_name("reference_results.json")
CANDIDATE = "embedding_power_consistency"
ORDINARY = "embedding_linear"
NEW_MODELS = (ORDINARY, CANDIDATE)
CONTROLS = ("baseline", "baseline_matched", "shared_qkv")
LABELS = {
    "baseline": "Full Transformer",
    "baseline_matched": "Similar-size Transformer",
    "shared_qkv": "Shared-QKV",
    ORDINARY: "Linear embeddings, ordinary training",
    CANDIDATE: "Linear embeddings + power-analogy dropout consistency (p = 0.5)",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_candidate(corpus, result_path, expected, cached=None, model=CANDIDATE):
    record = read_json(result_path)
    if record["model"] != model or record["corpus"] != "iwslt14":
        raise ValueError("Result belongs to another model or dataset.")
    config = record["manifest"]["train_config"]
    if (
        config["max_steps"] != 50000
        or config["batch_size"] != 160
        or config["warmup_steps"] != 4000
        or not math.isclose(config["learning_rate"], 0.0005)
        or not math.isclose(config["weight_decay"], 0.0001)
        or config["schedule"] != "inverse_sqrt"
        or config["selection"] != "best_dev_loss"
        or config["beam_size"] != 5
        or record["steps_run"] != 50000
        or not 0 < record["scored_step"] <= 50000
        or record.get("extra_steps", 0)
    ):
        raise ValueError("Expected the fresh, completed 50,000-update IWSLT14 recipe.")
    if record["parameters"] != 27241504:
        raise ValueError("Compact model parameter count differs from the approved design.")
    shape = record["manifest"]["model_config"]
    required_shape = {
        "d_model": 448,
        "d_ff": 856,
        "n_heads": 4,
        "vocab_size": 10000,
        "n_encoder_layers": 6,
        "n_decoder_layers": 6,
        "dropout": 0.3,
        "label_smoothing": 0.1,
        "max_positions": 256,
    }
    if any(shape.get(key) != value for key, value in required_shape.items()):
        raise ValueError("Compact model shape differs from the approved design.")
    context = record["manifest"]["context"]
    required_context = {
        "study_id": "power_consistency_iwslt_v1",
        "base_model": "embedding_linear",
        "parent_checkpoint_used": False,
        "power_scope": "training_loss_only",
        "fixed_power": 0.5,
        "learned_power": False,
        "trainable_power_count": 0,
        "consistency_coefficient": 1.0,
        "probability_uniform_floor_mass": 1e-6,
        "gradient_bearing_dropout_passes_per_update": 2,
        "inference_unchanged": True,
        "compression": "ordinary_tied_linear_336_to_448_embedding",
        "embedding_parameters": 3510528,
        "ordinary_attention_and_feed_forward": True,
        "sequence_caps": {"source": 96, "target": 96, "decode": 96, "max_positions": 256},
        "training_protocol": (
            "fresh_original_optimizer_schedule_with_two_pass_consistency_objective"
        ),
    }
    loss = context.get("loss_configuration", {})
    required_loss = {
        "name": "all_pairs_power_dropout_consistency",
        "power": 0.5,
        "weight": 1.0,
        "probability_floor_mass": 1e-6,
        "dropout_views": 2,
        "both_views_receive_gradients": True,
        "target_ignore_index": -100,
        "supervised_loss": "mean of two existing label-smoothed cross-entropies",
        "consistency_reduction": (
            "sum over centered vocabulary defects, mean over valid target positions"
        ),
    }
    if model == ORDINARY:
        required_context.update(
            {
                "power_scope": "not_used",
                "fixed_power": None,
                "consistency_coefficient": 0.0,
                "probability_uniform_floor_mass": None,
                "gradient_bearing_dropout_passes_per_update": 1,
                "training_protocol": (
                    "fresh_original_optimizer_schedule_with_single_pass_cross_entropy_objective"
                ),
                "training_objective": "ordinary_cross_entropy",
            }
        )
        required_loss = {
            "name": "ordinary_cross_entropy",
            "label_smoothing": 0.1,
            "target_ignore_index": -100,
        }
    if any(context.get(key) != value for key, value in required_context.items()) or any(
        loss.get(key) != value for key, value in required_loss.items()
    ):
        raise ValueError(f"{model} manifest differs from its declared training objective.")
    hypotheses = (
        (result_path.parent / "hypotheses.test.txt").read_text(encoding="utf-8").splitlines()
    )
    references = (
        (result_path.parent / "references.test.txt").read_text(encoding="utf-8").splitlines()
    )
    if references != expected or len(hypotheses) != len(expected) or not expected:
        raise ValueError("Saved predictions do not match the evaluation examples.")
    point = corpus.metric.score(hypotheses, references)
    if not math.isclose(point, record["scores"]["test"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Saved predictions and recorded score disagree.")
    if (
        cached
        and cached.get("source_record") == record
        and "interval" in cached
        and math.isclose(cached["interval"]["score"], point, rel_tol=0, abs_tol=1e-8)
    ):
        interval = cached["interval"]
    else:
        value = bootstrap_score(hypotheses, references, corpus.metric, resamples=1000, seed=12345)
        interval = {**asdict(value), "half_width": value.half_width}
    return {
        "model": model,
        "status": "completed",
        "registry_model": "embedding_linear",
        "parameters": record["parameters"],
        "d_model": 448,
        "d_ff": 856,
        "training_steps": 50000,
        "scored_step": record["scored_step"],
        "interval": interval,
        "source_results": str(result_path),
        "source_record": record,
        "source_hypotheses": str(result_path.parent / "hypotheses.test.txt"),
        "source_references": str(result_path.parent / "references.test.txt"),
    }


def load_or_pending(corpus, task_root, model, expected, cached=None):
    result_path = task_root / "runs" / f"iwslt14_{model}_seed42" / "results.json"
    if result_path.is_file() and "test" in read_json(result_path).get("scores", {}):
        return load_candidate(corpus, result_path, expected, cached, model)
    return {
        "model": model,
        "status": "pending",
        "registry_model": "embedding_linear",
        "parameters": 27241504,
        "parameter_count_source": "approved_model_design",
        "d_model": 448,
        "d_ff": 856,
        "training_steps": 50000,
        "scored_step": None,
        "interval": None,
        "source_results": str(result_path),
    }


def compare_compact(corpus, candidate, ordinary, cached=None):
    if any(row["status"] != "completed" for row in (candidate, ordinary)):
        return None
    records = [candidate["source_record"], ordinary["source_record"]]
    sources = [candidate["source_results"], ordinary["source_results"]]
    if (
        cached
        and cached.get("source_records") == records
        and cached.get("source_results") == sources
    ):
        return cached
    if (
        records[0]["manifest"]["model_config"] != records[1]["manifest"]["model_config"]
        or records[0]["manifest"]["train_config"] != records[1]["manifest"]["train_config"]
    ):
        raise ValueError(
            "The ordinary and powered compact models must share shape and base recipe."
        )
    hypotheses = Path(candidate["source_hypotheses"]).read_text(encoding="utf-8").splitlines()
    other = Path(ordinary["source_hypotheses"]).read_text(encoding="utf-8").splitlines()
    references = Path(candidate["source_references"]).read_text(encoding="utf-8").splitlines()
    other_references = Path(ordinary["source_references"]).read_text(encoding="utf-8").splitlines()
    if references != other_references:
        raise ValueError("Paired comparison references differ.")
    value = paired_bootstrap(
        hypotheses, other, references, corpus.metric, resamples=1000, seed=12345
    )
    return {
        "first_model": CANDIDATE,
        "second_model": ORDINARY,
        "direction": "power-trained minus ordinary compact model",
        "difference": value.difference,
        "low": value.low,
        "high": value.high,
        "n": value.n,
        "what_varies": value.what_varies,
        "source_results": sources,
        "source_records": records,
    }


def markdown(report):
    rows = [
        "# IWSLT14: power-analogy dropout consistency",
        "",
        "Test · BLEU/13a",
        "",
        "| Model | Parameters | Result |",
        "|---|---:|---:|",
    ]
    for model in CONTROLS:
        row = report["reported_references"]["models"][model]
        rows.append(
            f"| {LABELS[model]} | {row['reported_parameters']} | "
            f"{row['reported_score']:.2f} ± {row['reported_half_width']:.2f} |"
        )
    for model in NEW_MODELS:
        row = report["models"][model]
        interval = row["interval"]
        result = (
            f"{interval['score']:.2f} ± {interval['half_width']:.2f}"
            if interval is not None
            else "Pending"
        )
        rows.append(f"| {LABELS[model]} | {row['parameters']:,} | {result} |")
    rows += [
        "",
        (
            "The three controls are existing reported references from results/iwslt14.md,"
            " preserved at their reported table precision. Their original predictions are"
            " unavailable for this report. Completed compact rows are scored from saved test"
            " predictions; their ± is half the width of a 95% example-bootstrap interval (1,000"
            " resamples), with exact endpoints retained in JSON."
        ),
        "",
    ]
    differences = report["rounded_point_comparisons"]
    if differences:
        rows += [
            "Candidate minus rounded reported BLEU: "
            + "; ".join(
                f"{LABELS[model]} {differences[model]['difference']:+.2f}" for model in CONTROLS
            )
            + ". These are rounded-point comparisons, not paired tests.",
            "",
        ]
    paired = report["power_minus_ordinary"]
    if paired is not None:
        rows += [
            (
                f"Power-trained minus ordinary compact model: {paired['difference']:+.2f} BLEU; "
                f"95% paired-bootstrap interval [{paired['low']:+.2f}, {paired['high']:+.2f}]."
            ),
            "",
        ]
    else:
        rows += [
            (
                "The power-versus-ordinary paired comparison is pending both completed prediction"
                " files."
            ),
            "",
        ]
    rows += [
        "Both compact models use width 448, FFN width 856, tied 336→448 linear embeddings and"
        " independent full Q/K/V. The original 50,000-update IWSLT14 optimizer schedule is"
        " retained. The ordinary row uses ordinary translation training. The power row averages"
        " two dropout-pass translation losses and adds the all-pairs power-consistency penalty"
        " with fixed p = 0.5 and coefficient 1. The power affects training only, with no"
        " additional inference operations. Two gradient-bearing passes increase training work;"
        " this is not an equal-compute comparison."
    ]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="iwslt14", choices=("iwslt14",))
    parser.add_argument("--task-root", type=Path, default=NAS)
    args = parser.parse_args()
    task_root = args.task_root
    snapshot = read_json(REFERENCE_SNAPSHOT)
    if (
        snapshot["corpus"] != "iwslt14"
        or snapshot["split"] != "test"
        or snapshot["metric"] != "BLEU/13a"
        or set(snapshot["models"]) != set(CONTROLS)
    ):
        raise ValueError("Reference snapshot belongs to another comparison.")
    os.environ["ANA_DATA"] = str(task_root / "data")
    corpus = build_corpus("iwslt14")
    expected = [row.target for row in corpus.load_split("test")]
    reports = SERVER / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "iwslt14.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_path = reports / "iwslt14.json"
        previous = read_json(previous_path) if previous_path.is_file() else {}
        cache_valid = (
            previous.get("confidence") == 0.95
            and previous.get("resamples") == 1000
            and previous.get("bootstrap_seed") == 12345
        )
        candidate = load_or_pending(
            corpus,
            task_root,
            CANDIDATE,
            expected,
            previous.get("candidate") if cache_valid else None,
        )
        ordinary = load_or_pending(
            corpus,
            task_root,
            ORDINARY,
            expected,
            previous.get("ordinary") if cache_valid else None,
        )
        comparison = compare_compact(
            corpus,
            candidate,
            ordinary,
            previous.get("power_minus_ordinary") if cache_valid else None,
        )
        point = candidate["interval"]["score"] if candidate["interval"] is not None else None
        models = {ORDINARY: ordinary, CANDIDATE: candidate}
        completed = sum(row["status"] == "completed" for row in models.values())
        report = {
            "study_id": "power_consistency_iwslt_v1",
            "corpus": "iwslt14",
            "split": "test",
            "metric": corpus.metric.name,
            "confidence": 0.95,
            "resamples": 1000,
            "bootstrap_seed": 12345,
            "candidate": candidate,
            "ordinary": ordinary,
            "models": models,
            "power_minus_ordinary": comparison,
            "reported_references": snapshot,
            "reference_snapshot": str(REFERENCE_SNAPSHOT),
            "rounded_point_comparisons": {
                model: {
                    "type": "candidate_minus_rounded_reported_score",
                    "candidate_score": point,
                    "reference_reported_score": snapshot["models"][model]["reported_score"],
                    "difference": point - snapshot["models"][model]["reported_score"],
                    "reference_source_file": snapshot["source_file"],
                }
                for model in CONTROLS
                if point is not None
            },
            "reference_comparison_scope": "rounded point scores only; no paired intervals",
            "compact_comparison_scope": (
                "paired bootstrap from genuine predictions when both complete"
            ),
            "completed_new_models": completed,
            "total_new_models": 2,
            "complete": completed == 2,
        }
        atomic_write(
            reports / "iwslt14.json", json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        atomic_write(reports / "iwslt14.md", markdown(report))
    print(
        f"Reported {completed}/2 completed IWSLT14 compact models and three existing references.",
        flush=True,
    )


if __name__ == "__main__":
    main()
