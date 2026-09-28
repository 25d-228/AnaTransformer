"""Report completed Multi30k single-gradient power consistency and saved controls."""

import argparse
import fcntl
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path("/home/Yue_Ziran/workspace/ana-power-consistency-singlegrad-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-power-consistency-singlegrad-v1")
SOURCE = NAS.parent / "ana-compact-power-v1"
LINEAR_SOURCE = NAS.parent / "ana-embedding-analogy-v1"
TWO_GRADIENT_SOURCE = NAS.parent / "ana-power-consistency-v1"
CONTROLS = ("baseline_matched", "shared_qkv")
LINEAR = "embedding_linear"
TWO_GRADIENT = "embedding_power_consistency"
CANDIDATE = "embedding_power_consistency_singlegrad"
REUSED_MODELS = (*CONTROLS, LINEAR, TWO_GRADIENT)
MODELS = (*REUSED_MODELS, CANDIDATE)
LABELS = {
    "baseline_matched": "Similar-size Transformer",
    "shared_qkv": "Shared-QKV",
    LINEAR: "Transformer + linear compressed embeddings",
    TWO_GRADIENT: "Linear embeddings + power consistency, two gradient passes",
    CANDIDATE: "Linear embeddings + power consistency, one gradient pass",
}
PARAMETERS = {LINEAR: 2248512, TWO_GRADIENT: 2248512, CANDIDATE: 2248512}
SHAPES = {model: (128, 232) for model in PARAMETERS}
NOTE = (
    "± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints"
    " are retained in JSON."
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def folder(task_root, model):
    if model == CANDIDATE:
        root = task_root
    elif model == TWO_GRADIENT:
        root = TWO_GRADIENT_SOURCE
    elif model == LINEAR:
        root = LINEAR_SOURCE
    else:
        root = SOURCE
    return root / "runs" / f"multi30k_{model}_seed42"


def predictions(root):
    return tuple(
        (root / f"{kind}.test.txt").read_text(encoding="utf-8").splitlines()
        for kind in ("hypotheses", "references")
    )


def verify_objective(record, model):
    if model not in (CANDIDATE, TWO_GRADIENT):
        return
    single = model == CANDIDATE
    context = record["manifest"]["context"]
    required = {
        "study_id": "power_consistency_singlegrad_v1" if single else "power_consistency_v1",
        "base_model": "embedding_linear",
        "parent_checkpoint_used": False,
        "power_scope": "training_loss_only",
        "fixed_power": 0.5,
        "learned_power": False,
        "trainable_power_count": 0,
        "consistency_coefficient": 2.0 if single else 1.0,
        "probability_uniform_floor_mass": 1e-6,
        "gradient_bearing_dropout_passes_per_update": 1 if single else 2,
        "inference_unchanged": True,
    }
    required_loss = {
        "power": 0.5,
        "weight": 2.0 if single else 1.0,
        "probability_floor_mass": 1e-6,
        "dropout_views": 2,
        "both_views_receive_gradients": not single,
        "target_ignore_index": -100,
        "consistency_reduction": (
            "sum over centered vocabulary defects, mean over valid target positions"
        ),
    }
    if single:
        required["training_protocol"] = (
            "fresh_original_optimizer_schedule_with_single_gradient_consistency_objective"
        )
        required_loss.update(
            {
                "name": "all_pairs_power_dropout_consistency_single_gradient",
                "gradient_bearing_views": 1,
                "reference_training_mode": True,
                "reference_gradient": False,
                "reference_forward_order": "first",
                "supervised_loss": (
                    "existing label-smoothed cross-entropy of the gradient-bearing view"
                ),
            }
        )
    else:
        required["training_protocol"] = (
            "fresh_original_optimizer_schedule_with_two_pass_consistency_objective"
        )
        required_loss["name"] = "all_pairs_power_dropout_consistency"
        required_loss["supervised_loss"] = "mean of two existing label-smoothed cross-entropies"
    loss = context.get("loss_configuration", {})
    if any(context.get(key) != value for key, value in required.items()) or any(
        loss.get(key) != value for key, value in required_loss.items()
    ):
        raise ValueError("Saved power-consistency objective differs from this comparison.")


def load_row(corpus, task_root, model, expected, cached=None):
    root = folder(task_root, model)
    record = read_json(root / "results.json")
    if record["model"] != model or record["corpus"] != "multi30k":
        raise ValueError("Result belongs to another model or dataset.")
    config = record["manifest"]["train_config"]
    if (
        config["max_steps"] != 20000
        or config["batch_size"] != 256
        or config["warmup_steps"] != 2000
        or not math.isclose(config["learning_rate"], 0.005)
        or config["schedule"] != "inverse_sqrt"
        or config["selection"] != "best_dev_loss"
        or record["steps_run"] != 20000
        or not 0 < record["scored_step"] <= 20000
        or record.get("extra_steps", 0)
    ):
        raise ValueError("Expected the completed fresh 20,000-update recipe, not a continuation.")
    if model in PARAMETERS and record["parameters"] != PARAMETERS[model]:
        raise ValueError("Compressed-embedding parameter count differs from the design.")
    shape = record["manifest"]["model_config"]
    if model in SHAPES and (shape["d_model"], shape["d_ff"]) != SHAPES[model]:
        raise ValueError("Compressed-embedding model shape differs from the design.")
    verify_objective(record, model)
    hypotheses, references = predictions(root)
    if references != expected or len(hypotheses) != len(expected) or not expected:
        raise ValueError("Saved predictions do not match the evaluation examples.")
    point = corpus.metric.score(hypotheses, references)
    if not math.isclose(point, record["scores"]["test"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Saved predictions and recorded score disagree.")
    if (
        cached
        and cached.get("source_record") == record
        and math.isclose(cached["interval"]["score"], point, rel_tol=0, abs_tol=1e-8)
    ):
        interval = cached["interval"]
    else:
        value = bootstrap_score(hypotheses, references, corpus.metric, resamples=1000, seed=12345)
        interval = {**asdict(value), "half_width": value.half_width}
    full = record.get("standard_transformer_parameters", record.get("baseline_parameters"))
    if full != 2605568:
        raise ValueError("Savings must use the original full Transformer.")
    seconds = record["seconds"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Completed training must have a positive measured duration.")
    return {
        "model": model,
        "parameters": record["parameters"],
        "full_parameters": full,
        "d_model": shape["d_model"],
        "d_ff": shape["d_ff"],
        "saved_percent": 100 * (1 - record["parameters"] / full),
        "training_steps": config["max_steps"],
        "scored_step": record["scored_step"],
        "learning_rate": config["learning_rate"],
        "interval": interval,
        "training_seconds": seconds,
        "training_host": record["manifest"].get("training_host"),
        "training_gpu": record["manifest"].get("training_gpu"),
        "source_results": str(root / "results.json"),
        "source_record": record,
        "existing_run_reused": model in REUSED_MODELS,
    }


def compare(corpus, first, other, cached=None):
    sources = [first["source_results"], other["source_results"]]
    records = [first["source_record"], other["source_record"]]
    if (
        cached
        and cached.get("source_records") == records
        and cached.get("source_results") == sources
    ):
        return {
            key: value for key, value in cached.items() if key not in ("p_value", "significant")
        }
    hypotheses, references = predictions(Path(sources[0]).parent)
    other_hypotheses, other_references = predictions(Path(sources[1]).parent)
    if references != other_references or len(other_hypotheses) != len(references):
        raise ValueError("Paired comparison references differ.")
    result = paired_bootstrap(
        hypotheses, other_hypotheses, references, corpus.metric, resamples=1000, seed=12345
    )
    return {
        "difference": result.difference,
        "low": result.low,
        "high": result.high,
        "n": result.n,
        "what_varies": result.what_varies,
        "first_model": first["model"],
        "second_model": other["model"],
        "direction": "first_model minus second_model",
        "source_results": sources,
        "source_records": records,
    }


def markdown(report):
    rows = [
        "# Multi30k: single-gradient power consistency",
        "",
        "Test · BLEU",
        "",
        (
            "The candidate keeps the linear-compression Transformer: d_model = 128, "
            "d_ff = 232, four heads, four encoder/four decoder layers, independent "
            "full Q/K/V, and tied 96-to-128 linear embeddings. It has 2,248,512 "
            "parameters. Power adds no inference parameters or operations."
        ),
        "",
        (
            "Two independent train-mode dropout forwards predict the same target. "
            "P carries gradients; Q is a temporary, no-gradient reference from the "
            "same current network, not a separate or pretrained teacher. Training "
            "uses the existing label-smoothed CE(P) + 2 S(P, stop(Q)). The saved "
            "two-gradient reference instead uses 0.5(CE(P) + CE(Q)) + S(P, Q), "
            "with both views carrying gradients."
        ),
        "",
        (
            "For each valid target position, make probabilities positive through "
            "P_tilde = (1-epsilon)P + epsilon/V, and likewise for Q, with "
            "epsilon = 0.000001. For any two vocabulary entries i and j, the four "
            "terms are A = P_tilde_i, B = P_tilde_j, C = Q_tilde_i, D = Q_tilde_j. "
            "The numerical condition is A^p + D^p = B^p + C^p, with fixed p = 0.5. "
            "Write delta = sqrt(P_tilde) - sqrt(Q_tilde); then S is "
            "sum_v (delta_v - mean(delta))^2, averaged over valid target positions. "
            "This exactly aggregates all vocabulary pairs in O(V), without a V×V "
            "tensor. Power is training-only; the terms are prediction probabilities."
        ),
        "",
        (
            "The four-term condition follows "
            "[Lepage and Couceiro](https://arxiv.org/abs/2407.18770). "
            "[R-Drop](https://arxiv.org/abs/2106.14448) provides related dropout-view "
            "consistency using a different, bidirectional-KL penalty. The factor 2 "
            "in this one-gradient version preserves the two-gradient objective's "
            "expected raw gradient under independent, exchangeable dropout views; "
            "it does not require identical clipped gradients or optimizer paths."
        ),
        "",
        (
            "All rows use the original 20,000-update Multi30k base recipe: batch "
            "256, peak learning rate 0.005, 2,000 warmup updates and inverse-square-root "
            "decay. Ordinary single-pass development loss selects the checkpoint; "
            "test decoding uses beam 5. The four completed references are reused, "
            "not retrained. The matched Transformer and shared-QKV are the primary "
            "performance controls. Savings use the original full Transformer's "
            "2,605,568 parameters."
        ),
        "",
        "| Model | Parameters | Saved vs full | Result |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        row = report["models"][model]
        interval = row["interval"]
        rows.append(
            f"| {LABELS[model]} | {row['parameters']:,} | {row['saved_percent']:.2f}% |"
            f" {interval['score']:.2f} ± {interval['half_width']:.2f} |"
        )
    rows += [
        "",
        NOTE,
        "",
        "## Candidate minus saved references",
        "",
        "| Reference | BLEU difference | 95% paired interval |",
        "|---|---:|---:|",
    ]
    for model in REUSED_MODELS:
        result = report["comparisons"][model]
        rows.append(
            f"| {LABELS[model]} | {result['difference']:+.2f} | [{result['low']:+.2f},"
            f" {result['high']:+.2f}] |"
        )
    timing = report["training_time_comparison"]
    rows += [
        "",
        "## Measured training time",
        "",
        "| Model | Training seconds | Training minutes |",
        "|---|---:|---:|",
    ]
    for model in (TWO_GRADIENT, CANDIDATE):
        seconds = report["models"][model]["training_seconds"]
        rows.append(f"| {LABELS[model]} | {seconds:.2f} | {seconds / 60:.2f} |")
    rows += [
        "",
        (
            "Observed two-gradient/single-gradient duration ratio:"
            f" {timing['two_gradient_over_single_gradient']:.3f}×. Observed time reduction:"
            f" {timing['single_gradient_time_reduction_percent']:.2f}%."
        ),
        "",
        timing["interpretation"],
    ]
    return "\n".join(rows) + "\n"


def valid_cache(report):
    return (
        report.get("confidence") == 0.95
        and report.get("resamples") == 1000
        and report.get("bootstrap_seed") == 12345
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="multi30k", choices=("multi30k",))
    parser.add_argument("--task-root", type=Path, default=NAS)
    args = parser.parse_args()
    task_root = args.task_root
    # Do not create partial result tables when training or prediction is unfinished.
    for model in MODELS:
        path = folder(task_root, model) / "results.json"
        if not path.is_file() or "test" not in read_json(path).get("scores", {}):
            raise SystemExit(f"No report written: completed test result missing for {model}.")
    os.environ["ANA_DATA"] = str(task_root / "data")
    corpus = build_corpus("multi30k")
    expected = [row.target for row in corpus.load_split("test")]
    reports = task_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "multi30k.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_path = reports / "multi30k.json"
        previous = read_json(previous_path) if previous_path.is_file() else {}
        caches = [previous]
        for source in (TWO_GRADIENT_SOURCE, LINEAR_SOURCE, SOURCE):
            path = source / "reports/multi30k.json"
            if path.is_file():
                caches.append(read_json(path))
        caches = [cache for cache in caches if valid_cache(cache)]
        models = {}
        for model in MODELS:
            cached = next(
                (cache["models"][model] for cache in caches if model in cache.get("models", {})),
                None,
            )
            models[model] = load_row(corpus, task_root, model, expected, cached)
        configs = [row["source_record"]["manifest"]["train_config"] for row in models.values()]
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError("All reported models must share base optimizer and update settings.")
        single = models[CANDIDATE]
        two = models[TWO_GRADIENT]
        same_host = bool(
            single["training_host"] and single["training_host"] == two["training_host"]
        )
        same_gpu = bool(single["training_gpu"] and single["training_gpu"] == two["training_gpu"])
        timing = {
            "single_gradient_seconds": single["training_seconds"],
            "two_gradient_seconds": two["training_seconds"],
            "two_gradient_over_single_gradient": (
                two["training_seconds"] / single["training_seconds"]
            ),
            "single_gradient_time_reduction_percent": (
                100 * (1 - single["training_seconds"] / two["training_seconds"])
            ),
            "same_training_host": same_host,
            "same_gpu_model": same_gpu,
            "single_gradient_host": single["training_host"],
            "two_gradient_host": two["training_host"],
            "single_gradient_gpu": single["training_gpu"],
            "two_gradient_gpu": two["training_gpu"],
            "includes_training_and_periodic_development": True,
            "includes_final_decoding": False,
            "interpretation": (
                (
                    "Both runs used the same host and GPU model; their concurrent load may"
                    " differ. "
                    if same_host and same_gpu
                    else "The runs did not record matching host and GPU models. "
                )
                + "These are measured run durations, including periodic development but excluding "
                "final decoding, not a controlled throughput benchmark or guaranteed speedup."
            ),
        }
        report = {
            "study_id": "power_consistency_singlegrad_v1",
            "corpus": "multi30k",
            "split": "test",
            "metric": corpus.metric.name,
            "confidence": 0.95,
            "resamples": 1000,
            "bootstrap_seed": 12345,
            "training_steps": 20000,
            "full_transformer_parameters": 2605568,
            "base_model": "embedding_linear",
            "d_model": 128,
            "d_ff": 232,
            "n_heads": 4,
            "encoder_layers": 4,
            "decoder_layers": 4,
            "independent_full_projections": ["query", "key", "value"],
            "tied_input_and_output_embedding": True,
            "linear_basis_shape": [96, 128],
            "power_scope": "training_loss_only",
            "fixed_power": 0.5,
            "learned_power": False,
            "trainable_power_count": 0,
            "consistency_coefficient": 2.0,
            "probability_uniform_floor_mass": 1e-6,
            "dropout_forwards_per_update": 2,
            "gradient_bearing_dropout_passes_per_update": 1,
            "objective": "existing_CE(P) + 2 * mean_valid_positions(S(P, stop(Q)))",
            "reference_view": (
                "independent train-mode dropout view of the same current weights without gradients"
            ),
            "expected_raw_gradient_matches_two_gradient_objective": (
                "under independent exchangeable dropout views"
            ),
            "identical_clipped_gradient_or_optimizer_trajectory_claim": False,
            "positive_terms": {
                "A": "P_tilde_i",
                "B": "P_tilde_j",
                "C": "Q_tilde_i",
                "D": "Q_tilde_j",
            },
            "desired_power_identity": "A^p + D^p = B^p + C^p",
            "per_position_consistency": (
                "sum_v((delta_v - mean(delta))^2), delta=sqrt(P_tilde)-sqrt(Q_tilde)"
            ),
            "pair_coverage": "all vocabulary pairs exactly, O(V) per valid target position",
            "numerical_analogy_source": "https://arxiv.org/abs/2407.18770",
            "dropout_consistency_source": "https://arxiv.org/abs/2106.14448",
            "semantic_analogy_claim": False,
            "inference_unchanged": True,
            "additional_inference_parameters": 0,
            "parent_checkpoint_used": False,
            "checkpoint_selection": "ordinary single-pass development translation loss",
            "models": models,
            "completed_new_models": 1,
            "total_new_models": 1,
            "complete": True,
            "comparisons": {},
            "training_time_comparison": timing,
        }
        for model in REUSED_MODELS:
            cached = previous.get("comparisons", {}).get(model) if valid_cache(previous) else None
            report["comparisons"][model] = compare(corpus, single, models[model], cached)
        for directory in (reports, SERVER / "reports"):
            atomic_write(
                directory / "multi30k.json", json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            atomic_write(directory / "multi30k.md", markdown(report))
    print(
        "Reported completed Multi30k single-gradient power consistency and four saved references.",
        flush=True,
    )


if __name__ == "__main__":
    main()
