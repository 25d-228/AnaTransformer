"""Report completed Multi30k causal translation analogy and four saved references."""

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path("/home/Yue_Ziran/workspace/ana-causal-translation-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-causal-translation-v1")
SOURCE = NAS.parent / "ana-compact-power-v1"
LINEAR_SOURCE = NAS.parent / "ana-embedding-analogy-v1"
CONSISTENCY_SOURCE = NAS.parent / "ana-power-consistency-v1"
CONTROLS = ("baseline_matched", "shared_qkv")
LINEAR = "embedding_linear"
CONSISTENCY = "embedding_power_consistency"
CANDIDATE = "causal_translation_p05"
REUSED_MODELS = (*CONTROLS, LINEAR, CONSISTENCY)
MODELS = (*REUSED_MODELS, CANDIDATE)
LABELS = {
    "baseline_matched": "Similar-size Transformer",
    "shared_qkv": "Shared-QKV",
    LINEAR: "Transformer + linear compressed embeddings",
    CONSISTENCY: "Linear embeddings + power consistency, p = 0.5",
    CANDIDATE: "Linear embeddings + causal translation analogy, p = 0.5",
}
PARAMETERS = {LINEAR: 2248512, CONSISTENCY: 2248512, CANDIDATE: 2249545}
SHAPES = {model: (128, 232) for model in PARAMETERS}
ANALOGY_CONFIGURATION = {
    "power": 0.5,
    "learned_power": False,
    "feature_dim": 4,
    "positive_epsilon": 1e-6,
    "initial_gain": 0.1,
    "trainable_gain_count": 1,
    "maps": 2,
    "map_shape": [128, 4],
    "source_context": "last_decoder_cross_attention_output_before_residual_dropout",
    "reference_scope": "all_earlier_valid_target_steps_in_same_sentence",
    "condition": "A^p+D^p=B^p+C^p",
}
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
    elif model == CONSISTENCY:
        root = CONSISTENCY_SOURCE
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


def verify_method(record, model):
    if model not in (CANDIDATE, CONSISTENCY):
        return
    context = record["manifest"]["context"]
    if model == CANDIDATE:
        required = {
            "study_id": "causal_translation_v1",
            "base_model": CANDIDATE,
            "parent_checkpoint_used": False,
            "training_protocol": (
                "fresh_full_recipe_single_pass_cross_entropy_no_teacher_no_warm_start"
            ),
            "ordinary_attention_and_feed_forward": True,
            "compression": "tied_96_to_128_linear_embedding",
            "embedding_parameters": 972288,
        }
        analogy = context.get("analogy", {})
        if any(context.get(key) != value for key, value in required.items()) or any(
            analogy.get(key) != value for key, value in ANALOGY_CONFIGURATION.items()
        ):
            raise ValueError("Saved causal translation method differs from this design.")
        summary = record.get("scored_analogy_summary", {})
        gain = summary.get("gain")
        if (
            summary.get("fixed_power") != 0.5
            or isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(gain)
            or gain <= 0
        ):
            raise ValueError(
                "The scored checkpoint must record its fixed power and positive gain."
            )
        return
    required = {
        "study_id": "power_consistency_v1",
        "base_model": "embedding_linear",
        "parent_checkpoint_used": False,
        "power_scope": "training_loss_only",
        "fixed_power": 0.5,
        "learned_power": False,
        "trainable_power_count": 0,
        "consistency_coefficient": 1.0,
        "probability_uniform_floor_mass": 1e-6,
        "gradient_bearing_dropout_passes_per_update": 2,
        "training_protocol": (
            "fresh_original_optimizer_schedule_with_two_pass_consistency_objective"
        ),
        "inference_unchanged": True,
    }
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
    loss = context.get("loss_configuration", {})
    if any(context.get(key) != value for key, value in required.items()) or any(
        loss.get(key) != value for key, value in required_loss.items()
    ):
        raise ValueError("Saved power-consistency reference differs from the completed method.")


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
        raise ValueError("Parameter count differs from the reported design.")
    shape = record["manifest"]["model_config"]
    if model in SHAPES and (shape["d_model"], shape["d_ff"]) != SHAPES[model]:
        raise ValueError("Model shape differs from the reported design.")
    verify_method(record, model)
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
        "# Multi30k: causal translation-example analogy",
        "",
        "Test · BLEU",
        "",
        (
            "The candidate keeps independent full Q/K/V, ordinary feed-forward "
            "layers and tied linear compressed embeddings. Its width is 128, "
            "FFN width 232, with four heads and four encoder/four decoder layers. "
            "A small vocabulary-scoring branch adds 1,033 parameters, for "
            "2,249,545 total: below the similar-size Transformer."
        ),
        "",
        (
            "The branch uses earlier translation steps in the same sentence. "
            "A and C describe earlier and current source contexts; B and D "
            "describe the earlier translated token and a candidate next token. "
            "Two learned maps produce four positive features for each role. "
            "With fixed p = 0.5, a candidate receives a higher score when "
            "A^p + D^p is close to B^p + C^p, the four-term condition from "
            "[Lepage and Couceiro](https://arxiv.org/abs/2407.18770)."
        ),
        "",
        (
            "All earlier valid steps contribute, weighted by source-context "
            "similarity. Their average powered relation gives the exact weighted "
            "squared-mismatch score without comparing every vocabulary item "
            "separately with every reference. No future target tokens are read. "
            "The first position has no earlier example and uses ordinary logits. "
            "Power affects training and inference; training uses one forward "
            "pass and the ordinary translation cross-entropy."
        ),
        "",
        (
            "The references are reused completed runs. The power-consistency "
            "reference is a different method: two dropout passes and a "
            "training-only p = 0.5 penalty. All rows use the original "
            "20,000-update Multi30k base recipe: batch 256, peak learning rate "
            "0.005, 2,000 warmup updates and inverse-square-root decay. "
            "Lowest development loss selects the checkpoint; test decoding "
            "uses beam 5. Parameter savings use the full Transformer's "
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
    # No partial or placeholder table while training or test prediction is pending.
    for model in MODELS:
        path = folder(task_root, model) / "results.json"
        if not path.is_file() or "test" not in read_json(path).get("scores", {}):
            raise SystemExit(f"No report written: completed test result missing for {model}.")
    os.environ["ANA_DATA"] = str(task_root / "data")
    corpus = build_corpus("multi30k")
    expected = [row.target for row in corpus.load_split("test")]
    reports = SERVER / "reports"
    previous_path = reports / "multi30k.json"
    previous = read_json(previous_path) if previous_path.is_file() else {}
    caches = [previous]
    for source in (CONSISTENCY_SOURCE, LINEAR_SOURCE, SOURCE):
        path = source / "reports/multi30k.json"
        if path.is_file():
            caches.append(read_json(path))
    caches = [cache for cache in caches if valid_cache(cache)]
    models = {}
    for model in MODELS:
        cached = next(
            (cache["models"][model] for cache in caches if model in cache.get("models", {})), None
        )
        models[model] = load_row(corpus, task_root, model, expected, cached)
    configs = [row["source_record"]["manifest"]["train_config"] for row in models.values()]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("All reported models must share base optimizer and update settings.")
    report = {
        "study_id": "causal_translation_v1",
        "corpus": "multi30k",
        "split": "test",
        "metric": corpus.metric.name,
        "confidence": 0.95,
        "resamples": 1000,
        "bootstrap_seed": 12345,
        "training_steps": 20000,
        "full_transformer_parameters": 2605568,
        "base_model": CANDIDATE,
        "d_model": 128,
        "d_ff": 232,
        "n_heads": 4,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "independent_full_projections": ["query", "key", "value"],
        "tied_input_and_output_embedding": True,
        "linear_basis_shape": [96, 128],
        "power_scope": "causal_vocabulary_scoring_at_training_and_inference",
        "fixed_power": 0.5,
        "learned_power": False,
        "trainable_power_count": 0,
        "gradient_bearing_dropout_passes_per_update": 1,
        "objective": "existing single-pass label-smoothed translation cross-entropy",
        "analogy": ANALOGY_CONFIGURATION,
        "positive_terms": {
            "A": "earlier source-context features",
            "B": "known earlier target-token features",
            "C": "current source-context features",
            "D": "candidate next target-token features",
        },
        "desired_power_identity": "A^p + D^p = B^p + C^p",
        "feature_transform": "T(u)=(u^p-1)/p",
        "positive_encoding": "softplus(linear(parameter_free_layer_norm(input)))/log(2)+1e-6",
        "reference_weighting": (
            "softmax over all earlier valid source-context squared distances in powered feature"
            " space"
        ),
        "vocabulary_correction": (
            "gain/4 * (2*(T(C)+weighted_mean(T(B)-T(A))) dot T(D) - squared_norm(T(D)))"
        ),
        "reference_coverage": (
            "all earlier valid target steps within the same sentence, strict s<t"
        ),
        "numerical_analogy_source": "https://arxiv.org/abs/2407.18770",
        "semantic_analogy_claim": False,
        "inference_unchanged": False,
        "additional_inference_parameters": 1033,
        "parent_checkpoint_used": False,
        "checkpoint_selection": "ordinary single-pass development translation loss",
        "scored_analogy_summary": models[CANDIDATE]["source_record"]["scored_analogy_summary"],
        "models": models,
        "completed_new_models": 1,
        "total_new_models": 1,
        "complete": True,
        "comparisons": {},
    }
    for model in REUSED_MODELS:
        cached = previous.get("comparisons", {}).get(model) if valid_cache(previous) else None
        report["comparisons"][model] = compare(corpus, models[CANDIDATE], models[model], cached)
    atomic_write(reports / "multi30k.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write(reports / "multi30k.md", markdown(report))
    print(
        "Reported completed Multi30k causal translation analogy and four saved references.",
        flush=True,
    )


if __name__ == "__main__":
    main()
