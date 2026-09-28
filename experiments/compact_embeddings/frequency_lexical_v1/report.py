"""Report Multi30k frequency-aware embeddings with power consistency and saved references."""

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path("/home/Yue_Ziran/workspace/ana-frequency-lexical-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-frequency-lexical-v1")
SOURCE = NAS.parent / "ana-compact-power-v1"
LINEAR_SOURCE = NAS.parent / "ana-embedding-analogy-v1"
CONSISTENCY_SOURCE = NAS.parent / "ana-power-consistency-v1"
CONTROLS = ("baseline_matched", "shared_qkv")
LINEAR = "embedding_linear"
CONSISTENCY = "embedding_power_consistency"
CANDIDATE = "embedding_frequency_power_consistency"
REUSED_MODELS = (*CONTROLS, LINEAR, CONSISTENCY)
MODELS = (*REUSED_MODELS, CANDIDATE)
LABELS = {
    "baseline_matched": "Similar-size Transformer",
    "shared_qkv": "Shared-QKV",
    LINEAR: "Transformer + linear compressed embeddings",
    CONSISTENCY: "Linear embeddings + power consistency, p = 0.5",
    CANDIDATE: "Frequency-aware embeddings + power consistency, p = 0.5",
}
PARAMETERS = {LINEAR: 2248512, CONSISTENCY: 2248512, CANDIDATE: 2248512}
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
    candidate = model == CANDIDATE
    context = record["manifest"]["context"]
    required = {
        "study_id": "frequency_lexical_v1" if candidate else "power_consistency_v1",
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
    }
    if candidate:
        required.update(
            {
                "compression": "frequency_adaptive_tied_embedding",
                "embedding_parameters": 972288,
            }
        )
        allocation = context.get("frequency_allocation", {})
        expected_allocation = {
            "selection_split": "train",
            "selection_sides": "source_and_target",
            "selection_order": "descending_combined_count_then_token_id",
            "full_token_count": 3376,
            "compressed_token_count": 6624,
            "full_dimension": 128,
            "compressed_dimension": 80,
            "compressed_basis_shape": [80, 128],
            "reserved_special_ids": [0, 1, 2, 3],
        }
        full_ids = allocation.get("full_token_ids", [])
        if (
            any(allocation.get(key) != value for key, value in expected_allocation.items())
            or not isinstance(full_ids, list)
            or len(full_ids) != 3376
            or any(type(token_id) is not int or not 0 <= token_id < 10000 for token_id in full_ids)
            or len(set(full_ids)) != 3376
            or not {0, 1, 2, 3}.issubset(full_ids)
        ):
            raise ValueError("Frequency allocation differs from the fixed lexical budget.")
    else:
        required["inference_unchanged"] = True
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
    if (
        any(context.get(key) != value for key, value in required.items())
        or context.get("loss_configuration") != required_loss
    ):
        raise ValueError("Expected the original full-vocabulary two-gradient p=0.5 objective.")


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
        "# Multi30k: frequency-aware lexical compression and power consistency",
        "",
        "Test · BLEU",
        "",
        (
            "The candidate changes how the lexical table uses its parameter budget. "
            "It stores full 128-dimensional vectors for 3,376 vocabulary entries, "
            "and 80-dimensional codes plus a shared 80×128 basis for the other "
            "6,624. Special tokens receive full vectors; the remaining full-vector "
            "entries are chosen by combined source/target training frequency, "
            "with token-ID tie breaking. Development and test frequencies are "
            "not used. Token IDs, tied lookup/output weights and the global "
            "vocabulary softmax are retained."
        ),
        "",
        (
            "This costs 972,288 lexical parameters, exactly the same as the saved "
            "uniform 10,000×96 table and 96×128 basis. Both models have "
            "2,248,512 total parameters: width 128, FFN width 232, four heads, "
            "four encoder/four decoder layers and independent full Q/K/V. "
            "The allocation follows the frequency-dependent capacity idea in "
            "[Baevski and Auli](https://arxiv.org/abs/1809.10853)."
        ),
        "",
        (
            "Training retains the original successful two-gradient power-consistency "
            "objective. Two independent dropout passes produce distributions P and Q. "
            "A small uniform mixture makes them positive: "
            "P_tilde = (1-epsilon)P + epsilon/V, and likewise for Q, "
            "with epsilon = 0.000001. For vocabulary entries i,j, the four "
            "terms are A = P_tilde_i, B = P_tilde_j, C = Q_tilde_i, D = Q_tilde_j. "
            "With fixed p = 0.5, the penalty encourages "
            "A^p + D^p = B^p + C^p from "
            "[Lepage and Couceiro](https://arxiv.org/abs/2407.18770)."
        ),
        "",
        (
            "Subtract the distributions' square roots, center that difference "
            "across vocabulary entries, and sum its squares. This exactly "
            "covers every vocabulary pair in O(V), averaged over valid target "
            "positions. Add the penalty with coefficient 1 to the average of "
            "the two ordinary translation losses; both passes receive gradients. "
            "There is no target/rest term. Power remains training-only and adds "
            "no inference operation. [R-Drop](https://arxiv.org/abs/2106.14448) "
            "provides related two-view consistency with a different, KL-based penalty."
        ),
        "",
        (
            "The four references are reused completed runs. All rows use the "
            "original 20,000-update Multi30k base recipe: batch 256, peak "
            "learning rate 0.005, 2,000 warmup updates and inverse-square-root "
            "decay. Ordinary single-pass development loss selects the checkpoint; "
            "test decoding uses beam 5. Savings use the full Transformer's "
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
        "study_id": "frequency_lexical_v1",
        "corpus": "multi30k",
        "split": "test",
        "metric": corpus.metric.name,
        "confidence": 0.95,
        "resamples": 1000,
        "bootstrap_seed": 12345,
        "training_steps": 20000,
        "full_transformer_parameters": 2605568,
        "base_model": models[CANDIDATE]["source_record"]["manifest"]["context"]["base_model"],
        "d_model": 128,
        "d_ff": 232,
        "n_heads": 4,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "independent_full_projections": ["query", "key", "value"],
        "tied_input_and_output_embedding": True,
        "embedding_parameters": 972288,
        "full_embedding_rows": 3376,
        "full_embedding_width": 128,
        "compressed_embedding_rows": 6624,
        "compressed_code_width": 80,
        "compressed_basis_shape": [80, 128],
        "allocation": (
            "special tokens first, then descending combined source/target training token frequency"
            " with token-ID tie breaking"
        ),
        "tokenizer_and_output_order_unchanged": True,
        "global_vocabulary_softmax_retained": True,
        "power_scope": "training_loss_only",
        "fixed_power": 0.5,
        "learned_power": False,
        "trainable_power_count": 0,
        "consistency_coefficient": 1.0,
        "probability_uniform_floor_mass": 1e-6,
        "gradient_bearing_dropout_passes_per_update": 2,
        "objective": "mean of two existing label-smoothed cross-entropies + S_full",
        "both_views_receive_gradients": True,
        "full_vocabulary_consistency": (
            "sum_v((delta_v-mean(delta))^2), delta=sqrt(P_tilde)-sqrt(Q_tilde), averaged over"
            " valid target positions"
        ),
        "full_vocabulary_smoothing": "P_tilde=(1-epsilon)*P+epsilon/V, likewise for Q",
        "positive_terms": {
            "A": "P_tilde_i",
            "B": "P_tilde_j",
            "C": "Q_tilde_i",
            "D": "Q_tilde_j",
        },
        "desired_power_identity": "A^p + D^p = B^p + C^p",
        "pair_coverage": "all vocabulary pairs exactly, O(V) per valid target position",
        "frequency_embedding_source": "https://arxiv.org/abs/1809.10853",
        "numerical_analogy_source": "https://arxiv.org/abs/2407.18770",
        "dropout_consistency_source": "https://arxiv.org/abs/2106.14448",
        "semantic_analogy_claim": False,
        "additional_power_inference_parameters": 0,
        "additional_power_inference_operations": False,
        "parent_checkpoint_used": False,
        "checkpoint_selection": "ordinary single-pass development translation loss",
        "loss_configuration": models[CANDIDATE]["source_record"]["manifest"]["context"][
            "loss_configuration"
        ],
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
        "Reported completed Multi30k frequency-aware embeddings with power consistency and four"
        " saved references.",
        flush=True,
    )


if __name__ == "__main__":
    main()
