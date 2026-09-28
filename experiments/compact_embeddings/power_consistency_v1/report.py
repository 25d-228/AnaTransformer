"""Report fresh Multi30k power-analogy dropout consistency and saved references."""

import argparse
import fcntl
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path("/home/Yue_Ziran/workspace/ana-power-consistency-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-power-consistency-v1")
SOURCE = NAS.parent / "ana-compact-power-v1"
LINEAR_SOURCE = NAS.parent / "ana-embedding-analogy-v1"
CONTROLS = ("baseline_matched", "shared_qkv")
LINEAR = "embedding_linear"
CANDIDATE = "embedding_power_consistency"
REUSED_MODELS = (*CONTROLS, LINEAR)
NEW_MODELS = (CANDIDATE,)
MODELS = (*REUSED_MODELS, *NEW_MODELS)
LABELS = {
    "baseline_matched": "Similar-size Transformer",
    "shared_qkv": "Shared-QKV",
    LINEAR: "Transformer + linear compressed embeddings",
    CANDIDATE: "Linear embeddings + power-analogy dropout consistency (p = 0.5)",
}
PARAMETERS = {LINEAR: 2248512, CANDIDATE: 2248512}
SHAPES = {LINEAR: (128, 232), CANDIDATE: (128, 232)}
REGISTRY = {LINEAR: "embedding_linear", CANDIDATE: "embedding_linear"}
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
    root = task_root if model in NEW_MODELS else LINEAR_SOURCE if model == LINEAR else SOURCE
    return root / "runs" / f"multi30k_{model}_seed42"


def predictions(root):
    return tuple(
        (root / f"{kind}.test.txt").read_text(encoding="utf-8").splitlines()
        for kind in ("hypotheses", "references")
    )


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
        or not 0 < record["scored_step"] <= 20000
        or record.get("extra_steps", 0)
    ):
        raise ValueError("Expected the fresh 20,000-update Multi30k recipe, not a continuation.")
    if model in PARAMETERS and record["parameters"] != PARAMETERS[model]:
        raise ValueError("Compressed-embedding parameter count differs from the approved design.")
    shape = record["manifest"]["model_config"]
    if model in SHAPES and (shape["d_model"], shape["d_ff"]) != SHAPES[model]:
        raise ValueError("Compressed-embedding model shape differs from the approved design.")
    if model == CANDIDATE:
        context = record["manifest"]["context"]
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
            "inference_unchanged": True,
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
        if any(context.get(key) != value for key, value in required.items()) or any(
            loss.get(key) != value for key, value in required_loss.items()
        ):
            raise ValueError(
                "Expected the approved fresh, fixed-power two-pass training objective."
            )
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
        value = bootstrap_score(hypotheses, references, corpus.metric)
        interval = {**asdict(value), "half_width": value.half_width}
    full = record.get("standard_transformer_parameters", record.get("baseline_parameters"))
    if full != 2605568:
        raise ValueError("Savings must use the original full Transformer, not the narrower shape.")
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
    result = paired_bootstrap(hypotheses, other_hypotheses, references, corpus.metric)
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


def comparison_rows(title, values):
    if not values:
        return []
    rows = [
        "",
        title,
        "",
        "| Comparison model | Difference | 95% paired interval |",
        "|---|---:|---:|",
    ]
    for model, result in values.items():
        rows.append(
            f"| {LABELS[model]} | {result['difference']:+.2f} | [{result['low']:+.2f},"
            f" {result['high']:+.2f}] |"
        )
    return rows


def markdown(report):
    rows = [
        "# Multi30k: power-analogy dropout consistency",
        "",
        "Test · BLEU",
        "",
        (
            "One fresh candidate keeps the completed linear-compression architecture: "
            "d_model = 128, d_ff = 232, four heads, four encoder/four decoder layers, "
            "and independent full Q/K/V at all 12 attention sites. Each token stores "
            "96 coordinates, mapped through a learned 96×128 matrix. Input embeddings "
            "and output prediction weights remain tied. There are no additional "
            "inference parameters or inference operations relative to the linear control."
        ),
        "",
        (
            "Training uses two independent dropout passes on each example. Average "
            "their existing label-smoothed translation losses, then add a fixed "
            "coefficient of 1 times the power-analogy consistency penalty. At each "
            "valid target position, let P and Q be the two next-word distributions. "
            "Use P_tilde = (1-epsilon)P + epsilon/V and the same rule for Q, with "
            "epsilon = 0.000001 and vocabulary size V. This uniform mixture keeps "
            "all four comparison terms positive; it is separate from label smoothing."
        ),
        "",
        (
            "For every pair of vocabulary entries i and j, the four quantities are "
            "A = P_tilde_i, B = P_tilde_j, C = Q_tilde_i, D = Q_tilde_j. "
            "Their desired numerical relation is A^p + D^p = B^p + C^p, using "
            "fixed p = 0.5. Define delta = sqrt(P_tilde) - sqrt(Q_tilde). The "
            "penalty for a target position is S = sum_v (delta_v - mean(delta))^2; "
            "average S over non-padding target positions. This covers all vocabulary "
            "pairs through an O(V) reduction, not an explicit V×V comparison matrix."
        ),
        "",
        (
            "The four-term condition comes from "
            "[Lepage and Couceiro](https://arxiv.org/abs/2407.18770). "
            "[R-Drop](https://arxiv.org/abs/2106.14448) supplies the related idea of "
            "training consistency between independent dropout predictions, using "
            "bidirectional KL rather than this centered power-gap penalty. Those "
            "sources do not establish that this new penalty improves this model. "
            "Power is used only in training, is not learned, and is absent from "
            "the inference computation. The four terms are prediction probabilities, "
            "not four independently identified words forming a semantic analogy."
        ),
        "",
        (
            "The original base settings are retained: 20,000 optimizer updates, "
            "256 distinct examples per batch, peak learning rate 0.005, 2,000 "
            "warmup updates and inverse-square-root decay. Two gradient-bearing "
            "passes roughly double model training work per update; this is not an "
            "unchanged compute budget. Selection uses ordinary single-pass development "
            "translation loss; decoding remains beam 5. No teacher, pretrained "
            "checkpoint, continuation or additional dataset is used."
        ),
        "",
        (
            "The similar-size Transformer (d_model = 116, d_ff = 232) and shared-QKV (d_model ="
            " 128, d_ff = 256) controls are reused from the completed original full-recipe"
            " compact-power runs. The completed linear embedding model at 128/232 is reused as a"
            " third, structural reference, not retrained. All three references use saved"
            " predictions with the same base optimizer/update settings, but without the new"
            " two-pass consistency objective. Parameter savings are relative to the original full"
            " Transformer with d_model = 128, d_ff = 256, and 2,605,568 parameters. The original"
            " matched Transformer and shared-QKV remain the primary performance targets."
        ),
        "",
        "| Model | d_model | d_ff | Parameters | Saved vs full | Result |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = report["models"].get(model)
        if row is None:
            if model in SHAPES:
                width, ffn = SHAPES[model]
                count = PARAMETERS[model]
                saved = 100 * (1 - count / 2605568)
                rows.append(
                    f"| {LABELS[model]} | {width} | {ffn} | {count:,} | {saved:.2f}% | Pending |"
                )
            else:
                rows.append(f"| {LABELS[model]} | — | — | — | — | Pending |")
        else:
            interval = row["interval"]
            rows.append(
                f"| {LABELS[model]} | {row['d_model']} | {row['d_ff']} | {row['parameters']:,} |"
                f" {row['saved_percent']:.2f}% | {interval['score']:.2f} ±"
                f" {interval['half_width']:.2f} |"
            )
    rows += ["", NOTE]
    rows += comparison_rows(
        "## Primary: power-consistency training minus controls", report["comparisons"]
    )
    rows += comparison_rows(
        "## Structural: power-consistency training minus saved linear embeddings",
        report["variant_comparisons"],
    )
    rows += [
        "",
        f"New models completed: {report['completed_new_models']}/{report['total_new_models']}.",
    ]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="multi30k", choices=("multi30k",))
    parser.add_argument("--task-root", type=Path, default=NAS)
    args = parser.parse_args()
    task_root = args.task_root
    os.environ["ANA_DATA"] = str(task_root / "data")
    corpus = build_corpus("multi30k")
    expected = [row.target for row in corpus.load_split("test")]
    reports = task_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "multi30k.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous_path = reports / "multi30k.json"
        previous = read_json(previous_path) if previous_path.is_file() else {}
        old_path = SOURCE / "reports/multi30k.json"
        old_report = read_json(old_path) if old_path.is_file() else {}
        linear_path = LINEAR_SOURCE / "reports/multi30k.json"
        linear_report = read_json(linear_path) if linear_path.is_file() else {}
        old_cache_valid = (
            old_report.get("confidence") == 0.95
            and old_report.get("resamples") == 1000
            and old_report.get("bootstrap_seed") == 12345
        )
        linear_cache_valid = (
            linear_report.get("confidence") == 0.95
            and linear_report.get("resamples") == 1000
            and linear_report.get("bootstrap_seed") == 12345
        )
        models = {}
        for model in MODELS:
            path = folder(task_root, model) / "results.json"
            if path.is_file() and "test" in read_json(path).get("scores", {}):
                cached = previous.get("models", {}).get(model)
                if cached is None and old_cache_valid and model in CONTROLS:
                    cached = old_report.get("models", {}).get(model)
                if cached is None and linear_cache_valid and model == LINEAR:
                    cached = linear_report.get("models", {}).get(model)
                models[model] = load_row(corpus, task_root, model, expected, cached)
        configs = [row["source_record"]["manifest"]["train_config"] for row in models.values()]
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError("All reported models must share base optimizer and update settings.")
        completed = sum(model in models for model in NEW_MODELS)
        report = {
            "study_id": "power_consistency_v1",
            "corpus": "multi30k",
            "split": "test",
            "metric": corpus.metric.name,
            "confidence": 0.95,
            "resamples": 1000,
            "bootstrap_seed": 12345,
            "training_steps": 20000,
            "d_model": 128,
            "d_ff_by_model": {key: value[1] for key, value in SHAPES.items()},
            "n_heads": 4,
            "encoder_layers": 4,
            "decoder_layers": 4,
            "attention_sites": 12,
            "full_transformer_parameters": 2605568,
            "full_transformer_shape": {"d_model": 128, "d_ff": 256},
            "independent_full_projections": ["query", "key", "value"],
            "ffn_type": "ordinary GELU; full up/down weights",
            "tied_input_and_output_embedding": True,
            "stored_coordinates_per_token": 96,
            "expanded_coordinates_per_token": 128,
            "linear_basis_shape": [96, 128],
            "effective_embedding": "codes @ basis",
            "embedding_parameters": 972288,
            "analogy_domain": (
                "same_target_position_next_word_probabilities_under_two_dropout_views"
            ),
            "numerical_analogy_source": "https://arxiv.org/abs/2407.18770",
            "dropout_consistency_source": "https://arxiv.org/abs/2106.14448",
            "rdrop_uses_different_penalty": "bidirectional KL",
            "new_penalty_previously_validated": False,
            "positive_probability_rule": "P_tilde = (1-epsilon) * softmax(logits) + epsilon/V",
            "probability_uniform_floor_mass": 1e-6,
            "positive_terms": {
                "A": "P_tilde_i",
                "B": "P_tilde_j",
                "C": "Q_tilde_i",
                "D": "Q_tilde_j",
            },
            "desired_power_identity": "A^p + D^p = B^p + C^p",
            "power_scope": "training_loss_only",
            "fixed_power": 0.5,
            "learned_power": False,
            "trainable_power_count": 0,
            "gradient_bearing_dropout_passes_per_update": 2,
            "delta": "sqrt(P_tilde) - sqrt(Q_tilde)",
            "per_position_consistency": "sum_v((delta_v - mean(delta))^2)",
            "all_pairs_identity": (
                "sum_ij((delta_i-delta_j)^2)/(2*V) = sum_i((delta_i-mean(delta))^2)"
            ),
            "pair_coverage": "all vocabulary pairs exactly, O(V) per valid target position",
            "consistency_coefficient": 1.0,
            "objective": (
                "0.5 * (existing_CE_view1 + existing_CE_view2) + mean_valid_positions(consistency)"
            ),
            "target_ignore_index": -100,
            "semantic_analogy_claim": False,
            "inference_unchanged": True,
            "additional_inference_parameters": 0,
            "training_compute_budget_unchanged": False,
            "training_work_relative_to_single_pass": (
                "roughly doubled model work per update, not a measured wall-time ratio"
            ),
            "checkpoint_selection": "ordinary single-pass development translation loss",
            "parent_checkpoint_used": False,
            "stored_code_scale": "d_model**-0.5",
            "row_normalization": False,
            "linear_basis_initialization": "orthogonal rows, gain sqrt(d_model/code_dim)",
            "initialization_uses_corpus_examples": False,
            "biases_preserved": True,
            "registry_models": REGISTRY,
            "models": models,
            "completed_new_models": completed,
            "total_new_models": len(NEW_MODELS),
            "complete": all(model in models for model in MODELS),
            "comparisons": {},
            "variant_comparisons": {},
        }
        for key, first, others in (
            ("comparisons", CANDIDATE, CONTROLS),
            ("variant_comparisons", CANDIDATE, (LINEAR,)),
        ):
            if first in models:
                for model in others:
                    if model in models:
                        report[key][model] = compare(
                            corpus, models[first], models[model], previous.get(key, {}).get(model)
                        )
        for directory in (reports, SERVER / "reports"):
            atomic_write(
                directory / "multi30k.json", json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            atomic_write(directory / "multi30k.md", markdown(report))
    print(
        f"Reported Multi30k power-analogy dropout consistency; {completed}/{len(NEW_MODELS)} new"
        " models completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
