"""Report fresh Multi30k compressed-embedding runs and matching saved controls."""

import argparse
import fcntl
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

from ana.data.corpora import build_corpus
from ana.stats import bootstrap_score, paired_bootstrap

SERVER = Path("/home/Yue_Ziran/workspace/ana-embedding-analogy-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-embedding-analogy-v1")
SOURCE = NAS.parent / "ana-compact-power-v1"
CONTROLS = ("baseline_matched", "shared_qkv")
LINEAR = "embedding_linear"
CANDIDATE = "embedding_learned"
NEW_MODELS = (LINEAR, CANDIDATE)
MODELS = (*CONTROLS, *NEW_MODELS)
LABELS = {
    "baseline_matched": "Similar-size Transformer",
    "shared_qkv": "Shared-QKV",
    LINEAR: "Transformer + linear compressed embeddings",
    CANDIDATE: "Transformer + power-analogy embeddings (learned p)",
}
PARAMETERS = {LINEAR: 2248512, CANDIDATE: 2248528}
SHAPES = {LINEAR: (128, 232), CANDIDATE: (128, 230)}
REGISTRY = {LINEAR: "embedding_linear", CANDIDATE: "ana_embedding_learned"}
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
    root = task_root if model in NEW_MODELS else SOURCE
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
        "existing_run_reused": model in CONTROLS,
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
        "# Multi30k: power-analogy lexical compression",
        "",
        "Test · BLEU",
        "",
        (
            "The two new models train from scratch for 20,000 updates using the original "
            "Multi30k recipe: batch size 256, peak learning rate 0.005, 2,000 warmup updates, "
            "and inverse-square-root decay. The lowest-development-loss checkpoint is "
            "evaluated. Neither model loads a pretrained checkpoint or uses a teacher."
        ),
        "",
        (
            "Both new models use d_model = 128, four attention heads, and four encoder "
            "plus four decoder layers. All 12 attention sites retain independent full "
            "Q, K, and V projections. The parameter saving comes from the tied source, "
            "target, and output embedding table, not from attention or reconstructed "
            "FFN weights. The linear control uses d_ff = 232; the power-analogy model "
            "uses d_ff = 230 to keep their total parameter budgets nearly equal."
        ),
        "",
        (
            "The linear structural control stores 96 coordinates per token and maps "
            "them linearly to 128 coordinates using one learned 96×128 matrix. "
            "The power-analogy model also stores 96 coordinates per token: three "
            "for each of 32 groups. A group generates a positive radius R and "
            "u = 0.95 tanh(alpha), v = 0.95 tanh(beta), then the four positive terms "
            "A = R(1+u)^(1/p), D = R(1-u)^(1/p), B = R(1+v)^(1/p), and "
            "C = R(1-v)^(1/p). Their powered sums satisfy "
            "A^p + D^p = B^p + C^p = 2R^p without a radicand clamp. "
            "This is the four-term condition from the project's "
            "[numerical-analogy paper](https://arxiv.org/abs/2407.18770)."
        ),
        "",
        (
            "Fixed common centering and scaling turn those positive terms into "
            "signed latent features. A learned shared 128×128 basis maps them to "
            "the model's embedding coordinates. Input lookup and output prediction "
            "use the same effective table. The numerical equality holds in the "
            "positive latent quartet, not in the final rotated signed coordinates; "
            "the four terms are not labeled linguistic analogies. There are 32 "
            "powers shared across the vocabulary, initialized at 2 and bounded "
            "between 0.5 and 4. No D4, attention-cell branch, shared QKV, or "
            "zero-start correction gain is added."
        ),
        "",
        (
            "Both compressed models initialize stored codes at the ordinary "
            "embedding scale 1/sqrt(128). The analogy model uses fixed moments "
            "from synthetic samples and a blockwise-whitened initial basis; the "
            "linear control uses a scaled orthogonal row basis. Neither "
            "initialization uses corpus examples or a trained embedding table."
        ),
        "",
        (
            "The similar-size Transformer (d_model = 116, d_ff = 232) and shared-QKV "
            "(d_model = 128, d_ff = 256) controls are reused from the completed original "
            "full-recipe compact-power runs, not from later continuations. Parameter "
            "savings are relative to the original full Transformer with d_model = 128, "
            "d_ff = 256, and 2,605,568 parameters. The new linear embedding control "
            "tests the compression structure; it does not replace either original "
            "performance target."
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
        "## Primary: power-analogy embeddings minus controls", report["comparisons"]
    )
    rows += comparison_rows(
        "## Linear compressed embeddings minus controls", report["linear_comparisons"]
    )
    rows += comparison_rows(
        "## Structural comparison: power-analogy minus linear embeddings",
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
        old_cache_valid = (
            old_report.get("confidence") == 0.95
            and old_report.get("resamples") == 1000
            and old_report.get("bootstrap_seed") == 12345
        )
        models = {}
        for model in MODELS:
            path = folder(task_root, model) / "results.json"
            if path.is_file() and "test" in read_json(path).get("scores", {}):
                cached = previous.get("models", {}).get(model)
                if cached is None and old_cache_valid and model in CONTROLS:
                    cached = old_report.get("models", {}).get(model)
                models[model] = load_row(corpus, task_root, model, expected, cached)
        configs = [row["source_record"]["manifest"]["train_config"] for row in models.values()]
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError("All reported models must use the same Multi30k training recipe.")
        completed = sum(model in models for model in NEW_MODELS)
        report = {
            "study_id": "embedding_analogy_v1",
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
            "analogy_basis_shape": [128, 128],
            "analogy_domain": "positive_latent_lexical_features",
            "numerical_analogy_source": "https://arxiv.org/abs/2407.18770",
            "groups_per_token": 32,
            "stored_coordinates_per_group": 3,
            "positive_terms_per_group": 4,
            "radius": "softplus(raw_radius) + 0.0001",
            "u": "0.95 * tanh(alpha)",
            "v": "0.95 * tanh(beta)",
            "positive_terms": {
                "A": "R * (1+u)^(1/p)",
                "D": "R * (1-u)^(1/p)",
                "B": "R * (1+v)^(1/p)",
                "C": "R * (1-v)^(1/p)",
            },
            "power_identity": "A^p + D^p = B^p + C^p = 2 * R^p",
            "signed_features": "fixed common centering and scaling; then learned shared basis",
            "centering_is_per_token": False,
            "final_signed_embedding_obeys_power_identity": False,
            "learned_gain_count": 0,
            "learned_power_count": 32,
            "learned_power_range": [0.5, 4.0],
            "learned_initial_power": 2.0,
            "powers_shared_across_vocabulary": True,
            "stored_code_scale": "d_model**-0.5",
            "analogy_codes_normalized_before_chart": True,
            "feature_moments": "fixed common moments from private synthetic triplets at p=2",
            "moment_calibration_samples": 65536,
            "basis_initialization": "code scale times blockwise symmetric covariance whitening",
            "whitening_gain_cap": 8.0,
            "linear_basis_initialization": "orthogonal rows, gain sqrt(d_model/code_dim)",
            "initialization_uses_corpus_examples": False,
            "biases_preserved": True,
            "registry_models": REGISTRY,
            "models": models,
            "completed_new_models": completed,
            "total_new_models": len(NEW_MODELS),
            "complete": all(model in models for model in MODELS),
            "comparisons": {},
            "linear_comparisons": {},
            "variant_comparisons": {},
        }
        for key, first, others in (
            ("comparisons", CANDIDATE, CONTROLS),
            ("linear_comparisons", LINEAR, CONTROLS),
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
        f"Reported Multi30k compressed embeddings; {completed}/{len(NEW_MODELS)} new models"
        " completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
