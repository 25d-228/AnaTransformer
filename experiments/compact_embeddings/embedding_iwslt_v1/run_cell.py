"""Conditional IWSLT14 runner for one fresh residual-embedding candidate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
from trainer import _atomic_checkpoint_save, set_seed, train

from ana.config import Schedule, Selection, TrainConfig
from ana.data.corpora import build_corpus
from ana.experiment import encode_split, prepare, score_split
from ana.nn.attention import MultiHeadAttention
from ana.nn.embeddings.analogy_embedding import ResidualPowerAnalogyEmbedding
from ana.nn.layers import FeedForward
from ana.nn.projection import SeparateQKV
from ana.registry import baseline_parameters, build_model, count_parameters

SERVER = Path("/home/Yue_Ziran/workspace/ana-embedding-iwslt-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-embedding-iwslt-v1")
DATA = NAS / "data"
MODELS = {
    "embedding_residual_learned": "ana_embedding_residual_learned",
    "embedding_mix_learned": "ana_embedding_mix_learned",
}
EXPECTED_PARAMETERS = {
    "embedding_residual_learned": 27_241_616,
    "embedding_mix_learned": 27_243_396,
}
FF_WIDTHS = {"embedding_residual_learned": 856, "embedding_mix_learned": 855}
EMBEDDING_PARAMETERS = {
    "embedding_residual_learned": 3_510_640,
    "embedding_mix_learned": 3_523_184,
}
SPLITS = ("dev", "test")


def plain(value):
    return json.loads(json.dumps(value))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def recipe():
    values = json.loads((SERVER / "recipes" / "iwslt14.json").read_text())
    config = {
        field.name: values[field.name] for field in fields(TrainConfig) if field.name in values
    }
    config["schedule"] = Schedule(config["schedule"])
    config["selection"] = Selection(config["selection"])
    config["adam_betas"] = tuple(config["adam_betas"])
    result = TrainConfig(**config)
    if (
        result.max_steps != 50_000
        or result.learning_rate != 0.0005
        or result.warmup_steps != 4_000
        or result.batch_size != 160
        or result.weight_decay != 0.0001
        or result.schedule is not Schedule.INVERSE_SQRT
        or result.selection is not Selection.BEST_DEV_LOSS
    ):
        raise ValueError("Use the original full-budget normal-rate IWSLT14 recipe unchanged.")
    return result


def power_summary(model):
    values = model.embedding.powers().detach().float().cpu().flatten()
    return {
        "count": values.numel(),
        "min": values.min().item(),
        "mean": values.mean().item(),
        "max": values.max().item(),
    }


def construct(full_shape, model_name):
    assert full_shape.d_model == 512 and full_shape.d_ff == 1024
    assert full_shape.vocab_size == 10_000 and full_shape.n_heads == 4
    assert full_shape.n_encoder_layers == 6 and full_shape.n_decoder_layers == 6
    assert full_shape.dropout == 0.3 and full_shape.label_smoothing == 0.1
    assert full_shape.max_positions == 256
    assert baseline_parameters(full_shape) == 36_665_344
    shape = replace(full_shape, d_model=448, d_ff=FF_WIDTHS[model_name])
    model = build_model(MODELS[model_name], shape)
    mixing = model_name == "embedding_mix_learned"
    attentions = [layer.self_attention for layer in model.encoder]
    for layer in model.decoder:
        attentions.extend((layer.self_attention, layer.cross_attention))
    assert len(attentions) == 18
    for attention in attentions:
        assert type(attention) is MultiHeadAttention
        assert type(attention.projection) is SeparateQKV
    feed_forwards = [layer.feed_forward for layer in (*model.encoder, *model.decoder)]
    assert len(feed_forwards) == 12
    for block in feed_forwards:
        assert type(block) is FeedForward
        assert block.up.weight.shape == (shape.d_ff, 448)
        assert block.down.weight.shape == (448, shape.d_ff)
    assert type(model.embedding) is ResidualPowerAnalogyEmbedding
    assert model.embedding.normalize_rows is False
    assert model.embedding.learn_mixing is mixing
    assert model.embedding.codes.shape == (full_shape.vocab_size, 336)
    assert model.embedding.basis.shape == (336, 448)
    assert model.embedding.complement.shape == (112, 448)
    assert model.embedding.complement.requires_grad is False
    if mixing:
        assert model.embedding.feature_mixing.shape == (112, 112)
        assert model.embedding.feature_mixing.requires_grad is True
        assert torch.equal(model.embedding.feature_mixing, torch.eye(112))
    else:
        assert model.embedding.feature_mixing is None
    assert model.embedding.d_model == 448 and model.embedding.code_dim == 336
    powers = model.embedding.powers()
    assert powers.numel() == 112
    assert torch.equal(powers, torch.ones_like(powers))
    assert count_parameters(model) == EXPECTED_PARAMETERS[model_name]
    assert count_parameters(model) < 27_246_512 < baseline_parameters(full_shape)
    assert count_parameters(model.embedding) == EMBEDDING_PARAMETERS[model_name]
    assert model.activation_checkpointing is False
    return model


def capacity():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    cap_mib = int(os.environ.get("ANA_MAX_GPU_MIB", "16384"))
    free = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    ram_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if (
        free < cap_mib + 2048
        or ram_kib < 8 * 1024**2
        or min(shutil.disk_usage(SERVER).free, shutil.disk_usage(NAS).free) < 1024**3
    ):
        raise RuntimeError("Insufficient free resources; existing jobs remain untouched.")
    if not torch.cuda.is_available():
        raise RuntimeError("Working CUDA required; no CPU training fallback.")
    torch.cuda.set_per_process_memory_fraction(
        cap_mib * 1024**2 / torch.cuda.get_device_properties(0).total_memory,
        0,
    )
    return cap_mib


def run(model_name, describe=False):
    if Path.cwd() != SERVER:
        raise RuntimeError(f"Run from {SERVER}")
    os.environ["ANA_DATA"] = str(DATA)
    corpus = build_corpus("iwslt14")
    assert (corpus.max_source_length, corpus.max_target_length, corpus.max_decode_length) == (
        96,
        96,
        96,
    )
    if not Path(corpus.tokenizer_path(False)).is_file():
        raise FileNotFoundError("Reuse the existing tokenizer; do not train a new one.")
    tokenizer, splits = prepare(corpus, False, ("dev",))
    full_shape = corpus.model_config(len(tokenizer))
    config = recipe()
    set_seed(config.seed)
    model = construct(full_shape, model_name)
    mixing = model_name == "embedding_mix_learned"
    context = plain(
        {
            "study_id": "embedding_iwslt_v1",
            "corpus": "iwslt14",
            "model": model_name,
            "contextual_parent_study": (
                "ana-embedding-mix-v1" if mixing else "ana-embedding-residual-v1"
            ),
            "parent_checkpoint_used": False,
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "base_model": MODELS[model_name],
            "compression": (
                "tied_336_coefficient_lexical_table_with_retained_linear_path_and_learned_mixing_of_power_analogy_residuals"
                if mixing
                else "tied_336_coefficient_lexical_table_with_retained_linear_path_"
                "and_power_analogy_residual"
            ),
            "embedding_parameters": EMBEDDING_PARAMETERS[model_name],
            "ordinary_attention_and_feed_forward": True,
            "training_protocol": "fresh_full_recipe_no_teacher_no_warm_start",
            "initial_power": 1.0,
            "learned_power": True,
            "power_range": [0.75, 2.0],
            "power_granularity": "one_per_four_term_feature_group_shared_across_vocabulary",
            "power_count": 112,
            "trainable_power_count": 112,
            "sequence_caps": {"source": 96, "target": 96, "decode": 96, "max_positions": 256},
            "architecture_change": {
                "original_d_model": 512,
                "original_d_ff": 1024,
                "new_d_model": 448,
                "new_d_ff": FF_WIDTHS[model_name],
                "heads": 4,
                "head_width": 112,
                "attention_sites": 18,
                "ffn_sites": 12,
                "encoder_layers": 6,
                "decoder_layers": 6,
            },
            "embedding": {
                "stored_coefficients_per_token": 336,
                "generated_features_per_token": 448,
                "shared_basis_shape": [336, 448],
                "fixed_complement_shape": [112, 448],
                "feature_mixing_shape": [112, 112] if mixing else None,
                "feature_mixing_parameters": 12_544 if mixing else 0,
                "feature_mixing_shared_across_vocabulary": mixing,
                "feature_mixing_bias": False,
                "basis_bias": False,
                "tied_input_output": True,
                "row_normalization": {
                    "enabled": False,
                    "lookup": (
                        "unnormalized_effective_embedding_before_existing_sqrt(d_model)_scaling"
                    ),
                    "vocabulary_prediction": (
                        "same_unnormalized_effective_embedding_rows_as_lookup"
                    ),
                    "added_parameters": 0,
                },
                "input_scaling": "sqrt(d_model)",
                "full_trainable_embedding_table_retained": False,
                "feature_groups": 112,
                "terms_per_group": 4,
                "raw_linear_path_retained": True,
                "effective_embedding": (
                    "codes@basis + sigma*(D_p-D_1)@feature_mixing@fixed_complement"
                    if mixing
                    else "codes@basis + sigma*(D_p-D_1)@fixed_complement"
                ),
                "fixed_complement_orthogonality": (
                    "orthogonal_to_initial_linear_basis;"
                    " neither_recomputed_nor_constrained_after_updates"
                ),
                "zero_start_gate": False,
                "trainable_gain_count": 0,
                "padding": (
                    "near-zero initial effective embedding; suppress code-row gradient in lookup"
                    " but retain tied-output gradient"
                ),
            },
            "initialization": {
                "stored_code_scale": "sigma=d_model^(-1/2)",
                "stored_codes": "independent_N(0,sigma^2)",
                "basis": "orthogonal_rows_with_gain_sqrt(d_model/code_dim)",
                "fixed_complement": (
                    "remaining_112_columns_of_complete_QR_of_initial_basis_transpose_then_transposed;"
                    " saved_buffer_not_recomputed_during_training"
                ),
                "feature_mixing": (
                    "identity_112_by_112_matrix; no_random_draws" if mixing else None
                ),
                "power_parameterization": "1+1.25*(sigmoid(offset-log(4))-sigmoid(-log(4)))",
                "power_offset_initialization": 0.0,
                "residual_initialization": "D_p-D_1=0_at_initial_p=1; p_can_change_without_a_gate",
                "initial_feature_mixing_gradient": (
                    "zero_in_exact_arithmetic_because_h_starts_at_zero;"
                    " becomes_trainable_from_data_after_h_changes"
                    if mixing
                    else None
                ),
                "initial_power_gradient": "power_path_active_at_initial_p=1; no_zero_start_gate",
                "code_and_basis_rng": (
                    "same_initialization_sequence_as_ordinary_linear_compressed_embedding"
                ),
                "pretrained_checkpoint": None,
            },
            "analogy": {
                "source": "https://arxiv.org/abs/2407.18770",
                "condition": "A^p + D^p = B^p + C^p",
                "domain": (
                    "four_positive_latent_lexical_terms_used_to_compute_the_nonlinear_residual"
                ),
                "normalized_raw_codes": "stored_codes/sigma, where sigma=d_model^(-1/2)",
                "A": "softplus(normalized_raw_x)+1e-4",
                "positive_increment": "t=softplus(normalized_raw_y)+1e-4",
                "B": "A+t",
                "C": "softplus(normalized_raw_z)+1e-4",
                "D_p": "(B^p+C^p-A^p)^(1/p)",
                "D_1": "C+t, algebraically_equal_to_B+C-A",
                "positive_epsilon": 1e-4,
                "positive_radicand": "B>A_and_C>0_and_p>0",
                "h": "D_p-D_1",
                "stable_residual_computation": (
                    "exp(log(D_1))*expm1(log(D_p)-log(D_1));"
                    " completion_preserves_positive_increment_separately"
                ),
                "residual_reference_gradient_detached": False,
                "signed_final_embedding_claimed_to_satisfy_condition": False,
                "four_identified_words_or_verified_semantic_analogy": False,
                "power_features_raised_back_to_p_before_use": False,
            },
            "activation_checkpointing": False,
        }
    )
    print(
        json.dumps(
            {
                "phase": "configuration",
                **context,
                "parameters": count_parameters(model),
                "baseline_parameters": baseline_parameters(full_shape),
            }
        ),
        flush=True,
    )
    if describe:
        return
    folder = NAS / "runs" / f"iwslt14_{model_name}_seed{config.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = folder / "results.json"
        if result_path.is_file():
            record = json.loads(result_path.read_text())
            if record["manifest"]["context"] != context:
                raise ValueError("Existing results belong to another configuration.")
            if all(split in record["scores"] for split in SPLITS):
                print("Completed cell; no training repeated.", flush=True)
                return
        cap_mib = capacity()
        device = torch.device("cuda:0")
        marker = folder / "training_complete.json"
        resume = folder / "resume.pt"
        if marker.is_file():
            record = json.loads(marker.read_text())
            saved = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
            if saved["context"] != context:
                raise ValueError("Completed checkpoint configuration differs.")
            model.load_state_dict(saved["state_dict"])
            del saved
        else:
            started = time.monotonic()
            progress = SERVER / "logs" / f"{folder.name}.jsonl"
            if progress.is_file() and not resume.is_file():
                raise FileExistsError(
                    "Partial run without checkpoint; preserve before restarting."
                )

            def on_eval(step, loss, dev, best):
                event = {
                    "step": step,
                    "train_loss": loss,
                    "dev_loss": dev,
                    "best_dev_loss": best,
                    "session_seconds": time.monotonic() - started,
                    "power_summary": power_summary(model),
                }
                line = json.dumps(event, allow_nan=False)
                with progress.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                print(line, flush=True)

            outcome = train(
                model,
                encode_split(splits["train"], tokenizer, corpus),
                encode_split(splits["dev"], tokenizer, corpus),
                config,
                device,
                on_eval=on_eval,
                checkpoint_path=resume,
                checkpoint_context=context,
            )
            _atomic_checkpoint_save(
                {
                    "state_dict": model.state_dict(),
                    "context": context,
                    "step": outcome.scored_step,
                },
                folder / "weights.pt",
            )
            record = plain(
                {
                    "model": model_name,
                    "corpus": "iwslt14",
                    "metric": corpus.metric.name,
                    "parameters": count_parameters(model),
                    "baseline_parameters": baseline_parameters(full_shape),
                    "scores": {},
                    **asdict(outcome),
                    "scored_power_summary": power_summary(model),
                    "manifest": {
                        **context,
                        "context": context,
                        "seed": config.seed,
                        "seeded_before_model_init": True,
                        "smoke": False,
                        "data_directory": str(DATA),
                        "tokenizer": corpus.tokenizer_path(False),
                        "training_host": platform.node(),
                        "training_gpu": torch.cuda.get_device_name(0),
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "allocator_cap_mib": cap_mib,
                        "evaluated_splits": [],
                        "resumable_checkpoint_every": config.eval_every,
                    },
                }
            )
            write_json(marker, record)
            print(
                f"Training complete; saved step {outcome.scored_step}. Decoding next.", flush=True
            )
        model.to(device).eval()
        if result_path.is_file():
            record = json.loads(result_path.read_text())
        for split in SPLITS:
            if split in record["scores"]:
                continue
            examples = splits[split] if split in splits else corpus.load_split(split)
            score, hypotheses = score_split(
                model,
                corpus,
                examples,
                tokenizer,
                device,
                config.decode_batch_size,
                config.beam_size,
            )
            for kind, lines in (
                ("hypotheses", hypotheses),
                ("references", [e.target for e in examples]),
            ):
                (folder / f"{kind}.{split}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            record["scores"][split] = score
            record["manifest"]["evaluated_splits"] = list(record["scores"])
            write_json(result_path, record)
            print(f"iwslt14/{model_name}: {split} {score:.6f}", flush=True)
        print("EMBEDDING_IWSLT_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run(args.model, args.describe)
