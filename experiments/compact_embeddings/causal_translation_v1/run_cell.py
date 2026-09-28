"""Fresh Multi30k training with causal translation-example analogy scores."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
from trainer import _atomic_checkpoint_save, set_seed, train

from ana.causal_translation import CausalTranslationTransformer
from ana.config import Schedule, Selection, TrainConfig
from ana.data.corpora import build_corpus
from ana.experiment import encode_split, prepare, score_split
from ana.nn.attention import MultiHeadAttention
from ana.nn.embeddings.analogy_embedding import LinearCompressedEmbedding
from ana.nn.layers import FeedForward
from ana.nn.projection import SeparateQKV
from ana.registry import baseline_parameters, count_parameters

SERVER = Path("/home/Yue_Ziran/workspace/ana-causal-translation-v1")
NAS = Path("/mango/homes/YUE_Ziran/workspace/ana-causal-translation-v1")
DATA = NAS / "data"
MODELS = {"causal_translation_p05": "causal_translation_p05"}
EXPECTED_PARAMETERS = {"causal_translation_p05": 2_249_545}
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
    values = json.loads((SERVER / "recipes" / "multi30k.json").read_text())
    config = {
        field.name: values[field.name] for field in fields(TrainConfig) if field.name in values
    }
    config["schedule"] = Schedule(config["schedule"])
    config["selection"] = Selection(config["selection"])
    config["adam_betas"] = tuple(config["adam_betas"])
    result = TrainConfig(**config)
    if (
        result.max_steps != 20_000
        or result.learning_rate != 0.005
        or result.warmup_steps != 2_000
        or result.batch_size != 256
        or result.schedule is not Schedule.INVERSE_SQRT
        or result.selection is not Selection.BEST_DEV_LOSS
    ):
        raise ValueError("Use the original full-budget normal-rate Multi30k recipe unchanged.")
    return result


def analogy_summary(model):
    return {"fixed_power": 0.5, "gain": model.analogy_head.gain().detach().float().cpu().item()}


def construct(full_shape, model_name):
    assert model_name in MODELS
    assert full_shape.d_model == 128 and full_shape.d_ff == 256
    assert full_shape.n_heads == 4
    assert full_shape.n_encoder_layers == full_shape.n_decoder_layers == 4
    assert baseline_parameters(full_shape) == 2_605_568
    shape = replace(full_shape, d_model=128, d_ff=232)
    model = CausalTranslationTransformer(shape)
    attentions = [layer.self_attention for layer in model.encoder]
    for layer in model.decoder:
        attentions.extend((layer.self_attention, layer.cross_attention))
    assert len(attentions) == 12
    for attention in attentions:
        assert type(attention) is MultiHeadAttention
        assert type(attention.projection) is SeparateQKV
    feed_forwards = [layer.feed_forward for layer in (*model.encoder, *model.decoder)]
    assert len(feed_forwards) == 8
    for block in feed_forwards:
        assert type(block) is FeedForward
        assert block.up.weight.shape == (232, 128)
        assert block.down.weight.shape == (128, 232)
    assert type(model.embedding) is LinearCompressedEmbedding
    assert model.embedding.normalize_rows is False
    assert model.embedding.codes.shape == (full_shape.vocab_size, 96)
    assert model.embedding.basis.shape == (96, 128)
    assert model.embedding.d_model == 128 and model.embedding.code_dim == 96
    assert count_parameters(model.embedding) == 972_288
    assert count_parameters(model.analogy_head) == 1_033
    assert math.isclose(analogy_summary(model)["gain"], 0.1, rel_tol=1e-6)
    assert count_parameters(model) == EXPECTED_PARAMETERS[model_name]
    assert count_parameters(model) < 2_249_936 < baseline_parameters(full_shape)
    return model


def capacity():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    cap_mib = int(os.environ.get("ANA_MAX_GPU_MIB", "12288"))
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
        free < cap_mib + 1024
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
    corpus = build_corpus("multi30k")
    if not Path(corpus.tokenizer_path(False)).is_file():
        raise FileNotFoundError("Reuse the existing tokenizer; do not train a new one.")
    tokenizer, splits = prepare(corpus, False, ("dev",))
    full_shape = corpus.model_config(len(tokenizer))
    config = recipe()
    set_seed(config.seed)
    model = construct(full_shape, model_name)
    context = plain(
        {
            "study_id": "causal_translation_v1",
            "corpus": "multi30k",
            "model": model_name,
            "base_model": MODELS[model_name],
            "parent_checkpoint_used": False,
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "compression": "tied_96_to_128_linear_embedding",
            "embedding_parameters": 972_288,
            "ordinary_attention_and_feed_forward": True,
            "training_protocol": (
                "fresh_full_recipe_single_pass_cross_entropy_no_teacher_no_warm_start"
            ),
            "architecture_change": {
                "original_d_model": 128,
                "original_d_ff": 256,
                "new_d_model": 128,
                "new_d_ff": 232,
                "heads": 4,
                "head_width": 32,
                "attention_sites": 12,
                "ffn_sites": 8,
                "additional_analogy_parameters": 1_033,
            },
            "embedding": {
                "stored_coefficients_per_token": 96,
                "shared_basis_shape": [96, 128],
                "basis_bias": False,
                "tied_input_output": True,
                "row_normalization": False,
                "input_scaling": "sqrt(d_model)",
                "full_trainable_embedding_table_retained": False,
            },
            "analogy": {
                "source": "https://arxiv.org/abs/2407.18770",
                "condition": "A^p+D^p=B^p+C^p",
                "power": 0.5,
                "learned_power": False,
                "feature_dim": 4,
                "maps": 2,
                "map_shape": [128, 4],
                "map_bias": True,
                "input_normalization": "parameter_free_layer_norm",
                "positive_features": "softplus(map(LN(x)))/log(2)+1e-6",
                "positive_epsilon": 1e-6,
                "power_transform": "T(u)=(u^p-1)/p",
                "initial_gain": 0.1,
                "gain_parameterization": "softplus",
                "trainable_gain_count": 1,
                "source_context": "last_decoder_cross_attention_output_before_residual_dropout",
                "A": "positive_features_of_earlier_source_context_c_s",
                "B": "positive_features_of_known_earlier_target_token_y_s",
                "C": "positive_features_of_current_source_context_c_t",
                "D": "positive_features_of_candidate_target_token_v",
                "reference_scope": "all_earlier_valid_target_steps_in_same_sentence",
                "reference_weights": "softmax_s_of_-mean((T(C_t)-T(A_s))^2)_for_s<t",
                "reference_relation": "mu_t=sum_s(alpha_ts*(T(B_s)-T(A_s)))",
                "prediction_feature": "z_t=T(C_t)+mu_t",
                "logit_correction": "gain/4*(2*z_t_dot_T(D_v)-squared_norm(T(D_v)))",
                "first_position_correction": 0.0,
                "target_references_from": "completed_tokens_in_shifted_prefix_only",
                "beam_parent_reorders_reference_memory": True,
                "additional_training_loss": False,
                "used_at_inference": True,
                "verified_semantic_analogy_claim": False,
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
    folder = NAS / "runs" / f"multi30k_{model_name}_seed{config.seed}"
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
                print("CAUSAL_TRANSLATION_CELL_COMPLETE", flush=True)
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
                    "analogy_summary": analogy_summary(model),
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
                    "corpus": "multi30k",
                    "metric": corpus.metric.name,
                    "parameters": count_parameters(model),
                    "baseline_parameters": baseline_parameters(full_shape),
                    "scores": {},
                    **asdict(outcome),
                    "scored_analogy_summary": analogy_summary(model),
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
            print(f"multi30k/{model_name}: {split} {score:.6f}", flush=True)
        print("CAUSAL_TRANSLATION_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run(args.model, args.describe)
