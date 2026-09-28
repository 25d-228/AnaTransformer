"""Run one compact model with reference-rescaled adaptive-power consistency."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
from models import MODEL_NAMES, construct
from trainer import _atomic_checkpoint_save, consistency_loss_config, set_seed, train

from ana.config import Schedule, Selection, TrainConfig
from ana.data.corpora import build_corpus
from ana.experiment import encode_split, prepare, score_split
from ana.registry import baseline_parameters, count_parameters

SERVER = Path(
    os.environ.get("ANA_SERVER", "/home/Yue_Ziran/workspace/ana-adaptive-power-balanced-v1")
).resolve()
NAS = Path(
    os.environ.get("ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-adaptive-power-balanced-v1")
).resolve()
DATA = Path(os.environ.get("ANA_DATA", str(NAS / "data"))).resolve()
SPLITS = {"multi30k": ("dev", "test"), "multi30k_enfr": ("dev", "test"), "cogs": ("gen",)}


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


def recipe(corpus):
    values = json.loads((SERVER / "recipes" / f"{corpus}.json").read_text())
    config = {
        field.name: values[field.name] for field in fields(TrainConfig) if field.name in values
    }
    config["schedule"] = Schedule(config["schedule"])
    config["selection"] = Selection(config["selection"])
    config["adam_betas"] = tuple(config["adam_betas"])
    result = TrainConfig(**config)
    expected = (
        (50_000, 128, 0.0001, 0, Schedule.CONSTANT, Selection.FINAL)
        if corpus == "cogs"
        else (20_000, 256, 0.005, 2_000, Schedule.INVERSE_SQRT, Selection.BEST_DEV_LOSS)
    )
    actual = (
        result.max_steps,
        result.batch_size,
        result.learning_rate,
        result.warmup_steps,
        result.schedule,
        result.selection,
    )
    if actual != expected or result.eval_every != 1_000 or result.beam_size != 5:
        raise ValueError("Retain the original corpus recipe, checkpoint interval, and beam size.")
    return result


def capacity():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    cap_mib = int(os.environ.get("ANA_MAX_GPU_MIB", "8192"))
    if cap_mib <= 0:
        raise ValueError("GPU memory cap must be positive.")
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


def run(corpus_name, model_name, describe=False):
    if Path.cwd() != SERVER:
        raise RuntimeError(f"Run from {SERVER}")
    os.environ["ANA_DATA"] = str(DATA)
    corpus = build_corpus(corpus_name)
    if not Path(corpus.tokenizer_path(False)).is_file():
        raise FileNotFoundError("Reuse the existing tokenizer; do not train a new one.")
    tokenizer, splits = prepare(corpus, False, ("dev",))
    full_shape = corpus.model_config(len(tokenizer))
    config = recipe(corpus_name)
    set_seed(config.seed)
    model = construct(corpus_name, model_name, full_shape)
    if corpus_name == "cogs" and model.config.label_smoothing != 0.0:
        raise ValueError("COGS keeps zero label smoothing.")
    adaptive = model_name == "embedding_adaptive_power_balanced"
    context = plain(
        {
            "study_id": "adaptive_power_balanced_v1",
            "corpus": corpus_name,
            "model": model_name,
            "model_config": asdict(model.config),
            "train_config": asdict(config),
            "base_model": "embedding_linear",
            "embedding_class": type(model.embedding).__name__,
            "embedding_code_dim": model.embedding.code_dim,
            "embedding_parameters": count_parameters(model.embedding),
            "parent_checkpoint_used": False,
            "training_protocol": "fresh_original_recipe",
            "training_objective": (
                "reference_rescaled_adaptive_power_dropout_consistency"
                if adaptive
                else "two_dropout_view_cross_entropy"
            ),
            "loss_configuration": consistency_loss_config(adaptive=adaptive),
            "initial_power": 0.5 if adaptive else None,
            "fixed_power": None,
            "adaptive_power": adaptive,
            "power_scope": "training_loss_only" if adaptive else "not_used",
            "gradient_bearing_dropout_passes_per_update": 2,
            "consistency_coefficient": 1.0 if adaptive else 0.0,
            "probability_uniform_floor_mass": 1e-6 if adaptive else None,
            "train_pair_count": len(splits["train"]),
            "dev_pair_count": len(splits["dev"]),
            "reported_splits": list(SPLITS[corpus_name]),
            "inference_unchanged_by_training_loss": True,
            "activation_checkpointing": False,
            "micro_batch_size": (
                int(os.environ.get("ANA_MICRO_BATCH_SIZE", "0")) or config.batch_size
            ),
            "gradient_accumulation_weighting": (
                "valid target tokens; one optimizer update per original batch"
            ),
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
    folder = NAS / "runs" / f"{corpus_name}_{model_name}_seed{config.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = folder / "results.json"
        if result_path.is_file():
            record = json.loads(result_path.read_text())
            if record["manifest"]["context"] != context:
                raise ValueError("Existing results belong to another configuration.")
            if all(split in record["scores"] for split in SPLITS[corpus_name]):
                print("Completed cell; no training repeated.", flush=True)
                print("BALANCED_ADAPTIVE_POWER_CELL_COMPLETE", flush=True)
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
                adaptive=adaptive,
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
                    "corpus": corpus_name,
                    "metric": corpus.metric.name,
                    "parameters": count_parameters(model),
                    "baseline_parameters": baseline_parameters(full_shape),
                    "scores": {},
                    **asdict(outcome),
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
        for split in SPLITS[corpus_name]:
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
            print(f"{corpus_name}/{model_name}: {split} {score:.6f}", flush=True)
        print("BALANCED_ADAPTIVE_POWER_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=tuple(SPLITS))
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run(args.corpus, args.model, args.describe)
