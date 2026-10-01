"""Train and score one analogy-preserving projection model with decode progress updates."""

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
from models import MODEL_NAMES, construct, model_details, models_for_corpus
from trainer import _atomic_checkpoint_save, plain, set_seed, train

from ana.config import Schedule, Selection, TrainConfig
from ana.data.corpora import build_corpus
from ana.decoding import beam_decode
from ana.experiment import encode_split, prepare
from ana.registry import baseline_parameters, count_parameters
from ana.trainer import fixed_batches

SERVER = Path(os.environ.get(
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-combined-v4"
)).resolve()
NAS = Path(os.environ.get(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-combined-v4"
)).resolve()
DATA = Path(os.environ.get("ANA_DATA", str(NAS / "data"))).resolve()
SPLITS = {
    "multi30k": ("dev", "test"),
    "multi30k_enfr": ("dev", "test"),
    "cogs": ("gen",),
    "iwslt14": ("dev", "test"),
}


def score_split(model, corpus, examples, tokenizer, device, batch_size, beam_size, *, split):
    """Same beam decoding and output order as the shared scorer, with progress."""
    encoded = encode_split(examples, tokenizer, corpus)
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index][0]))
    batches = fixed_batches([encoded[index] for index in order], batch_size, model.config.pad_id)
    generated_in_order = []
    started = time.monotonic()
    for index, batch in enumerate(batches, start=1):
        generated = beam_decode(
            model, batch.source_ids.to(device), batch.source_mask.to(device),
            corpus.max_decode_length, beam_size,
        )
        generated_in_order.extend(tokenizer.decode(row.tolist()).strip() for row in generated)
        if index == 1 or index % 25 == 0 or index == len(batches):
            print(json.dumps({
                "phase": "decoding_progress", "split": split,
                "batch": index, "total_batches": len(batches),
                "decoded_examples": len(generated_in_order), "total_examples": len(examples),
                "elapsed_seconds": time.monotonic() - started,
            }, allow_nan=False), flush=True)
    hypotheses = [""] * len(examples)
    for position, original in enumerate(order):
        hypotheses[original] = generated_in_order[position]
    references = [example.target for example in examples]
    return corpus.metric.score(hypotheses, references), hypotheses


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
    expected = {
        "cogs": (50_000, 128, 0.0001, 0, Schedule.CONSTANT, Selection.FINAL),
        "iwslt14": (
            50_000, 160, 0.0005, 4_000,
            Schedule.INVERSE_SQRT, Selection.BEST_DEV_LOSS,
        ),
        "multi30k": (
            20_000, 256, 0.005, 2_000,
            Schedule.INVERSE_SQRT, Selection.BEST_DEV_LOSS,
        ),
        "multi30k_enfr": (
            20_000, 256, 0.005, 2_000,
            Schedule.INVERSE_SQRT, Selection.BEST_DEV_LOSS,
        ),
    }[corpus]
    actual = (
        result.max_steps, result.batch_size, result.learning_rate, result.warmup_steps,
        result.schedule, result.selection,
    )
    if actual != expected or result.eval_every != 1_000 or result.beam_size != 5:
        raise ValueError("Keep the original corpus recipe, checkpoint interval, and beam size.")
    if corpus == "iwslt14" and (
        result.weight_decay != 0.0001 or result.adam_betas != (0.9, 0.98)
    ):
        raise ValueError("Keep the original IWSLT14 optimiser recipe.")
    return result


def capacity():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    cap_mib = int(os.environ.get("ANA_MAX_GPU_MIB", "9216"))
    free = int(subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip())
    ram_kib = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if (
        cap_mib <= 0 or free < cap_mib + 1024 or ram_kib < 8 * 1024**2
        or min(shutil.disk_usage(SERVER).free, shutil.disk_usage(NAS).free) < 1024**3
    ):
        raise RuntimeError("Insufficient free resources; existing jobs remain untouched.")
    if not torch.cuda.is_available():
        raise RuntimeError("Working CUDA required; no CPU training fallback.")
    torch.cuda.set_per_process_memory_fraction(
        cap_mib * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0
    )
    return cap_mib


def run(corpus_name, model_name, describe=False, *, study_id="analogy_combined_v4"):
    if model_name not in models_for_corpus(corpus_name):
        raise ValueError("This model is not scheduled for this corpus.")
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
    model.activation_checkpointing = (
        os.environ.get("ANA_ACTIVATION_CHECKPOINTING", "0") == "1"
    )
    micro_defaults = {"cogs": 64, "iwslt14": 20}
    micro_batch_size = int(os.environ.get(
        "ANA_MICRO_BATCH_SIZE", str(micro_defaults.get(corpus_name, 128))
    ))
    decode_batch_size = int(os.environ.get(
        "ANA_DECODE_BATCH_SIZE", str(config.decode_batch_size)
    ))
    if not 0 < micro_batch_size <= config.batch_size:
        raise ValueError("Invalid microbatch size.")
    if not 0 < decode_batch_size <= config.decode_batch_size:
        raise ValueError("Decode microbatch must not exceed the corpus recipe.")
    if corpus_name == "cogs" and (
        model.config.d_model != 512 or model.config.label_smoothing != 0.0
    ):
        raise ValueError("COGS retains the original width 512 and zero label smoothing.")
    context = plain({
        "study_id": study_id, "corpus": corpus_name, "model": model_name,
        "model_config": asdict(model.config), "train_config": asdict(config),
        "model_details": model_details(model_name),
        "base_model": "shared_qkv", "embedding_class": type(model.embedding).__name__,
        "parent_checkpoint_used": False, "training_protocol": "fresh_original_recipe",
        "training_objective": "ordinary_single_pass_cross_entropy",
        "gradient_bearing_dropout_passes_per_example": 1,
        "optimizer_learning_rate_scales": {
            "routing_controllers": getattr(model, "analogy_routing_lr_scale", 0.1),
            "backbone_and_readout": 1.0,
        },
        "best_dev_checkpoint": "best.pt; diagnostic only when corpus selection is final",
        "train_pair_count": len(splits["train"]), "dev_pair_count": len(splits["dev"]),
        "reported_splits": list(SPLITS[corpus_name]),
        "activation_checkpointing": model.activation_checkpointing,
        "micro_batch_size": micro_batch_size,
        "decode_batch_size": decode_batch_size,
        "decode_batch_override": decode_batch_size != config.decode_batch_size,
        "gradient_accumulation_weighting": "valid target tokens; one update per original batch",
    })
    print(json.dumps({
        "phase": "configuration", **context, "parameters": count_parameters(model),
        "baseline_parameters": baseline_parameters(full_shape),
    }), flush=True)
    if describe:
        return
    folder = NAS / "runs" / f"{corpus_name}_{model_name}_seed{config.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    (SERVER / "logs").mkdir(parents=True, exist_ok=True)
    with (folder / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = folder / "results.json"
        if result_path.is_file():
            record = json.loads(result_path.read_text())
            if record["manifest"]["context"] != context:
                raise ValueError("Existing result belongs to another configuration.")
            if all(split in record["scores"] for split in SPLITS[corpus_name]):
                print("ANALOGY_COMBINED_CELL_COMPLETE already_completed", flush=True)
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

            def on_eval(step, loss, dev, best):
                line = json.dumps({
                    "step": step, "train_loss": loss, "dev_loss": dev,
                    "best_dev_loss": best, "session_seconds": time.monotonic() - started,
                }, allow_nan=False)
                with progress.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                print(line, flush=True)

            outcome = train(
                model, encode_split(splits["train"], tokenizer, corpus),
                encode_split(splits["dev"], tokenizer, corpus), config, device,
                on_eval=on_eval, checkpoint_path=resume, checkpoint_context=context,
                micro_batch_size=micro_batch_size,
            )
            _atomic_checkpoint_save({
                "state_dict": model.state_dict(), "context": context,
                "step": outcome.scored_step,
            }, folder / "weights.pt")
            record = plain({
                "model": model_name, "corpus": corpus_name, "metric": corpus.metric.name,
                "parameters": count_parameters(model),
                "baseline_parameters": baseline_parameters(full_shape), "scores": {},
                **asdict(outcome),
                "manifest": {
                    **context, "context": context, "seed": config.seed,
                    "seeded_before_model_init": True, "smoke": False,
                    "data_directory": str(DATA), "tokenizer": corpus.tokenizer_path(False),
                    "training_host": platform.node(),
                    "training_gpu": torch.cuda.get_device_name(0),
                    "python": platform.python_version(), "torch": str(torch.__version__),
                    "allocator_cap_mib": cap_mib, "evaluated_splits": [],
                    "resumable_checkpoint_every": config.eval_every,
                },
            })
            write_json(marker, record)
            print(f"Training complete; decoding saved step {outcome.scored_step}.", flush=True)
        model.to(device).eval()
        if result_path.is_file():
            record = json.loads(result_path.read_text())
        for split in SPLITS[corpus_name]:
            if split in record["scores"]:
                continue
            examples = splits[split] if split in splits else corpus.load_split(split)
            score, hypotheses = score_split(
                model, corpus, examples, tokenizer, device,
                decode_batch_size, config.beam_size, split=split,
            )
            for kind, lines in (
                ("hypotheses", hypotheses), ("references", [row.target for row in examples]),
            ):
                path = folder / f"{kind}.{split}.txt"
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            record["scores"][split] = score
            record["manifest"]["evaluated_splits"] = list(record["scores"])
            write_json(result_path, record)
            print(f"{corpus_name}/{model_name}: {split} {score:.6f}", flush=True)
        print("ANALOGY_COMBINED_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=tuple(SPLITS))
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run(args.corpus, args.model, args.describe)
