"""Site-specific key-value projection sharing screen on Multi30k.

This module owns the frozen matrix, mandatory server preflight, one-cell execution guard,
strict run-manifest validation, aggregation, classification, and deterministic compact reports.
The test split is hashed as committed data but is never loaded, tokenized, decoded, or scored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from typing import Any

import torch
from torch import Tensor

from ana import recipes
from ana.config import ModelConfig, Site, TrainConfig
from ana.data.corpora import build_corpus
from ana.data.tokenizer import train_or_load
from ana.experiment import run_cell
from ana.nn.projection import SeparateQKV, SharedKeyValueProjection
from ana.registry import REGISTRY, build_model, count_parameters, model_parameters
from ana.trainer import set_seed

STUDY_ID = "multi30k_site_specific_key_value_projection_sharing_screen"
CORPUS = "multi30k"
DEVELOPMENT_SPLIT = "dev"
EVALUATED_SPLITS = (DEVELOPMENT_SPLIT,)
SEEDS = (42, 43, 44)
TRAINING_STEPS = 20_000
PREFLIGHT_FILENAME = "preflight.json"

MODEL_DESCRIPTIONS = {
    "baseline": "Standard separate query, key, and value projection baseline",
    "encoder_self_attention_key_value_sharing": (
        "Encoder self-attention key–value projection sharing model"
    ),
    "decoder_self_attention_key_value_sharing": (
        "Decoder self-attention key–value projection sharing model"
    ),
    "cross_attention_key_value_sharing": "Cross-attention key–value projection sharing model",
    "all_attention_key_value_sharing": "All-attention-site key–value projection sharing model",
}
MODELS = tuple(MODEL_DESCRIPTIONS)
SHARING_MODELS = MODELS[1:]
SHARING_SITES = {
    "baseline": frozenset(),
    "encoder_self_attention_key_value_sharing": frozenset({Site.ENCODER_SELF}),
    "decoder_self_attention_key_value_sharing": frozenset({Site.DECODER_SELF}),
    "cross_attention_key_value_sharing": frozenset({Site.CROSS}),
    "all_attention_key_value_sharing": frozenset(Site),
}
EXPECTED_PARAMETERS = {
    "baseline": 2_605_568,
    "encoder_self_attention_key_value_sharing": 2_539_520,
    "decoder_self_attention_key_value_sharing": 2_539_520,
    "cross_attention_key_value_sharing": 2_539_520,
    "all_attention_key_value_sharing": 2_407_424,
}
EXPECTED_SPLIT_COUNTS = {"train": 29_000, DEVELOPMENT_SPLIT: 1_014}
EXPECTED_MODEL_CONFIGURATION = {
    "vocab_size": 10_000,
    "d_model": 128,
    "n_heads": 4,
    "d_ff": 256,
    "n_encoder_layers": 4,
    "n_decoder_layers": 4,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "max_positions": 256,
    "pad_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "unk_id": 3,
}
CACHE_ABSOLUTE_TOLERANCE = 1e-6
CACHE_RELATIVE_TOLERANCE = 1e-5
PASSING_MEAN_DIFFERENCE = -0.30
PASSING_SEED_DIFFERENCE = -0.50
NEGATIVE_MEAN_DIFFERENCE = -0.50


def experiment_matrix() -> list[dict[str, Any]]:
    """Return the fixed five-model, three-seed, 20,000-step matrix."""
    return [
        {
            "description": MODEL_DESCRIPTIONS[model],
            "model": model,
            "seeds": list(SEEDS),
            "training_steps_per_seed": TRAINING_STEPS,
        }
        for model in MODELS
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, value: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _exact_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot resolve the exact repository commit") from error
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"expected a full Git commit, found {commit!r}")
    return commit


def _require_clean_repository() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise ValueError("scientific runtime requires a clean exact-commit worktree")


def _require_approved_cell(model: str, seed: int) -> None:
    if model not in MODELS:
        raise ValueError(f"unapproved model {model!r}; choose from {', '.join(MODELS)}")
    if seed not in SEEDS:
        raise ValueError(f"unapproved seed {seed}; choose from {', '.join(map(str, SEEDS))}")


def _screen_train_config(seed: int) -> TrainConfig:
    corpus = build_corpus(CORPUS)
    recipe = recipes.load(CORPUS, corpus.recipe)
    if not recipe.verified:
        raise ValueError("the frozen Multi30k recipe is not verified")
    config = TrainConfig(
        max_steps=recipe.max_steps,
        batch_size=recipe.batch_size,
        learning_rate=recipe.learning_rate,
        warmup_steps=recipe.warmup_steps,
        schedule=recipe.schedule,
        selection=recipe.selection,
        weight_decay=recipe.weight_decay,
        adam_betas=recipe.adam_betas,
        seed=seed,
    )
    expected = {
        "max_steps": TRAINING_STEPS,
        "batch_size": 256,
        "learning_rate": 0.005,
        "warmup_steps": 2_000,
        "schedule": "inverse_sqrt",
        "selection": "best_dev_loss",
        "weight_decay": 0.0,
        "adam_betas": [0.9, 0.98],
        "eval_every": 1_000,
        "beam_size": 5,
    }
    observed = _normalized(asdict(config))
    for field, wanted in expected.items():
        if observed.get(field) != wanted:
            raise ValueError(
                f"the frozen Multi30k recipe has {field}={observed.get(field)!r}, "
                f"expected {wanted!r}"
            )
    return config


def _verify_dataset_manifest(data_root: str) -> dict[str, Any]:
    manifest_path = os.path.join(data_root, "MANIFEST.json")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("the committed dataset manifest has no file entries")

    verified = []
    for relative_path, expected in sorted(files.items()):
        path = os.path.join(data_root, relative_path)
        if not os.path.isfile(path):
            raise ValueError(f"committed dataset file is missing: {relative_path}")
        size = os.path.getsize(path)
        digest = _sha256(path)
        if size != expected.get("bytes") or digest != expected.get("sha256"):
            raise ValueError(f"committed dataset hash mismatch: {relative_path}")
        verified.append(relative_path)

    counts = manifest.get("counts", {}).get(CORPUS)
    if counts is None:
        raise ValueError("the dataset manifest lacks Multi30k split counts")
    return {
        "path": os.path.abspath(manifest_path),
        "sha256": _sha256(manifest_path),
        "verified_files": verified,
        "declared_multi30k_split_counts": counts,
    }


def _tokenizer_record(corpus, tokenizer) -> dict[str, Any]:
    model_path = corpus.tokenizer_path(smoke=False)
    if not os.path.isfile(model_path):
        raise ValueError(f"the Multi30k tokenizer artifact is missing: {model_path}")
    vocab_path = model_path.removesuffix(".model") + ".vocab"
    return {
        "model_path": os.path.abspath(model_path),
        "model_sha256": _sha256(model_path),
        "vocabulary_path": os.path.abspath(vocab_path) if os.path.isfile(vocab_path) else None,
        "vocabulary_sha256": _sha256(vocab_path) if os.path.isfile(vocab_path) else None,
        "vocabulary_size": len(tokenizer),
        "configuration": {
            "joint_source_target": True,
            "requested_vocabulary_size": corpus.vocab_size,
            "model_type": corpus.tokenizer_type,
            "character_coverage": corpus.character_coverage,
        },
    }


def logical_cache_accounting(model: str, config: ModelConfig) -> dict[str, Any]:
    """Count cached representation tensors, without claiming allocated byte savings."""
    shared = SHARING_SITES[model]
    decoder_self = 1 if Site.DECODER_SELF in shared else 2
    cross = 1 if Site.CROSS in shared else 2
    return {
        "encoder_self_attention": {
            "layers": config.n_encoder_layers,
            "cached_representation_tensors_per_layer": 0,
            "total_cached_representation_tensors": 0,
        },
        "decoder_self_attention": {
            "layers": config.n_decoder_layers,
            "cached_representation_tensors_per_layer": decoder_self,
            "total_cached_representation_tensors": config.n_decoder_layers * decoder_self,
        },
        "cross_attention": {
            "layers": config.n_decoder_layers,
            "cached_representation_tensors_per_layer": cross,
            "total_cached_representation_tensors": config.n_decoder_layers * cross,
        },
        "total_decoder_cached_representation_tensors": config.n_decoder_layers
        * (decoder_self + cross),
    }


def _attention_modules(model) -> list[tuple[int, Site, Any]]:
    modules = []
    for layer, block in enumerate(model.encoder, start=1):
        modules.append((layer, Site.ENCODER_SELF, block.self_attention))
    for layer, block in enumerate(model.decoder, start=1):
        modules.append((layer, Site.DECODER_SELF, block.self_attention))
        modules.append((layer, Site.CROSS, block.cross_attention))
    return modules


def _expected_attention_site_wiring(model_name: str) -> list[dict[str, Any]]:
    rows = []
    for layer in range(1, EXPECTED_MODEL_CONFIGURATION["n_encoder_layers"] + 1):
        site = Site.ENCODER_SELF
        rows.append(
            {
                "layer": layer,
                "site": site.value,
                "projection": (
                    "SharedKeyValueProjection"
                    if site in SHARING_SITES[model_name]
                    else "SeparateQKV"
                ),
            }
        )
    for layer in range(1, EXPECTED_MODEL_CONFIGURATION["n_decoder_layers"] + 1):
        for site in (Site.DECODER_SELF, Site.CROSS):
            rows.append(
                {
                    "layer": layer,
                    "site": site.value,
                    "projection": (
                        "SharedKeyValueProjection"
                        if site in SHARING_SITES[model_name]
                        else "SeparateQKV"
                    ),
                }
            )
    return rows


def _validate_projection_structure(model_name: str, model) -> list[dict[str, Any]]:
    rows = []
    for layer, site, attention in _attention_modules(model):
        projection = attention.projection
        shares_here = site in SHARING_SITES[model_name]
        expected_type = SharedKeyValueProjection if shares_here else SeparateQKV
        if type(projection) is not expected_type:
            raise ValueError(
                f"{model_name} layer {layer} {site.value} uses "
                f"{type(projection).__name__}, expected {expected_type.__name__}"
            )
        if shares_here:
            if projection.query is projection.shared_key_value:
                raise ValueError(f"{model_name} layer {layer} {site.value} shares its query")
            if projection.query.weight is projection.shared_key_value.weight:
                raise ValueError(
                    f"{model_name} layer {layer} {site.value} aliases projection weights"
                )
        rows.append(
            {
                "layer": layer,
                "site": site.value,
                "projection": type(projection).__name__,
            }
        )
    return rows


def _validate_shared_projection_semantics(model_name: str, model, device: torch.device) -> None:
    sample = torch.arange(2 * 3 * model.config.d_model, device=device, dtype=torch.float32)
    sample = sample.reshape(2, 3, model.config.d_model) / model.config.d_model
    mask = torch.ones(2, 3, dtype=torch.long, device=device)
    for layer, site, attention in _attention_modules(model):
        projection = attention.projection
        if not isinstance(projection, SharedKeyValueProjection):
            continue
        calls = 0

        def count_call(module, inputs, output) -> None:
            nonlocal calls
            calls += 1

        handle = projection.shared_key_value.register_forward_hook(count_call)
        try:
            query, key, value = projection(sample, sample, mask, mask)
        finally:
            handle.remove()
        if key is not value or calls != 1:
            raise ValueError(
                f"{model_name} layer {layer} {site.value} does not return one exact "
                "shared key-value tensor from one projection call"
            )
        if query is key:
            raise ValueError(f"{model_name} layer {layer} {site.value} aliases query and key")


def _synthetic_inputs(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    source_ids = torch.tensor([[4, 5, 6, 2], [7, 8, 9, 2]], device=device)
    source_mask = torch.ones_like(source_ids)
    labels = torch.tensor([[10, 11, 2], [12, 13, 2]], device=device)
    target_ids = torch.tensor([[1, 10, 11, 2], [1, 12, 13, 2]], device=device)
    return source_ids, source_mask, labels, target_ids


def _validate_finite_forward_backward(model, device: torch.device) -> None:
    source_ids, source_mask, labels, _ = _synthetic_inputs(device)
    model.train()
    model.zero_grad(set_to_none=True)
    loss, logits = model(source_ids, source_mask, labels)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
        raise ValueError("synthetic forward computation is non-finite")
    loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(f"synthetic backward gradient is missing or non-finite: {name}")
    model.zero_grad(set_to_none=True)


def _validate_incremental_cache(model_name: str, model, device: torch.device) -> None:
    source_ids, source_mask, _, target_ids = _synthetic_inputs(device)
    model.eval()
    with torch.no_grad():
        memory = model.encode(source_ids, source_mask)
        target_mask = torch.ones_like(target_ids)
        full_hidden = model.decode(target_ids, memory, source_mask, target_mask)
        full_logits = model.embedding.project(full_hidden)

        call_counts: dict[tuple[int, Site], int] = {}
        handles = []
        for layer, site, attention in _attention_modules(model):
            projection = attention.projection
            if not isinstance(projection, SharedKeyValueProjection):
                continue
            key = (layer, site)
            call_counts[key] = 0

            def count_call(module, inputs, output, *, key=key) -> None:
                call_counts[key] += 1

            handles.append(projection.shared_key_value.register_forward_hook(count_call))

        cache = model.new_cache()
        incremental = []
        try:
            for position in range(target_ids.size(1)):
                token = target_ids[:, position : position + 1]
                incremental.append(model.decode_step(token, memory, source_mask, cache, position))
        finally:
            for handle in handles:
                handle.remove()
        incremental_logits = torch.stack(incremental, dim=1)

    if not torch.allclose(
        full_logits,
        incremental_logits,
        atol=CACHE_ABSOLUTE_TOLERANCE,
        rtol=CACHE_RELATIVE_TOLERANCE,
    ):
        raise ValueError(f"{model_name} full-prefix and incremental logits differ")

    for layer, (self_cache, cross_cache) in enumerate(cache, start=1):
        for site, held in (
            (Site.DECODER_SELF, self_cache),
            (Site.CROSS, cross_cache),
        ):
            if held.key is None:
                raise ValueError(f"{model_name} layer {layer} {site.value} lacks a key cache")
            shares_here = site in SHARING_SITES[model_name]
            if shares_here and held.value is not None:
                raise ValueError(
                    f"{model_name} layer {layer} {site.value} duplicates its shared cache"
                )
            if not shares_here and held.value is None:
                raise ValueError(
                    f"{model_name} layer {layer} {site.value} lacks its separate value cache"
                )
            if not shares_here and held.key.data_ptr() == held.value.data_ptr():
                raise ValueError(f"{model_name} layer {layer} {site.value} aliases separate caches")

    for (layer, site), calls in call_counts.items():
        expected = target_ids.size(1) if site is Site.DECODER_SELF else 1
        if site is Site.ENCODER_SELF:
            continue
        if calls != expected:
            raise ValueError(
                f"{model_name} layer {layer} {site.value} made {calls} shared key-value "
                f"projection calls during incremental decoding, expected {expected}"
            )


def _execute_server_preflight(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested {device}, but CUDA is unavailable")
    _require_clean_repository()

    corpus = build_corpus(CORPUS)
    dataset_manifest = _verify_dataset_manifest(corpus.data_dir())
    loaded_splits = {
        "train": corpus.load_split("train"),
        DEVELOPMENT_SPLIT: corpus.load_split(DEVELOPMENT_SPLIT),
    }
    observed_counts = {split: len(rows) for split, rows in loaded_splits.items()}
    if observed_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Multi30k train/development counts are {observed_counts}, "
            f"expected {EXPECTED_SPLIT_COUNTS}"
        )

    tokenizer_text = [row.source for row in loaded_splits["train"]]
    tokenizer_text += [row.target for row in loaded_splits["train"]]
    tokenizer = train_or_load(
        corpus.tokenizer_path(smoke=False),
        tokenizer_text,
        corpus.vocab_size,
        corpus.character_coverage,
        corpus.tokenizer_type,
    )
    tokenizer_manifest = _tokenizer_record(corpus, tokenizer)
    if len(tokenizer) != corpus.vocab_size:
        raise ValueError(
            f"Multi30k tokenizer has {len(tokenizer)} pieces, expected {corpus.vocab_size}"
        )

    model_config = corpus.model_config(len(tokenizer))
    model_rows = {}
    for model_name in MODELS:
        if model_name not in REGISTRY:
            raise ValueError(f"approved model is not registered: {model_name}")
        set_seed(SEEDS[0])
        model = build_model(model_name, model_config).to(device)
        declared = model_parameters(model_name, model_config)
        built = count_parameters(model)
        expected = EXPECTED_PARAMETERS[model_name]
        if declared != built or declared != expected:
            raise ValueError(
                f"{model_name} parameters: declared {declared:,}, built {built:,}, "
                f"expected {expected:,}"
            )
        wiring = _validate_projection_structure(model_name, model)
        _validate_shared_projection_semantics(model_name, model, device)
        _validate_finite_forward_backward(model, device)
        _validate_incremental_cache(model_name, model, device)
        model_rows[model_name] = {
            "declared_parameters": declared,
            "built_parameters": built,
            "attention_site_wiring": wiring,
            "logical_cache_accounting": logical_cache_accounting(model_name, model_config),
            "shared_projection_semantics_passed": True,
            "finite_forward_backward_passed": True,
            "full_prefix_incremental_equivalence_passed": True,
            "cache_storage_passed": True,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    training_configurations = {
        str(seed): _normalized(asdict(_screen_train_config(seed))) for seed in SEEDS
    }
    environment = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "accelerator": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "central processor"
        ),
    }
    return {
        "dataset_manifest": dataset_manifest,
        "tokenizer": tokenizer_manifest,
        "loaded_splits": list(loaded_splits),
        "observed_split_counts": observed_counts,
        "evaluated_splits": list(EVALUATED_SPLITS),
        "test_split_loaded_tokenized_decoded_or_evaluated": False,
        "model_configuration": _normalized(asdict(model_config)),
        "training_configurations": training_configurations,
        "environment": environment,
        "models": model_rows,
        "checks": {
            "all_committed_dataset_hashes": True,
            "five_approved_models_constructed": True,
            "attention_site_projection_types": True,
            "independent_query_and_shared_key_value_projection": True,
            "one_exact_shared_key_value_tensor": True,
            "declared_and_built_parameter_counts": True,
            "finite_forward_and_backward": True,
            "full_prefix_and_incremental_logits": True,
            "single_tensor_shared_caches": True,
            "multi30k_train_and_development_counts": True,
            "development_only_run_configuration": True,
        },
        "cache_comparison_tolerances": {
            "absolute": CACHE_ABSOLUTE_TOLERANCE,
            "relative": CACHE_RELATIVE_TOLERANCE,
        },
    }


def validate_preflight(preflight: dict[str, Any], expected_commit: str | None = None) -> None:
    """Require one complete, passed, exact-commit server preflight."""
    if preflight.get("study_id") != STUDY_ID or preflight.get("status") != "passed":
        raise ValueError("the required server preflight is absent or did not pass")
    if (
        preflight.get("runtime_validation_not_scientific_evidence") is not True
        or not preflight.get("started_at")
        or not preflight.get("ended_at")
    ):
        raise ValueError("the server preflight lacks runtime-validation provenance")
    if preflight.get("cache_comparison_tolerances") != {
        "absolute": CACHE_ABSOLUTE_TOLERANCE,
        "relative": CACHE_RELATIVE_TOLERANCE,
    }:
        raise ValueError("the server preflight used wrong cache comparison tolerances")
    commit = preflight.get("implementation_git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("the server preflight lacks an exact implementation commit")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("the server preflight and current implementation commits differ")
    if preflight.get("loaded_splits") != ["train", DEVELOPMENT_SPLIT]:
        raise ValueError("the server preflight did not load exactly train and development")
    if preflight.get("observed_split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("the server preflight has incorrect Multi30k split counts")
    if preflight.get("evaluated_splits") != [DEVELOPMENT_SPLIT]:
        raise ValueError("the server preflight does not declare development-only evaluation")
    if preflight.get("test_split_loaded_tokenized_decoded_or_evaluated") is not False:
        raise ValueError("the server preflight indicates prohibited test-split access")
    if preflight.get("model_configuration") != EXPECTED_MODEL_CONFIGURATION:
        raise ValueError("the server preflight has the wrong Multi30k model configuration")
    expected_training = {
        str(seed): _normalized(asdict(_screen_train_config(seed))) for seed in SEEDS
    }
    if preflight.get("training_configurations") != expected_training:
        raise ValueError("the server preflight has the wrong frozen training configurations")
    tokenizer = preflight.get("tokenizer", {})
    expected_tokenizer_configuration = {
        "joint_source_target": True,
        "requested_vocabulary_size": 10_000,
        "model_type": "bpe",
        "character_coverage": 0.9995,
    }
    if (
        tokenizer.get("vocabulary_size") != 10_000
        or tokenizer.get("configuration") != expected_tokenizer_configuration
        or not tokenizer.get("model_sha256")
    ):
        raise ValueError("the server preflight has incomplete tokenizer provenance")
    if (
        not os.path.isfile(tokenizer.get("model_path", ""))
        or _sha256(tokenizer["model_path"]) != tokenizer["model_sha256"]
    ):
        raise ValueError("the tokenizer artifact differs from the server preflight")
    dataset_manifest = preflight.get("dataset_manifest", {})
    if (
        not dataset_manifest.get("sha256")
        or not dataset_manifest.get("verified_files")
        or dataset_manifest.get("declared_multi30k_split_counts", {}).get("train") != 29_000
        or dataset_manifest.get("declared_multi30k_split_counts", {}).get("dev") != 1_014
    ):
        raise ValueError("the server preflight has incomplete dataset-manifest provenance")
    manifest_path = dataset_manifest.get("path", "")
    if not os.path.isfile(manifest_path) or _sha256(manifest_path) != dataset_manifest["sha256"]:
        raise ValueError("the dataset manifest differs from the server preflight")
    manifest_files = _load_json(manifest_path).get("files", {})
    if sorted(manifest_files) != dataset_manifest["verified_files"]:
        raise ValueError("the server preflight did not verify every manifested dataset file")
    environment = preflight.get("environment", {})
    environment_fields = ("host", "platform", "python", "torch", "device")
    if not all(environment.get(field) for field in environment_fields):
        raise ValueError("the server preflight has incomplete runtime-environment provenance")
    checks = preflight.get("checks", {})
    if len(checks) != 11 or not all(value is True for value in checks.values()):
        raise ValueError("the server preflight does not contain eleven passed checks")
    rows = preflight.get("models", {})
    if set(rows) != set(MODELS):
        raise ValueError("the server preflight does not contain exactly the five approved models")
    for model_name, expected in EXPECTED_PARAMETERS.items():
        row = rows[model_name]
        if row.get("declared_parameters") != expected or row.get("built_parameters") != expected:
            raise ValueError(f"the server preflight has wrong parameters for {model_name}")
        if row.get("attention_site_wiring") != _expected_attention_site_wiring(model_name):
            raise ValueError(f"the server preflight has wrong site wiring for {model_name}")
        expected_cache = logical_cache_accounting(
            model_name, ModelConfig(**EXPECTED_MODEL_CONFIGURATION)
        )
        if row.get("logical_cache_accounting") != expected_cache:
            raise ValueError(f"the server preflight has wrong cache accounting for {model_name}")
        for field in (
            "shared_projection_semantics_passed",
            "finite_forward_backward_passed",
            "full_prefix_incremental_equivalence_passed",
            "cache_storage_passed",
        ):
            if row.get(field) is not True:
                raise ValueError(f"the server preflight lacks {field} for {model_name}")


def run_server_preflight(output_dir: str, device: torch.device) -> dict[str, Any]:
    """Run and preserve the single mandatory runtime validation before training."""
    path = os.path.join(output_dir, PREFLIGHT_FILENAME)
    if os.path.exists(path):
        raise ValueError(f"a server preflight record already exists at {path}; do not rerun it")

    record: dict[str, Any] = {
        "study_id": STUDY_ID,
        "implementation_git_commit": _exact_git_commit(),
        "status": "running",
        "started_at": _now(),
        "device": str(device),
        "runtime_validation_not_scientific_evidence": True,
    }
    _write_json(path, record)
    try:
        record.update(_execute_server_preflight(device))
        _require_clean_repository()
        if _exact_git_commit() != record["implementation_git_commit"]:
            raise ValueError("the repository commit changed during the server preflight")
        record["status"] = "passed"
        record["ended_at"] = _now()
        validate_preflight(record, expected_commit=record["implementation_git_commit"])
    except BaseException as error:
        record["status"] = "failed"
        record["ended_at"] = _now()
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(path, record)
        raise
    _write_json(path, record)
    return record


def _scientific_configuration_hash(
    model: str,
    seed: int,
    train_config: TrainConfig,
    implementation_commit: str,
) -> str:
    value = {
        "study_id": STUDY_ID,
        "corpus": CORPUS,
        "evaluated_splits": list(EVALUATED_SPLITS),
        "model": model,
        "seed": seed,
        "train_config": _normalized(asdict(train_config)),
        "implementation_git_commit": implementation_commit,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _attempt_history(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    value = _load_json(path).get("attempts")
    if not isinstance(value, list):
        raise ValueError(f"{path} has no attempt list")
    return value


def _write_attempt_history(path: str, attempts: list[dict[str, Any]]) -> None:
    _write_json(path, {"study_id": STUDY_ID, "attempts": attempts})


def _artifact_references(cell_dir: str) -> dict[str, str]:
    return {
        "results": os.path.join(cell_dir, "results.json"),
        "selected_checkpoint": os.path.join(cell_dir, "weights.pt"),
        "development_predictions": os.path.join(cell_dir, "hypotheses.dev.txt"),
        "development_references": os.path.join(cell_dir, "references.dev.txt"),
        "attempt_history": os.path.join(cell_dir, "attempt_history.json"),
        "log": "standard output and standard error of exact_repository_command",
    }


def run_screen_cell(
    model: str,
    seed: int,
    output_dir: str,
    device: torch.device,
    exact_repository_command: str,
) -> dict[str, Any]:
    """Run one approved cell after the exact-commit preflight and preserve its provenance."""
    _require_approved_cell(model, seed)
    _require_clean_repository()
    implementation_commit = _exact_git_commit()
    preflight_path = os.path.join(output_dir, PREFLIGHT_FILENAME)
    preflight = _load_json(preflight_path)
    validate_preflight(preflight, expected_commit=implementation_commit)

    train_config = _screen_train_config(seed)
    configuration_hash = _scientific_configuration_hash(
        model, seed, train_config, implementation_commit
    )
    cell_dir = os.path.join(output_dir, f"{CORPUS}_{model}_seed{seed}")
    results_path = os.path.join(cell_dir, "results.json")
    attempt_path = os.path.join(cell_dir, "attempt_history.json")

    if os.path.isfile(results_path):
        record = _load_json(results_path)
        validate_run_record(record, preflight, cell_dir)
        return record

    attempts = _attempt_history(attempt_path)
    if attempts and attempts[-1].get("state") == "running":
        attempts[-1]["state"] = "interrupted"
        attempts[-1]["ended_at"] = _now()
        attempts[-1]["failure"] = "no complete result artifact was produced"
    if any(attempt.get("state") == "failed" for attempt in attempts):
        raise ValueError("a failed scientific run must be reported, not retried")
    if len(attempts) >= 2:
        raise ValueError("the one permitted exact retry has already been used")
    if any(
        attempt.get("scientific_configuration_sha256") != configuration_hash for attempt in attempts
    ):
        raise ValueError("the interrupted attempt does not match this exact scientific cell")

    started_at = _now()
    attempts.append(
        {
            "attempt": len(attempts) + 1,
            "state": "running",
            "started_at": started_at,
            "exact_repository_command": exact_repository_command,
            "scientific_configuration_sha256": configuration_hash,
            "implementation_git_commit": implementation_commit,
        }
    )
    _write_attempt_history(attempt_path, attempts)

    try:
        record = run_cell(
            model,
            CORPUS,
            train_config,
            output_dir,
            smoke=False,
            device=device,
            study_id=STUDY_ID,
            score_dev=True,
            evaluated_splits=EVALUATED_SPLITS,
        )
        _require_clean_repository()
        if _exact_git_commit() != implementation_commit:
            raise ValueError("the repository commit changed during the scientific run")
        ended_at = _now()
        attempts[-1]["state"] = "completed"
        attempts[-1]["ended_at"] = ended_at
        _write_attempt_history(attempt_path, attempts)

        tokenizer = preflight["tokenizer"]
        if _sha256(tokenizer["model_path"]) != tokenizer["model_sha256"]:
            raise ValueError("the tokenizer artifact changed after the server preflight")
        manifest = record["manifest"]
        manifest.update(
            {
                "implementation_git_commit": implementation_commit,
                "model_description": MODEL_DESCRIPTIONS[model],
                "attention_site_key_value_sharing": sorted(
                    site.value for site in SHARING_SITES[model]
                ),
                "loaded_splits": ["train", DEVELOPMENT_SPLIT],
                "evaluated_splits": [DEVELOPMENT_SPLIT],
                "test_split_loaded_tokenized_decoded_or_evaluated": False,
                "dataset_manifest": preflight["dataset_manifest"],
                "tokenizer": tokenizer,
                "runtime_environment": preflight["environment"],
                "exact_repository_command": exact_repository_command,
                "started_at": started_at,
                "ended_at": ended_at,
                "exit_state": "completed",
                "training_stability": "completed_with_finite_recorded_metrics",
                "logical_cache_accounting": logical_cache_accounting(
                    model, ModelConfig(**manifest["model_config"])
                ),
                "artifact_references": _artifact_references(cell_dir),
                "retry_history": attempts,
                "deviations": [],
            }
        )
        _write_json(results_path, record)
        validate_run_record(record, preflight, cell_dir)
        return record
    except BaseException as error:
        attempts[-1]["state"] = "failed"
        attempts[-1]["ended_at"] = _now()
        attempts[-1]["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_attempt_history(attempt_path, attempts)
        raise


def _require_finite_numbers(value: Any, path: str = "record") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{path}[{index}]")


def validate_run_record(
    record: dict[str, Any],
    preflight: dict[str, Any],
    cell_dir: str,
) -> None:
    """Reject a mixed, partial, retried-for-score, or non-development-only cell."""
    _require_finite_numbers(record)
    model = record.get("model")
    manifest = record.get("manifest", {})
    seed = manifest.get("seed")
    _require_approved_cell(model, seed)
    label = f"{model}/seed{seed}"
    if record.get("corpus") != CORPUS or manifest.get("study_id") != STUDY_ID:
        raise ValueError(f"{label} has wrong corpus or study provenance")
    if manifest.get("implementation_git_commit") != preflight.get("implementation_git_commit"):
        raise ValueError(f"{label} and the server preflight use different commits")
    short_commit = manifest.get("git_commit")
    if not isinstance(short_commit, str) or not preflight["implementation_git_commit"].startswith(
        short_commit
    ):
        raise ValueError(f"{label} has inconsistent short and exact commit provenance")
    if manifest.get("seeded_before_model_init") is not True or manifest.get("smoke") is not False:
        raise ValueError(f"{label} is not a full pre-construction-seeded run")
    if manifest.get("score_dev") is not True:
        raise ValueError(f"{label} did not explicitly request development scoring")
    if (
        manifest.get("loaded_splits") != ["train", DEVELOPMENT_SPLIT]
        or manifest.get("evaluated_splits") != [DEVELOPMENT_SPLIT]
        or set(record.get("scores", {})) != {DEVELOPMENT_SPLIT}
        or manifest.get("test_split_loaded_tokenized_decoded_or_evaluated") is not False
    ):
        raise ValueError(f"{label} is not strictly train/development-only")

    expected_train = _normalized(asdict(_screen_train_config(seed)))
    if manifest.get("train_config") != expected_train:
        raise ValueError(f"{label} does not use the exact frozen training configuration")
    if manifest.get("model_config") != preflight.get("model_configuration"):
        raise ValueError(f"{label} model configuration differs from the server preflight")
    model_config = ModelConfig(**manifest["model_config"])
    expected_parameters = model_parameters(model, model_config)
    if (
        record.get("parameters") != expected_parameters
        or expected_parameters != EXPECTED_PARAMETERS[model]
    ):
        raise ValueError(f"{label} has incorrect parameter accounting")
    if record.get("baseline_parameters") != EXPECTED_PARAMETERS["baseline"]:
        raise ValueError(f"{label} has an incorrect baseline parameter reference")
    expected_saved = (EXPECTED_PARAMETERS["baseline"] - expected_parameters) / EXPECTED_PARAMETERS[
        "baseline"
    ]
    if not math.isclose(record.get("saved_fraction", math.nan), expected_saved, abs_tol=1e-15):
        raise ValueError(f"{label} has an incorrect parameter-saving fraction")

    if manifest.get("dataset_manifest") != preflight.get("dataset_manifest"):
        raise ValueError(f"{label} dataset provenance differs from the server preflight")
    if manifest.get("tokenizer") != preflight.get("tokenizer"):
        raise ValueError(f"{label} tokenizer provenance differs from the server preflight")
    if manifest.get("runtime_environment") != preflight.get("environment"):
        raise ValueError(f"{label} environment provenance differs from the server preflight")
    if manifest.get("model_description") != MODEL_DESCRIPTIONS[model]:
        raise ValueError(f"{label} has an incorrect descriptive model name")
    if manifest.get("attention_site_key_value_sharing") != sorted(
        site.value for site in SHARING_SITES[model]
    ):
        raise ValueError(f"{label} has incorrect site-sharing provenance")
    if manifest.get("logical_cache_accounting") != logical_cache_accounting(model, model_config):
        raise ValueError(f"{label} has incorrect logical cache accounting")

    if (
        manifest.get("exit_state") != "completed"
        or manifest.get("training_stability") != "completed_with_finite_recorded_metrics"
        or not manifest.get("started_at")
        or not manifest.get("ended_at")
        or not manifest.get("exact_repository_command")
        or manifest.get("deviations") != []
    ):
        raise ValueError(f"{label} has incomplete execution provenance")
    attempts = manifest.get("retry_history", [])
    if not 1 <= len(attempts) <= 2 or attempts[-1].get("state") != "completed":
        raise ValueError(f"{label} has invalid retry history")
    if sum(attempt.get("state") == "completed" for attempt in attempts) != 1:
        raise ValueError(f"{label} retry history has multiple completed runs")
    if any(attempt.get("state") not in {"interrupted", "completed"} for attempt in attempts):
        raise ValueError(f"{label} retry history contains an invalid attempt state")
    configuration_hash = _scientific_configuration_hash(
        model,
        seed,
        _screen_train_config(seed),
        preflight["implementation_git_commit"],
    )
    for attempt in attempts:
        if (
            attempt.get("scientific_configuration_sha256") != configuration_hash
            or attempt.get("implementation_git_commit") != preflight["implementation_git_commit"]
            or not attempt.get("exact_repository_command")
            or not attempt.get("started_at")
            or not attempt.get("ended_at")
        ):
            raise ValueError(f"{label} has incomplete exact-retry provenance")
    if attempts[-1]["exact_repository_command"] != manifest["exact_repository_command"]:
        raise ValueError(f"{label} completed command differs from its run manifest")

    if (
        record.get("steps_run") != TRAINING_STEPS
        or not 1 <= record.get("scored_step", 0) <= TRAINING_STEPS
        or record.get("selection") != "best_dev_loss"
        or record.get("seconds", 0) <= 0
    ):
        raise ValueError(f"{label} has incomplete or incorrect training outcomes")
    if record.get("best_step") != record.get("scored_step"):
        raise ValueError(f"{label} did not score the lowest-development-loss checkpoint")

    artifacts = manifest.get("artifact_references", {})
    expected_artifacts = _artifact_references(cell_dir)
    if artifacts != expected_artifacts:
        raise ValueError(f"{label} has incorrect artifact references")
    for key in (
        "results",
        "selected_checkpoint",
        "development_predictions",
        "development_references",
        "attempt_history",
    ):
        if not os.path.isfile(artifacts[key]):
            raise ValueError(f"{label} is missing {key}")
    if _attempt_history(artifacts["attempt_history"]) != attempts:
        raise ValueError(f"{label} attempt history differs from its run manifest")
    for forbidden in ("hypotheses.test.txt", "references.test.txt"):
        if os.path.exists(os.path.join(cell_dir, forbidden)):
            raise ValueError(f"{label} contains prohibited test-set output")
    with open(artifacts["development_predictions"], encoding="utf-8") as handle:
        if sum(1 for _ in handle) != EXPECTED_SPLIT_COUNTS[DEVELOPMENT_SPLIT]:
            raise ValueError(f"{label} has an incomplete development prediction artifact")


def load_records(run_dir: str, preflight: dict[str, Any]) -> list[dict[str, Any]]:
    """Load exactly the fifteen approved cells from the isolated study directory."""
    records = []
    missing = []
    expected_folders = {f"{CORPUS}_{model}_seed{seed}" for model, seed in product(MODELS, SEEDS)}
    for model, seed in product(MODELS, SEEDS):
        folder = os.path.join(run_dir, f"{CORPUS}_{model}_seed{seed}")
        path = os.path.join(folder, "results.json")
        if not os.path.isfile(path):
            missing.append(f"{model}/seed{seed}")
            continue
        record = _load_json(path)
        validate_run_record(record, preflight, folder)
        records.append(record)
    if missing:
        raise ValueError(f"{STUDY_ID} is incomplete; missing {', '.join(missing)}")

    unexpected = []
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if (
            os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "results.json"))
            and name not in expected_folders
        ):
            unexpected.append(name)
    if unexpected:
        raise ValueError(f"the isolated study directory has unapproved cells: {unexpected}")
    if len(records) != len(MODELS) * len(SEEDS):
        raise ValueError("the screen must contain exactly fifteen validated run manifests")
    return records


def classify_screen(
    paired_differences: dict[str, list[float]],
    invalid_reasons: list[str] | None = None,
) -> str:
    """Apply the preregistered success, negative, inconclusive, and invalid rules."""
    if invalid_reasons:
        return "invalid"
    passing = []
    for model in SHARING_MODELS:
        differences = paired_differences[model]
        passing.append(
            statistics.mean(differences) >= PASSING_MEAN_DIFFERENCE
            and sum(value >= PASSING_SEED_DIFFERENCE for value in differences) >= 2
        )
    if any(passing):
        return "success"
    if all(
        statistics.mean(paired_differences[model]) < NEGATIVE_MEAN_DIFFERENCE
        for model in SHARING_MODELS
    ):
        return "negative"
    return "inconclusive"


def build_artifact(records: list[dict[str, Any]], preflight: dict[str, Any]) -> dict[str, Any]:
    """Aggregate same-seed development results into a compact deterministic artifact."""
    if len(records) != len(MODELS) * len(SEEDS):
        raise ValueError("the screen requires exactly fifteen records")
    cells = {(row["model"], row["manifest"]["seed"]): row for row in records}
    expected_cells = set(product(MODELS, SEEDS))
    if set(cells) != expected_cells:
        raise ValueError("the screen records do not match the fixed matrix")

    ordered_runs = []
    aggregates = {}
    paired_differences: dict[str, list[float]] = {}
    baseline_scores = {
        seed: float(cells[("baseline", seed)]["scores"][DEVELOPMENT_SPLIT]) for seed in SEEDS
    }
    baseline_parameters = int(cells[("baseline", SEEDS[0])]["parameters"])

    for model in MODELS:
        scores = []
        losses = []
        differences = []
        for seed in SEEDS:
            record = cells[(model, seed)]
            if int(record["parameters"]) != int(cells[(model, SEEDS[0])]["parameters"]):
                raise ValueError(f"{model} parameter counts differ across seeds")
            score = float(record["scores"][DEVELOPMENT_SPLIT])
            difference = score - baseline_scores[seed]
            scores.append(score)
            losses.append(float(record["best_dev_loss"]))
            differences.append(difference)
            ordered_runs.append(
                {
                    "model": model,
                    "model_description": MODEL_DESCRIPTIONS[model],
                    "seed": seed,
                    "development_bleu": score,
                    "paired_development_bleu_difference_from_baseline": difference,
                    "development_loss": float(record["best_dev_loss"]),
                    "selected_checkpoint_step": int(record["scored_step"]),
                    "trainable_parameters": int(record["parameters"]),
                    "parameter_saving_from_baseline": baseline_parameters
                    - int(record["parameters"]),
                    "logical_cache_accounting": record["manifest"]["logical_cache_accounting"],
                    "exit_state": record["manifest"]["exit_state"],
                    "development_prediction_artifact": record["manifest"]["artifact_references"][
                        "development_predictions"
                    ],
                    "run_manifest": record["manifest"],
                }
            )
        paired_differences[model] = differences
        model_passes = (
            model != "baseline"
            and statistics.mean(differences) >= PASSING_MEAN_DIFFERENCE
            and sum(value >= PASSING_SEED_DIFFERENCE for value in differences) >= 2
        )
        aggregates[model] = {
            "mean_development_bleu": statistics.mean(scores),
            "sample_standard_deviation_development_bleu": statistics.stdev(scores),
            "mean_development_loss": statistics.mean(losses),
            "sample_standard_deviation_development_loss": statistics.stdev(losses),
            "paired_development_bleu_differences_from_baseline": differences,
            "mean_paired_development_bleu_difference_from_baseline": statistics.mean(differences),
            "paired_seed_differences_at_least_minus_0_50": sum(
                value >= PASSING_SEED_DIFFERENCE for value in differences
            ),
            "passes_screen": model_passes,
            "trainable_parameters": int(cells[(model, SEEDS[0])]["parameters"]),
            "parameter_saving_from_baseline": baseline_parameters
            - int(cells[(model, SEEDS[0])]["parameters"]),
            "logical_cache_accounting": ordered_runs[-1]["logical_cache_accounting"],
        }

    classification = classify_screen(paired_differences)
    artifact = {
        "study_id": STUDY_ID,
        "implementation_git_commit": preflight["implementation_git_commit"],
        "corpus": CORPUS,
        "exact_matrix": experiment_matrix(),
        "total_training_runs": len(MODELS) * len(SEEDS),
        "total_optimizer_steps": len(MODELS) * len(SEEDS) * TRAINING_STEPS,
        "evaluated_splits": [DEVELOPMENT_SPLIT],
        "test_split_loaded_tokenized_decoded_or_evaluated": False,
        "server_preflight": preflight,
        "runs": ordered_runs,
        "run_manifests": [row["run_manifest"] for row in ordered_runs],
        "aggregates": aggregates,
        "contextual_references": {
            "included": False,
            "source_artifact": None,
            "reason": "optional contextual references were not included",
        },
        "classification": {
            "outcome": classification,
            "passing_models": [
                model for model in SHARING_MODELS if aggregates[model]["passes_screen"]
            ],
            "rules": {
                "success": (
                    "at least one sharing model has mean paired development BLEU difference "
                    "at least -0.30 and at least two paired seed differences at least -0.50"
                ),
                "negative": "every sharing model has mean paired difference below -0.50",
                "inconclusive": "no model passes and the negative rule is not met",
                "invalid": (
                    "scientific provenance, implementation, data, metric, or artifact failure"
                ),
            },
        },
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require_finite_numbers(artifact, "artifact")
    if artifact.get("study_id") != STUDY_ID:
        raise ValueError("the compact artifact has the wrong study identifier")
    if artifact.get("exact_matrix") != experiment_matrix():
        raise ValueError("the compact artifact has a stale experiment matrix")
    if (
        artifact.get("total_training_runs") != 15
        or artifact.get("total_optimizer_steps") != 300_000
    ):
        raise ValueError("the compact artifact has a wrong scientific budget")
    if (
        artifact.get("evaluated_splits") != [DEVELOPMENT_SPLIT]
        or artifact.get("test_split_loaded_tokenized_decoded_or_evaluated") is not False
    ):
        raise ValueError("the compact artifact is not development-only")
    validate_preflight(
        artifact.get("server_preflight", {}),
        expected_commit=artifact.get("implementation_git_commit"),
    )
    runs = artifact.get("runs", [])
    manifests = artifact.get("run_manifests", [])
    if len(runs) != 15 or len(manifests) != 15:
        raise ValueError("the compact artifact does not contain all fifteen run manifests")
    observed = {(row.get("model"), row.get("seed")) for row in runs}
    if observed != set(product(MODELS, SEEDS)):
        raise ValueError("the compact artifact does not contain the exact approved cells")
    outcome = artifact.get("classification", {}).get("outcome")
    if outcome not in {"success", "negative", "inconclusive", "invalid"}:
        raise ValueError("the compact artifact has an unknown screen classification")


def markdown_report(artifact: dict[str, Any]) -> str:
    """Render deterministic individual, aggregate, accounting, and classification tables."""
    validate_artifact(artifact)
    lines = [
        "# Multi30k site-specific key-value projection sharing screen",
        "",
        "Development-only screening results. Paired differences are model minus the same-seed "
        "standard baseline. The server preflight is runtime validation, not scientific evidence.",
        "",
        "## Individual runs",
        "",
        "| model | seed | dev BLEU | paired Δ | dev loss | selected step | parameters | "
        "saved | exit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in artifact["runs"]:
        lines.append(
            f"| `{row['model']}` | {row['seed']} | {row['development_bleu']:.2f} | "
            f"{row['paired_development_bleu_difference_from_baseline']:+.2f} | "
            f"{row['development_loss']:.4f} | {row['selected_checkpoint_step']:,} | "
            f"{row['trainable_parameters']:,} | {row['parameter_saving_from_baseline']:,} | "
            f"{row['exit_state']} |"
        )

    lines += [
        "",
        "## Aggregates",
        "",
        "| model | dev BLEU mean ± sample SD | mean paired Δ | paired seeds ≥ -0.50 | passes |",
        "|---|---:|---:|---:|---|",
    ]
    for model in MODELS:
        row = artifact["aggregates"][model]
        lines.append(
            f"| `{model}` | {row['mean_development_bleu']:.2f} ± "
            f"{row['sample_standard_deviation_development_bleu']:.2f} | "
            f"{row['mean_paired_development_bleu_difference_from_baseline']:+.2f} | "
            f"{row['paired_seed_differences_at_least_minus_0_50']}/3 | "
            f"{'yes' if row['passes_screen'] else 'no'} |"
        )

    lines += [
        "",
        "## Parameter and logical cache accounting",
        "",
        "| model | parameters | saved | decoder self cache tensors/layer | "
        "cross cache tensors/layer |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = artifact["aggregates"][model]
        cache = row["logical_cache_accounting"]
        lines.append(
            f"| `{model}` | {row['trainable_parameters']:,} | "
            f"{row['parameter_saving_from_baseline']:,} | "
            f"{cache['decoder_self_attention']['cached_representation_tensors_per_layer']} | "
            f"{cache['cross_attention']['cached_representation_tensors_per_layer']} |"
        )

    decision = artifact["classification"]
    passing = ", ".join(f"`{model}`" for model in decision["passing_models"]) or "none"
    lines += [
        "",
        "## Screen classification",
        "",
        f"**{decision['outcome']}**",
        "",
        f"Passing sharing models: {passing}.",
        "",
        "The test split was not loaded, tokenized, decoded, or evaluated. No significance test, "
        "hardware-speed claim, throughput claim, energy claim, or wall-clock claim is included.",
        "",
    ]
    return "\n".join(lines)


def write_results(
    run_dir: str,
    json_path: str,
    markdown_path: str,
) -> dict[str, Any]:
    """Validate the full matrix and write deterministic compact JSON and Markdown artifacts."""
    preflight = _load_json(os.path.join(run_dir, PREFLIGHT_FILENAME))
    validate_preflight(preflight)
    records = load_records(run_dir, preflight)
    artifact = build_artifact(records, preflight)
    _write_json(json_path, artifact)
    parent = os.path.dirname(os.path.abspath(markdown_path))
    os.makedirs(parent, exist_ok=True)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report(artifact))
    return artifact
