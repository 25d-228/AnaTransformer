"""Development-only screen of D4 against fixed matched permutation families."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import torch

from ana.config import ModelConfig
from ana.d4_diagnostic import (
    REPRODUCTION_TOLERANCE,
    collect_router_statistics,
    decode_condition,
    load_checkpoint_sources,
    load_model,
    require_reproduction,
)
from ana.d4_diagnostic import (
    validate_artifact as validate_d4_artifact,
)
from ana.data.corpora import build_corpus
from ana.data.tokenizer import Tokenizer
from ana.experiment import encode_split, git_commit
from ana.nn.grouping import (
    D4_PERMUTATIONS,
    N_PERMUTATIONS,
    PERM_CONTROL_A,
    PERM_CONTROL_B,
    PERM_CONTROL_C,
    PERMUTATION_FAMILIES,
    Permutation,
    PermutationFamily,
)
from ana.registry import build_model
from ana.trainer import set_seed

STUDY_ID = "multi30k_permutation_family_v1"
CORPUS = "multi30k"
SPLIT = "dev"
SEEDS = (42, 43, 44)
CONTROL_MODELS = ("perm_ctrl_a_enc", "perm_ctrl_b_enc", "perm_ctrl_c_enc")
FAMILY_MODELS = ("ana_d4_enc", *CONTROL_MODELS)
REFERENCE_MODELS = ("baseline_matched", "shared_qkv", "ana_d4_enc")
SOURCE_FACTORIAL_STUDY = "multi30k_factorial_v1"
SOURCE_DIAGNOSTIC_STUDY = "multi30k_d4_checkpoint_diagnostic_v1"
PREFLIGHT_FILENAME = "preflight.json"
CONDITIONS = ("original", "hard_argmax")
FAMILY_LABELS = {
    "ana_d4_enc": "D4",
    "perm_ctrl_a_enc": "control A",
    "perm_ctrl_b_enc": "control B",
    "perm_ctrl_c_enc": "control C",
}
FAMILIES: dict[str, PermutationFamily] = {
    "ana_d4_enc": D4_PERMUTATIONS,
    "perm_ctrl_a_enc": PERM_CONTROL_A,
    "perm_ctrl_b_enc": PERM_CONTROL_B,
    "perm_ctrl_c_enc": PERM_CONTROL_C,
}


def _compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[index]] for index in range(4))  # type: ignore[return-value]


def generated_closure(family: PermutationFamily) -> set[Permutation]:
    closure = set(family)
    while True:
        expanded = {_compose(left, right) for left in tuple(closure) for right in tuple(closure)}
        if expanded <= closure:
            return closure
        closure.update(expanded)


def _cycle_type(permutation: Permutation) -> str:
    visited: set[int] = set()
    lengths = []
    for start in range(4):
        if start in visited:
            continue
        current = start
        length = 0
        while current not in visited:
            visited.add(current)
            current = permutation[current]
            length += 1
        if length > 1:
            lengths.append(length)
    cycle = tuple(sorted(lengths))
    names = {
        (): "identity",
        (2,): "transposition",
        (2, 2): "double_transposition",
        (3,): "three_cycle",
        (4,): "four_cycle",
    }
    if cycle not in names:
        raise ValueError(f"unexpected cycle type {cycle} for {permutation}")
    return names[cycle]


def _hamming_histogram(family: PermutationFamily) -> dict[str, int]:
    counts = Counter(
        sum(left[index] != right[index] for index in range(4))
        for left, right in combinations(family, 2)
    )
    return {str(distance): counts[distance] for distance in sorted(counts)}


def _family_hash(family: PermutationFamily) -> str:
    encoded = json.dumps(family, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def family_properties(model: str) -> dict[str, Any]:
    family = FAMILIES[model]
    closure = generated_closure(family)
    cycle_counts = Counter(_cycle_type(permutation) for permutation in family)
    return {
        "member_count": len(family),
        "distinct_member_count": len(set(family)),
        "identity_count": cycle_counts["identity"],
        "cycle_type_counts": {
            "identity": cycle_counts["identity"],
            "transposition": cycle_counts["transposition"],
            "double_transposition": cycle_counts["double_transposition"],
            "three_cycle": cycle_counts["three_cycle"],
            "four_cycle": cycle_counts["four_cycle"],
        },
        "closed_under_composition": len(closure) == len(family),
        "generated_closure_size": len(closure),
        "d4_intersection_size": len(set(family) & set(D4_PERMUTATIONS)),
        "hamming_distance_histogram": _hamming_histogram(family),
    }


def family_definitions() -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "label": FAMILY_LABELS[model],
            "router_order": [list(permutation) for permutation in FAMILIES[model]],
            "sha256": _family_hash(FAMILIES[model]),
            "properties": family_properties(model),
        }
        for model in FAMILY_MODELS
    ]


def control_pairwise_non_core_intersections() -> list[dict[str, Any]]:
    d4 = set(D4_PERMUTATIONS)
    return [
        {
            "left_model": left,
            "right_model": right,
            "non_core_intersection_size": len((set(FAMILIES[left]) & set(FAMILIES[right])) - d4),
        }
        for left, right in combinations(CONTROL_MODELS, 2)
    ]


def validate_preregistered_families() -> None:
    if FAMILIES != PERMUTATION_FAMILIES:
        raise ValueError("the study family registry differs from the model family registry")

    expected_cycles = {
        "identity": 1,
        "transposition": 2,
        "double_transposition": 3,
        "three_cycle": 0,
        "four_cycle": 2,
    }
    expected_control_histogram = {"2": 8, "3": 4, "4": 16}
    expected_core = {
        (0, 1, 2, 3),
        (3, 2, 1, 0),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
    }

    for model in FAMILY_MODELS:
        family = FAMILIES[model]
        properties = family_properties(model)
        if len(family) != N_PERMUTATIONS or len(set(family)) != N_PERMUTATIONS:
            raise ValueError(f"{model} must contain eight distinct permutations")
        if properties["cycle_type_counts"] != expected_cycles:
            raise ValueError(f"{model} has the wrong cycle-type composition")

    d4 = family_properties("ana_d4_enc")
    if not d4["closed_under_composition"] or d4["generated_closure_size"] != 8:
        raise ValueError("the D4 reference is not a closed eight-element group")
    if d4["hamming_distance_histogram"] != {"2": 8, "4": 20}:
        raise ValueError("the D4 reference has an unexpected Hamming-distance histogram")

    for model in CONTROL_MODELS:
        properties = family_properties(model)
        if properties["closed_under_composition"]:
            raise ValueError(f"{model} is unexpectedly closed")
        if properties["generated_closure_size"] != 24:
            raise ValueError(f"{model} does not generate all of S4")
        if set(FAMILIES[model]) & set(D4_PERMUTATIONS) != expected_core:
            raise ValueError(f"{model} does not have the fixed four-element D4 core")
        if properties["hamming_distance_histogram"] != expected_control_histogram:
            raise ValueError(f"{model} has the wrong Hamming-distance histogram")

    non_core_overlaps = {}
    for left, right in combinations(CONTROL_MODELS, 2):
        overlap = (set(FAMILIES[left]) & set(FAMILIES[right])) - expected_core
        non_core_overlaps[f"{left}__{right}"] = len(overlap)
    if non_core_overlaps != {
        "perm_ctrl_a_enc__perm_ctrl_b_enc": 2,
        "perm_ctrl_a_enc__perm_ctrl_c_enc": 1,
        "perm_ctrl_b_enc__perm_ctrl_c_enc": 1,
    }:
        raise ValueError("the control families do not have the preregistered diversity")


def trainable_initialization_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_hashes(config: ModelConfig) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        hashes = {}
        for model_name in FAMILY_MODELS:
            set_seed(seed)
            model = build_model(model_name, config)
            hashes[model_name] = trainable_initialization_hash(model)
        if len(set(hashes.values())) != 1:
            raise ValueError(f"seed {seed} does not initialize all four families identically")
        rows.append({"seed": seed, "common_sha256": next(iter(hashes.values())), "models": hashes})
    return rows


def _require_finite_numbers(value: Any, path: str = "artifact") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite numeric value")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{path}[{index}]")


def validate_preflight(preflight: dict[str, Any]) -> None:
    _require_finite_numbers(preflight, "preflight")
    if preflight.get("study_id") != STUDY_ID:
        raise ValueError("the compatibility preflight has the wrong study ID")
    if not preflight.get("implementation_git_commit"):
        raise ValueError("the compatibility preflight lacks implementation provenance")
    if preflight.get("source_factorial_study") != SOURCE_FACTORIAL_STUDY:
        raise ValueError("the compatibility preflight has stale source-study provenance")
    if not preflight.get("source_experiment_commit"):
        raise ValueError("the compatibility preflight lacks source experiment provenance")

    rows = preflight.get("d4_checkpoint_reproductions", [])
    expected = {(model, seed) for model, seed in product(("ana_d4_enc", "ana_feat_enc"), SEEDS)}
    observed = {(row.get("model"), row.get("seed")) for row in rows}
    if len(rows) != 6 or observed != expected:
        raise ValueError("the compatibility preflight must reproduce exactly six checkpoints")
    for row in rows:
        if abs(row.get("difference", math.inf)) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"{row.get('model')}/seed{row.get('seed')} fails the preflight guard")

    hashes = preflight.get("initialization_hashes", [])
    if [row.get("seed") for row in hashes] != list(SEEDS):
        raise ValueError("the preflight lacks same-seed initialization hashes")
    for row in hashes:
        models = row.get("models", {})
        if set(models) != set(FAMILY_MODELS) or len(set(models.values())) != 1:
            raise ValueError(f"seed {row.get('seed')} initialization hashes do not match")
        if row.get("common_sha256") != next(iter(models.values())):
            raise ValueError(f"seed {row.get('seed')} has an incorrect common hash")


def run_compatibility_preflight(
    source_run_dir: str,
    output_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    """Strict-load and reproduce all six prior checkpoints before any control training."""
    validate_preregistered_families()
    sources, source_commit = load_checkpoint_sources(source_run_dir)
    corpus = build_corpus(CORPUS)
    tokenizer_path = corpus.tokenizer_path(smoke=False)
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"the existing Multi30k tokenizer is required at {tokenizer_path}")
    tokenizer = Tokenizer(tokenizer_path)
    development = corpus.load_split(SPLIT)

    rows = []
    model_config: ModelConfig | None = None
    for source in sources:
        model = load_model(source, device)
        if model_config is None:
            model_config = model.config
        if model.config != model_config:
            raise ValueError("the six source checkpoints do not share one model configuration")
        train_config = source.results["manifest"]["train_config"]
        observed = decode_condition(
            model,
            "original",
            corpus,
            development,
            tokenizer,
            device,
            int(train_config["decode_batch_size"]),
            int(train_config["beam_size"]),
        )
        difference = require_reproduction(source.stored_development_bleu, observed)
        rows.append(
            {
                "model": source.model,
                "seed": source.seed,
                "stored_development_bleu": source.stored_development_bleu,
                "reproduced_development_bleu": observed,
                "difference": difference,
            }
        )
        print(
            f"preflight {source.model}/seed{source.seed}: "
            f"{source.stored_development_bleu:.4f} -> {observed:.4f} ({difference:+.4f})",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if model_config is None:
        raise ValueError("the compatibility preflight found no source model configuration")
    preflight = {
        "study_id": STUDY_ID,
        "implementation_git_commit": git_commit(),
        "source_factorial_study": SOURCE_FACTORIAL_STUDY,
        "source_experiment_commit": source_commit,
        "reproduction_tolerance_bleu": REPRODUCTION_TOLERANCE,
        "d4_checkpoint_reproductions": rows,
        "initialization_hashes": initialization_hashes(model_config),
    }
    validate_preflight(preflight)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PREFLIGHT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(preflight, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"compatibility and initialization preflight passed; wrote {path}", flush=True)
    return preflight


@dataclass(frozen=True)
class NewCheckpointSource:
    model: str
    seed: int
    folder: str
    results: dict[str, Any]
    weights_metadata: dict[str, Any]

    @property
    def stored_development_bleu(self) -> float:
        return float(self.results["scores"]["dev"])

    @property
    def weights_path(self) -> str:
        return os.path.join(self.folder, "weights.pt")


def _load_weights(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def load_new_checkpoint_sources(run_dir: str) -> tuple[list[NewCheckpointSource], str]:
    sources = []
    commits = set()
    recipes = set()
    for model, seed in product(CONTROL_MODELS, SEEDS):
        folder = os.path.join(run_dir, f"{CORPUS}_{model}_seed{seed}")
        results_path = os.path.join(folder, "results.json")
        weights_path = os.path.join(folder, "weights.pt")
        if not os.path.isfile(results_path) or not os.path.isfile(weights_path):
            raise ValueError(f"missing new checkpoint or results for {model}/seed{seed}")
        if any(
            os.path.exists(os.path.join(folder, filename))
            for filename in ("hypotheses.test.txt", "references.test.txt")
        ):
            raise ValueError(f"{model}/seed{seed} contains forbidden test-set output")

        with open(results_path, encoding="utf-8") as handle:
            record = json.load(handle)
        manifest = record.get("manifest", {})
        label = f"{model}/seed{seed}"
        if record.get("model") != model or manifest.get("seed") != seed:
            raise ValueError(f"{label} has mismatched model or seed provenance")
        if record.get("corpus") != CORPUS or manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"{label} has stale corpus or study provenance")
        if (
            manifest.get("seeded_before_model_init") is not True
            or manifest.get("smoke") is not False
        ):
            raise ValueError(f"{label} lacks full seeded-run provenance")
        if manifest.get("evaluated_splits") != [SPLIT] or set(record.get("scores", {})) != {SPLIT}:
            raise ValueError(f"{label} is not strictly development-only")
        if manifest.get("score_dev") is not True:
            raise ValueError(f"{label} did not explicitly request development scoring")
        train_config = manifest.get("train_config", {})
        if train_config.get("seed") != seed or train_config.get("max_steps") != 20_000:
            raise ValueError(f"{label} does not use the frozen seed and step budget")

        metadata = _load_weights(weights_path)
        if metadata.get("model") != model or metadata.get("corpus") != CORPUS:
            raise ValueError(f"{label} checkpoint metadata does not match its cell")
        if metadata.get("model_config") != manifest.get("model_config"):
            raise ValueError(f"{label} checkpoint and manifest model configurations differ")
        if metadata.get("step") != record.get("scored_step"):
            raise ValueError(f"{label} checkpoint and record selected steps differ")
        if set(metadata) != {
            "state_dict",
            "model",
            "corpus",
            "model_config",
            "step",
            "dev_loss",
        }:
            raise ValueError(f"{label} checkpoint has unexpected or incomplete metadata")

        commits.add(manifest.get("git_commit"))
        recipes.add(
            json.dumps(
                {key: value for key, value in train_config.items() if key != "seed"},
                sort_keys=True,
            )
        )
        sources.append(NewCheckpointSource(model, seed, folder, record, metadata))

    if len(sources) != 9:
        raise ValueError("the permutation-family study must contain exactly nine new cells")
    if len(commits) != 1 or None in commits:
        raise ValueError(f"new cells do not share one experiment commit: {commits}")
    if len(recipes) != 1:
        raise ValueError("new cells do not share one frozen recipe")
    return sources, str(next(iter(commits)))


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_reused_references(
    factorial_path: str,
    diagnostic_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factorial = _load_json(factorial_path)
    diagnostic = _load_json(diagnostic_path)
    _require_finite_numbers(factorial, "factorial_source")
    validate_d4_artifact(diagnostic)

    if (
        factorial.get("study_id") != SOURCE_FACTORIAL_STUDY
        or factorial.get("corpus") != CORPUS
        or factorial.get("seeds") != list(SEEDS)
    ):
        raise ValueError("the factorial source artifact has stale study provenance")
    factorial_runs = factorial.get("runs", [])
    expected_factorial = {
        (model, seed)
        for model, seed in product(
            ("baseline_matched", "shared_qkv", "ana_mag_enc", "ana_d4_enc", "ana_feat_enc"),
            SEEDS,
        )
    }
    observed_factorial = {(row.get("model"), row.get("seed")) for row in factorial_runs}
    if len(factorial_runs) != 15 or observed_factorial != expected_factorial:
        raise ValueError("the factorial source artifact is incomplete")
    factorial_commits = {row.get("git_commit") for row in factorial_runs}
    if len(factorial_commits) != 1 or None in factorial_commits:
        raise ValueError("the factorial source artifact has mixed experiment commits")
    source_experiment_commit = str(next(iter(factorial_commits)))
    if diagnostic.get("source_experiment_commit") != source_experiment_commit:
        raise ValueError("the source artifacts do not describe the same experiment commit")

    factorial_by_cell = {(row["model"], row["seed"]): row for row in factorial_runs}
    diagnostic_by_cell = {(row["model"], row["seed"]): row for row in diagnostic["checkpoints"]}
    references = []
    for model, seed in product(REFERENCE_MODELS, SEEDS):
        factorial_row = factorial_by_cell[(model, seed)]
        reference = {
            "origin": "reused reference",
            "model": model,
            "seed": seed,
            "development_bleu": factorial_row["development_bleu"],
            "parameters": factorial_row["parameters"],
            "wall_clock_seconds": factorial_row["wall_clock_seconds"],
            "source_study": SOURCE_FACTORIAL_STUDY,
            "source_experiment_commit": source_experiment_commit,
            "conditions": [
                {
                    "condition": "original",
                    "development_bleu": factorial_row["development_bleu"],
                    "difference_from_original": 0.0,
                }
            ],
            "modules": None,
        }
        if model == "ana_d4_enc":
            diagnostic_row = diagnostic_by_cell[(model, seed)]
            if not math.isclose(
                diagnostic_row["stored_development_bleu"],
                factorial_row["development_bleu"],
                abs_tol=1e-12,
            ):
                raise ValueError(f"ana_d4_enc/seed{seed} differs across source artifacts")
            hard = next(
                row for row in diagnostic_row["interventions"] if row["condition"] == "hard_argmax"
            )
            modules = copy.deepcopy(diagnostic_row["modules"])
            for module in modules:
                module["nearest_family_distance"] = module.pop("nearest_d4_distance")
            reference.update(
                {
                    "source_diagnostic_study": SOURCE_DIAGNOSTIC_STUDY,
                    "source_diagnostic_commit": diagnostic["analysis_git_commit"],
                    "conditions": [
                        {
                            "condition": "original",
                            "development_bleu": diagnostic_row["original_reproduction_bleu"],
                            "difference_from_original": 0.0,
                        },
                        {
                            "condition": "hard_argmax",
                            "development_bleu": hard["development_bleu"],
                            "difference_from_original": hard["difference_from_original"],
                        },
                    ],
                    "modules": modules,
                }
            )
        references.append(reference)

    provenance = {
        "factorial": {
            "path": factorial_path,
            "study_id": SOURCE_FACTORIAL_STUDY,
            "source_experiment_commit": source_experiment_commit,
        },
        "d4_diagnostic": {
            "path": diagnostic_path,
            "study_id": SOURCE_DIAGNOSTIC_STUDY,
            "analysis_git_commit": diagnostic["analysis_git_commit"],
            "source_experiment_commit": diagnostic["source_experiment_commit"],
        },
    }
    return references, provenance


def _load_new_model(source: NewCheckpointSource, device: torch.device) -> torch.nn.Module:
    config = ModelConfig(**source.weights_metadata["model_config"])
    model = build_model(source.model, config)
    checkpoint = _load_weights(source.weights_path, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def _condition(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(condition for condition in row["conditions"] if condition["condition"] == name)


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


MODULE_SCALARS = (
    "gate_strength",
    "normalized_router_entropy",
    "maximum_route_probability",
    "effective_number_of_routes",
    "identity_probability",
    "nearest_family_distance",
    "token_conditioned_routing_variation",
    "qkv_role_differentiation",
)


def aggregate_artifact(
    new_checkpoints: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    references_by_cell = {(row["model"], row["seed"]): row for row in references}
    new_by_cell = {(row["model"], row["seed"]): row for row in new_checkpoints}
    family_by_cell = {
        **{("ana_d4_enc", seed): references_by_cell[("ana_d4_enc", seed)] for seed in SEEDS},
        **new_by_cell,
    }

    family_summaries = []
    for model in FAMILY_MODELS:
        rows = [family_by_cell[(model, seed)] for seed in SEEDS]
        soft = [_condition(row, "original")["development_bleu"] for row in rows]
        hard = [_condition(row, "hard_argmax")["development_bleu"] for row in rows]
        hard_differences = [
            _condition(row, "hard_argmax")["difference_from_original"] for row in rows
        ]
        family_summaries.append(
            {
                "model": model,
                "mean_soft_development_bleu": _mean(soft),
                "sample_standard_deviation_soft_development_bleu": _sample_sd(soft),
                "mean_hard_development_bleu": _mean(hard),
                "sample_standard_deviation_hard_development_bleu": _sample_sd(hard),
                "mean_hard_minus_soft_bleu": _mean(hard_differences),
                "paired_hard_minus_soft_bleu": hard_differences,
                "parameters": rows[0]["parameters"],
                "mean_wall_clock_seconds": _mean(
                    [float(row["wall_clock_seconds"]) for row in rows]
                ),
            }
        )

    d4_specific = []
    for seed in SEEDS:
        d4 = _condition(family_by_cell[("ana_d4_enc", seed)], "original")["development_bleu"]
        controls = {
            model: _condition(family_by_cell[(model, seed)], "original")["development_bleu"]
            for model in CONTROL_MODELS
        }
        control_average = _mean(list(controls.values()))
        d4_specific.append(
            {
                "seed": seed,
                "d4_development_bleu": d4,
                "control_development_bleu": controls,
                "control_average_development_bleu": control_average,
                "d4_specific_effect": d4 - control_average,
            }
        )

    paired_d4 = []
    for model in CONTROL_MODELS:
        differences = [
            _condition(family_by_cell[("ana_d4_enc", seed)], "original")["development_bleu"]
            - _condition(family_by_cell[(model, seed)], "original")["development_bleu"]
            for seed in SEEDS
        ]
        paired_d4.append(
            {
                "control_model": model,
                "paired_d4_minus_control_bleu": differences,
                "mean_d4_minus_control_bleu": _mean(differences),
            }
        )

    contextual = []
    for model in FAMILY_MODELS:
        for reference_model in ("shared_qkv", "baseline_matched"):
            differences = [
                _condition(family_by_cell[(model, seed)], "original")["development_bleu"]
                - references_by_cell[(reference_model, seed)]["development_bleu"]
                for seed in SEEDS
            ]
            contextual.append(
                {
                    "model": model,
                    "reference_model": reference_model,
                    "paired_differences_bleu": differences,
                    "mean_difference_bleu": _mean(differences),
                }
            )

    module_summaries = []
    for model, layer, role in product(FAMILY_MODELS, range(1, 5), ("Q", "K", "V")):
        rows = [
            next(
                module
                for module in family_by_cell[(model, seed)]["modules"]
                if module["layer"] == layer and module["role"] == role
            )
            for seed in SEEDS
        ]
        summary = {key: _mean([float(row[key]) for row in rows]) for key in MODULE_SCALARS}
        summary["mean_route_distribution"] = [
            _mean([row["mean_route_distribution"][index] for row in rows])
            for index in range(N_PERMUTATIONS)
        ]
        module_summaries.append(
            {
                "model": model,
                "layer": layer,
                "role": role,
                "seeds": list(SEEDS),
                **summary,
            }
        )

    mean_effect = _mean([row["d4_specific_effect"] for row in d4_specific])
    positive_seeds = sum(row["d4_specific_effect"] > 0 for row in d4_specific)
    family_means = {row["model"]: row["mean_soft_development_bleu"] for row in family_summaries}
    d4_hard_delta = next(
        row["mean_hard_minus_soft_bleu"] for row in family_summaries if row["model"] == "ana_d4_enc"
    )
    promising = (
        mean_effect >= 0.30
        and positive_seeds >= 2
        and all(family_means["ana_d4_enc"] > family_means[control] for control in CONTROL_MODELS)
        and abs(d4_hard_delta) <= 0.15
    )
    shared_mean = _mean(
        [references_by_cell[("shared_qkv", seed)]["development_bleu"] for seed in SEEDS]
    )
    generic = abs(mean_effect) <= 0.20 or any(
        family_means[control] >= family_means["ana_d4_enc"] and family_means[control] > shared_mean
        for control in CONTROL_MODELS
    )
    if promising:
        decision = "d4_specific_signal_remains_promising"
    elif generic:
        decision = "generic_permutation_routing_not_d4_specifically"
    else:
        decision = "no_useful_routing_signal"

    return {
        "family_summaries": family_summaries,
        "d4_specific_by_seed": d4_specific,
        "mean_d4_specific_effect": mean_effect,
        "positive_d4_specific_seeds": positive_seeds,
        "paired_d4_against_controls": paired_d4,
        "contextual_paired_comparisons": contextual,
        "modules_across_seeds": module_summaries,
        "screening_decision": decision,
    }


def build_artifact(
    new_checkpoints: list[dict[str, Any]],
    references: list[dict[str, Any]],
    source_artifacts: dict[str, Any],
    preflight: dict[str, Any],
    analysis_git_commit: str,
    experiment_git_commit: str,
) -> dict[str, Any]:
    new_ordered = sorted(
        new_checkpoints,
        key=lambda row: (CONTROL_MODELS.index(row["model"]), SEEDS.index(row["seed"])),
    )
    reference_ordered = sorted(
        references,
        key=lambda row: (REFERENCE_MODELS.index(row["model"]), SEEDS.index(row["seed"])),
    )
    artifact = {
        "study_id": STUDY_ID,
        "analysis_git_commit": analysis_git_commit,
        "experiment_git_commit": experiment_git_commit,
        "corpus": CORPUS,
        "evaluated_splits": [SPLIT],
        "test_split_evaluated": False,
        "models": list(FAMILY_MODELS),
        "new_models": list(CONTROL_MODELS),
        "seeds": list(SEEDS),
        "reproduction_tolerance_bleu": REPRODUCTION_TOLERANCE,
        "family_selection_rationale": (
            "three preregistered eight-member non-closed families match D4's cycle-type "
            "composition and unavoidable four-member intersection; their non-core overlaps "
            "are 2, 1, and 1. Controls retain four distance-3 pairs whereas D4 replaces them "
            "with distance-4 pairs, so this screen tests the complete family rather than "
            "closure in isolation"
        ),
        "family_definitions": family_definitions(),
        "control_pairwise_non_core_intersections": (control_pairwise_non_core_intersections()),
        "source_artifacts": source_artifacts,
        "compatibility_preflight": preflight,
        "new_runs": new_ordered,
        "reused_references": reference_ordered,
        "aggregates": aggregate_artifact(new_ordered, reference_ordered),
    }
    validate_artifact(artifact)
    return artifact


def _validate_modules(row: dict[str, Any], label: str) -> None:
    modules = row.get("modules")
    expected = set(product(range(1, 5), ("Q", "K", "V")))
    observed = {(module.get("layer"), module.get("role")) for module in modules or []}
    if len(modules or []) != 12 or observed != expected:
        raise ValueError(f"{label} must contain 12 labeled router-statistics rows")
    for module in modules:
        if module.get("real_token_count", 0) <= 0:
            raise ValueError(f"{label} has an empty router-statistics row")
        if len(module.get("mean_route_distribution", [])) != N_PERMUTATIONS:
            raise ValueError(f"{label} has an incomplete family route distribution")
        if "nearest_family_distance" not in module or "nearest_d4_distance" in module:
            raise ValueError(f"{label} does not use family-generic distance naming")
        if module.get("magnitude") is not None:
            raise ValueError(f"{label} unexpectedly reports magnitude statistics")


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require_finite_numbers(artifact)
    validate_preregistered_families()
    if artifact.get("study_id") != STUDY_ID:
        raise ValueError("artifact has the wrong study ID")
    if not artifact.get("analysis_git_commit") or not artifact.get("experiment_git_commit"):
        raise ValueError("artifact lacks analysis or experiment Git provenance")
    if (
        artifact.get("corpus") != CORPUS
        or artifact.get("evaluated_splits") != [SPLIT]
        or artifact.get("test_split_evaluated") is not False
    ):
        raise ValueError("artifact is not strictly Multi30k development-only")
    if artifact.get("models") != list(FAMILY_MODELS) or artifact.get("new_models") != list(
        CONTROL_MODELS
    ):
        raise ValueError("artifact has stale model definitions")
    if artifact.get("seeds") != list(SEEDS):
        raise ValueError("artifact has stale seeds")
    if artifact.get("family_definitions") != family_definitions():
        raise ValueError("artifact family definitions or hashes differ from preregistration")
    if (
        artifact.get("control_pairwise_non_core_intersections")
        != control_pairwise_non_core_intersections()
    ):
        raise ValueError("artifact has stale control-family diversity metadata")

    source_artifacts = artifact.get("source_artifacts", {})
    factorial = source_artifacts.get("factorial", {})
    diagnostic = source_artifacts.get("d4_diagnostic", {})
    if (
        factorial.get("study_id") != SOURCE_FACTORIAL_STUDY
        or diagnostic.get("study_id") != SOURCE_DIAGNOSTIC_STUDY
        or not factorial.get("source_experiment_commit")
        or diagnostic.get("source_experiment_commit") != factorial.get("source_experiment_commit")
    ):
        raise ValueError("artifact has stale or inconsistent source provenance")

    preflight = artifact.get("compatibility_preflight", {})
    validate_preflight(preflight)
    if preflight.get("implementation_git_commit") != artifact.get("experiment_git_commit"):
        raise ValueError("preflight and new runs use different implementation commits")
    if preflight.get("source_experiment_commit") != factorial.get("source_experiment_commit"):
        raise ValueError("preflight and reused references have different source commits")

    new_runs = artifact.get("new_runs", [])
    expected_new = set(product(CONTROL_MODELS, SEEDS))
    observed_new = {(row.get("model"), row.get("seed")) for row in new_runs}
    if len(new_runs) != 9 or observed_new != expected_new:
        raise ValueError("artifact must contain exactly nine new control cells")
    for row in new_runs:
        label = f"{row.get('model')}/seed{row.get('seed')}"
        if row.get("origin") != "new run":
            raise ValueError(f"{label} is not labeled as a new run")
        manifest = row.get("run_manifest", {})
        if (
            manifest.get("study_id") != STUDY_ID
            or manifest.get("evaluated_splits") != [SPLIT]
            or manifest.get("git_commit") != artifact.get("experiment_git_commit")
        ):
            raise ValueError(f"{label} has stale or non-development-only run provenance")
        conditions = row.get("conditions", [])
        if [condition.get("condition") for condition in conditions] != list(CONDITIONS):
            raise ValueError(f"{label} has missing or reordered evaluation conditions")
        if abs(row.get("original_reproduction_difference", math.inf)) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"{label} fails the reproduction guard")
        original = _condition(row, "original")["development_bleu"]
        for condition in conditions:
            if not math.isclose(
                condition["difference_from_original"],
                condition["development_bleu"] - original,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{label} has an incorrect paired condition difference")
        _validate_modules(row, label)

    references = artifact.get("reused_references", [])
    expected_references = set(product(REFERENCE_MODELS, SEEDS))
    observed_references = {(row.get("model"), row.get("seed")) for row in references}
    if len(references) != 9 or observed_references != expected_references:
        raise ValueError("artifact must contain all nine reused reference cells")
    for row in references:
        label = f"{row.get('model')}/seed{row.get('seed')}"
        if row.get("origin") != "reused reference":
            raise ValueError(f"{label} is not labeled as a reused reference")
        if row.get("source_experiment_commit") != factorial.get("source_experiment_commit"):
            raise ValueError(f"{label} has stale source experiment provenance")
        expected_conditions = list(CONDITIONS) if row["model"] == "ana_d4_enc" else ["original"]
        if [condition.get("condition") for condition in row.get("conditions", [])] != (
            expected_conditions
        ):
            raise ValueError(f"{label} has missing reused evaluation conditions")
        if row["model"] == "ana_d4_enc":
            _validate_modules(row, label)
        elif row.get("modules") is not None:
            raise ValueError(f"{label} unexpectedly contains router statistics")

    expected_aggregates = aggregate_artifact(new_runs, references)
    if artifact.get("aggregates") != expected_aggregates:
        raise ValueError("artifact aggregates do not match the individual cells")


def _aggregate_row(artifact: dict[str, Any], model: str) -> dict[str, Any]:
    return next(row for row in artifact["aggregates"]["family_summaries"] if row["model"] == model)


def _context_row(
    artifact: dict[str, Any],
    model: str,
    reference: str,
) -> dict[str, Any]:
    return next(
        row
        for row in artifact["aggregates"]["contextual_paired_comparisons"]
        if row["model"] == model and row["reference_model"] == reference
    )


def markdown_report(artifact: dict[str, Any]) -> str:
    validate_artifact(artifact)
    lines = [
        f"# {STUDY_ID}",
        "",
        "Multi30k development-only screening comparison of the analogy-equivalent D4 family "
        "against three preregistered, cycle-type-matched, non-closed permutation families. "
        "The test split was not loaded, decoded, or reported.",
        "",
        "## Fixed families and matching checks",
        "",
        "| family | SHA-256 | closed | generated closure | D4 overlap | Hamming histogram |",
        "|---|---|---:|---:|---:|---|",
    ]
    for family in artifact["family_definitions"]:
        properties = family["properties"]
        histogram = ", ".join(
            f"d={distance}: {count}"
            for distance, count in properties["hamming_distance_histogram"].items()
        )
        lines.append(
            f"| `{family['model']}` | `{family['sha256']}` | "
            f"{str(properties['closed_under_composition']).lower()} | "
            f"{properties['generated_closure_size']} | "
            f"{properties['d4_intersection_size']} | {histogram} |"
        )
    lines += [
        "",
        "Every family has one identity, two transpositions, three double transpositions, two "
        "four-cycles, and no three-cycles. Each control generates all 24 elements of S4 and "
        "shares exactly identity plus the three double transpositions with D4. The controls "
        "have four distance-3 pairs; D4 instead has twenty distance-4 pairs rather than "
        "sixteen. This residual geometry means the screen tests the complete D4 family, not "
        "group closure in isolation.",
        "",
        "## Compatibility and initialization preflight",
        "",
        "All six prior D4 diagnostic checkpoints strictly loaded and reproduced within "
        f"{artifact['reproduction_tolerance_bleu']:.2f} BLEU before any control training began.",
        "",
        "| seed | common trainable initialization SHA-256 |",
        "|---:|---|",
    ]
    for row in artifact["compatibility_preflight"]["initialization_hashes"]:
        lines.append(f"| {row['seed']} | `{row['common_sha256']}` |")

    lines += [
        "",
        "## Individual development results",
        "",
        "| origin | model | seed | soft BLEU | hard BLEU | hard - soft | params | hours |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    family_rows = [
        row for row in artifact["reused_references"] if row["model"] == "ana_d4_enc"
    ] + artifact["new_runs"]
    family_rows.sort(key=lambda row: (FAMILY_MODELS.index(row["model"]), row["seed"]))
    for row in family_rows:
        soft = _condition(row, "original")
        hard = _condition(row, "hard_argmax")
        lines.append(
            f"| {row['origin']} | `{row['model']}` | {row['seed']} | "
            f"{soft['development_bleu']:.2f} | {hard['development_bleu']:.2f} | "
            f"{hard['difference_from_original']:+.2f} | {row['parameters']:,} | "
            f"{row['wall_clock_seconds'] / 3600:.2f} |"
        )

    lines += [
        "",
        "## Family means",
        "",
        "| model | soft BLEU mean ± sample SD | hard BLEU mean ± sample SD | "
        "mean hard - soft | mean hours |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["family_summaries"]:
        lines.append(
            f"| `{row['model']}` | {row['mean_soft_development_bleu']:.2f} ± "
            f"{row['sample_standard_deviation_soft_development_bleu']:.2f} | "
            f"{row['mean_hard_development_bleu']:.2f} ± "
            f"{row['sample_standard_deviation_hard_development_bleu']:.2f} | "
            f"{row['mean_hard_minus_soft_bleu']:+.2f} | "
            f"{row['mean_wall_clock_seconds'] / 3600:.2f} |"
        )

    lines += [
        "",
        "## D4-specific contrast",
        "",
        "| seed | D4 | control A | control B | control C | control average | "
        "D4 - control average |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["d4_specific_by_seed"]:
        controls = row["control_development_bleu"]
        lines.append(
            f"| {row['seed']} | {row['d4_development_bleu']:.2f} | "
            f"{controls['perm_ctrl_a_enc']:.2f} | {controls['perm_ctrl_b_enc']:.2f} | "
            f"{controls['perm_ctrl_c_enc']:.2f} | "
            f"{row['control_average_development_bleu']:.2f} | "
            f"{row['d4_specific_effect']:+.2f} |"
        )
    lines.append(
        f"| **mean** | | | | | | **{artifact['aggregates']['mean_d4_specific_effect']:+.2f}** |"
    )

    lines += [
        "",
        "## Paired contextual comparisons",
        "",
        "| model | comparison | seed 42 | seed 43 | seed 44 | mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["paired_d4_against_controls"]:
        differences = row["paired_d4_minus_control_bleu"]
        lines.append(
            f"| `ana_d4_enc` | minus `{row['control_model']}` | "
            f"{differences[0]:+.2f} | {differences[1]:+.2f} | {differences[2]:+.2f} | "
            f"{row['mean_d4_minus_control_bleu']:+.2f} |"
        )
    for row in artifact["aggregates"]["contextual_paired_comparisons"]:
        differences = row["paired_differences_bleu"]
        lines.append(
            f"| `{row['model']}` | minus `{row['reference_model']}` | "
            f"{differences[0]:+.2f} | {differences[1]:+.2f} | {differences[2]:+.2f} | "
            f"{row['mean_difference_bleu']:+.2f} |"
        )

    lines += [
        "",
        "## Router statistics across seeds",
        "",
        "| family | layer | role | gate | H/log8 | max p | effective | identity | "
        "nearest family | token JS | QKV JS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["modules_across_seeds"]:
        lines.append(
            f"| `{row['model']}` | {row['layer']} | {row['role']} | "
            f"{row['gate_strength']:.3f} | {row['normalized_router_entropy']:.3f} | "
            f"{row['maximum_route_probability']:.3f} | "
            f"{row['effective_number_of_routes']:.2f} | "
            f"{row['identity_probability']:.3f} | "
            f"{row['nearest_family_distance']:.3f} | "
            f"{row['token_conditioned_routing_variation']:.4f} | "
            f"{row['qkv_role_differentiation']:.4f} |"
        )

    d4 = _aggregate_row(artifact, "ana_d4_enc")
    control_means = {
        model: _aggregate_row(artifact, model)["mean_soft_development_bleu"]
        for model in CONTROL_MODELS
    }
    d4_vs_controls = {
        row["control_model"]: row["mean_d4_minus_control_bleu"]
        for row in artifact["aggregates"]["paired_d4_against_controls"]
    }
    control_vs_shared = {
        model: _context_row(artifact, model, "shared_qkv")["mean_difference_bleu"]
        for model in CONTROL_MODELS
    }
    control_router_profiles = {}
    for model in CONTROL_MODELS:
        modules = [
            row for row in artifact["aggregates"]["modules_across_seeds"] if row["model"] == model
        ]
        control_router_profiles[model] = {
            "maximum_route_probability": _mean(
                [row["maximum_route_probability"] for row in modules]
            ),
            "nearest_family_distance": _mean([row["nearest_family_distance"] for row in modules]),
            "hard_minus_soft": _aggregate_row(artifact, model)["mean_hard_minus_soft_bleu"],
        }
    family_vs_baseline = {
        model: _context_row(artifact, model, "baseline_matched")["mean_difference_bleu"]
        for model in FAMILY_MODELS
    }
    best_control = max(control_means, key=control_means.get)
    decision = artifact["aggregates"]["screening_decision"]
    decision_text = {
        "d4_specific_signal_remains_promising": (
            "D4-specific signal remains promising. Stop for audit; the likely next experiment "
            "is D4 against a parameter/compute-matched generic local 4x4 mixer."
        ),
        "generic_permutation_routing_not_d4_specifically": (
            "The screen supports generic token- and role-conditioned permutation routing, not "
            "D4 specifically."
        ),
        "no_useful_routing_signal": (
            "No useful family-specific routing signal remains; stop this architecture branch."
        ),
    }[decision]
    lines += [
        "",
        "## Decision questions",
        "",
        f"- **Does D4 beat the average matched non-closed family?** Its mean paired effect is "
        f"{artifact['aggregates']['mean_d4_specific_effect']:+.2f} BLEU and is positive on "
        f"{artifact['aggregates']['positive_d4_specific_seeds']}/3 seeds.",
        f"- **Does D4 beat A, B, and C separately?** Mean paired differences are "
        f"{d4_vs_controls['perm_ctrl_a_enc']:+.2f}, "
        f"{d4_vs_controls['perm_ctrl_b_enc']:+.2f}, and "
        f"{d4_vs_controls['perm_ctrl_c_enc']:+.2f} BLEU.",
        f"- **Is any D4 advantage consistent across seeds?** "
        f"{artifact['aggregates']['positive_d4_specific_seeds']}/3 seed-level effects against "
        "the control average are positive.",
        f"- **Do arbitrary families improve over `shared_qkv`?** Mean paired effects are "
        f"{control_vs_shared['perm_ctrl_a_enc']:+.2f}, "
        f"{control_vs_shared['perm_ctrl_b_enc']:+.2f}, and "
        f"{control_vs_shared['perm_ctrl_c_enc']:+.2f} BLEU.",
        f"- **Do arbitrary families become near-discrete and retain BLEU under hard argmax?** "
        f"Mean maximum route probability / nearest-family distance / hard-minus-soft BLEU is "
        f"{control_router_profiles['perm_ctrl_a_enc']['maximum_route_probability']:.3f} / "
        f"{control_router_profiles['perm_ctrl_a_enc']['nearest_family_distance']:.3f} / "
        f"{control_router_profiles['perm_ctrl_a_enc']['hard_minus_soft']:+.2f} for A, "
        f"{control_router_profiles['perm_ctrl_b_enc']['maximum_route_probability']:.3f} / "
        f"{control_router_profiles['perm_ctrl_b_enc']['nearest_family_distance']:.3f} / "
        f"{control_router_profiles['perm_ctrl_b_enc']['hard_minus_soft']:+.2f} for B, and "
        f"{control_router_profiles['perm_ctrl_c_enc']['maximum_route_probability']:.3f} / "
        f"{control_router_profiles['perm_ctrl_c_enc']['nearest_family_distance']:.3f} / "
        f"{control_router_profiles['perm_ctrl_c_enc']['hard_minus_soft']:+.2f} for C. "
        f"The best-soft control is `{best_control}` at {control_means[best_control]:.2f} BLEU.",
        f"- **Does any permutation family approach `baseline_matched`?** D4 is "
        f"{family_vs_baseline['ana_d4_enc']:+.2f} BLEU relative to it; the control "
        f"differences are {family_vs_baseline['perm_ctrl_a_enc']:+.2f}, "
        f"{family_vs_baseline['perm_ctrl_b_enc']:+.2f}, and "
        f"{family_vs_baseline['perm_ctrl_c_enc']:+.2f}.",
        "",
        f"**Screening interpretation:** {decision_text}",
        "",
        f"D4 soft-to-hard mean change is {d4['mean_hard_minus_soft_bleu']:+.2f} BLEU. "
        "These are three-seed screening effects; no p-values, bootstrap tests, or significance "
        "claims are reported.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    run_dir: str,
    factorial_path: str,
    diagnostic_path: str,
    json_path: str,
    markdown_path: str,
    device: torch.device,
) -> dict[str, Any]:
    sources, experiment_commit = load_new_checkpoint_sources(run_dir)
    preflight = _load_json(os.path.join(run_dir, PREFLIGHT_FILENAME))
    validate_preflight(preflight)
    if preflight["implementation_git_commit"] != experiment_commit:
        raise ValueError("preflight and training runs were not made from the same commit")
    references, provenance = load_reused_references(factorial_path, diagnostic_path)

    corpus = build_corpus(CORPUS)
    tokenizer_path = corpus.tokenizer_path(smoke=False)
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"the existing Multi30k tokenizer is required at {tokenizer_path}")
    tokenizer = Tokenizer(tokenizer_path)
    development = corpus.load_split(SPLIT)
    encoded_development = encode_split(development, tokenizer, corpus)

    checkpoints = []
    print("phase 1: reproducing all nine new development scores", flush=True)
    for source in sources:
        model = _load_new_model(source, device)
        train_config = source.results["manifest"]["train_config"]
        observed = decode_condition(
            model,
            "original",
            corpus,
            development,
            tokenizer,
            device,
            int(train_config["decode_batch_size"]),
            int(train_config["beam_size"]),
        )
        difference = require_reproduction(source.stored_development_bleu, observed)
        checkpoints.append(
            {
                "origin": "new run",
                "model": source.model,
                "seed": source.seed,
                "selected_step": source.results["scored_step"],
                "stored_development_bleu": source.stored_development_bleu,
                "original_reproduction_difference": difference,
                "parameters": source.results["parameters"],
                "wall_clock_seconds": source.results["seconds"],
                "run_manifest": source.results["manifest"],
                "conditions": [
                    {
                        "condition": "original",
                        "development_bleu": observed,
                        "difference_from_original": 0.0,
                    }
                ],
                "modules": None,
            }
        )
        print(
            f"  {source.model}/seed{source.seed}: "
            f"{source.stored_development_bleu:.4f} -> {observed:.4f} ({difference:+.4f})",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("phase 2: all guards passed; collecting family statistics and hard argmax", flush=True)
    for source, checkpoint in zip(sources, checkpoints, strict=True):
        model = _load_new_model(source, device)
        train_config = source.results["manifest"]["train_config"]
        checkpoint["modules"] = collect_router_statistics(
            model,
            encoded_development,
            int(train_config["decode_batch_size"]),
            device,
            distance_key="nearest_family_distance",
        )
        hard = decode_condition(
            model,
            "hard_argmax",
            corpus,
            development,
            tokenizer,
            device,
            int(train_config["decode_batch_size"]),
            int(train_config["beam_size"]),
        )
        original = checkpoint["conditions"][0]["development_bleu"]
        checkpoint["conditions"].append(
            {
                "condition": "hard_argmax",
                "development_bleu": hard,
                "difference_from_original": hard - original,
            }
        )
        print(
            f"  {source.model}/seed{source.seed}/hard_argmax: {hard:.4f} ({hard - original:+.4f})",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    artifact = build_artifact(
        checkpoints,
        references,
        provenance,
        preflight,
        git_commit(),
        experiment_commit,
    )
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(markdown_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report(artifact))
    return artifact


validate_preregistered_families()
