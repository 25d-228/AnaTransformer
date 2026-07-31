"""Development-only inference diagnostic for the shared V4 permutation core."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
from torch import Tensor

from ana.config import ModelConfig
from ana.d4_diagnostic import (
    REPRODUCTION_TOLERANCE,
    labeled_encoder_permutation_roles,
    require_reproduction,
)
from ana.d4_diagnostic import (
    validate_artifact as validate_d4_artifact,
)
from ana.data.corpora import build_corpus
from ana.data.tokenizer import Tokenizer
from ana.experiment import encode_split, git_commit, score_split
from ana.model import Seq2SeqTransformer
from ana.nn.grouping import (
    D4_PERMUTATIONS,
    GROUP_SIZE,
    N_PERMUTATIONS,
    PERMUTATION_FAMILIES,
    V4_CORE,
    Permutation,
    PermutationFamily,
    permutation_matrices,
)
from ana.nn.roles import (
    FAMILY_SUBSET_INTERVENTIONS,
    D4Intervention,
    D4RoleTransform,
    family_subset_masks,
    temporary_family_subset_intervention,
)
from ana.permutation_family import (
    FAMILIES,
    FAMILY_LABELS,
    FAMILY_MODELS,
    SEEDS,
)
from ana.permutation_family import (
    STUDY_ID as FAMILY_STUDY_ID,
)
from ana.permutation_family import (
    validate_artifact as validate_family_artifact,
)
from ana.registry import build_model
from ana.trainer import fixed_batches

STUDY_ID = "multi30k_v4_core_diagnostic_v1"
CORPUS = "multi30k"
SPLIT = "dev"
D4_SOURCE_STUDY = "multi30k_factorial_v1"
D4_DIAGNOSTIC_STUDY = "multi30k_d4_checkpoint_diagnostic_v1"
CONDITIONS: tuple[D4Intervention, ...] = FAMILY_SUBSET_INTERVENTIONS
REUSED_CONDITIONS = ("original", "all_family_hard")
ROLES = ("Q", "K", "V")
SUBSET_MASS_EPSILON = 1e-12
GATE_INITIAL_LOGIT = -2.0

CONTRASTS: dict[str, tuple[str, str]] = {
    "core_hard_minus_all_family_hard": ("core_hard", "all_family_hard"),
    "core_soft_minus_original": ("core_soft", "original"),
    "core_hard_minus_noncore_hard": ("core_hard", "noncore_hard"),
    "core_soft_minus_noncore_soft": ("core_soft", "noncore_soft"),
    "core_soft_minus_core_uniform": ("core_soft", "core_uniform"),
}


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _require_finite_numbers(value: Any, path: str = "artifact") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite_numbers(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite_numbers(nested, f"{path}[{index}]")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def core_definition() -> dict[str, Any]:
    return {
        "router_order": [list(permutation) for permutation in V4_CORE],
        "sha256": _sha256_json(V4_CORE),
    }


def _permutation_from_matrix(matrix: Tensor) -> Permutation:
    cpu = matrix.detach().to(device="cpu")
    expected_shape = (GROUP_SIZE, GROUP_SIZE)
    if tuple(cpu.shape) != expected_shape:
        raise ValueError(
            f"permutation matrix has shape {tuple(cpu.shape)}, expected {expected_shape}"
        )
    if not torch.equal(cpu.sum(dim=0), torch.ones(GROUP_SIZE, dtype=cpu.dtype)) or not torch.equal(
        cpu.sum(dim=1), torch.ones(GROUP_SIZE, dtype=cpu.dtype)
    ):
        raise ValueError("a family buffer row is not a permutation matrix")
    if not torch.equal(cpu, (cpu == 1).to(cpu.dtype)):
        raise ValueError("a family buffer row is not exactly binary")
    values = tuple(int(value) for value in cpu.argmax(dim=1).tolist())
    return values  # type: ignore[return-value]


def family_from_buffer(permutations: Tensor) -> PermutationFamily:
    expected_shape = (N_PERMUTATIONS, GROUP_SIZE, GROUP_SIZE)
    if tuple(permutations.shape) != expected_shape:
        raise ValueError(
            f"permutation buffer has shape {tuple(permutations.shape)}, expected {expected_shape}"
        )
    family = tuple(_permutation_from_matrix(matrix) for matrix in permutations)
    if len(set(family)) != N_PERMUTATIONS:
        raise ValueError("the active family buffer does not contain eight distinct permutations")
    return family


def partition_from_buffer(permutations: Tensor) -> dict[str, Any]:
    """Describe a family partition, deriving every index from the live buffer."""
    core_mask, noncore_mask = family_subset_masks(permutations)
    family = family_from_buffer(permutations)
    core_indices = [family.index(permutation) for permutation in V4_CORE]
    noncore_indices = [
        index for index, selected in enumerate(noncore_mask.tolist()) if bool(selected)
    ]
    if [bool(core_mask[index]) for index in core_indices] != [True] * len(V4_CORE):
        raise ValueError("tuple-derived core indices disagree with the live-buffer mask")
    return {
        "family_router_order": [list(permutation) for permutation in family],
        "core_router_indices_in_v4_order": core_indices,
        "noncore_router_indices_in_family_order": noncore_indices,
        "noncore_router_order": [list(family[index]) for index in noncore_indices],
    }


def family_partitions() -> list[dict[str, Any]]:
    rows = []
    for model in FAMILY_MODELS:
        family = FAMILIES[model]
        partition = partition_from_buffer(permutation_matrices(family))
        rows.append({"model": model, "label": FAMILY_LABELS[model], **partition})
    return rows


def validate_family_partitions() -> None:
    if FAMILIES != PERMUTATION_FAMILIES:
        raise ValueError("the diagnostic and model family registries differ")
    core = set(V4_CORE)
    if len(V4_CORE) != 4 or len(core) != 4:
        raise ValueError("V4_CORE must contain four distinct tuples")
    for model in FAMILY_MODELS:
        family = FAMILIES[model]
        if len(family) != N_PERMUTATIONS or len(set(family)) != N_PERMUTATIONS:
            raise ValueError(f"{model} does not contain eight distinct permutations")
        if not core <= set(family) or len(set(family) - core) != 4:
            raise ValueError(f"{model} does not partition into four core and four non-core tuples")
        partition_from_buffer(permutation_matrices(family))


def centroid_metadata() -> list[dict[str, Any]]:
    reference = permutation_matrices(D4_PERMUTATIONS).to(torch.float64).mean(dim=0)
    identity = torch.eye(GROUP_SIZE, dtype=torch.float64)
    gate = float(torch.sigmoid(torch.tensor(GATE_INITIAL_LOGIT, dtype=torch.float64)))
    rows = []
    for model in FAMILY_MODELS:
        centroid = permutation_matrices(FAMILIES[model]).to(torch.float64).mean(dim=0)
        effective = identity + gate * (centroid - identity)
        rows.append(
            {
                "model": model,
                "uniform_router_centroid": centroid.tolist(),
                "frobenius_distance_from_d4_all_quarters": float(
                    torch.linalg.matrix_norm(centroid - reference)
                ),
                "singular_values": torch.linalg.svdvals(centroid).tolist(),
                "initial_gate_strength": gate,
                "initial_effective_residual_matrix_before_diagonal_scale": effective.tolist(),
            }
        )
    return rows


@dataclass(frozen=True)
class CoreCheckpointSource:
    model: str
    seed: int
    folder: str
    results: dict[str, Any]
    weights_metadata: dict[str, Any]
    source_study: str
    source_experiment_commit: str
    artifact_reference: dict[str, Any]

    @property
    def weights_path(self) -> str:
        return os.path.join(self.folder, "weights.pt")

    @property
    def stored_development_bleu(self) -> float:
        return float(self.artifact_reference["original_development_bleu"])

    @property
    def selected_step(self) -> int:
        return int(self.results["scored_step"])


def _load_weights(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def _artifact_references(
    d4_artifact_path: str,
    family_artifact_path: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any], dict[str, Any]]:
    d4 = _load_json(d4_artifact_path)
    family = _load_json(family_artifact_path)
    validate_d4_artifact(d4)
    validate_family_artifact(family)
    if family.get("source_artifacts", {}).get("d4_diagnostic", {}).get(
        "analysis_git_commit"
    ) != d4.get("analysis_git_commit"):
        raise ValueError("the family screen does not reference the supplied D4 diagnostic")

    references: dict[tuple[str, int], dict[str, Any]] = {}
    d4_by_cell = {
        (row["model"], row["seed"]): row
        for row in d4["checkpoints"]
        if row["model"] == "ana_d4_enc"
    }
    family_by_cell = {(row["model"], row["seed"]): row for row in family["new_runs"]}
    for model, seed in product(FAMILY_MODELS, SEEDS):
        if model == "ana_d4_enc":
            row = d4_by_cell[(model, seed)]
            hard = next(item for item in row["interventions"] if item["condition"] == "hard_argmax")
            references[(model, seed)] = {
                "original_development_bleu": row["original_reproduction_bleu"],
                "all_family_hard_development_bleu": hard["development_bleu"],
                "selected_step": row["selected_step"],
                "modules": copy.deepcopy(row["modules"]),
                "source_study": D4_SOURCE_STUDY,
                "source_experiment_commit": d4["source_experiment_commit"],
                "source_analysis_study": D4_DIAGNOSTIC_STUDY,
                "source_analysis_commit": d4["analysis_git_commit"],
                "run_manifest": None,
            }
        else:
            row = family_by_cell[(model, seed)]
            original = next(item for item in row["conditions"] if item["condition"] == "original")
            hard = next(item for item in row["conditions"] if item["condition"] == "hard_argmax")
            references[(model, seed)] = {
                "original_development_bleu": original["development_bleu"],
                "all_family_hard_development_bleu": hard["development_bleu"],
                "selected_step": row["selected_step"],
                "modules": copy.deepcopy(row["modules"]),
                "source_study": FAMILY_STUDY_ID,
                "source_experiment_commit": family["experiment_git_commit"],
                "source_analysis_study": FAMILY_STUDY_ID,
                "source_analysis_commit": family["analysis_git_commit"],
                "run_manifest": copy.deepcopy(row["run_manifest"]),
            }
    if set(references) != set(product(FAMILY_MODELS, SEEDS)):
        raise ValueError("source artifacts do not contain exactly the 12 required cells")
    return references, d4, family


def _validate_source_folder(
    model: str,
    seed: int,
    run_dir: str,
    reference: dict[str, Any],
    expected_model_config: dict[str, Any],
) -> CoreCheckpointSource:
    folder = os.path.join(run_dir, f"{CORPUS}_{model}_seed{seed}")
    results_path = os.path.join(folder, "results.json")
    weights_path = os.path.join(folder, "weights.pt")
    if not os.path.isfile(results_path) or not os.path.isfile(weights_path):
        raise ValueError(f"missing checkpoint or results for {model}/seed{seed}")
    with open(results_path, encoding="utf-8") as handle:
        record = json.load(handle)
    manifest = record.get("manifest", {})
    label = f"{model}/seed{seed}"
    if record.get("model") != model or manifest.get("seed") != seed:
        raise ValueError(f"{label} has mismatched model or seed provenance")
    if record.get("corpus") != CORPUS or manifest.get("study_id") != reference["source_study"]:
        raise ValueError(f"{label} has stale corpus or source-study provenance")
    if manifest.get("seeded_before_model_init") is not True or manifest.get("smoke") is not False:
        raise ValueError(f"{label} lacks valid pre-construction full-run provenance")
    if manifest.get("git_commit") != reference["source_experiment_commit"]:
        raise ValueError(f"{label} has a stale source experiment commit")
    if manifest.get("model_config") != expected_model_config:
        raise ValueError(f"{label} has a mismatched model configuration")
    if record.get("scored_step") != reference["selected_step"]:
        raise ValueError(f"{label} has a mismatched selected checkpoint step")
    if not math.isclose(
        float(record.get("scores", {}).get("dev", math.nan)),
        float(reference["original_development_bleu"]),
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} development BLEU differs from the committed artifact")
    if model != "ana_d4_enc" and (
        manifest.get("evaluated_splits") != [SPLIT] or set(record.get("scores", {})) != {SPLIT}
    ):
        raise ValueError(f"{label} control source is not strictly development-only")

    metadata = _load_weights(weights_path)
    if metadata.get("model") != model or metadata.get("corpus") != CORPUS:
        raise ValueError(f"{label} checkpoint metadata does not match its source record")
    if metadata.get("model_config") != expected_model_config:
        raise ValueError(f"{label} checkpoint has a mismatched model configuration")
    if metadata.get("step") != reference["selected_step"]:
        raise ValueError(f"{label} checkpoint metadata has a mismatched selected step")
    if set(metadata) != {
        "state_dict",
        "model",
        "corpus",
        "model_config",
        "step",
        "dev_loss",
    }:
        raise ValueError(f"{label} checkpoint has unexpected or incomplete metadata")
    return CoreCheckpointSource(
        model=model,
        seed=seed,
        folder=folder,
        results=record,
        weights_metadata=metadata,
        source_study=reference["source_study"],
        source_experiment_commit=reference["source_experiment_commit"],
        artifact_reference=reference,
    )


def load_checkpoint_sources(
    d4_run_dir: str,
    family_run_dir: str,
    d4_artifact_path: str,
    family_artifact_path: str,
) -> tuple[list[CoreCheckpointSource], dict[str, Any]]:
    """Resolve and cross-check exactly 12 checkpoints against both committed diagnostics."""
    references, d4_artifact, family_artifact = _artifact_references(
        d4_artifact_path,
        family_artifact_path,
    )
    first_control = family_artifact["new_runs"][0]
    expected_model_config = first_control["run_manifest"]["model_config"]
    sources = []
    for model, seed in product(FAMILY_MODELS, SEEDS):
        root = d4_run_dir if model == "ana_d4_enc" else family_run_dir
        source = _validate_source_folder(
            model,
            seed,
            root,
            references[(model, seed)],
            expected_model_config,
        )
        if (
            source.artifact_reference["run_manifest"] is not None
            and source.results["manifest"] != source.artifact_reference["run_manifest"]
        ):
            raise ValueError(f"{model}/seed{seed} manifest differs from the committed artifact")
        sources.append(source)
    if len(sources) != 12:
        raise ValueError("the V4-core diagnostic must resolve exactly 12 checkpoints")
    return sources, {
        "d4_diagnostic": {
            "path": d4_artifact_path,
            "study_id": d4_artifact["study_id"],
            "analysis_git_commit": d4_artifact["analysis_git_commit"],
            "source_experiment_commit": d4_artifact["source_experiment_commit"],
        },
        "permutation_family": {
            "path": family_artifact_path,
            "study_id": family_artifact["study_id"],
            "analysis_git_commit": family_artifact["analysis_git_commit"],
            "experiment_git_commit": family_artifact["experiment_git_commit"],
        },
    }


def load_model(source: CoreCheckpointSource, device: torch.device) -> Seq2SeqTransformer:
    config = ModelConfig(**source.weights_metadata["model_config"])
    model = build_model(source.model, config)
    checkpoint = _load_weights(source.weights_path, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _entropy(probabilities: Tensor) -> Tensor:
    return -(
        probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=-1)


class CoreUsageAccumulator:
    """Collect conditional core/complement statistics for one routed role."""

    def __init__(self, permutations: Tensor) -> None:
        partition = partition_from_buffer(permutations)
        self.core_indices = partition["core_router_indices_in_v4_order"]
        self.noncore_indices = partition["noncore_router_indices_in_family_order"]
        self.noncore_order = partition["noncore_router_order"]
        self.count = 0
        self.argmax_core_count = 0
        self.core_masses: list[Tensor] = []
        self.core_conditioned_count = 0
        self.noncore_conditioned_count = 0
        self.sum_core_distribution = torch.zeros(4, dtype=torch.float64)
        self.sum_noncore_distribution = torch.zeros(4, dtype=torch.float64)
        self.sum_core_conditional_entropy = 0.0
        self.sum_noncore_conditional_entropy = 0.0

    def _add_conditional(
        self,
        selected: Tensor,
        indices: list[int],
        *,
        core: bool,
    ) -> None:
        subset = selected[:, indices]
        mass = subset.sum(dim=-1)
        valid = mass > SUBSET_MASS_EPSILON
        if not bool(valid.any()):
            return
        conditioned = subset[valid] / mass[valid, None]
        conditioned_cpu = conditioned.detach().to(device="cpu", dtype=torch.float64)
        entropy = _entropy(conditioned)
        if core:
            self.core_conditioned_count += conditioned.size(0)
            self.sum_core_distribution += conditioned_cpu.sum(dim=0)
            self.sum_core_conditional_entropy += float(entropy.sum())
        else:
            self.noncore_conditioned_count += conditioned.size(0)
            self.sum_noncore_distribution += conditioned_cpu.sum(dim=0)
            self.sum_noncore_conditional_entropy += float(entropy.sum())

    def add(self, probabilities: Tensor, real_mask: Tensor) -> None:
        selected = probabilities[real_mask.bool()]
        if selected.numel() == 0:
            return
        if tuple(selected.shape[1:]) != (N_PERMUTATIONS,):
            raise ValueError("learned router probabilities do not have eight routes")
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("learned router probabilities contain non-finite values")

        core_mass = selected[:, self.core_indices].sum(dim=-1)
        family_core_mask = torch.zeros(
            N_PERMUTATIONS,
            dtype=torch.bool,
            device=selected.device,
        )
        family_core_mask[self.core_indices] = True
        self.count += selected.size(0)
        self.argmax_core_count += int(family_core_mask[selected.argmax(dim=-1)].sum())
        self.core_masses.append(core_mass.detach().to(device="cpu", dtype=torch.float64))
        self._add_conditional(selected, self.core_indices, core=True)
        self._add_conditional(selected, self.noncore_indices, core=False)

    def finish(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("core-usage statistics contain no real source tokens")
        masses = torch.cat(self.core_masses)
        quantiles = torch.quantile(
            masses,
            torch.tensor([0.05, 0.50, 0.95], dtype=masses.dtype),
        )
        core_distribution = (
            self.sum_core_distribution / self.core_conditioned_count
            if self.core_conditioned_count
            else torch.zeros(4, dtype=torch.float64)
        )
        noncore_distribution = (
            self.sum_noncore_distribution / self.noncore_conditioned_count
            if self.noncore_conditioned_count
            else torch.zeros(4, dtype=torch.float64)
        )
        mean_core = float(masses.mean())
        return {
            "real_token_count": self.count,
            "mean_core_probability_mass": mean_core,
            "core_probability_mass_p05": float(quantiles[0]),
            "core_probability_mass_p50": float(quantiles[1]),
            "core_probability_mass_p95": float(quantiles[2]),
            "mean_noncore_probability_mass": 1.0 - mean_core,
            "all_family_argmax_core_frequency": self.argmax_core_count / self.count,
            "core_route_distribution_v4_order": core_distribution.tolist(),
            "noncore_router_order": copy.deepcopy(self.noncore_order),
            "noncore_route_distribution_family_order": noncore_distribution.tolist(),
            "normalized_conditional_core_entropy": (
                self.sum_core_conditional_entropy / self.core_conditioned_count / math.log(4)
                if self.core_conditioned_count
                else 0.0
            ),
            "normalized_conditional_noncore_entropy": (
                self.sum_noncore_conditional_entropy / self.noncore_conditioned_count / math.log(4)
                if self.noncore_conditioned_count
                else 0.0
            ),
            "core_conditioned_token_count": self.core_conditioned_count,
            "noncore_conditioned_token_count": self.noncore_conditioned_count,
        }


class CoreUsageCollector:
    def __init__(self, model: Seq2SeqTransformer) -> None:
        self.labeled = labeled_encoder_permutation_roles(model)
        self.accumulators = {
            (layer, role): CoreUsageAccumulator(module.permutations)
            for layer, role, module in self.labeled
        }
        self.handles: list[Any] = []

    def _hook(
        self,
        layer: int,
        role: str,
        module: D4RoleTransform,
        inputs: tuple[Any, ...],
    ) -> None:
        z, pad_mask = inputs
        readout = module.grouping.router_readout(z, pad_mask)
        probabilities = module.learned_route_probabilities(readout)
        self.accumulators[(layer, role)].add(probabilities, pad_mask)

    def __enter__(self) -> CoreUsageCollector:
        for layer, role, module in self.labeled:
            self.handles.append(
                module.register_forward_pre_hook(
                    lambda held, inputs, layer=layer, role=role: self._hook(
                        layer,
                        role,
                        held,
                        inputs,
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish(self) -> list[dict[str, Any]]:
        return [
            {
                "layer": layer,
                "role": role,
                **self.accumulators[(layer, role)].finish(),
            }
            for layer, role, _ in self.labeled
        ]


@torch.no_grad()
def collect_core_usage(
    model: Seq2SeqTransformer,
    encoded_development: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    batches = fixed_batches(encoded_development, batch_size, model.config.pad_id)
    collector = CoreUsageCollector(model)
    with collector:
        for batch in batches:
            model.encode(batch.source_ids.to(device), batch.source_mask.to(device))
    return collector.finish()


@torch.no_grad()
def decode_subset_condition(
    model: Seq2SeqTransformer,
    condition: D4Intervention,
    corpus,
    examples,
    tokenizer: Tokenizer,
    device: torch.device,
    batch_size: int,
    beam_size: int,
) -> float:
    model.eval()
    with temporary_family_subset_intervention(model, condition):
        score, _ = score_split(
            model,
            corpus,
            examples,
            tokenizer,
            device,
            batch_size,
            beam_size,
        )
    if not math.isfinite(score):
        raise ValueError(f"{condition} produced a non-finite development BLEU")
    return score


@torch.no_grad()
def decode_original(
    model: Seq2SeqTransformer,
    corpus,
    examples,
    tokenizer: Tokenizer,
    device: torch.device,
    batch_size: int,
    beam_size: int,
) -> float:
    model.eval()
    score, _ = score_split(
        model,
        corpus,
        examples,
        tokenizer,
        device,
        batch_size,
        beam_size,
    )
    if not math.isfinite(score):
        raise ValueError("original reproduction produced a non-finite development BLEU")
    return score


CORE_USAGE_SCALARS = (
    "mean_core_probability_mass",
    "core_probability_mass_p05",
    "core_probability_mass_p50",
    "core_probability_mass_p95",
    "mean_noncore_probability_mass",
    "all_family_argmax_core_frequency",
    "normalized_conditional_core_entropy",
    "normalized_conditional_noncore_entropy",
)


def _mean(values: Iterable[float]) -> float:
    return statistics.mean(values)


def _sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_core_usage(rows: list[dict[str, Any]], include_noncore: bool = True) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty set of core-usage rows")
    summary = {key: _mean(float(row[key]) for row in rows) for key in CORE_USAGE_SCALARS}
    summary.update(
        {
            "module_row_count": len(rows),
            "real_token_count_sum": sum(int(row["real_token_count"]) for row in rows),
            "core_conditioned_token_count_sum": sum(
                int(row["core_conditioned_token_count"]) for row in rows
            ),
            "noncore_conditioned_token_count_sum": sum(
                int(row["noncore_conditioned_token_count"]) for row in rows
            ),
            "core_route_distribution_v4_order": [
                _mean(float(row["core_route_distribution_v4_order"][index]) for row in rows)
                for index in range(4)
            ],
        }
    )
    if include_noncore:
        orders = {json.dumps(row["noncore_router_order"]) for row in rows}
        if len(orders) != 1:
            raise ValueError("cannot aggregate non-core routes with different tuple orderings")
        summary["noncore_router_order"] = copy.deepcopy(rows[0]["noncore_router_order"])
        summary["noncore_route_distribution_family_order"] = [
            _mean(float(row["noncore_route_distribution_family_order"][index]) for row in rows)
            for index in range(4)
        ]
    return summary


def aggregate_core_usage(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint = []
    for checkpoint in checkpoints:
        by_checkpoint.append(
            {
                "model": checkpoint["model"],
                "seed": checkpoint["seed"],
                **_mean_core_usage(checkpoint["core_usage"]),
            }
        )

    by_family_layer_role = []
    for model, layer, role in product(FAMILY_MODELS, range(1, 5), ROLES):
        rows = [
            next(
                module
                for module in checkpoint["core_usage"]
                if module["layer"] == layer and module["role"] == role
            )
            for checkpoint in checkpoints
            if checkpoint["model"] == model
        ]
        by_family_layer_role.append(
            {
                "model": model,
                "layer": layer,
                "role": role,
                "seeds": list(SEEDS),
                **_mean_core_usage(rows),
            }
        )

    by_family = []
    for model in FAMILY_MODELS:
        rows = [
            module
            for checkpoint in checkpoints
            if checkpoint["model"] == model
            for module in checkpoint["core_usage"]
        ]
        by_family.append({"model": model, **_mean_core_usage(rows)})

    by_layer_role = []
    for layer, role in product(range(1, 5), ROLES):
        rows = [
            module
            for checkpoint in checkpoints
            for module in checkpoint["core_usage"]
            if module["layer"] == layer and module["role"] == role
        ]
        by_layer_role.append(
            {
                "layer": layer,
                "role": role,
                **_mean_core_usage(rows, include_noncore=False),
            }
        )

    every = [module for checkpoint in checkpoints for module in checkpoint["core_usage"]]
    return {
        "by_checkpoint": by_checkpoint,
        "by_family_layer_role": by_family_layer_role,
        "by_family": by_family,
        "by_layer_role_across_all_checkpoints": by_layer_role,
        "across_all_12_checkpoints": _mean_core_usage(every, include_noncore=False),
    }


def _score(checkpoint: dict[str, Any], condition: str) -> float:
    if condition == "original":
        return float(checkpoint["reused_references"][0]["development_bleu"])
    if condition == "all_family_hard":
        return float(checkpoint["reused_references"][1]["development_bleu"])
    return float(
        next(
            row["development_bleu"]
            for row in checkpoint["new_interventions"]
            if row["condition"] == condition
        )
    )


def aggregate_scores(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint_contrasts = []
    for checkpoint in checkpoints:
        values = {
            name: _score(checkpoint, left) - _score(checkpoint, right)
            for name, (left, right) in CONTRASTS.items()
        }
        checkpoint_contrasts.append(
            {"model": checkpoint["model"], "seed": checkpoint["seed"], **values}
        )

    reported_conditions = ("original", "all_family_hard", *CONDITIONS)
    family_condition_summaries = []
    for model in FAMILY_MODELS:
        rows = [row for row in checkpoints if row["model"] == model]
        for condition in reported_conditions:
            scores = [_score(row, condition) for row in rows]
            originals = [_score(row, "original") for row in rows]
            family_condition_summaries.append(
                {
                    "model": model,
                    "condition": condition,
                    "mean_development_bleu": _mean(scores),
                    "sample_standard_deviation_development_bleu": _sample_sd(scores),
                    "paired_development_bleu": scores,
                    "paired_difference_from_original": [
                        score - original for score, original in zip(scores, originals, strict=True)
                    ],
                    "mean_difference_from_original": _mean(
                        score - original for score, original in zip(scores, originals, strict=True)
                    ),
                }
            )

    family_contrasts = []
    for model in FAMILY_MODELS:
        rows = [row for row in checkpoint_contrasts if row["model"] == model]
        family_contrasts.append(
            {
                "model": model,
                "seeds": list(SEEDS),
                **{
                    name: {
                        "paired_values": [float(row[name]) for row in rows],
                        "mean": _mean(float(row[name]) for row in rows),
                    }
                    for name in CONTRASTS
                },
            }
        )

    aggregate_contrasts = {
        name: {
            "paired_values": [float(row[name]) for row in checkpoint_contrasts],
            "mean": _mean(float(row[name]) for row in checkpoint_contrasts),
        }
        for name in CONTRASTS
    }
    family_favorable = sum(
        row["core_hard_minus_noncore_hard"]["mean"] > 0 for row in family_contrasts
    )
    core_hard_all = aggregate_contrasts["core_hard_minus_all_family_hard"]["mean"]
    core_soft_original = aggregate_contrasts["core_soft_minus_original"]["mean"]
    core_hard_noncore = aggregate_contrasts["core_hard_minus_noncore_hard"]["mean"]
    core_soft_noncore = aggregate_contrasts["core_soft_minus_noncore_soft"]["mean"]

    all_hard_original = _mean(
        _score(row, "all_family_hard") - _score(row, "original") for row in checkpoints
    )
    noncore_soft_original = _mean(
        _score(row, "noncore_soft") - _score(row, "original") for row in checkpoints
    )
    noncore_hard_all = _mean(
        _score(row, "noncore_hard") - _score(row, "all_family_hard") for row in checkpoints
    )
    sufficient = (
        core_hard_all >= -0.15
        and core_soft_original >= -0.20
        and core_hard_noncore >= 0.30
        and family_favorable >= 3
    )
    interchangeable = (
        abs(core_hard_noncore) <= 0.20
        and abs(core_soft_noncore) <= 0.20
        and core_soft_original >= -0.20
        and noncore_soft_original >= -0.20
        and core_hard_all >= -0.20
        and noncore_hard_all >= -0.20
    )
    full_coverage = (
        core_soft_original < -0.50 and noncore_soft_original < -0.50 and all_hard_original >= -0.15
    )
    noncore_stronger = (
        core_hard_noncore < 0
        and core_soft_noncore < 0
        and sum(row["core_hard_minus_noncore_hard"]["mean"] < 0 for row in family_contrasts) >= 3
    )
    if sufficient:
        decision = "common_v4_core_is_sufficient"
    elif interchangeable:
        decision = "core_and_noncore_are_interchangeable"
    elif full_coverage:
        decision = "full_family_coverage_is_necessary"
    elif noncore_stronger:
        decision = "noncore_is_stronger"
    else:
        decision = "mixed_or_inconclusive"

    return {
        "family_condition_summaries": family_condition_summaries,
        "checkpoint_contrasts": checkpoint_contrasts,
        "family_contrasts": family_contrasts,
        "aggregate_contrasts": aggregate_contrasts,
        "supporting_means": {
            "all_family_hard_minus_original": all_hard_original,
            "noncore_soft_minus_original": noncore_soft_original,
            "noncore_hard_minus_all_family_hard": noncore_hard_all,
            "families_with_favorable_core_hard_comparison": family_favorable,
        },
        "screening_decision": decision,
    }


def aggregate_artifact(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "core_usage": aggregate_core_usage(checkpoints),
        "scores": aggregate_scores(checkpoints),
    }


def _source_provenance(source: CoreCheckpointSource) -> dict[str, Any]:
    manifest = source.results["manifest"]
    return {
        "source_study": source.source_study,
        "source_experiment_commit": source.source_experiment_commit,
        "source_analysis_study": source.artifact_reference["source_analysis_study"],
        "source_analysis_commit": source.artifact_reference["source_analysis_commit"],
        "selected_step": source.selected_step,
        "model_config": copy.deepcopy(source.weights_metadata["model_config"]),
        "seeded_before_model_init": manifest["seeded_before_model_init"],
        "smoke": manifest["smoke"],
    }


def build_artifact(
    checkpoints: list[dict[str, Any]],
    source_artifacts: dict[str, Any],
    analysis_git_commit: str,
) -> dict[str, Any]:
    ordered = sorted(
        checkpoints,
        key=lambda row: (FAMILY_MODELS.index(row["model"]), SEEDS.index(row["seed"])),
    )
    artifact = {
        "study_id": STUDY_ID,
        "analysis_git_commit": analysis_git_commit,
        "corpus": CORPUS,
        "evaluated_splits": [SPLIT],
        "test_split_evaluated": False,
        "models": list(FAMILY_MODELS),
        "seeds": list(SEEDS),
        "reproduction_tolerance_bleu": REPRODUCTION_TOLERANCE,
        "core_definition": core_definition(),
        "conditional_subset_convention": {
            "epsilon": SUBSET_MASS_EPSILON,
            "rule": (
                "token-subset pairs with subset probability mass <= epsilon are excluded from "
                "conditional route distributions and conditional entropy; if none remain, the "
                "reported distribution is four zeros and normalized entropy is zero"
            ),
        },
        "family_partitions": family_partitions(),
        "initial_function_centroids": centroid_metadata(),
        "source_artifacts": source_artifacts,
        "execution_guard": {
            "all_original_reproductions_completed_before_interventions": True,
            "original_reproduction_decode_count": len(ordered),
            "new_intervention_decode_count": sum(len(row["new_interventions"]) for row in ordered),
            "all_family_hard_decode_count": 0,
        },
        "checkpoints": ordered,
        "aggregates": aggregate_artifact(ordered),
    }
    validate_artifact(artifact)
    return artifact


def _validate_router_rows(rows: Any, label: str) -> None:
    expected = set(product(range(1, 5), ROLES))
    observed = {(row.get("layer"), row.get("role")) for row in rows or []}
    if len(rows or []) != 12 or observed != expected:
        raise ValueError(f"{label} must contain 12 labeled router-statistics rows")
    for row in rows:
        if row.get("real_token_count", 0) <= 0:
            raise ValueError(f"{label} contains an empty router-statistics row")
        if len(row.get("mean_route_distribution", [])) != N_PERMUTATIONS:
            raise ValueError(f"{label} contains an incomplete route distribution")


def _validate_core_usage_rows(
    rows: Any,
    model: str,
    label: str,
) -> None:
    expected = set(product(range(1, 5), ROLES))
    observed = {(row.get("layer"), row.get("role")) for row in rows or []}
    if len(rows or []) != 12 or observed != expected:
        raise ValueError(f"{label} must contain 12 labeled core-usage rows")
    partition = next(row for row in family_partitions() if row["model"] == model)
    for row in rows:
        if row.get("real_token_count", 0) <= 0:
            raise ValueError(f"{label} contains an empty core-usage row")
        if row.get("noncore_router_order") != partition["noncore_router_order"]:
            raise ValueError(f"{label} contains a stale non-core tuple ordering")
        core_distribution = row.get("core_route_distribution_v4_order", [])
        noncore_distribution = row.get("noncore_route_distribution_family_order", [])
        if len(core_distribution) != 4 or len(noncore_distribution) != 4:
            raise ValueError(f"{label} contains an incomplete conditional route distribution")
        if row.get("core_conditioned_token_count", 0):
            if not math.isclose(sum(core_distribution), 1.0, abs_tol=1e-8):
                raise ValueError(f"{label} core conditional distribution does not sum to one")
        elif core_distribution != [0.0] * 4:
            raise ValueError(f"{label} has a nonzero undefined core conditional distribution")
        if row.get("noncore_conditioned_token_count", 0):
            if not math.isclose(sum(noncore_distribution), 1.0, abs_tol=1e-8):
                raise ValueError(f"{label} non-core conditional distribution does not sum to one")
        elif noncore_distribution != [0.0] * 4:
            raise ValueError(f"{label} has a nonzero undefined non-core distribution")
        mean_core = row.get("mean_core_probability_mass", math.nan)
        mean_noncore = row.get("mean_noncore_probability_mass", math.nan)
        if not math.isclose(mean_core + mean_noncore, 1.0, abs_tol=1e-10):
            raise ValueError(f"{label} core and non-core probability masses do not sum to one")
        p05 = row.get("core_probability_mass_p05", math.nan)
        p50 = row.get("core_probability_mass_p50", math.nan)
        p95 = row.get("core_probability_mass_p95", math.nan)
        if not 0 <= p05 <= p50 <= p95 <= 1:
            raise ValueError(f"{label} has invalid core-mass quantiles")
        for key in (
            "mean_core_probability_mass",
            "mean_noncore_probability_mass",
            "all_family_argmax_core_frequency",
            "normalized_conditional_core_entropy",
            "normalized_conditional_noncore_entropy",
        ):
            if not 0 <= row.get(key, math.nan) <= 1:
                raise ValueError(f"{label} has an invalid {key}")


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require_finite_numbers(artifact)
    validate_family_partitions()
    if artifact.get("study_id") != STUDY_ID:
        raise ValueError("artifact has the wrong study ID")
    if not artifact.get("analysis_git_commit"):
        raise ValueError("artifact lacks analysis Git provenance")
    if (
        artifact.get("corpus") != CORPUS
        or artifact.get("evaluated_splits") != [SPLIT]
        or artifact.get("test_split_evaluated") is not False
    ):
        raise ValueError("artifact is not strictly Multi30k development-only")
    if artifact.get("models") != list(FAMILY_MODELS) or artifact.get("seeds") != list(SEEDS):
        raise ValueError("artifact has stale model or seed definitions")
    if artifact.get("reproduction_tolerance_bleu") != REPRODUCTION_TOLERANCE:
        raise ValueError("artifact has a stale reproduction tolerance")
    if artifact.get("core_definition") != core_definition():
        raise ValueError("artifact has a stale V4 core definition or hash")
    if artifact.get("family_partitions") != family_partitions():
        raise ValueError("artifact has stale family partitions")
    if artifact.get("initial_function_centroids") != centroid_metadata():
        raise ValueError("artifact has stale initial-function centroid metadata")
    convention = artifact.get("conditional_subset_convention", {})
    if convention.get("epsilon") != SUBSET_MASS_EPSILON or not convention.get("rule"):
        raise ValueError("artifact lacks the fixed negligible-subset convention")

    sources = artifact.get("source_artifacts", {})
    d4 = sources.get("d4_diagnostic", {})
    family = sources.get("permutation_family", {})
    if (
        d4.get("study_id") != D4_DIAGNOSTIC_STUDY
        or family.get("study_id") != FAMILY_STUDY_ID
        or not d4.get("analysis_git_commit")
        or not d4.get("source_experiment_commit")
        or not family.get("analysis_git_commit")
        or not family.get("experiment_git_commit")
    ):
        raise ValueError("artifact has missing or stale source-artifact provenance")

    checkpoints = artifact.get("checkpoints", [])
    expected_cells = set(product(FAMILY_MODELS, SEEDS))
    observed_cells = {(row.get("model"), row.get("seed")) for row in checkpoints}
    if len(checkpoints) != 12 or observed_cells != expected_cells:
        raise ValueError("artifact must contain exactly the 12 prescribed checkpoints")

    for checkpoint in checkpoints:
        model = checkpoint["model"]
        seed = checkpoint["seed"]
        label = f"{model}/seed{seed}"
        provenance = checkpoint.get("source_checkpoint", {})
        expected_study = D4_SOURCE_STUDY if model == "ana_d4_enc" else FAMILY_STUDY_ID
        expected_commit = (
            d4["source_experiment_commit"]
            if model == "ana_d4_enc"
            else family["experiment_git_commit"]
        )
        if (
            provenance.get("source_study") != expected_study
            or provenance.get("source_experiment_commit") != expected_commit
            or provenance.get("seeded_before_model_init") is not True
            or provenance.get("smoke") is not False
            or provenance.get("selected_step") != checkpoint["reproduction"]["selected_step"]
            or not provenance.get("model_config")
        ):
            raise ValueError(f"{label} has stale or incomplete checkpoint provenance")

        reproduction = checkpoint.get("reproduction", {})
        if abs(reproduction.get("difference", math.inf)) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"{label} fails the original-score reproduction guard")
        if not math.isclose(
            reproduction["difference"],
            reproduction["reproduced_development_bleu"] - reproduction["stored_development_bleu"],
            abs_tol=1e-10,
        ):
            raise ValueError(f"{label} has an incorrect reproduction difference")

        reused = checkpoint.get("reused_references", [])
        if [row.get("condition") for row in reused] != list(REUSED_CONDITIONS):
            raise ValueError(f"{label} has missing or reordered reused references")
        if any(row.get("origin") != "reused reference" for row in reused):
            raise ValueError(f"{label} has an incorrectly labeled reused reference")
        if not math.isclose(
            reused[0]["development_bleu"],
            reproduction["stored_development_bleu"],
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} original reference differs from the reproduction source")

        interventions = checkpoint.get("new_interventions", [])
        if [row.get("condition") for row in interventions] != list(CONDITIONS):
            raise ValueError(f"{label} has missing or reordered subset interventions")
        original = reused[0]["development_bleu"]
        for row in interventions:
            if row.get("origin") != "new intervention":
                raise ValueError(f"{label}/{row.get('condition')} has the wrong origin label")
            if not math.isclose(
                row["difference_from_original"],
                row["development_bleu"] - original,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{label}/{row['condition']} has an incorrect paired difference")

        hashes = checkpoint.get("state_dict_sha256", {})
        if len(hashes.get("before", "")) != 64 or hashes.get("before") != hashes.get("after"):
            raise ValueError(f"{label} state dictionary changed during intervention evaluation")
        _validate_router_rows(checkpoint.get("existing_router_statistics"), label)
        _validate_core_usage_rows(checkpoint.get("core_usage"), model, label)

    execution = artifact.get("execution_guard", {})
    if execution != {
        "all_original_reproductions_completed_before_interventions": True,
        "original_reproduction_decode_count": 12,
        "new_intervention_decode_count": 60,
        "all_family_hard_decode_count": 0,
    }:
        raise ValueError("artifact does not prove the required 12-then-60 execution order")
    if artifact.get("aggregates") != aggregate_artifact(checkpoints):
        raise ValueError("artifact aggregates do not match the individual checkpoint rows")


def _format_matrix(matrix: list[list[float]]) -> str:
    return "[" + "; ".join(", ".join(f"{value:.3f}" for value in row) for row in matrix) + "]"


def _family_condition(
    artifact: dict[str, Any],
    model: str,
    condition: str,
) -> dict[str, Any]:
    return next(
        row
        for row in artifact["aggregates"]["scores"]["family_condition_summaries"]
        if row["model"] == model and row["condition"] == condition
    )


def _family_contrast(artifact: dict[str, Any], model: str) -> dict[str, Any]:
    return next(
        row for row in artifact["aggregates"]["scores"]["family_contrasts"] if row["model"] == model
    )


def markdown_report(artifact: dict[str, Any]) -> str:
    validate_artifact(artifact)
    lines = [
        f"# {STUDY_ID}",
        "",
        "Inference-only diagnostic of whether the four permutations shared by D4 and all three "
        "matched control families carry their near-discrete routing behavior. All 72 decodes "
        "used the full Multi30k development set under evaluation mode and no-gradient inference. "
        "The test split was not loaded, decoded, or reported.",
        "",
        "## Fixed V4 core and family partitions",
        "",
        f"V4 core SHA-256: `{artifact['core_definition']['sha256']}`",
        "",
        "| model | core router indices in V4 order | non-core indices | non-core tuple order |",
        "|---|---|---|---|",
    ]
    for row in artifact["family_partitions"]:
        lines.append(
            f"| `{row['model']}` | {row['core_router_indices_in_v4_order']} | "
            f"{row['noncore_router_indices_in_family_order']} | "
            f"`{row['noncore_router_order']}` |"
        )

    lines += [
        "",
        "Every mask is derived by matching tuples against the active module's nonpersistent "
        "permutation buffer. Each family contains all four core members and exactly four "
        "family-specific non-core members.",
        "",
        "## Initial-function caveat",
        "",
        "Issue #6 matched trainable initialization, not the initial function: uniform routing "
        "averages each fixed family into its own centroid. Matrices below use row-major order.",
        "",
        "| model | centroid | distance from D4 centroid | singular values | "
        "initial effective residual |",
        "|---|---|---:|---|---|",
    ]
    for row in artifact["initial_function_centroids"]:
        lines.append(
            f"| `{row['model']}` | `{_format_matrix(row['uniform_router_centroid'])}` | "
            f"{row['frobenius_distance_from_d4_all_quarters']:.4f} | "
            f"{', '.join(f'{value:.4f}' for value in row['singular_values'])} | "
            f"`{_format_matrix(row['initial_effective_residual_matrix_before_diagonal_scale'])}` |"
        )

    lines += [
        "",
        f"The initial residual gate is `sigmoid(-2) = "
        f"{artifact['initial_function_centroids'][0]['initial_gate_strength']:.6f}`; learned "
        "diagonal scaling is excluded from the displayed effective matrices.",
        "",
        "## Original-score reproduction guard",
        "",
        "All 12 original reproductions completed before any subset intervention began. Existing "
        "all-family hard-argmax scores were loaded from committed artifacts and not rerun.",
        "",
        "| model | seed | selected step | stored dev BLEU | reproduced | difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in artifact["checkpoints"]:
        reproduction = checkpoint["reproduction"]
        lines.append(
            f"| `{checkpoint['model']}` | {checkpoint['seed']} | "
            f"{reproduction['selected_step']:,} | "
            f"{reproduction['stored_development_bleu']:.4f} | "
            f"{reproduction['reproduced_development_bleu']:.4f} | "
            f"{reproduction['difference']:+.4f} |"
        )

    lines += [
        "",
        "## Core usage in trained routers",
        "",
        "Subset-conditional distributions and entropies exclude token-subset pairs with mass at "
        f"most {SUBSET_MASS_EPSILON:g}; zero is reported only if no valid pair remains.",
        "",
        "| family | core mass | p05 | p50 | p95 | argmax in core | "
        "H(core)/log4 | H(non-core)/log4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["core_usage"]["by_family"]:
        lines.append(
            f"| `{row['model']}` | {row['mean_core_probability_mass']:.3f} | "
            f"{row['core_probability_mass_p05']:.3f} | "
            f"{row['core_probability_mass_p50']:.3f} | "
            f"{row['core_probability_mass_p95']:.3f} | "
            f"{row['all_family_argmax_core_frequency']:.3f} | "
            f"{row['normalized_conditional_core_entropy']:.3f} | "
            f"{row['normalized_conditional_noncore_entropy']:.3f} |"
        )
    overall_usage = artifact["aggregates"]["core_usage"]["across_all_12_checkpoints"]
    lines.append(
        f"| **all 12** | **{overall_usage['mean_core_probability_mass']:.3f}** | "
        f"**{overall_usage['core_probability_mass_p05']:.3f}** | "
        f"**{overall_usage['core_probability_mass_p50']:.3f}** | "
        f"**{overall_usage['core_probability_mass_p95']:.3f}** | "
        f"**{overall_usage['all_family_argmax_core_frequency']:.3f}** | "
        f"**{overall_usage['normalized_conditional_core_entropy']:.3f}** | "
        f"**{overall_usage['normalized_conditional_noncore_entropy']:.3f}** |"
    )

    lines += [
        "",
        "### Family, layer, and role means",
        "",
        "| family | layer | role | core mass | argmax in core | "
        "core distribution (V4 order) | non-core distribution |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for row in artifact["aggregates"]["core_usage"]["by_family_layer_role"]:
        core_distribution = ", ".join(
            f"{value:.3f}" for value in row["core_route_distribution_v4_order"]
        )
        noncore_distribution = ", ".join(
            f"{value:.3f}" for value in row["noncore_route_distribution_family_order"]
        )
        lines.append(
            f"| `{row['model']}` | {row['layer']} | {row['role']} | "
            f"{row['mean_core_probability_mass']:.3f} | "
            f"{row['all_family_argmax_core_frequency']:.3f} | "
            f"{core_distribution} | {noncore_distribution} |"
        )

    lines += [
        "",
        "## Development BLEU by checkpoint",
        "",
        "Original and all-family hard are reused references. The five subset columns are new "
        "inference-only decodes from unchanged checkpoints.",
        "",
        "| model | seed | original | all hard | core soft | core hard | core uniform | "
        "non-core soft | non-core hard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in artifact["checkpoints"]:
        lines.append(
            f"| `{checkpoint['model']}` | {checkpoint['seed']} | "
            f"{_score(checkpoint, 'original'):.2f} | "
            f"{_score(checkpoint, 'all_family_hard'):.2f} | "
            f"{_score(checkpoint, 'core_soft'):.2f} | "
            f"{_score(checkpoint, 'core_hard'):.2f} | "
            f"{_score(checkpoint, 'core_uniform'):.2f} | "
            f"{_score(checkpoint, 'noncore_soft'):.2f} | "
            f"{_score(checkpoint, 'noncore_hard'):.2f} |"
        )

    lines += [
        "",
        "## Primary paired contrasts",
        "",
        "| model | seed | core hard − all hard | core soft − original | "
        "core hard − non-core hard | core soft − non-core soft | "
        "core soft − core uniform |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["scores"]["checkpoint_contrasts"]:
        lines.append(
            f"| `{row['model']}` | {row['seed']} | "
            f"{row['core_hard_minus_all_family_hard']:+.2f} | "
            f"{row['core_soft_minus_original']:+.2f} | "
            f"{row['core_hard_minus_noncore_hard']:+.2f} | "
            f"{row['core_soft_minus_noncore_soft']:+.2f} | "
            f"{row['core_soft_minus_core_uniform']:+.2f} |"
        )
    aggregate = artifact["aggregates"]["scores"]["aggregate_contrasts"]
    lines.append(
        "| **all 12 mean** | | "
        f"**{aggregate['core_hard_minus_all_family_hard']['mean']:+.2f}** | "
        f"**{aggregate['core_soft_minus_original']['mean']:+.2f}** | "
        f"**{aggregate['core_hard_minus_noncore_hard']['mean']:+.2f}** | "
        f"**{aggregate['core_soft_minus_noncore_soft']['mean']:+.2f}** | "
        f"**{aggregate['core_soft_minus_core_uniform']['mean']:+.2f}** |"
    )

    lines += [
        "",
        "### Family means",
        "",
        "| family | core hard − all hard | core soft − original | "
        "core hard − non-core hard | core soft − non-core soft | core soft − core uniform |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["scores"]["family_contrasts"]:
        lines.append(
            f"| `{row['model']}` | "
            f"{row['core_hard_minus_all_family_hard']['mean']:+.2f} | "
            f"{row['core_soft_minus_original']['mean']:+.2f} | "
            f"{row['core_hard_minus_noncore_hard']['mean']:+.2f} | "
            f"{row['core_soft_minus_noncore_soft']['mean']:+.2f} | "
            f"{row['core_soft_minus_core_uniform']['mean']:+.2f} |"
        )

    support = artifact["aggregates"]["scores"]["supporting_means"]
    favorable = support["families_with_favorable_core_hard_comparison"]
    lines += [
        "",
        "## Decision questions",
        "",
        f"- **Do trained routers favor the shared core?** Across all checkpoints and modules, "
        f"mean core probability mass is {overall_usage['mean_core_probability_mass']:.3f} and "
        f"all-family argmax selects the core on "
        f"{overall_usage['all_family_argmax_core_frequency']:.3f} of real source tokens.",
        f"- **Does core soft retain original BLEU?** The aggregate paired difference is "
        f"{aggregate['core_soft_minus_original']['mean']:+.2f} BLEU.",
        f"- **Does core hard retain all-family hard BLEU?** The aggregate paired difference is "
        f"{aggregate['core_hard_minus_all_family_hard']['mean']:+.2f} BLEU.",
        f"- **Does learned core selection beat core uniform?** The aggregate paired difference "
        f"is {aggregate['core_soft_minus_core_uniform']['mean']:+.2f} BLEU.",
        f"- **Does the core beat the non-core complement?** Core minus non-core is "
        f"{aggregate['core_hard_minus_noncore_hard']['mean']:+.2f} BLEU under hard selection "
        f"and {aggregate['core_soft_minus_noncore_soft']['mean']:+.2f} BLEU under soft routing.",
        f"- **Is the comparison consistent across families?** Core hard exceeds non-core hard "
        f"in {favorable}/4 family means.",
        "",
        "## Screening interpretation",
        "",
        f"Decision: **`{artifact['aggregates']['scores']['screening_decision']}`**.",
        "",
    ]
    decision = artifact["aggregates"]["scores"]["screening_decision"]
    explanations = {
        "common_v4_core_is_sufficient": (
            "The common V4 core meets every preregistered sufficiency threshold. The next "
            "training comparison should use a four-route V4 model against parameter-matched "
            "rank-4 and continuous local controls."
        ),
        "core_and_noncore_are_interchangeable": (
            "Core and complement both retain performance and remain within the preregistered "
            "0.20-BLEU interchangeability window. The evidence supports generic discrete "
            "permutation routing rather than a special shared subgroup."
        ),
        "full_family_coverage_is_necessary": (
            "Both four-member subsets lose more than 0.50 BLEU while all-family hard selection "
            "retains performance. The next comparison should test full S4 coverage against a "
            "learned basis rather than train a V4-only model."
        ),
        "noncore_is_stronger": (
            "The family-specific complements consistently outperform the shared core. The "
            "shared V4 subgroup does not explain the result; generic adapter controls are the "
            "appropriate next step."
        ),
        "mixed_or_inconclusive": (
            "The aggregate pattern does not satisfy any preregistered directional rule. Stop "
            "for audit before selecting a new training comparison."
        ),
    }
    lines += [
        explanations[decision],
        "",
        "These are descriptive 12-checkpoint screening contrasts. No p-values or bootstrap "
        "significance claims are reported.",
        "",
    ]
    return "\n".join(lines)


def expected_new_intervention_decode_count() -> int:
    return len(FAMILY_MODELS) * len(SEEDS) * len(CONDITIONS)


def require_complete_reproduction_phase(
    checkpoints: list[dict[str, Any]],
    failures: list[str],
) -> None:
    """Block every intervention until all 12 finite original-score guards have run and passed."""
    if len(checkpoints) != 12:
        raise ValueError("interventions cannot start until all 12 original reproductions complete")
    if failures:
        raise ValueError(
            "all 12 original reproductions completed, but intervention evaluation is blocked:\n"
            + "\n".join(failures)
        )


def iter_new_interventions() -> Iterable[tuple[str, int, D4Intervention]]:
    yield from product(FAMILY_MODELS, SEEDS, CONDITIONS)


def run_diagnostic(
    d4_run_dir: str,
    family_run_dir: str,
    d4_artifact_path: str,
    family_artifact_path: str,
    json_path: str,
    markdown_path: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run the mandatory 12-score guard, then exactly 60 subset intervention decodes."""
    sources, source_artifacts = load_checkpoint_sources(
        d4_run_dir,
        family_run_dir,
        d4_artifact_path,
        family_artifact_path,
    )
    corpus = build_corpus(CORPUS)
    tokenizer_path = corpus.tokenizer_path(smoke=False)
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"the existing Multi30k tokenizer is required at {tokenizer_path}; "
            "the diagnostic will not train a tokenizer"
        )
    tokenizer = Tokenizer(tokenizer_path)
    development = corpus.load_split(SPLIT)
    encoded_development = encode_split(development, tokenizer, corpus)

    checkpoints: list[dict[str, Any]] = []
    reproduction_failures: list[str] = []
    print(
        "phase 1: validating core usage and all 12 original reproductions",
        flush=True,
    )
    for source in sources:
        train_config = source.results["manifest"]["train_config"]
        model = load_model(source, device)
        if len(tokenizer) != model.config.vocab_size:
            raise ValueError(
                f"{source.model}/seed{source.seed} tokenizer vocabulary does not match checkpoint"
            )
        before_hash = _state_dict_sha256(model)
        core_usage = collect_core_usage(
            model,
            encoded_development,
            int(train_config["decode_batch_size"]),
            device,
        )
        reproduced = decode_original(
            model,
            corpus,
            development,
            tokenizer,
            device,
            int(train_config["decode_batch_size"]),
            int(train_config["beam_size"]),
        )
        try:
            difference = require_reproduction(source.stored_development_bleu, reproduced)
        except ValueError as error:
            difference = reproduced - source.stored_development_bleu
            reproduction_failures.append(f"{source.model}/seed{source.seed}: {error}")
        if _state_dict_sha256(model) != before_hash:
            raise ValueError(
                f"{source.model}/seed{source.seed} state dictionary changed during preflight"
            )
        reused_modules = copy.deepcopy(source.artifact_reference["modules"])
        checkpoints.append(
            {
                "model": source.model,
                "seed": source.seed,
                "source_checkpoint": _source_provenance(source),
                "reproduction": {
                    "selected_step": source.selected_step,
                    "stored_development_bleu": source.stored_development_bleu,
                    "reproduced_development_bleu": reproduced,
                    "difference": difference,
                },
                "reused_references": [
                    {
                        "origin": "reused reference",
                        "condition": "original",
                        "development_bleu": source.stored_development_bleu,
                    },
                    {
                        "origin": "reused reference",
                        "condition": "all_family_hard",
                        "development_bleu": source.artifact_reference[
                            "all_family_hard_development_bleu"
                        ],
                    },
                ],
                "existing_router_statistics": reused_modules,
                "core_usage": core_usage,
                "new_interventions": [],
                "state_dict_sha256": {
                    "before": before_hash,
                    "after": None,
                },
            }
        )
        print(
            f"  {source.model}/seed{source.seed}: "
            f"{source.stored_development_bleu:.4f} -> {reproduced:.4f} "
            f"({difference:+.4f})",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    require_complete_reproduction_phase(checkpoints, reproduction_failures)

    print(
        "phase 2: all reproduction guards passed; running exactly 60 subset decodes",
        flush=True,
    )
    for source, checkpoint in zip(sources, checkpoints, strict=True):
        train_config = source.results["manifest"]["train_config"]
        model = load_model(source, device)
        expected_hash = checkpoint["state_dict_sha256"]["before"]
        if _state_dict_sha256(model) != expected_hash:
            raise ValueError(
                f"{source.model}/seed{source.seed} fresh checkpoint differs from preflight"
            )
        original = checkpoint["reused_references"][0]["development_bleu"]
        for condition in CONDITIONS:
            score = decode_subset_condition(
                model,
                condition,
                corpus,
                development,
                tokenizer,
                device,
                int(train_config["decode_batch_size"]),
                int(train_config["beam_size"]),
            )
            if _state_dict_sha256(model) != expected_hash:
                raise ValueError(
                    f"{source.model}/seed{source.seed}/{condition} changed checkpoint state"
                )
            checkpoint["new_interventions"].append(
                {
                    "origin": "new intervention",
                    "condition": condition,
                    "development_bleu": score,
                    "difference_from_original": score - original,
                }
            )
            print(
                f"  {source.model}/seed{source.seed}/{condition}: "
                f"{score:.4f} ({score - original:+.4f})",
                flush=True,
            )
        checkpoint["state_dict_sha256"]["after"] = _state_dict_sha256(model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if sum(len(row["new_interventions"]) for row in checkpoints) != (
        expected_new_intervention_decode_count()
    ):
        raise ValueError("the diagnostic did not produce exactly 60 new intervention rows")

    artifact = build_artifact(checkpoints, source_artifacts, git_commit())
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(markdown_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report(artifact))
    return artifact


validate_family_partitions()
