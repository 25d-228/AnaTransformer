"""Development-only diagnostics for the six Multi30k factorial D4 checkpoints."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
from torch import Tensor

from ana.config import ModelConfig
from ana.data.corpora import build_corpus
from ana.data.tokenizer import Tokenizer
from ana.experiment import encode_split, git_commit, score_split
from ana.model import Seq2SeqTransformer
from ana.nn.grouping import D4_PERMUTATIONS, FEATURE, N_PERMUTATIONS, permutation_matrices
from ana.nn.projection import SharedQKV
from ana.nn.roles import (
    D4Intervention,
    D4Mixing,
    D4RoleTransform,
    temporary_d4_intervention,
)
from ana.registry import build_model
from ana.trainer import fixed_batches

STUDY_ID = "multi30k_d4_checkpoint_diagnostic_v1"
SOURCE_STUDY_ID = "multi30k_factorial_v1"
CORPUS = "multi30k"
SPLIT = "dev"
MODELS = ("ana_d4_enc", "ana_feat_enc")
SEEDS = (42, 43, 44)
ROLES = ("Q", "K", "V")
REPRODUCTION_TOLERANCE = 0.05
COMMON_CONDITIONS: tuple[D4Intervention, ...] = (
    "original",
    "gate_zero",
    "uniform_router",
    "identity_router",
    "hard_argmax",
)
CONDITIONS: dict[str, tuple[D4Intervention, ...]] = {
    "ana_d4_enc": COMMON_CONDITIONS,
    "ana_feat_enc": (*COMMON_CONDITIONS, "magnitude_one"),
}

DEFINITIONS = {
    "gate_strength": "sigmoid(gate)",
    "hard_argmax_scope": (
        "hard argmax makes routed matrix P an exact D4 permutation; the full role remains "
        "diag(scale)[(1-g)I + gP]z in ana_d4_enc and additionally contains a learned positive "
        "magnitude in ana_feat_enc"
    ),
    "identity_router_gate_zero_equivalence": (
        "in ana_d4_enc, identity_router and gate_zero are the same effective ablation because "
        "magnitude is one and identity mixing gives mixed = z"
    ),
    "normalized_router_entropy": "mean_token H(p) / log(8)",
    "maximum_route_probability": "mean_token max_k p_k",
    "effective_number_of_routes": "mean_token exp(H(p))",
    "identity_probability": "mean_token p_0, where D4 element 0 is identity",
    "mean_route_distribution": "mean_token p in the fixed D4 order",
    "nearest_d4_distance": "mean_token min_k ||sum_j p_j P_j - P_k||_F / sqrt(8)",
    "token_conditioned_routing_variation": ("(H(mean_token p) - mean_token H(p)) / log(8)"),
    "qkv_role_differentiation": (
        "mean_token mean_pair JS(p_Q, p_K, p_V) / log(2), over Q-K, Q-V, and K-V"
    ),
    "magnitude": (
        "positive softplus magnitude over real source tokens: mean, sample standard deviation, "
        "p05, p50, p95"
    ),
}


@dataclass(frozen=True)
class CheckpointSource:
    model: str
    seed: int
    folder: str
    results: dict[str, Any]
    weights_metadata: dict[str, Any]

    @property
    def weights_path(self) -> str:
        return os.path.join(self.folder, "weights.pt")

    @property
    def stored_development_bleu(self) -> float:
        return float(self.results["scores"]["dev"])

    @property
    def selected_step(self) -> int:
        return int(self.results["scored_step"])


def _entropy(probabilities: Tensor) -> Tensor:
    return -(
        probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=-1)


def _sample_standard_deviation(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


class _ModuleAccumulator:
    """Online sufficient statistics for one encoder-layer role."""

    def __init__(self, permutations: Tensor, has_magnitude: bool) -> None:
        self.permutations = permutations.detach()
        self.has_magnitude = has_magnitude
        self.count = 0
        self.sum_probabilities = torch.zeros(N_PERMUTATIONS, dtype=torch.float64)
        self.sum_entropy = 0.0
        self.sum_maximum = 0.0
        self.sum_effective = 0.0
        self.sum_nearest_distance = 0.0
        self.magnitudes: list[Tensor] = []

    def add(
        self,
        probabilities: Tensor,
        real_mask: Tensor,
        magnitude: Tensor | None,
    ) -> None:
        selected = probabilities[real_mask.bool()]
        if selected.numel() == 0:
            return

        entropy = _entropy(selected)
        matrix = torch.einsum("nc,cji->nji", selected, self.permutations)
        distances = (matrix[:, None] - self.permutations[None]).square().sum(
            dim=(-1, -2)
        ).sqrt() / math.sqrt(N_PERMUTATIONS)

        selected_cpu = selected.detach().to(dtype=torch.float64, device="cpu")
        self.count += selected.size(0)
        self.sum_probabilities += selected_cpu.sum(dim=0)
        self.sum_entropy += float(entropy.sum())
        self.sum_maximum += float(selected.max(dim=-1).values.sum())
        self.sum_effective += float(entropy.exp().sum())
        self.sum_nearest_distance += float(distances.min(dim=-1).values.sum())

        if self.has_magnitude:
            if magnitude is None:
                raise ValueError("a learned-magnitude D4 role did not provide magnitudes")
            self.magnitudes.append(
                magnitude.squeeze(-1)[real_mask.bool()]
                .detach()
                .to(dtype=torch.float64, device="cpu")
            )

    def finish(self, gate_strength: float) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("router statistics contain no real source tokens")

        mean_probabilities = self.sum_probabilities / self.count
        mean_entropy = self.sum_entropy / self.count
        entropy_of_mean = float(_entropy(mean_probabilities))
        result: dict[str, Any] = {
            "real_token_count": self.count,
            "gate_strength": gate_strength,
            "normalized_router_entropy": mean_entropy / math.log(N_PERMUTATIONS),
            "maximum_route_probability": self.sum_maximum / self.count,
            "effective_number_of_routes": self.sum_effective / self.count,
            "identity_probability": float(mean_probabilities[0]),
            "mean_route_distribution": mean_probabilities.tolist(),
            "nearest_d4_distance": self.sum_nearest_distance / self.count,
            "token_conditioned_routing_variation": max(
                0.0,
                (entropy_of_mean - mean_entropy) / math.log(N_PERMUTATIONS),
            ),
            "magnitude": None,
        }
        if self.has_magnitude:
            values = torch.cat(self.magnitudes)
            quantiles = torch.quantile(
                values,
                torch.tensor([0.05, 0.50, 0.95], dtype=values.dtype),
            )
            result["magnitude"] = {
                "mean": float(values.mean()),
                "sample_standard_deviation": float(values.std(unbiased=True))
                if values.numel() > 1
                else 0.0,
                "p05": float(quantiles[0]),
                "p50": float(quantiles[1]),
                "p95": float(quantiles[2]),
            }
        return result


class _RoleDifferentiationAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_divergence = 0.0

    def add(self, probabilities: dict[str, Tensor], real_mask: Tensor) -> None:
        divergences = []
        for left, right in (("Q", "K"), ("Q", "V"), ("K", "V")):
            p, q = probabilities[left], probabilities[right]
            middle = (p + q) / 2
            divergences.append(_entropy(middle) - (_entropy(p) + _entropy(q)) / 2)
        per_token = torch.stack(divergences).mean(dim=0) / math.log(2)
        selected = per_token[real_mask.bool()]
        self.count += selected.numel()
        self.sum_divergence += float(selected.sum())

    def finish(self) -> float:
        if self.count == 0:
            raise ValueError("role differentiation contains no real source tokens")
        return self.sum_divergence / self.count


def labeled_encoder_d4_roles(
    model: Seq2SeqTransformer,
) -> list[tuple[int, str, D4RoleTransform]]:
    """Return every encoder layer and Q/K/V D4 role exactly once, in fixed order."""
    labeled = []
    for layer_index, layer in enumerate(model.encoder, start=1):
        projection = layer.self_attention.projection
        if not isinstance(projection, SharedQKV):
            raise ValueError(f"encoder layer {layer_index} does not use SharedQKV")
        for role, attribute in (
            ("Q", "query_role"),
            ("K", "key_role"),
            ("V", "value_role"),
        ):
            module = getattr(projection, attribute)
            if not isinstance(module, D4RoleTransform):
                raise ValueError(f"encoder layer {layer_index} role {role} is not a D4 router")
            if module.grouping is not FEATURE:
                raise ValueError(
                    f"encoder layer {layer_index} role {role} is not routed per source token"
                )
            labeled.append((layer_index, role, module))

    expected = set(product(range(1, model.config.n_encoder_layers + 1), ROLES))
    observed = {(layer, role) for layer, role, _ in labeled}
    if observed != expected or len(labeled) != len(expected):
        raise ValueError(f"D4 module labels are incomplete or duplicated: {sorted(observed)}")

    every_d4 = {module for module in model.modules() if isinstance(module, D4RoleTransform)}
    labeled_modules = {module for _, _, module in labeled}
    if every_d4 != labeled_modules:
        raise ValueError("the checkpoint contains D4 modules outside the labeled encoder roles")
    return labeled


class RouterStatisticsCollector:
    """Forward hooks that retain sufficient statistics, not activation dumps."""

    def __init__(self, model: Seq2SeqTransformer) -> None:
        self.model = model
        self.labeled = labeled_encoder_d4_roles(model)
        self.modules = {
            (layer, role): _ModuleAccumulator(
                module.permutations,
                isinstance(module, D4Mixing),
            )
            for layer, role, module in self.labeled
        }
        self.role_differentiation = {
            layer: _RoleDifferentiationAccumulator()
            for layer in range(1, model.config.n_encoder_layers + 1)
        }
        self.pending: dict[int, dict[str, tuple[Tensor, Tensor]]] = defaultdict(dict)
        self.handles: list[Any] = []

    def _hook(
        self, layer: int, role: str, module: D4RoleTransform, inputs: tuple[Any, ...]
    ) -> None:
        z, pad_mask = inputs
        readout = module.grouping.router_readout(z, pad_mask)
        probabilities = module.learned_route_probabilities(readout)
        magnitude = module.routed_magnitude(readout) if isinstance(module, D4Mixing) else None
        self.modules[(layer, role)].add(probabilities, pad_mask, magnitude)
        self.pending[layer][role] = (probabilities, pad_mask)

        if set(self.pending[layer]) == set(ROLES):
            masks = [self.pending[layer][name][1] for name in ROLES]
            if not all(torch.equal(masks[0], mask) for mask in masks[1:]):
                raise ValueError(f"layer {layer} Q/K/V masks do not align")
            routed = {name: self.pending[layer][name][0] for name in ROLES}
            self.role_differentiation[layer].add(routed, masks[0])
            self.pending[layer].clear()

    def __enter__(self) -> RouterStatisticsCollector:
        for layer, role, module in self.labeled:
            self.handles.append(
                module.register_forward_pre_hook(
                    lambda held, inputs, layer=layer, role=role: self._hook(
                        layer, role, held, inputs
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish(self) -> list[dict[str, Any]]:
        unfinished = {layer: sorted(roles) for layer, roles in self.pending.items() if roles}
        if unfinished:
            raise ValueError(f"incomplete Q/K/V hook groups: {unfinished}")

        rows = []
        for layer, role, module in self.labeled:
            row = {
                "layer": layer,
                "role": role,
                **self.modules[(layer, role)].finish(float(torch.sigmoid(module.gate))),
                "qkv_role_differentiation": self.role_differentiation[layer].finish(),
            }
            rows.append(row)
        return rows


@torch.no_grad()
def collect_router_statistics(
    model: Seq2SeqTransformer,
    encoded_development: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Collect Part A statistics over real source tokens only."""
    model.eval()
    batches = fixed_batches(encoded_development, batch_size, model.config.pad_id)
    collector = RouterStatisticsCollector(model)
    with collector:
        for batch in batches:
            model.encode(batch.source_ids.to(device), batch.source_mask.to(device))
    return collector.finish()


def _load_weights(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def load_checkpoint_sources(run_dir: str) -> tuple[list[CheckpointSource], str]:
    """Validate exactly the six prescribed factorial checkpoints and their sibling records."""
    sources = []
    commits = set()
    recipes = set()

    for model, seed in product(MODELS, SEEDS):
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
        if record.get("corpus") != CORPUS:
            raise ValueError(f"{label} does not use {CORPUS}")
        if manifest.get("study_id") != SOURCE_STUDY_ID:
            raise ValueError(f"{label} is not marked as source study {SOURCE_STUDY_ID}")
        if manifest.get("seeded_before_model_init") is not True:
            raise ValueError(f"{label} lacks pre-construction seed provenance")
        if manifest.get("smoke") is not False:
            raise ValueError(f"{label} is a smoke or unmarked run")
        if manifest.get("score_dev") is not True or "dev" not in record.get("scores", {}):
            raise ValueError(f"{label} lacks factorial development BLEU")

        metadata = _load_weights(weights_path)
        if metadata.get("model") != model or metadata.get("corpus") != CORPUS:
            raise ValueError(f"{label} checkpoint metadata does not match its folder")
        if metadata.get("model_config") != manifest.get("model_config"):
            raise ValueError(f"{label} checkpoint and results disagree on model configuration")
        if metadata.get("step") != record.get("scored_step"):
            raise ValueError(f"{label} checkpoint and results disagree on selected step")
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
        train_config = manifest.get("train_config", {})
        recipes.add(
            json.dumps(
                {key: value for key, value in train_config.items() if key != "seed"},
                sort_keys=True,
            )
        )
        sources.append(CheckpointSource(model, seed, folder, record, metadata))

    if len(sources) != len(MODELS) * len(SEEDS):
        raise ValueError("the checkpoint diagnostic did not resolve exactly six checkpoints")
    if len(commits) != 1 or None in commits:
        raise ValueError(f"source checkpoints do not share one experiment commit: {commits}")
    if len(recipes) != 1:
        raise ValueError("source checkpoints do not share one factorial recipe")
    return sources, str(next(iter(commits)))


def load_model(source: CheckpointSource, device: torch.device) -> Seq2SeqTransformer:
    config = ModelConfig(**source.weights_metadata["model_config"])
    model = build_model(source.model, config)
    checkpoint = _load_weights(source.weights_path, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def require_reproduction(
    stored: float,
    observed: float,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> float:
    if not math.isfinite(stored) or not math.isfinite(observed):
        raise ValueError("stored and reproduced development BLEU must be finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("the reproduction tolerance must be finite and non-negative")
    difference = observed - stored
    if abs(difference) > tolerance:
        raise ValueError(
            f"original development BLEU {observed:.6f} differs from stored {stored:.6f} "
            f"by {difference:+.6f}, beyond the {tolerance:.2f} guard"
        )
    return difference


@torch.no_grad()
def decode_condition(
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
    with temporary_d4_intervention(model, condition):
        score, _ = score_split(
            model,
            corpus,
            examples,
            tokenizer,
            device,
            batch_size,
            beam_size,
        )
    return score


def _mean_module_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "gate_strength",
        "normalized_router_entropy",
        "maximum_route_probability",
        "effective_number_of_routes",
        "identity_probability",
        "nearest_d4_distance",
        "token_conditioned_routing_variation",
        "qkv_role_differentiation",
    )
    summary = {key: statistics.mean(row[key] for row in rows) for key in scalar_keys}
    summary["mean_route_distribution"] = [
        statistics.mean(row["mean_route_distribution"][index] for row in rows)
        for index in range(N_PERMUTATIONS)
    ]
    magnitudes = [row["magnitude"] for row in rows if row["magnitude"] is not None]
    summary["magnitude"] = (
        {key: statistics.mean(magnitude[key] for magnitude in magnitudes) for key in magnitudes[0]}
        if magnitudes
        else None
    )
    return summary


def aggregate_checkpoints(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    intervention_rows = []
    for model in MODELS:
        selected = [checkpoint for checkpoint in checkpoints if checkpoint["model"] == model]
        for condition in CONDITIONS[model]:
            rows = [
                next(row for row in checkpoint["interventions"] if row["condition"] == condition)
                for checkpoint in selected
            ]
            scores = [row["development_bleu"] for row in rows]
            differences = [row["difference_from_original"] for row in rows]
            intervention_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "mean_development_bleu": statistics.mean(scores),
                    "sample_standard_deviation_development_bleu": (
                        _sample_standard_deviation(scores)
                    ),
                    "mean_difference_from_original": statistics.mean(differences),
                    "paired_differences_from_original": differences,
                }
            )

    module_rows = []
    for model, layer, role in product(MODELS, range(1, 5), ROLES):
        rows = [
            next(
                row
                for row in checkpoint["modules"]
                if row["layer"] == layer and row["role"] == role
            )
            for checkpoint in checkpoints
            if checkpoint["model"] == model
        ]
        module_rows.append(
            {
                "model": model,
                "layer": layer,
                "role": role,
                "seeds": list(SEEDS),
                **_mean_module_summary(rows),
            }
        )
    return {
        "interventions_across_seeds": intervention_rows,
        "modules_across_seeds": module_rows,
    }


def build_artifact(
    checkpoints: list[dict[str, Any]],
    analysis_git_commit: str,
    source_experiment_commit: str,
) -> dict[str, Any]:
    ordered = sorted(
        checkpoints,
        key=lambda row: (MODELS.index(row["model"]), SEEDS.index(row["seed"])),
    )
    for checkpoint in ordered:
        checkpoint["checkpoint_module_mean"] = _mean_module_summary(checkpoint["modules"])

    artifact = {
        "study_id": STUDY_ID,
        "analysis_git_commit": analysis_git_commit,
        "source_factorial_study": SOURCE_STUDY_ID,
        "source_experiment_commit": source_experiment_commit,
        "corpus": CORPUS,
        "split": SPLIT,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "reproduction_tolerance_bleu": REPRODUCTION_TOLERANCE,
        "decode_count": sum(len(CONDITIONS[model]) for model in MODELS) * len(SEEDS),
        "d4_order": [
            {"index": index, "permutation": list(permutation)}
            for index, permutation in enumerate(D4_PERMUTATIONS)
        ],
        "definitions": DEFINITIONS,
        "checkpoints": ordered,
        "aggregates": aggregate_checkpoints(ordered),
    }
    validate_artifact(artifact)
    return artifact


def _require_finite_numbers(value: Any, path: str = "artifact") -> None:
    """Reject non-finite numeric values anywhere in a diagnostic artifact."""
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


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Refuse partial checkpoints, conditions, labels, or failed reproductions."""
    _require_finite_numbers(artifact)
    if artifact.get("study_id") != STUDY_ID:
        raise ValueError(f"artifact is not marked as {STUDY_ID}")
    if artifact.get("source_factorial_study") != SOURCE_STUDY_ID:
        raise ValueError("artifact has the wrong source factorial study")
    if not artifact.get("analysis_git_commit") or not artifact.get("source_experiment_commit"):
        raise ValueError("artifact lacks analysis or source Git provenance")
    if artifact.get("corpus") != CORPUS or artifact.get("split") != SPLIT:
        raise ValueError("the diagnostic must contain Multi30k development results only")
    if artifact.get("decode_count") != 33:
        raise ValueError("the diagnostic must declare exactly 33 development decodes")

    checkpoints = artifact.get("checkpoints", [])
    expected_checkpoints = set(product(MODELS, SEEDS))
    observed_checkpoints = {(row.get("model"), row.get("seed")) for row in checkpoints}
    if len(checkpoints) != 6 or observed_checkpoints != expected_checkpoints:
        raise ValueError("artifact must contain exactly the six prescribed checkpoints")

    expected_modules = set(product(range(1, 5), ROLES))
    for checkpoint in checkpoints:
        model, seed = checkpoint["model"], checkpoint["seed"]
        interventions = checkpoint.get("interventions", [])
        observed_conditions = [row.get("condition") for row in interventions]
        if observed_conditions != list(CONDITIONS[model]):
            raise ValueError(f"{model}/seed{seed} has missing, extra, or reordered conditions")
        difference = checkpoint.get("original_reproduction_difference")
        if difference is None or abs(difference) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"{model}/seed{seed} fails the reproduction guard")
        for row in interventions:
            expected_difference = row["development_bleu"] - checkpoint["original_reproduction_bleu"]
            if not math.isclose(
                row["difference_from_original"],
                expected_difference,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{model}/seed{seed} has an incorrect paired BLEU difference")

        modules = checkpoint.get("modules", [])
        observed_modules = {(row.get("layer"), row.get("role")) for row in modules}
        if len(modules) != 12 or observed_modules != expected_modules:
            raise ValueError(f"{model}/seed{seed} must contain 12 unique encoder role rows")
        for row in modules:
            if row.get("real_token_count", 0) <= 0:
                raise ValueError(f"{model}/seed{seed} has an empty router-statistics row")
            if len(row.get("mean_route_distribution", [])) != N_PERMUTATIONS:
                raise ValueError(f"{model}/seed{seed} has an incomplete D4 route distribution")
            if model == "ana_feat_enc" and row.get("magnitude") is None:
                raise ValueError(f"{model}/seed{seed} lacks magnitude statistics")
            if model == "ana_d4_enc" and row.get("magnitude") is not None:
                raise ValueError(f"{model}/seed{seed} unexpectedly has magnitude statistics")


def _aggregate_row(artifact: dict[str, Any], model: str, condition: str) -> dict[str, Any]:
    return next(
        row
        for row in artifact["aggregates"]["interventions_across_seeds"]
        if row["model"] == model and row["condition"] == condition
    )


def _module_average(artifact: dict[str, Any], model: str, key: str) -> float:
    rows = [row for row in artifact["aggregates"]["modules_across_seeds"] if row["model"] == model]
    return statistics.mean(row[key] for row in rows)


def markdown_report(artifact: dict[str, Any]) -> str:
    validate_artifact(artifact)
    lines = [
        f"# {STUDY_ID}",
        "",
        "Inference-only analysis of six trained Multi30k D4 checkpoints. All statistics and "
        "decodes use the full development set; the test set is not loaded or decoded.",
        "",
        "## Checkpoint reproduction",
        "",
        "| model | seed | selected step | stored dev BLEU | reproduced | difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in artifact["checkpoints"]:
        lines.append(
            f"| `{checkpoint['model']}` | {checkpoint['seed']} | "
            f"{checkpoint['selected_step']:,} | {checkpoint['stored_development_bleu']:.4f} | "
            f"{checkpoint['original_reproduction_bleu']:.4f} | "
            f"{checkpoint['original_reproduction_difference']:+.4f} |"
        )

    lines += [
        "",
        "## Development-set interventions",
        "",
        "| model | seed | condition | dev BLEU | Δ original |",
        "|---|---:|---|---:|---:|",
    ]
    for checkpoint in artifact["checkpoints"]:
        for row in checkpoint["interventions"]:
            lines.append(
                f"| `{checkpoint['model']}` | {checkpoint['seed']} | `{row['condition']}` | "
                f"{row['development_bleu']:.2f} | {row['difference_from_original']:+.2f} |"
            )

    lines += [
        "",
        "## Intervention means across seeds",
        "",
        "| model | condition | dev BLEU mean ± sample SD | mean Δ | paired Δ by seed |",
        "|---|---|---:|---:|---|",
    ]
    for row in artifact["aggregates"]["interventions_across_seeds"]:
        differences = ", ".join(
            f"{seed}: {difference:+.2f}"
            for seed, difference in zip(
                SEEDS,
                row["paired_differences_from_original"],
                strict=True,
            )
        )
        lines.append(
            f"| `{row['model']}` | `{row['condition']}` | "
            f"{row['mean_development_bleu']:.2f} ± "
            f"{row['sample_standard_deviation_development_bleu']:.2f} | "
            f"{row['mean_difference_from_original']:+.2f} | {differences} |"
        )

    lines += [
        "",
        "## Router statistics across seeds",
        "",
        "Each row is the mean of the same labeled module over seeds 42, 43, and 44.",
        "",
        "| model | layer | role | gate | H/log8 | max p | effective | identity | "
        "nearest D4 | token JS | QKV JS | magnitude mean |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in artifact["aggregates"]["modules_across_seeds"]:
        magnitude = row["magnitude"]
        magnitude_mean = "—" if magnitude is None else f"{magnitude['mean']:.3f}"
        lines.append(
            f"| `{row['model']}` | {row['layer']} | {row['role']} | "
            f"{row['gate_strength']:.3f} | {row['normalized_router_entropy']:.3f} | "
            f"{row['maximum_route_probability']:.3f} | "
            f"{row['effective_number_of_routes']:.2f} | "
            f"{row['identity_probability']:.3f} | {row['nearest_d4_distance']:.3f} | "
            f"{row['token_conditioned_routing_variation']:.4f} | "
            f"{row['qkv_role_differentiation']:.4f} | {magnitude_mean} |"
        )

    gate_d4 = _module_average(artifact, "ana_d4_enc", "gate_strength")
    gate_feat = _module_average(artifact, "ana_feat_enc", "gate_strength")
    entropy_d4 = _module_average(artifact, "ana_d4_enc", "normalized_router_entropy")
    entropy_feat = _module_average(artifact, "ana_feat_enc", "normalized_router_entropy")
    max_d4 = _module_average(artifact, "ana_d4_enc", "maximum_route_probability")
    max_feat = _module_average(artifact, "ana_feat_enc", "maximum_route_probability")
    token_d4 = _module_average(artifact, "ana_d4_enc", "token_conditioned_routing_variation")
    token_feat = _module_average(artifact, "ana_feat_enc", "token_conditioned_routing_variation")
    role_d4 = _module_average(artifact, "ana_d4_enc", "qkv_role_differentiation")
    role_feat = _module_average(artifact, "ana_feat_enc", "qkv_role_differentiation")
    distance_d4 = _module_average(artifact, "ana_d4_enc", "nearest_d4_distance")
    distance_feat = _module_average(artifact, "ana_feat_enc", "nearest_d4_distance")

    def delta(model: str, condition: str) -> dict[str, Any]:
        return _aggregate_row(artifact, model, condition)

    lines += [
        "",
        "## Decision questions",
        "",
        f"- **Did the gates open enough for the branch to matter?** Mean gate strength is "
        f"{gate_d4:.3f} for `ana_d4_enc` and {gate_feat:.3f} for `ana_feat_enc` "
        f"(initial value {torch.sigmoid(torch.tensor(-2.0)):.3f}); the causal check is the "
        "gate-zero intervention below.",
        f"- **Does disabling the routed branch reduce BLEU?** Gate zero changes mean BLEU by "
        f"{delta('ana_d4_enc', 'gate_zero')['mean_difference_from_original']:+.2f} and "
        f"{delta('ana_feat_enc', 'gate_zero')['mean_difference_from_original']:+.2f}, "
        "respectively.",
        f"- **Did routing move away from uniform?** Mean normalized entropy / maximum "
        f"probability is {entropy_d4:.3f} / {max_d4:.3f} for `ana_d4_enc` and "
        f"{entropy_feat:.3f} / {max_feat:.3f} for `ana_feat_enc`; uniform routing is "
        "1.000 / 0.125.",
        f"- **Are routes token-conditioned?** Generalized token JS averages {token_d4:.4f} "
        f"and {token_feat:.4f}; zero is token-constant routing.",
        f"- **Did Q, K, and V differentiate?** Normalized role JS averages {role_d4:.4f} and "
        f"{role_feat:.4f}; zero means identical role distributions.",
        f"- **Are soft matrices close to exact D4 permutations?** Mean normalized nearest-form "
        f"distance is {distance_d4:.3f} and {distance_feat:.3f}; zero is an exact form.",
        f"- **Does hard argmax retain BLEU?** Its mean difference is "
        f"{delta('ana_d4_enc', 'hard_argmax')['mean_difference_from_original']:+.2f} for "
        f"`ana_d4_enc` and "
        f"{delta('ana_feat_enc', 'hard_argmax')['mean_difference_from_original']:+.2f} for "
        "`ana_feat_enc`. Hard argmax makes only the routed matrix `P` an exact D4 "
        "permutation; the learned gate, diagonal scale, and (for `ana_feat_enc`) magnitude "
        "remain active.",
        f"- **Does learned routing beat uniform and identity?** Relative to original, uniform / "
        f"identity change mean BLEU by "
        f"{delta('ana_d4_enc', 'uniform_router')['mean_difference_from_original']:+.2f} / "
        f"{delta('ana_d4_enc', 'identity_router')['mean_difference_from_original']:+.2f} for "
        f"`ana_d4_enc`, and "
        f"{delta('ana_feat_enc', 'uniform_router')['mean_difference_from_original']:+.2f} / "
        f"{delta('ana_feat_enc', 'identity_router')['mean_difference_from_original']:+.2f} for "
        "`ana_feat_enc`.",
        "- **Are gate zero and identity routing independent checks?** Not for `ana_d4_enc`: "
        "magnitude is fixed to one, so identity mixing gives `mixed = z`, exactly as gate zero "
        "does. Their identical scores are one effective ablation, not independent evidence.",
        f"- **Does the combined model rely on magnification?** Setting magnitude to one changes "
        f"mean BLEU by "
        f"{delta('ana_feat_enc', 'magnitude_one')['mean_difference_from_original']:+.2f}.",
        "",
        "## Interpretation for the next discussion",
        "",
        "Descriptively, this matches the first decision pattern: hard argmax retains nearly all "
        "BLEU, routing is strongly token- and role-differentiated, the soft matrices lie near "
        "individual D4 forms, and effective gate/router interventions matter on every seed. "
        "This is evidence for a near-discrete, token- and role-conditioned D4 routing component. "
        "It does not make the whole role transform exactly D4: under hard argmax the D4-only "
        "role is `diag(scale)[(1-g)I + gP]z`, while the combined role also contains a learned "
        "magnitude. The full transform is therefore not shown to lie exactly in an "
        "analogy-preserving `D4 × R+` orbit. Exact D4 routing remains a live direction for a "
        "later comparison with random permutation families and generic local mixers. Those "
        "comparisons are not implemented here.",
        "",
        "The combined checkpoints also rely heavily on their learned magnitudes, but the "
        "factorial screen did not show a useful combined-model advantage. Checkpoint reliance "
        "therefore demonstrates co-adaptation, not that the magnifier improves the architecture.",
        "",
        "These three-seed effects are descriptive screening evidence. No p-values or bootstrap "
        "significance tests are reported.",
        "",
    ]
    return "\n".join(lines)


def run_diagnostic(
    run_dir: str,
    json_path: str,
    markdown_path: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run the six-checkpoint, 33-decode diagnostic and write compact artifacts."""
    sources, source_commit = load_checkpoint_sources(run_dir)
    corpus = build_corpus(CORPUS)
    tokenizer_path = corpus.tokenizer_path(smoke=False)
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"the existing Multi30k tokenizer is required at {tokenizer_path}; "
            "the diagnostic will not train a new tokenizer"
        )
    tokenizer = Tokenizer(tokenizer_path)
    development = corpus.load_split(SPLIT)
    encoded_development = encode_split(development, tokenizer, corpus)

    partial: list[dict[str, Any]] = []
    print("phase 1: validating router statistics and all six original reproductions", flush=True)
    for source in sources:
        train_config = source.results["manifest"]["train_config"]
        model = load_model(source, device)
        if len(tokenizer) != model.config.vocab_size:
            raise ValueError(
                f"{source.model}/seed{source.seed} tokenizer vocabulary does not match checkpoint"
            )
        modules = collect_router_statistics(
            model,
            encoded_development,
            int(train_config["decode_batch_size"]),
            device,
        )
        original = decode_condition(
            model,
            "original",
            corpus,
            development,
            tokenizer,
            device,
            int(train_config["decode_batch_size"]),
            int(train_config["beam_size"]),
        )
        difference = require_reproduction(source.stored_development_bleu, original)
        partial.append(
            {
                "model": source.model,
                "seed": source.seed,
                "selected_step": source.selected_step,
                "stored_development_bleu": source.stored_development_bleu,
                "original_reproduction_bleu": original,
                "original_reproduction_difference": difference,
                "modules": modules,
                "interventions": [
                    {
                        "condition": "original",
                        "development_bleu": original,
                        "difference_from_original": 0.0,
                    }
                ],
            }
        )
        print(
            f"  {source.model}/seed{source.seed}: stored {source.stored_development_bleu:.4f}, "
            f"reproduced {original:.4f}, difference {difference:+.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("phase 2: all reproduction guards passed; running inference interventions", flush=True)
    for source, checkpoint in zip(sources, partial, strict=True):
        train_config = source.results["manifest"]["train_config"]
        model = load_model(source, device)
        original = checkpoint["original_reproduction_bleu"]
        for condition in CONDITIONS[source.model][1:]:
            score = decode_condition(
                model,
                condition,
                corpus,
                development,
                tokenizer,
                device,
                int(train_config["decode_batch_size"]),
                int(train_config["beam_size"]),
            )
            checkpoint["interventions"].append(
                {
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
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    artifact = build_artifact(partial, git_commit(), source_commit)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(markdown_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_report(artifact))
    return artifact


def expected_decode_count() -> int:
    return sum(len(CONDITIONS[model]) for model in MODELS) * len(SEEDS)


def iter_conditions() -> Iterable[tuple[str, int, D4Intervention]]:
    for model in MODELS:
        for seed in SEEDS:
            for condition in CONDITIONS[model]:
                yield model, seed, condition


assert expected_decode_count() == 33
assert permutation_matrices().shape == (N_PERMUTATIONS, 4, 4)
