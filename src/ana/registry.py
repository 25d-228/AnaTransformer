"""The eight models, held as data rather than as a hierarchy of classes.

Read from `shared_qkv` downwards. Its per-role operator is a diagonal. The `ana_*` operator
replaces that diagonal with a routed blend over the eight permutations of D4, and then the
remaining rows vary one thing each: which axis is cut into fours, which axis indexes the
routing decision, whether an exponent is applied to a group before it is mixed, and at how many
attention sites the operator runs.

    baseline           an ordinary transformer, in the corpus's published configuration
    baseline_matched   the same transformer, narrowed until it is the size of the shared ones
    shared_qkv         Kowsher et al. 2024: one projection, a diagonal per role. PRIOR WORK.
    ana_seq_enc        ours: cuts 4 tokens,   routed per token-GROUP
    ana_feat_enc       ours: cuts 4 channels, routed per TOKEN
    ana_feat_1_enc     ours: cuts 4 channels, routed per channel-GROUP
    ana_feat_2_enc     ours: ana_feat_1_enc, and a routed exponent per group per token
    ana_feat_all       ours: ana_feat_enc, at every attention site

WHAT EACH COMPARISON AMONG THE ana_* MODELS ACTUALLY ISOLATES

The names are short and the distinction they turn on is not, so it is written here rather than
carried in the identifiers. Two of the three questions above move together unless the grid is
built to hold them apart.

`ana_seq_enc` and `ana_feat_1_enc` both route each group from that group's own four members and
broadcast the matrix across the other axis. They differ in the AXIS and in nothing else, so a
difference between them is the axis.

`ana_feat_enc` cuts the same axis as `ana_feat_1_enc` but indexes its decision by the other one:
one matrix per token, applied to all of that token's groups. So it differs from `ana_feat_1_enc`
in the ROUTING GRANULARITY and in nothing else.

`ana_feat_enc` against `ana_seq_enc` therefore varies BOTH at once, and a difference between
those two attributes to neither. It is the pair that looks like the obvious comparison and is
not, which is why the third model exists.

`ana_feat_2_enc` is `ana_feat_1_enc` with one addition: each group of four channels is raised to
a routed exponent before the permutation blend and lowered by the reciprocal exponent after it.
Its blend, its magnitude and its sites are `ana_feat_1_enc`'s, so it differs from it in the
EXPONENT and in nothing else. At an exponent of one the two compute the same thing, and the
exponent router is initialised to emit one.

`baseline` is the reference the literature has a number for, and the only model the calibration
gate is checked against. `shared_qkv` is the method being improved on. `baseline_matched` is an
ordinary transformer at that same size: it separates the operator from the parameter budget.

HOW THE MATCHED BASELINE IS BUILT, AND WHY IT NARROWS d_model

Every model except `baseline` must be the same size, or a difference between them is capacity
and not architecture. The matched baseline is an ordinary transformer shrunk to that size.

It shrinks by narrowing `d_model`, with `d_ff` held in the same proportion to it — the same
transformer, thinner. Narrowing `d_ff` alone, which is the obvious thing to try, does not work
in general: on COGS the sharing removes 3.1M parameters while the entire feed-forward stack is
only 2.1M, so there is no value of `d_ff` — zero included — that shrinks the baseline far
enough. The knob has to be one that can absorb an arbitrary fraction of the model, and `d_model`
is the only one that can.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from ana.config import ALL_SITES, ENCODER_ONLY, ModelConfig, Site
from ana.model import Seq2SeqTransformer
from ana.nn.attention import MultiHeadAttention
from ana.nn.grouping import FEATURE, FEATURE_PER_GROUP, SEQUENCE, Grouping
from ana.nn.projection import SeparateQKV, SharedQKV
from ana.nn.roles import D4Mixing, D4MixingPowered, DiagonalRescale, RoleTransform

Mixer = Callable[[int, Grouping], RoleTransform]


@dataclass(frozen=True)
class ModelSpec:
    """One model. `shared` decides the projection; the rest decide the per-role operator."""

    name: str
    purpose: str
    shared: bool
    matched: bool = False
    mixer: Mixer | None = None
    grouping: Grouping | None = None
    sites: frozenset[Site] = ALL_SITES

    def __post_init__(self) -> None:
        if self.mixer is None:
            return
        if self.grouping is None:
            raise ValueError(f"{self.name}: a mixer needs a grouping")
        if not self.grouping.moves_information_between_positions:
            return
        illegal = {site for site in self.sites if not site.query_stream_is_complete}
        if illegal:
            names = ", ".join(sorted(s.value for s in illegal))
            raise ValueError(
                f"{self.name}: this grouping moves information between query positions, "
                f"so it cannot run at {names}. There it would let a position read from "
                f"later ones, and during generation there is only one token in hand."
            )


REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(
            name="baseline",
            purpose="an ordinary transformer; the published configuration, and the only "
            "model the calibration gate checks",
            shared=False,
        ),
        ModelSpec(
            name="baseline_matched",
            purpose="an ordinary transformer narrowed to the size of the shared models; the "
            "control that decides whether an improvement over shared_qkv means anything",
            shared=False,
            matched=True,
        ),
        ModelSpec(
            name="shared_qkv",
            purpose="Kowsher et al. 2024: one projection, a diagonal per role. Prior work, "
            "and the method being improved on",
            shared=True,
        ),
        ModelSpec(
            name="ana_seq_enc",
            purpose="mixing routed over the eight forms, four tokens at a time",
            shared=True,
            mixer=D4Mixing,
            grouping=SEQUENCE,
            sites=ENCODER_ONLY,
        ),
        ModelSpec(
            name="ana_feat_enc",
            purpose="the same, four channels at a time, routed once per TOKEN; against "
            "ana_feat_1_enc it isolates the routing granularity",
            shared=True,
            mixer=D4Mixing,
            grouping=FEATURE,
            sites=ENCODER_ONLY,
        ),
        ModelSpec(
            name="ana_feat_1_enc",
            purpose="the same four channels, routed once per channel GROUP rather than once "
            "per token; the mirror of ana_seq_enc, and against it the axis is the only thing "
            "that changes",
            shared=True,
            mixer=D4Mixing,
            grouping=FEATURE_PER_GROUP,
            sites=ENCODER_ONLY,
        ),
        ModelSpec(
            name="ana_feat_2_enc",
            purpose="ana_feat_1_enc with a routed exponent applied to each group of four "
            "channels before it is mixed and undone after; against ana_feat_1_enc it isolates "
            "the exponent",
            shared=True,
            mixer=D4MixingPowered,
            grouping=FEATURE_PER_GROUP,
            sites=ENCODER_ONLY,
        ),
        ModelSpec(
            name="ana_feat_all",
            purpose="ana_feat_enc at every attention site; isolates the coverage",
            shared=True,
            mixer=D4Mixing,
            grouping=FEATURE,
            sites=ALL_SITES,
        ),
    ]
}


def baseline_parameters(config: ModelConfig) -> int:
    """How many parameters an ordinary transformer of this shape has, without building it.

    Counted rather than measured so that the matched baseline can be solved for directly:
    building a model for each candidate width would mean allocating tens of millions of
    parameters a hundred times over, once per run. `tests/test_params.py` checks this against
    an actual `count_parameters(build_model("baseline", ...))`, so the two cannot drift.
    """
    d, f = config.d_model, config.d_ff

    embedding = config.vocab_size * d
    attention = 4 * (d * d + d)  # query, key, value, output
    feed_forward = (d * f + f) + (f * d + d)  # up, down
    norm = 2 * d  # LayerNorm has a weight and a bias

    encoder_layer = attention + feed_forward + 2 * norm
    decoder_layer = 2 * attention + feed_forward + 3 * norm

    return (
        embedding
        + config.n_encoder_layers * encoder_layer
        + config.n_decoder_layers * decoder_layer
        + 2 * norm  # the final encoder and decoder norms
    )


def mixing_sites(spec: ModelSpec, config: ModelConfig) -> int:
    """How many attention sites this model puts a D4 mixing operator at."""
    if spec.mixer is None:
        return 0
    count = 0
    if Site.ENCODER_SELF in spec.sites:
        count += config.n_encoder_layers
    if Site.DECODER_SELF in spec.sites:
        count += config.n_decoder_layers
    if Site.CROSS in spec.sites:
        count += config.n_decoder_layers
    return count


def shared_parameters(spec: ModelSpec, config: ModelConfig) -> int:
    """A shared-projection model's parameter count, without building it.

    It is the baseline less what sharing removes, plus what the role transforms add back. Each
    transform declares its own cost beyond the `DiagonalRescale` it replaces, in
    `RoleTransform.extra_parameters`, and that cost is paid at three roles per mixing site.

    A `D4Mixing` costs 9w + 10, where w is the width the grouping's routing decision is read
    from: d_model for a grouping that summarises a token, and 4 for one that reads a group's own
    members. A `D4MixingPowered` costs a further GROUP_SIZE + 1 for its exponent router.
    """
    extra = 0
    if spec.mixer is not None:
        per_role = spec.mixer.extra_parameters(config.d_model, spec.grouping)
        extra = mixing_sites(spec, config) * 3 * per_role

    return baseline_parameters(config) - config.shared_qkv_saving + extra


def matched_config(config: ModelConfig) -> ModelConfig:
    """An ordinary transformer narrowed until it is the size of `shared_qkv`.

    `d_model` is swept over its legal values — the multiples of `n_heads` — with `d_ff` held in
    the same ratio to it, and the width whose parameter count comes closest to the target wins.
    The sweep is exhaustive rather than clever because it is arithmetic, not model building:
    there are at most a few hundred candidates and it costs nothing.

    The target is `shared_qkv`, which is the size the whole comparison is set at. The routers are
    not quite free, so the `ana_*` models come out one or two percent above it and the match is
    quantised besides — `d_model` has to stay a multiple of `n_heads`, so an exact hit is not on
    offer. `tests/test_params.py` holds that residual under MAX_SPREAD and the report prints
    every model's size, so it is stated rather than assumed away.
    """
    target = shared_parameters(REGISTRY["shared_qkv"], config)
    ratio = config.d_ff / config.d_model
    step = config.n_heads

    best, best_gap = config, None
    for width in range(step, config.d_model + step, step):
        trial = replace(config, d_model=width, d_ff=max(step, round(ratio * width)))
        gap = abs(baseline_parameters(trial) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = trial, gap

    return best


def model_parameters(name: str, config: ModelConfig) -> int:
    """Any registered model's parameter count, from arithmetic rather than from a built model."""
    spec = REGISTRY[name]
    if spec.matched:
        return baseline_parameters(matched_config(config))
    if not spec.shared:
        return baseline_parameters(config)
    return shared_parameters(spec, config)


def _attention_factory(
    spec: ModelSpec, config: ModelConfig
) -> Callable[[Site], MultiHeadAttention]:
    def build(site: Site) -> MultiHeadAttention:
        if not spec.shared:
            projection = SeparateQKV(config.d_model)
        else:
            mixes_here = spec.mixer is not None and site in spec.sites

            def role(d_model: int) -> RoleTransform:
                if mixes_here:
                    return spec.mixer(d_model, spec.grouping)
                # Where the mixing does not run, fall back to the diagonal, not to the
                # identity. The identity would make the query, key and value streams the
                # same tensor and the attention scores permanently symmetric.
                return DiagonalRescale(d_model)

            projection = SharedQKV(config.d_model, role)

        return MultiHeadAttention(config.d_model, config.n_heads, projection, site, config.dropout)

    return build


def config_for(name: str, config: ModelConfig) -> ModelConfig:
    """The shape this model is actually built at. Only the matched baseline differs."""
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(REGISTRY)}")
    return matched_config(config) if REGISTRY[name].matched else config


def build_model(name: str, config: ModelConfig) -> Seq2SeqTransformer:
    shape = config_for(name, config)
    return Seq2SeqTransformer(shape, _attention_factory(REGISTRY[name], shape))


def count_parameters(model: Seq2SeqTransformer) -> int:
    return sum(p.numel() for p in model.parameters())
