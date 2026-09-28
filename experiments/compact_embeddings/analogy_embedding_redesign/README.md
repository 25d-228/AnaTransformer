# Redesign: compact lexical representations through power analogy

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/analogy_embedding_redesign](../../../runs/analogy_embedding_redesign/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Design, 2026-09-15. Implemented in src/ana/nn/embeddings/analogy_embedding.py and launched
as embedding_analogy_v1 at 00:44:12 JST: learned power on exp15, linear control
on exp16. Both completed at 01:01 JST: learned power 39.19 +/- 1.67 BLEU,
linear compression 40.31 +/- 1.64. A fresh unit-row normalization follow-up,
embedding_unit_v1, started at 01:03:16 JST and was stopped at the user's
request at 01:11:30 JST, with both 9,000-update checkpoints preserved and
no final test scores. No job is queued or automatically resumed. Actual parameter counts match this
design. The preceding paired-gate experiment on exp15 remains stopped at the
user's request, with its checkpoint retained; it must not resume automatically.

## Decision and reason

Move the parameter saving from Q/K/V or feed-forward weight sharing to the
tied token-embedding/output table. Preserve independent full Q, K and V and
the original Multi30k model width of 128. This is a new compression location,
not another D4 variant or an arbitrary larger permutation block.

The original model has 2,605,568 parameters. Its tied 10,000 by 128 lexical
table accounts for 1,280,000, or 49.13%. The same table supplies source tokens,
target tokens and output vocabulary scores. Saving one quarter of its stored
coordinates provides 320,000 parameters without sharing attention roles.

Recent completed learned-power key, value and FFN reconstruction models scored
39.86, 39.79 and 39.75 BLEU, respectively, below the original full-recipe
similar-size control's 40.12. This does not prove their failure mechanism.
One structural concern is that they constrain weights around copied references
and encode inputs near one: 1 + 0.15*tanh(w/sigma). In the local expansion of
their completion, the leading transferred difference is independent of p.
The learned powers and gains did change during training; they were not inactive.

The proposed power operation instead constructs features used directly by both
token lookup and vocabulary prediction. There is no zero-start correction gate,
copied attention head or reconstructed FFN column.

## Four-term construction

Each token stores 96 real coefficients, arranged as 32 groups of three. A group
provides a positive radius R and two signed contrasts u and v. Use
R = softplus(raw_radius) + epsilon, u = 0.95*tanh(raw_u), and
v = 0.95*tanh(raw_v). Generate four positive features in the order A, B, C, D:

- A = R * (1 + u)^(1/p)
- B = R * (1 + v)^(1/p)
- C = R * (1 - v)^(1/p)
- D = R * (1 - u)^(1/p)

Both opposing powered sums equal 2*R^p, so A^p + D^p = B^p + C^p exactly.
All root bases are positive by construction; no invalid-root clamp is needed.
There are 32 learned powers shared across vocabulary items, not one power per
token. Proposed initialization is p=2, bounded to (0.5,4).

Power is not just a common gain: A/D = ((1+u)/(1-u))^(1/p), which changes
with p. Do not raise the generated features back to p before using them;
that would remove the intended nonlinear effect.

These are four learned numerical feature terms, not four identified words or
verified semantic analogies. The equation has the paper's eight equivalent
forms; the network is not claimed to be invariant under arbitrary permutations.
The group size is four for this reason. The count 32 is simply 128 divided by 4.

## Use inside the Transformer

Apply a fixed common centering/scaling to the positive features, followed by a
single learned bias-free 128 by 128 basis M shared by all vocabulary items.
If Z contains the generated, centered features, the tied embedding is E=Z M.
Input lookup uses rows of E, with the existing sqrt(128) scaling; vocabulary
prediction uses hidden states times E-transpose. The same factors and powers
serve both directions. The numerical equality applies before the centering
and basis, not to the final rotated signed embedding coordinates.

No full learned embedding table is retained. During training, input lookup can
generate only requested rows, while the output layer generates the full table.
The generated table is temporary state, not an additional trainable parameter.
This saves stored parameters, not necessarily peak activation memory or latency.
Actual throughput must be measured on the first real training run.

Initialization matters: p=1 creates an exact linear dependency in every group.
Even p=2 can produce a weak fourth direction. Initialize the shared basis with
blockwise covariance scaling derived from the chosen initialization distribution,
with conservative bounds on amplification. Use a fixed global offset, never
each token's own quartet mean: quartet-wise mean subtraction would introduce
another exact zero-sum constraint. One focused check must verify finite forward
and backward values, sensible initial scale and cached/full decoding agreement.
This does not call for a new training smoke run or a full test suite.

## Multi30k pilot budget

Counts below were confirmed against instantiated models in the focused check.
All models retain four encoder layers, four decoder layers and four heads.

| Model | Width / FFN | Parameters | Status or test BLEU |
|---|---:|---:|---|
| Similar-size ordinary Transformer | 116 / 232 | 2,249,936 | 40.12 +/- 1.71 |
| Shared QKV | 128 / 256 | 2,213,888 | 38.65 +/- 1.65 |
| Ordinary linear compressed embedding, 96 to 128 | 128 / 232 | 2,248,512 | 40.31 +/- 1.64 |
| Power-analogy embedding, 96 to 128, then shared basis | 128 / 230 | 2,248,528 | 39.19 +/- 1.67 |

Candidate embedding parameters: 10,000*96 + 32 + 128*128 = 976,416.
Its ordinary non-embedding backbone at FFN230 has 1,272,112 parameters.
Total: 2,248,528, which is 1,408 below the existing similar-size control.
The slightly reduced FFN230 is budget-driven, not an analogy group size.

The new linear-embedding control is useful because it tests the same decision
to compress lexical storage and preserve full attention. It does not replace
the two primary targets. It stores a 10,000 by 96 table plus a 96 by 128 map;
FFN232 keeps its total within 16 parameters of the powered candidate.

Use the existing full Multi30k recipe, data, trainer and best-dev-loss selection.
Reuse saved predictions from ana-compact-power-v1 for the two primary controls.
Report test BLEU with the existing symmetric 95% bootstrap half-width, plus
saved paired score intervals. No new significance symbols, packages, infrastructure
or extra datasets are needed. COGS/IWSLT14 should wait for a promising translation
result. The first pair completed, and the unit-row pair was stopped on Multi30k;
no other dataset or follow-up is queued.

## Interpretation and risks

The proposed story is: numerical analogy reconstructs part of the lexical
representation, reducing storage while retaining independent attention.
Nonunit power can supply a nonlinear fourth feature where p=1 imposes a linear
dependency. This can increase the span of the generated table relative to
ordinary linear compression, but does not guarantee useful features or BLEU gains.

The main risk moves to lexical capacity: both input representations and word
classification use the compressed table. Learned latent features need not align
with useful translation distinctions. This design also does not automatically
provide enough savings on small-vocabulary COGS; its budget must be reconsidered
before claiming that the same shape meets every dataset's size target.

## Sources

- [Lepage and Couceiro, numerical analogy in power p](https://arxiv.org/abs/2407.18770):
  the four-term positive numerical condition and its equivalent forms.
- [Lioutas et al., Distilled Embedding](https://arxiv.org/html/1910.06720v2):
  prior work compressing tied input/output embeddings in translation and the
  runtime tradeoff of reconstructing an output embedding table. Their trained
  models and distillation procedure are not results for this proposal.
