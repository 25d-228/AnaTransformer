# Proposed redesign: causal translation-example analogy

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/causal_translation_analogy_design](../../../runs/causal_translation_analogy_design/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Designed 2026-09-15. Implemented in src/ana/causal_translation.py during the
subsequent active-goal continuation. One focused check passed in 1.65 seconds;
the actual parameter count is 2,249,545. A fresh Multi30k run launched on exp15
at 07:42:51 JST under runs/causal_translation_v1, without resuming any stopped task.
It completed at 08:06:56 with 39.93 +/- 1.67 BLEU, below the matched-size
control and best completed powered reference. It is not promoted; see the
[completed report](../../../results/compact_embeddings/causal_translation_v1/multi30k.md).
The latest user instruction was to stop the current experiment and redesign.
The quarter-power consistency worker on exp15 was stopped at 07:29:04 JST;
its 6,000-update checkpoint remains preserved. Do not resume any stopped task.

## Decision

Keep independent full Q/K/V, ordinary FFNs and the completed linear compressed
tied embedding backbone (d128, FFN232, four encoder/four decoder layers).
Add a small causal vocabulary-scoring branch that transfers earlier
source-to-target relationships to the current translation step.

This is not D4 projection sharing, an 8x8 block mixer, hard reconstruction of
weights/embeddings, or another exponent change in the dropout loss. Compression
comes from lexical storage. Numerical analogy supplies an additional prediction
score, at training and inference. The first pilot would use ordinary single-pass
translation loss, not the two-pass consistency objective.

The completed linear backbone scored 40.31 +/- 1.64 Multi30k BLEU; adding the
earlier two-gradient power-consistency loss scored 40.90 +/- 1.64. Both results
are retained, not replaced by this untested proposal. Primary controls remain
matched-size 40.12 +/- 1.71 and shared-QKV 38.65 +/- 1.65.

## Four roles, tied to actual translation steps

Let c_t be the last decoder cross-attention output at the position predicting
y_t, before the decoder layer's residual dropout. It is a source-conditioned
vector, not a verified hard word alignment. For each completed earlier step s<t:

| Role | Quantity |
|---|---|
| A | Positive features of earlier source context c_s |
| B | Positive features of the known earlier target token y_s |
| C | Positive features of current source context c_t |
| D | Positive features of a candidate next target token v |

Use two learned 128-to-4 maps, one shared by A/C and one shared by B/D.
Their inputs receive parameter-free LayerNorm. Each map's output z becomes
u=softplus(z)/log(2)+1e-6. Thus all four terms are strictly positive without
restricting the Transformer's ordinary signed representations.

Four feature coordinates are a small parameter-budget choice, not the reason
an analogy has four terms: each coordinate separately has A, B, C and D.
Use fixed p=0.5 for the initial pilot and T(u)=(u^p-1)/p.
The per-reference mismatch is the mean squared difference between
T(B)-T(A) and T(D)-T(C). A zero mismatch is exactly
A^p+D^p=B^p+C^p, coordinatewise. No fourth-root completion or radicand clamp
is required. The condition and its eight equivalent forms come from
[Lepage and Couceiro, Sections 2.3-2.5](https://arxiv.org/html/2407.18770v1).
The whole directional translation network is not claimed to be D4 invariant.

## All earlier examples, without a vocabulary-by-reference tensor

Use all earlier valid target steps in the same sentence. Weight them with
alpha_ts=softmax over s<t of -mean((T(C_t)-T(A_s))^2).
These weights depend on source-context similarity, not the unknown next token.
No reference memory crosses sentences or uses future target tokens.

Compute the weighted mean relation mu_t=sum_s alpha_ts*(T(B_s)-T(A_s)).
Set z_t=T(C_t)+mu_t and U_v=T(D_v). Add
g/4 * (2*z_t dot U_v - ||U_v||^2) to the ordinary vocabulary logits.
Here g is a learned positive scalar, initialized to 0.1 through softplus.

This is exactly the weighted negative squared four-term mismatch, after
dropping a candidate-independent constant. It does not approximate a
vocabulary-by-reference comparison. Source routing costs O(B*T^2*r), and
vocabulary scoring costs O(B*T*V*r), with r=4. Never materialize B*T*V*r
or B*T*T*V tensors. The extra vocabulary dot-product MAC count is about
4/128 of the ordinary width-128 output projection; this is not a measured
end-to-end speed claim. There is no second Transformer pass.

## Causality and parameter budget

Position t consumes y_(t-1) and predicts y_t, using zero-based indices.
At t=0 there is no completed reference, so use ordinary logits alone.
At inference, receiving y_(t-1) finalizes the preceding pending source context
into a reference pair. Store source features and relation features for previous
steps, and reorder both with beam parents. In training, get the same earlier
tokens from the shifted prefix and use the strict s<t mask. The branch must
never gather y_t as the current answer. Padding is excluded.

Analytic parameter estimate: 2,248,512 + 2*(128+1)*4 + 1 = 2,249,545,
391 below the matched-size control. The candidate keeps FFN232 unchanged.
This count was verified on construction in the subsequent implementation.
The vocabulary feature table can be computed once per inference call from
current weights; it is not another trainable embedding table.

## Risk and minimal next experiment

The key hypothesis is that previous translated contexts offer useful examples
for the next word, rather than forcing arbitrary adjacent weights to correspond.
The main risks are inaccurate context-to-token correspondence, repeated-word
bias when similar contexts recur, and averaging incompatible relationships.
The canceled variance term does not create automatic confidence rejection.
Learned maps may collapse or the gain may become negligible; cross-entropy
does not guarantee meaningful semantic analogy. No BLEU improvement is assumed.

Proposed next step is one fresh Multi30k candidate, unchanged 20,000-update
data/optimizer/selection/decoding recipe. Reuse saved control predictions and
report the existing symmetric bootstrap +/- alongside parameter count. Keep
40.90 as the best completed powered reference. No COGS/IWSLT14 queue or p grid.
Before any launch, use one focused numerical/gradient and causal cached-vs-full
decoding check, plus the actual model count; no broad suite or training smoke.
Implementation and launch occurred in the subsequent active-goal continuation;
the initial stop response itself launched nothing.
