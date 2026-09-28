# Redesign: power-analogy consistency during translation training

2026-09-15. This design was explained to the user as a training-only use of
power. The subsequent active-goal continuation moved it into implementation
as power_consistency_v1. One fresh Multi30k worker launched on exp15 at
02:17:13 JST (wrapper 1943596, worker 1943602), after one focused check
passed in 1.21 seconds and the actual 2,248,512 parameters were confirmed.
No IWSLT14 run is launched. The earlier mixed embedding run had already
completed at 02:04:58 JST when the stop-and-redesign request was checked;
its processes were verified absent at 02:05:34. All checkpoints are retained.
Do not resume stopped experiments. Details of the design follow below.

## Evidence guiding the choice

| Original-recipe Multi30k model | Parameters | BLEU |
|---|---:|---:|
| Similar-size Transformer | 2,249,936 | 40.12 ± 1.71 |
| Shared QKV | 2,213,888 | 38.65 ± 1.65 |
| Linear compressed embeddings, independent Q/K/V | 2,248,512 | 40.31 ± 1.64 |
| Mandatory power-analogy embeddings | 2,248,528 | 39.19 ± 1.67 |
| Linear embeddings plus power residual | 2,248,544 | 40.20 ± 1.62 |
| Linear embeddings plus mixed power residual | 2,249,568 | 39.86 ± 1.65 |

Sources: ../embedding_analogy_v1/reports/multi30k.json and
../embedding_mix_v1/reports/multi30k.json. Symmetric ± is the half-width
of a 95% example-bootstrap interval. The matched and shared models remain
the primary targets; the linear model is an additional structural reference.

These observations motivate retaining unrestricted linear lexical features
and full independent Q/K/V. They do not establish a universal reason that
analogy architectures fail. Reconstructing weights, adding another mixer,
or moving the same residual into a local four-channel coupling is not the
preferred next experiment. Invertibility of a coupling would not establish
good conditioning or recover arbitrary independent Q/K/V projections.

## Proposed computation

Keep the completed linear embedding architecture: width 128, FFN 232,
four heads, four encoder and four decoder layers, tied 96-to-128 compressed
embeddings. No power residual, D4, shared QKV, teacher, or extra projector.

During training, run the same sentence and reference prefix twice with
independent dropout masks. Let P and Q be the two next-word probability
vectors. For any vocabulary candidates i and j, the four terms are:

- A = P(i), B = P(j): preference between the two words in the first view.
- C = Q(i), D = Q(j): the same preference in the second view.

Encourage A^p + D^p to match B^p + C^p. This applies the four-term condition
from [Lepage and Couceiro](https://arxiv.org/html/2407.18770v1), section 2.3.
The quartets concern prediction consistency, not a claim that the two word
types constitute a supervised linguistic analogy. Four quantities have
specific roles; no arbitrary 8-by-8 analogy block is introduced.

First pilot: fixed p = 0.5, so the comparison uses square roots of
probabilities. Define delta_i = sqrt(P(i)) - sqrt(Q(i)). The penalty at
one non-padding target position is S = sum_i (delta_i - mean(delta))^2.
Exactly, S = (1/V) sum_{i<j} (A^p + D^p - B^p - C^p)^2. Thus an O(V)
reduction covers every vocabulary pair without a V-by-V tensor. Do not
replace the final sum with a mean: that would shrink the penalty by V.

Use half the sum of the two ordinary label-smoothed translation losses,
plus lambda times S averaged over non-padding target positions. Proposed
initial lambda = 1 is an engineering starting point, not a value prescribed
by either paper. For numerical safety, use float32 and a small normalized
uniform mixture P' = (1-epsilon)P + epsilon/V (and likewise Q') only for
the power penalty; proposed epsilon = 1e-6. Both distributions then remain
normalized and strictly positive. Do not detach either prediction.

At p=0.5, 0 <= S <= 2. S vanishes exactly when the distributions agree:
otherwise a constant nonzero delta would make every P(i) larger than Q(i),
or every one smaller, contradicting normalization. The supervised loss
still supplies the correct translations; consistency alone is not sufficient.
This penalty is related to, but is not exactly, Hellinger distance because
it removes the common mean difference between square-root probabilities.

Keep p fixed initially. If p appeared only in this penalty and were freely
optimized, it could weaken the penalty by changing its numerical scale.
Learning p requires a separate defensible design; do not silently introduce
it or describe this fixed-p pilot as learned-p training.

## Why this is a different direction

[R-Drop](https://arxiv.org/abs/2106.14448) uses two dropout views and
bidirectional KL consistency, with reported translation improvements.
That is evidence for trying prediction consistency, not evidence that this
new powered relation penalty will improve Multi30k. Our proposed penalty
replaces KL with a specific all-pair four-term analogy defect.

Unlike prior mandatory identities, these four numbers are actual model
predictions with a reason to agree: they answer the same translation question.
Unlike earlier distill_power_v1, there is no frozen teacher or soft-label
distillation, and the new loss itself contains the numerical analogy.

Important scope change: p is used during training only. This is a new
training regularizer for a compact Transformer, not a new inference-time
attention architecture. It is appropriate only if training-time use of the
professor's power analogy fits the user's intended story. Do not present it
as preserving inference-time p or the earlier shared-QKV/D4 mechanism.

## Small first experiment, if this direction is accepted

One fresh Multi30k pilot, existing tokenizer/data and 20,000 optimizer
updates. Keep the current batch size, optimizer schedule, dropout, label
smoothing, development-loss checkpoint choice, and beam decoding. The
two passes change training work: expect roughly twice the forward/backward
work, not an unchanged compute budget. No added inference cost or trainable
parameters; total remains 2,248,512, 13.70% below the original full model.

Reuse saved primary controls and the linear reference. Report actual BLEU
and bootstrap ±; no invented comparisons or significance marks. No COGS,
IWSLT14, hyperparameter grid, automatic continuation, or baseline retraining
is included in this proposal. If useful, a later ordinary R-Drop comparison
would test whether the proposed penalty is competitive with the established
consistency method; that comparison has not been performed.
