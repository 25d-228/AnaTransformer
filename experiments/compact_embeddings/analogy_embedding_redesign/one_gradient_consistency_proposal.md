# Proposed redesign: cheaper four-term power-consistency training

2026-09-15. The subsequent active-goal continuation moved this proposal into
implementation as power_consistency_singlegrad_v1, a fresh Multi30k task,
not a restart of any stopped task. The user explicitly requested
stopping the current experiment and redesigning. IWSLT14 wrapper 945911 and
worker 945917 on exp18 were stopped at 06:17:05 JST and confirmed absent at
06:17:29. Its update-16,000 checkpoint is retained. The new task's actual
status is in ../power_consistency_singlegrad_v1/launch.json. No additional
dataset or automatic follow-up is queued.

## Evidence and choice

The completed compact linear-embedding Transformer with two-gradient-view
power consistency scored 40.90 ± 1.64 on Multi30k, versus 40.12 ± 1.71 for
the similar-size Transformer and 38.65 ± 1.65 for shared QKV. Its linear
embedding structural reference scored 40.31 ± 1.64. These are completed
test results, not the incomplete IWSLT14 evaluation.

Forcing power-analogy embedding coordinates scored 39.19, a power residual
40.20, and residual mixing 39.86. These observations favor preserving the
successful compact backbone rather than constraining another weight matrix.
They do not establish the cause of every earlier failure.

The recommended redesign targets training work. Keep the full independent
Q/K/V, ordinary FFNs, and tied linear-compressed embedding. At the existing
Multi30k shape the trainable parameter count remains 2,248,512. No D4,
8-by-8 block, teacher model, or inference-time module is added.

## Computation

Use the same batch twice, with independent dropout and model training mode
enabled for both passes. One pass is computed without a gradient graph and
is a temporary reference; the other is the trainable pass. No moving-average
teacher or pretrained checkpoint is introduced. Discard both predictions
after the update.

For two vocabulary choices i,j, set A=P(i), B=P(j), C=Q(i), D=Q(j), where
P is the trainable prediction and Q is the temporary reference. Keep the
same positive uniform floor, fixed p=0.5, and exact all-pairs penalty S:
delta=sqrt(P)-sqrt(Q); S=sum_v((delta_v-mean_v(delta))^2), averaged over
non-padding target positions. This is the four-term powered defect for
every pair, computed in O(V).

The proposed loss is CE(P) + 2*S(P, stop_gradient(Q)). The factor two is
intentional. The completed method used 0.5*(CE(P)+CE(Q)) + S(P,Q), with
gradients through both views. Conditional on a batch and current weights,
independent identically distributed dropout and symmetry of S imply that
the proposed estimator has the same expected raw gradient. A realized
update, its variance, gradient clipping, Adam history, and BLEU are not
therefore identical. It is a separate experiment, not a transparent resume
optimization. The numerical loss value is not directly comparable either.

The reference pass must remain in training mode; switching it to evaluation
mode would break the exchangeability argument. Power remains fixed and used
only during training. It is not present in inference.

## Expected benefit and risk

There are still two forward passes, but only one backward pass and one set
of retained backbone activations. Lower training work and activation memory
are expected; speed and peak memory have not been measured. Logit/reference
tensors still require memory. The noisier gradient estimate may lose some
of the completed Multi30k improvement.

## Bounded next experiment, only after approval

One fresh Multi30k run, original 20,000-update recipe, normal development
checkpoint selection, saved test predictions, and bootstrap ±. Reuse the
completed controls and the completed two-gradient-view result. Check only
the small loss/gradient computation needed for this change; no broad suite,
training smoke, environment change, or new infrastructure.

Compare both BLEU and measured training cost. Do not launch IWSLT14 or COGS
automatically, and do not resume the stopped IWSLT14 task. Consider a second
translation run only if this pilot keeps a useful advantage over both primary
controls and provides a worthwhile speed improvement.

Evidence: [completed power-consistency report](../../../results/compact_embeddings/power_consistency_v1/multi30k.md)
and [embedding comparison](../../../results/compact_embeddings/embedding_mix_v1/multi30k.md).

Method sources: https://arxiv.org/html/2407.18770v1 for the numerical
four-term condition; https://arxiv.org/abs/2106.14448 for related two-dropout
consistency training with a different, KL-based penalty. The one-gradient
estimator above is our proposed change, not a result attributed to either
paper.
