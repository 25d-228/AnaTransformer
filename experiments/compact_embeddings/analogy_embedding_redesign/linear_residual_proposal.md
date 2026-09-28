# Redesign proposal: retain linear features, add power completion

2026-09-15. The next active-goal continuation moved this newly proposed
design into implementation as embedding_residual_v1. Its single new Multi30k
worker launched detached on exp15 at 01:24:00 JST, after one focused check
passed in 1.64 seconds. It completed at 01:41:39 JST with 40.20 +/- 1.62
BLEU: +0.08 against the primary similar-size control and +1.55 against
shared QKV. A separate learned-feature-mixing follow-up is being prepared
to seek a larger margin before IWSLT14. The user requested stopping the preceding
experiment and rethinking the design. Both
embedding_unit_v1 workers were stopped at 01:11:30 JST, verified gone by
01:12:01, with their 9,000-update checkpoints retained. Do not resume them.
The new residual task is a separate experiment, not a continuation of either
stopped unit-row model.

## Evidence and decision

Completed original-recipe Multi30k results:

| Model | Parameters | Test BLEU |
|---|---:|---:|
| Similar-size ordinary Transformer | 2,249,936 | 40.12 ± 1.71 |
| Shared QKV | 2,213,888 | 38.65 ± 1.65 |
| Linear compressed embeddings | 2,248,512 | 40.31 ± 1.64 |
| Previous power-analogy embeddings | 2,248,528 | 39.19 ± 1.67 |

Source: ../embedding_analogy_v1/reports/multi30k.json. The powered-minus-linear
difference was -1.12 BLEU, paired 95% interval [-1.93, -0.31]. The stopped
unit-row models have no final test scores and are not included above.

Keep the useful linear compressed representation completely unrestricted.
Do not force every lexical coordinate through the positive quartet chart.
Use power analogy only to supply additional nonlinear features. Parameter
saving still comes from the lexical table; independent full Q/K/V and the
ordinary FFN remain. This continues compact numerical representations as a
research objective, but is a different compression mechanism from the paper's
shared-QKV/D4 attention. It must not be described as the same architecture.

## Exact proposed computation

Keep X[V,96] and the learned linear map L[96,128] exactly as in the completed
linear model, including its FFN width 232. Set sigma=1/sqrt(128). Split each
row X/sigma into 32 triples (x,y,z). For each triple form positive values:

- A = softplus(x) + epsilon.
- Delta = softplus(y) + epsilon, and B = A + Delta.
- C = softplus(z) + epsilon.
- D_p = (B^p + C^p - A^p)^(1/p).

Because B>A, the radicand is positive for every allowed p>0. These four
positive terms satisfy A^p + D_p^p = B^p + C^p exactly in real arithmetic.
They are latent numerical features, not four identified words. The number
four comes from the analogy equation; eight equivalent orderings do not
justify an 8-by-8 representation block.

Compute h_p = D_p - D_1, where D_1 = B+C-A. Generate the embedding as
E = X L + sigma h_p N. N[32,128] is a fixed orthonormal complement of the
initial row space of L, obtained without additional random draws. N is not
recomputed as L trains. The same effective E serves lookup and tied output
prediction. No per-row normalization, tanh bottleneck, whitening, learned
correction gain, or extra feature projector is added.

Learn 32 powers, shared across all vocabulary rows. Proposed numerical range
is (0.75,2), initialized at exactly 1; this range is an engineering choice,
not a requirement from the paper. At p=1 the added feature is exactly zero,
so the model contains the full ordinary linear model as a special case.
This is equality of model functions, not a promise to reproduce its trained
BLEU after optimizing a different model from scratch.

Unlike a zero output gate, p=1 does not generically block the power gradient:
d(h_p)/dp at 1 is B log B + C log C - A log A - D_1 log D_1.
For (A,B,C)=(1,2,3), D_1=4, D_2=sqrt(12)=3.4641, so h_2=-0.5359;
the derivative at 1 is -0.8630. Changing p changes the nonlinear feature
shape across triples, not merely a common scale. No global identifiability
claim is made when every other model weight can also change.

## Budget and implementation cautions

Proposed trainable count: 960,000 codes + 12,288 linear-map entries + 32
powers + 1,276,224 backbone parameters = 2,248,544. This is 1,392 below
the similar-size control and 13.70% below the 2,605,568 full model.
The focused check confirmed this count on an instantiated new model.
N adds 4,096 fixed buffer entries (16 KiB in float32), not trainable weights.

If implemented, retain Delta separately to avoid recovering a small number
by B-A. Use stable log/expm1 completion, and evaluate the p=1 reference with
the same differentiable kernel. Do not special-case p==1 with a zero branch,
detach the reference's code gradients, or multiply by a zero-initialized gate.
These mistakes could hide or block the intended first-step derivative.

N stays bounded and initially points outside the linear embedding span;
L can later rotate, so permanent orthogonality and full effective rank are
not guaranteed. Extra nonlinear features can still hurt optimization or be
irrelevant. Keeping the linear route prevents a mandatory representation
restriction; it does not guarantee a better trained translation score.

## Next experiment under the active implement-and-try goal

One Multi30k candidate from scratch, original recipe and development-loss
checkpoint selection, compared with the saved primary controls and the
completed linear model. Fixed p=1 is functionally the linear model, not a
new expensive architecture to train. Before launch, one focused check should
verify this identity, actual translation-loss power gradients, tied decoding,
finite values and the count. No broad test suite or extra training smoke.
No COGS/IWSLT14, checkpoint continuation, or hyperparameter grid is queued.

An alternative considered was four-term consistency between two dropout
predictions, inspired by R-Drop. That has a meaningful translation training
role, but would put p only in a new training loss and require two forward
passes. It is deferred here to preserve inference-time power computation.

Sources: [Lepage and Couceiro](https://arxiv.org/html/2407.18770v1), sections
2.3/2.5 and 4.1 for four-term power analogy and completion;
[R-Drop](https://arxiv.org/abs/2106.14448) for the deferred consistency idea.
Neither paper supplies empirical evidence for this proposed model.
