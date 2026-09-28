# Multi30k: unit-normalized power-analogy embeddings

Test · BLEU

The two new models train from scratch for 20,000 updates using the original Multi30k recipe: batch size 256, peak learning rate 0.005, 2,000 warmup updates, and inverse-square-root decay. The lowest-development-loss checkpoint is evaluated. Neither model loads a pretrained checkpoint or uses a teacher.

Both new models use d_model = 128, four attention heads, and four encoder plus four decoder layers. All 12 attention sites retain independent full Q, K, and V projections. The parameter saving comes from the tied source, target, and output embedding table, not from attention or reconstructed FFN weights. The linear control uses d_ff = 232; the power-analogy model uses d_ff = 230 to keep their total parameter budgets nearly equal.

The linear structural control stores 96 coordinates per token and maps them linearly to 128 coordinates using one learned 96×128 matrix. The power-analogy model also stores 96 coordinates per token: three for each of 32 groups. A group generates a positive radius R and u = 0.95 tanh(alpha), v = 0.95 tanh(beta), then the four positive terms A = R(1+u)^(1/p), D = R(1-u)^(1/p), B = R(1+v)^(1/p), and C = R(1-v)^(1/p). Their powered sums satisfy A^p + D^p = B^p + C^p = 2R^p without a radicand clamp. This is the four-term condition from the project's [numerical-analogy paper](https://arxiv.org/abs/2407.18770).

Fixed common centering and scaling turn those positive terms into signed latent features. A learned shared 128×128 basis maps them to the model's embedding coordinates. Both new models then divide each non-padding effective embedding row by its L2 norm, with a denominator floor of 0.000001. Rows above that floor have unit L2 norm and coordinate RMS 1/sqrt(128). The padding row is left unnormalized to avoid amplifying a near-zero vector. Input lookup and output prediction use the same effective table, including the same padding exception. The existing sqrt(d_model) input multiplier is unchanged. No new learned scale is added; the existing final decoder LayerNorm remains trainable. The numerical equality holds in the positive latent quartet, not in the final rotated signed coordinates; the four terms are not labeled linguistic analogies. There are 32 powers shared across the vocabulary, initialized at 2 and bounded between 0.5 and 4. No D4, attention-cell branch, shared QKV, or zero-start correction gain is added.

Both compressed models initialize stored codes at the ordinary embedding scale 1/sqrt(128). The analogy model uses fixed moments from synthetic samples and a blockwise-whitened initial basis; the linear control uses a scaled orthogonal row basis. Neither initialization uses corpus examples or a trained embedding table.

The similar-size Transformer (d_model = 116, d_ff = 232) and shared-QKV (d_model = 128, d_ff = 256) controls are reused from the completed original full-recipe compact-power runs, not from later continuations. Parameter savings are relative to the original full Transformer with d_model = 128, d_ff = 256, and 2,605,568 parameters. The new linear embedding control tests the compression structure; it does not replace either original performance target.

| Model | d_model | d_ff | Parameters | Saved vs full | Result |
|---|---:|---:|---:|---:|---:|
| Similar-size Transformer | 116 | 232 | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 128 | 256 | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + unit-normalized linear embeddings | 128 | 232 | 2,248,512 | 13.70% | Pending |
| Transformer + unit-normalized power-analogy embeddings (learned p) | 128 | 230 | 2,248,528 | 13.70% | Pending |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

New models completed: 0/2.
