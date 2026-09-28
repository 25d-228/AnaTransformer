# Multi30k: linear embeddings with a power-completion residual

Test · BLEU

One new candidate trains from scratch for 20,000 updates using the original Multi30k recipe: batch size 256, peak learning rate 0.005, 2,000 warmup updates, and inverse-square-root decay. The lowest-development-loss checkpoint is evaluated. No pretrained checkpoint, teacher, continuation, or additional dataset is used.

The candidate uses d_model = 128, d_ff = 232, four attention heads, and four encoder plus four decoder layers. All 12 attention sites retain independent full Q, K, and V projections. The parameter saving comes from the tied source, target, and output embedding table, not from attention or reconstructed FFN weights. It retains the completed linear control's 96 stored coordinates per token and unrestricted learned 96×128 linear map L.

For each token, divide its stored codes X by sigma = 1/sqrt(128) and split them into 32 triples (x,y,z). With epsilon = 0.0001, form A = softplus(x) + epsilon, Delta = softplus(y) + epsilon, B = A + Delta, and C = softplus(z) + epsilon. Complete D_p = (B^p + C^p - A^p)^(1/p). Because B > A and C > 0, the radicand is positive for every allowed p. The four terms satisfy A^p + D_p^p = B^p + C^p. This positive numerical completion comes from the project's [numerical-analogy paper](https://arxiv.org/abs/2407.18770).

Set h_p = D_p - D_1, with D_1 = C+Delta, and use the effective embedding E = X L + sigma h_p N. N is a fixed 32×128 orthonormal complement of L's initial row space. It is obtained without extra random draws and is not recomputed while L trains. The ordinary linear route remains unrestricted; only 32 powers are added. They are shared across the vocabulary, initialized at 1, and bounded between 0.75 and 2. At p = 1, the residual is exactly zero and the model function is the ordinary linear embedding model. The power derivative is generally nonzero there, so the residual can learn from its first update.

The same effective E is used for input lookup and output prediction. The positive four-term identity describes the completion features, not the final signed embedding coordinates or four labeled words. There is no row normalization, tanh bottleneck, whitening, learned correction gain, or additional learned feature projector. N contributes 4,096 fixed buffer entries, not trainable parameters. Initial orthogonality does not guarantee orthogonality after L changes.

The similar-size Transformer (d_model = 116, d_ff = 232) and shared-QKV (d_model = 128, d_ff = 256) controls are reused from the completed original full-recipe compact-power runs. The completed linear embedding model at 128/232 is reused as a third, structural reference, not retrained. All three references use saved predictions from the same normal recipe. Parameter savings are relative to the original full Transformer with d_model = 128, d_ff = 256, and 2,605,568 parameters. Keeping the linear model as a special case does not guarantee that a new optimization run will reproduce or improve its trained score.

| Model | d_model | d_ff | Parameters | Saved vs full | Result |
|---|---:|---:|---:|---:|---:|
| Similar-size Transformer | 116 | 232 | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 128 | 256 | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + linear compressed embeddings | 128 | 232 | 2,248,512 | 13.70% | 40.31 ± 1.64 |
| Transformer + linear embeddings and power-completion residual | 128 | 232 | 2,248,544 | 13.70% | 40.20 ± 1.62 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

## Primary: power-completion residual minus controls

| Comparison model | Difference | 95% paired interval |
|---|---:|---:|
| Similar-size Transformer | +0.08 | [-0.79, +0.95] |
| Shared-QKV | +1.55 | [+0.57, +2.38] |

## Structural: power-completion residual minus saved linear embeddings

| Comparison model | Difference | 95% paired interval |
|---|---:|---:|
| Transformer + linear compressed embeddings | -0.11 | [-1.05, +0.82] |

New models completed: 1/1.
