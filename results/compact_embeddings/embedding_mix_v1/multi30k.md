# Multi30k: mixing power-completion residual features

Test · BLEU

One new candidate trains from scratch for 20,000 updates using the original Multi30k recipe: batch size 256, peak learning rate 0.005, 2,000 warmup updates, and inverse-square-root decay. The lowest-development-loss checkpoint is evaluated. No pretrained checkpoint, teacher, continuation, or additional dataset is used.

The candidate uses d_model = 128, d_ff = 232, four attention heads, and four encoder plus four decoder layers. All 12 attention sites retain independent full Q, K, and V projections. The parameter saving comes from the tied source, target, and output embedding table, not from attention or reconstructed FFN weights. It retains the completed linear control's 96 stored coordinates per token and unrestricted learned 96×128 linear map L.

For each token, divide its stored codes X by sigma = 1/sqrt(128) and split them into 32 triples (x,y,z). With epsilon = 0.0001, form A = softplus(x) + epsilon, Delta = softplus(y) + epsilon, B = A + Delta, and C = softplus(z) + epsilon. Complete D_p = (B^p + C^p - A^p)^(1/p). Because B > A and C > 0, the radicand is positive for every allowed p. The four terms satisfy A^p + D_p^p = B^p + C^p. This positive numerical completion comes from the project's [numerical-analogy paper](https://arxiv.org/abs/2407.18770).

Set h_p = D_p - D_1, with D_1 = C+Delta, and use the effective embedding E = X L + sigma h_p H N. H is a learned 32×32 matrix, initialized to the identity. It mixes the 32 residual features before they enter the fixed complement N. N is a fixed 32×128 orthonormal complement of L's initial row space. It is obtained without extra random draws and is not recomputed while L trains. The ordinary linear route remains unrestricted. The 32 powers are shared across the vocabulary, initialized at 1, and bounded between 0.75 and 2. At p = 1, the residual is exactly zero and the model function is the ordinary linear embedding model. The power derivative is generally nonzero there, so the residual can learn from its first update.

This adds 1,024 learned mixing entries to the preceding 2,248,544-parameter residual model, for 2,249,568 total parameters: 368 below the similar-size Transformer. H starts as identity and adds no new random initialization draws. It lets each power feature contribute to combinations of N's fixed directions; it does not enlarge N's span. Initially h_p is zero, so H's gradient starts at zero while p can move immediately. Once residual features become nonzero, H can learn their mapping.

The same effective E is used for input lookup and output prediction. The positive four-term identity describes the completion features, not the final signed embedding coordinates or four labeled words. There is no row normalization, tanh bottleneck, whitening, learned scalar correction gate, or second learned lexical table. H is the only addition to the previous residual design. N contributes 4,096 fixed buffer entries, not trainable parameters. Initial orthogonality does not guarantee orthogonality after L changes.

The similar-size Transformer (d_model = 116, d_ff = 232) and shared-QKV (d_model = 128, d_ff = 256) controls are reused from the completed original full-recipe compact-power runs. The completed linear embedding model at 128/232 and the completed unmixed power-residual model are reused as structural references, not retrained. All four references use saved predictions from the same normal recipe. Parameter savings are relative to the original full Transformer with d_model = 128, d_ff = 256, and 2,605,568 parameters. Keeping the linear model as a special case does not guarantee that a new optimization run will reproduce or improve its trained score.

| Model | d_model | d_ff | Parameters | Saved vs full | Result |
|---|---:|---:|---:|---:|---:|
| Similar-size Transformer | 116 | 232 | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 128 | 256 | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + linear compressed embeddings | 128 | 232 | 2,248,512 | 13.70% | 40.31 ± 1.64 |
| Transformer + linear embeddings and power-completion residual | 128 | 232 | 2,248,544 | 13.70% | 40.20 ± 1.62 |
| Transformer + linear embeddings and mixed power-completion residual | 128 | 232 | 2,249,568 | 13.66% | 39.86 ± 1.65 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

## Primary: mixed power-completion residual minus controls

| Comparison model | Difference | 95% paired interval |
|---|---:|---:|
| Similar-size Transformer | -0.26 | [-1.26, +0.68] |
| Shared-QKV | +1.21 | [+0.14, +2.22] |

## Structural: mixed residual minus saved linear and unmixed residual models

| Comparison model | Difference | 95% paired interval |
|---|---:|---:|
| Transformer + linear compressed embeddings | -0.45 | [-1.34, +0.48] |
| Transformer + linear embeddings and power-completion residual | -0.34 | [-1.16, +0.54] |

New models completed: 1/1.
