# Multi30k: single-gradient power consistency

Test · BLEU

The candidate keeps the linear-compression Transformer: d_model = 128, d_ff = 232, four heads, four encoder/four decoder layers, independent full Q/K/V, and tied 96-to-128 linear embeddings. It has 2,248,512 parameters. Power adds no inference parameters or operations.

Two independent train-mode dropout forwards predict the same target. P carries gradients; Q is a temporary, no-gradient reference from the same current network, not a separate or pretrained teacher. Training uses the existing label-smoothed CE(P) + 2 S(P, stop(Q)). The saved two-gradient reference instead uses 0.5(CE(P) + CE(Q)) + S(P, Q), with both views carrying gradients.

For each valid target position, make probabilities positive through P_tilde = (1-epsilon)P + epsilon/V, and likewise for Q, with epsilon = 0.000001. For any two vocabulary entries i and j, the four terms are A = P_tilde_i, B = P_tilde_j, C = Q_tilde_i, D = Q_tilde_j. The numerical condition is A^p + D^p = B^p + C^p, with fixed p = 0.5. Write delta = sqrt(P_tilde) - sqrt(Q_tilde); then S is sum_v (delta_v - mean(delta))^2, averaged over valid target positions. This exactly aggregates all vocabulary pairs in O(V), without a V×V tensor. Power is training-only; the terms are prediction probabilities.

The four-term condition follows [Lepage and Couceiro](https://arxiv.org/abs/2407.18770). [R-Drop](https://arxiv.org/abs/2106.14448) provides related dropout-view consistency using a different, bidirectional-KL penalty. The factor 2 in this one-gradient version preserves the two-gradient objective's expected raw gradient under independent, exchangeable dropout views; it does not require identical clipped gradients or optimizer paths.

All rows use the original 20,000-update Multi30k base recipe: batch 256, peak learning rate 0.005, 2,000 warmup updates and inverse-square-root decay. Ordinary single-pass development loss selects the checkpoint; test decoding uses beam 5. The four completed references are reused, not retrained. The matched Transformer and shared-QKV are the primary performance controls. Savings use the original full Transformer's 2,605,568 parameters.

| Model | Parameters | Saved vs full | Result |
|---|---:|---:|---:|
| Similar-size Transformer | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + linear compressed embeddings | 2,248,512 | 13.70% | 40.31 ± 1.64 |
| Linear embeddings + power consistency, two gradient passes | 2,248,512 | 13.70% | 40.90 ± 1.64 |
| Linear embeddings + power consistency, one gradient pass | 2,248,512 | 13.70% | 40.00 ± 1.64 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

## Candidate minus saved references

| Reference | BLEU difference | 95% paired interval |
|---|---:|---:|
| Similar-size Transformer | -0.12 | [-1.01, +0.73] |
| Shared-QKV | +1.35 | [+0.50, +2.14] |
| Transformer + linear compressed embeddings | -0.31 | [-1.20, +0.66] |
| Linear embeddings + power consistency, two gradient passes | -0.90 | [-1.57, -0.10] |

## Measured training time

| Model | Training seconds | Training minutes |
|---|---:|---:|
| Linear embeddings + power consistency, two gradient passes | 2933.36 | 48.89 |
| Linear embeddings + power consistency, one gradient pass | 2081.40 | 34.69 |

Observed two-gradient/single-gradient duration ratio: 1.409×. Observed time reduction: 29.04%.

Both runs used the same host and GPU model; their concurrent load may differ. These are measured run durations, including periodic development but excluding final decoding, not a controlled throughput benchmark or guaranteed speedup.
