# Multi30k: power-analogy dropout consistency

Test · BLEU

One fresh candidate keeps the completed linear-compression architecture: d_model = 128, d_ff = 232, four heads, four encoder/four decoder layers, and independent full Q/K/V at all 12 attention sites. Each token stores 96 coordinates, mapped through a learned 96×128 matrix. Input embeddings and output prediction weights remain tied. There are no additional inference parameters or inference operations relative to the linear control.

Training uses two independent dropout passes on each example. Average their existing label-smoothed translation losses, then add a fixed coefficient of 1 times the power-analogy consistency penalty. At each valid target position, let P and Q be the two next-word distributions. Use P_tilde = (1-epsilon)P + epsilon/V and the same rule for Q, with epsilon = 0.000001 and vocabulary size V. This uniform mixture keeps all four comparison terms positive; it is separate from label smoothing.

For every pair of vocabulary entries i and j, the four quantities are A = P_tilde_i, B = P_tilde_j, C = Q_tilde_i, D = Q_tilde_j. Their desired numerical relation is A^p + D^p = B^p + C^p, using fixed p = 0.5. Define delta = sqrt(P_tilde) - sqrt(Q_tilde). The penalty for a target position is S = sum_v (delta_v - mean(delta))^2; average S over non-padding target positions. This covers all vocabulary pairs through an O(V) reduction, not an explicit V×V comparison matrix.

The four-term condition comes from [Lepage and Couceiro](https://arxiv.org/abs/2407.18770). [R-Drop](https://arxiv.org/abs/2106.14448) supplies the related idea of training consistency between independent dropout predictions, using bidirectional KL rather than this centered power-gap penalty. Those sources do not establish that this new penalty improves this model. Power is used only in training, is not learned, and is absent from the inference computation. The four terms are prediction probabilities, not four independently identified words forming a semantic analogy.

The original base settings are retained: 20,000 optimizer updates, 256 distinct examples per batch, peak learning rate 0.005, 2,000 warmup updates and inverse-square-root decay. Two gradient-bearing passes roughly double model training work per update; this is not an unchanged compute budget. Selection uses ordinary single-pass development translation loss; decoding remains beam 5. No teacher, pretrained checkpoint, continuation or additional dataset is used.

The similar-size Transformer (d_model = 116, d_ff = 232) and shared-QKV (d_model = 128, d_ff = 256) controls are reused from the completed original full-recipe compact-power runs. The completed linear embedding model at 128/232 is reused as a third, structural reference, not retrained. All three references use saved predictions with the same base optimizer/update settings, but without the new two-pass consistency objective. Parameter savings are relative to the original full Transformer with d_model = 128, d_ff = 256, and 2,605,568 parameters. The original matched Transformer and shared-QKV remain the primary performance targets.

| Model | d_model | d_ff | Parameters | Saved vs full | Result |
|---|---:|---:|---:|---:|---:|
| Similar-size Transformer | 116 | 232 | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 128 | 256 | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + linear compressed embeddings | 128 | 232 | 2,248,512 | 13.70% | 40.31 ± 1.64 |
| Linear embeddings + power-analogy dropout consistency (p = 0.5) | 128 | 232 | 2,248,512 | 13.70% | 40.90 ± 1.64 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

## Primary: power-consistency training minus controls

| Comparison model | Difference | 95% paired interval |
|---|---:|---:|
| Similar-size Transformer | +0.78 | [-0.08, +1.58] |
| Shared-QKV | +2.24 | [+1.32, +3.02] |

## Structural: power-consistency training minus saved linear embeddings

| Comparison model | Difference | 95% paired interval |
|---|---:|---:|
| Transformer + linear compressed embeddings | +0.58 | [-0.28, +1.44] |

New models completed: 1/1.
