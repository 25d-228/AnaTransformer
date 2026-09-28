# Multi30k English → German: compact embeddings and lookahead adaptive power

All six rows use the same compact model structure and parameter count. The first five rows reuse completed runs. The new row keeps the exact scale-balanced analogy loss, but chooses p using three-step temporary training trials and three further training batches for prediction-loss comparison. It changes p only when a candidate improves enough over the current p and wins on at least two of the three scoring batches. Development and test data do not guide p. Power remains training-only.

| Training method | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 40.31 ± 1.64 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 40.12 ± 1.67 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 40.90 ± 1.64 |
| Same compact model + previous adaptive-p training | 2,248,512 | 39.59 ± 1.66 |
| Same compact model + scale-balanced adaptive-p training | 2,248,512 | 40.44 ± 1.63 |
| Same compact model + lookahead adaptive-p training | 2,248,512 | 40.82 ± 1.62 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Previous adaptive p at the end of training: 0.5800.

Scale-balanced adaptive p at the end of training: 0.6400.

Lookahead adaptive p at the end of training: 0.7100.

## Lookahead adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | +0.50 | [-0.33, +1.40] |
| Same compact model, two dropout passes without analogy | +0.69 | [-0.19, +1.57] |
| Same compact model + analogy training, fixed p = 0.5 | -0.08 | [-0.79, +0.72] |
| Same compact model + previous adaptive-p training | +1.22 | [+0.43, +2.03] |
| Same compact model + scale-balanced adaptive-p training | +0.38 | [-0.13, +1.07] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
