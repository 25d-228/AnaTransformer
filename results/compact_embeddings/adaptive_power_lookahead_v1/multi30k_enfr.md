# Multi30k English → French: compact embeddings and lookahead adaptive power

All six rows use the same compact model structure and parameter count. The first five rows reuse completed runs. The new row keeps the exact scale-balanced analogy loss, but chooses p using three-step temporary training trials and three further training batches for prediction-loss comparison. It changes p only when a candidate improves enough over the current p and wins on at least two of the three scoring batches. Development and test data do not guide p. Power remains training-only.

| Training method | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 60.72 ± 1.72 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 60.66 ± 1.65 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 60.37 ± 1.66 |
| Same compact model + previous adaptive-p training | 2,248,512 | 56.83 ± 1.63 |
| Same compact model + scale-balanced adaptive-p training | 2,248,512 | 59.85 ± 1.64 |
| Same compact model + lookahead adaptive-p training | 2,248,512 | 60.03 ± 1.62 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Previous adaptive p at the end of training: 0.6400.

Scale-balanced adaptive p at the end of training: 0.7500.

Lookahead adaptive p at the end of training: 0.7400.

## Lookahead adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | -0.69 | [-1.45, +0.00] |
| Same compact model, two dropout passes without analogy | -0.63 | [-1.36, +0.14] |
| Same compact model + analogy training, fixed p = 0.5 | -0.35 | [-0.99, +0.30] |
| Same compact model + previous adaptive-p training | +3.19 | [+2.42, +3.94] |
| Same compact model + scale-balanced adaptive-p training | +0.17 | [-0.07, +0.41] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
