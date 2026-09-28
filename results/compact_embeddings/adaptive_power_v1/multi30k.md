# Multi30k English → German: compact embeddings and adaptive power

All four rows use the same compact model structure and parameter count. Ordinary training and fixed-p training reuse completed runs. The two new rows use two independently dropped-out supervised passes, with either no analogy penalty or an analogy penalty whose power adapts during training. Power and its adaptation are absent from inference.

| Training method | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 40.31 ± 1.64 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 40.12 ± 1.67 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 40.90 ± 1.64 |
| Same compact model + analogy training, adaptive p | 2,248,512 | 39.59 ± 1.66 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Adaptive p at the end of training: 0.5800.

## Adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | -0.72 | [-1.66, +0.23] |
| Same compact model, two dropout passes without analogy | -0.53 | [-1.60, +0.38] |
| Same compact model + analogy training, fixed p = 0.5 | -1.31 | [-2.15, -0.47] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
