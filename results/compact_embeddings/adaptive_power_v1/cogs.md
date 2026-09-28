# COGS generalization: compact embeddings and adaptive power

All four rows use the same compact model structure and parameter count. Ordinary training and fixed-p training reuse completed runs. The two new rows use two independently dropped-out supervised passes, with either no analogy penalty or an analogy penalty whose power adapts during training. Power and its adaptation are absent from inference.

| Training method | Parameters | Exact match (%) |
|---|---:|---:|
| Compact embeddings, ordinary training | 5,689,236 | 78.88 ± 0.55 |
| Same compact model, two dropout passes without analogy | 5,689,236 | 81.52 ± 0.50 |
| Same compact model + analogy training, fixed p = 0.5 | 5,689,236 | 81.10 ± 0.53 |
| Same compact model + analogy training, adaptive p | 5,689,236 | 76.05 ± 0.55 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Adaptive p at the end of training: 0.4500.

## Adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | -2.82 | [-3.13, -2.50] |
| Same compact model, two dropout passes without analogy | -5.47 | [-5.79, -5.12] |
| Same compact model + analogy training, fixed p = 0.5 | -5.05 | [-5.36, -4.74] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
