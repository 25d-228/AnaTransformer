# Multi30k English → French: compact embeddings and adaptive power

All four rows use the same compact model structure and parameter count. Ordinary training and fixed-p training reuse completed runs. The two new rows use two independently dropped-out supervised passes, with either no analogy penalty or an analogy penalty whose power adapts during training. Power and its adaptation are absent from inference.

| Training method | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 60.72 ± 1.72 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 60.66 ± 1.65 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 60.37 ± 1.66 |
| Same compact model + analogy training, adaptive p | 2,248,512 | 56.83 ± 1.63 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Adaptive p at the end of training: 0.6400.

## Adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | -3.88 | [-4.77, -3.03] |
| Same compact model, two dropout passes without analogy | -3.83 | [-4.67, -2.95] |
| Same compact model + analogy training, fixed p = 0.5 | -3.54 | [-4.39, -2.75] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
