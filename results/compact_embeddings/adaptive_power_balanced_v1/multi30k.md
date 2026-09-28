# Multi30k English → German: compact embeddings and scale-balanced adaptive power

All five rows use the same compact model structure and parameter count. The first four rows reuse completed runs. The new row keeps the previous adaptive-p rule, but rescales its batch penalty to the p=0.5 reference magnitude using a detached scalar ratio. This changes the penalty's gradient scale, not the model architecture. Power remains training-only.

| Training method | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 40.31 ± 1.64 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 40.12 ± 1.67 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 40.90 ± 1.64 |
| Same compact model + previous adaptive-p training | 2,248,512 | 39.59 ± 1.66 |
| Same compact model + scale-balanced adaptive-p training | 2,248,512 | 40.44 ± 1.63 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Previous adaptive p at the end of training: 0.5800.

Scale-balanced adaptive p at the end of training: 0.6400.

## Scale-balanced adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | +0.12 | [-0.84, +0.94] |
| Same compact model, two dropout passes without analogy | +0.31 | [-0.68, +1.11] |
| Same compact model + analogy training, fixed p = 0.5 | -0.46 | [-1.23, +0.37] |
| Same compact model + previous adaptive-p training | +0.84 | [+0.00, +1.67] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
