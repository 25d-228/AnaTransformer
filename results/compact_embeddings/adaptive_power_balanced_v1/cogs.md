# COGS generalization: compact embeddings and scale-balanced adaptive power

All five rows use the same compact model structure and parameter count. The first four rows reuse completed runs. The new row keeps the previous adaptive-p rule, but rescales its batch penalty to the p=0.5 reference magnitude using a detached scalar ratio. This changes the penalty's gradient scale, not the model architecture. Power remains training-only.

| Training method | Parameters | Exact match (%) |
|---|---:|---:|
| Compact embeddings, ordinary training | 5,689,236 | 78.88 ± 0.55 |
| Same compact model, two dropout passes without analogy | 5,689,236 | 81.52 ± 0.50 |
| Same compact model + analogy training, fixed p = 0.5 | 5,689,236 | 81.10 ± 0.53 |
| Same compact model + previous adaptive-p training | 5,689,236 | 76.05 ± 0.55 |
| Same compact model + scale-balanced adaptive-p training | 5,689,236 | 80.99 ± 0.54 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Previous adaptive p at the end of training: 0.4500.

Scale-balanced adaptive p at the end of training: 0.4300.

## Scale-balanced adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | +2.11 | [+1.85, +2.34] |
| Same compact model, two dropout passes without analogy | -0.53 | [-0.72, -0.35] |
| Same compact model + analogy training, fixed p = 0.5 | -0.12 | [-0.28, +0.05] |
| Same compact model + previous adaptive-p training | +4.93 | [+4.60, +5.26] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
