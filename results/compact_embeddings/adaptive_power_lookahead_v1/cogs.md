# COGS generalization: compact embeddings and lookahead adaptive power

All six rows use the same compact model structure and parameter count. The first five rows reuse completed runs. The new row keeps the exact scale-balanced analogy loss, but chooses p using three-step temporary training trials and three further training batches for prediction-loss comparison. It changes p only when a candidate improves enough over the current p and wins on at least two of the three scoring batches. Development and test data do not guide p. Power remains training-only.

| Training method | Parameters | Exact match (%) |
|---|---:|---:|
| Compact embeddings, ordinary training | 5,689,236 | 78.88 ± 0.55 |
| Same compact model, two dropout passes without analogy | 5,689,236 | 81.52 ± 0.50 |
| Same compact model + analogy training, fixed p = 0.5 | 5,689,236 | 81.10 ± 0.53 |
| Same compact model + previous adaptive-p training | 5,689,236 | 76.05 ± 0.55 |
| Same compact model + scale-balanced adaptive-p training | 5,689,236 | 80.99 ± 0.54 |
| Same compact model + lookahead adaptive-p training | 5,689,236 | 81.74 ± 0.51 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

Previous adaptive p at the end of training: 0.4500.

Scale-balanced adaptive p at the end of training: 0.4300.

Lookahead adaptive p at the end of training: 0.3500.

## Lookahead adaptive-power training minus comparison rows

| Comparison row | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | +2.86 | [+2.62, +3.11] |
| Same compact model, two dropout passes without analogy | +0.22 | [+0.03, +0.40] |
| Same compact model + analogy training, fixed p = 0.5 | +0.63 | [+0.48, +0.81] |
| Same compact model + previous adaptive-p training | +5.69 | [+5.36, +6.02] |
| Same compact model + scale-balanced adaptive-p training | +0.75 | [+0.57, +0.94] |

Completed rows are scored from their saved predictions. Original result records and reference-run sources are retained in JSON. Pending rows have no scores. Each corpus keeps its original base recipe; adaptive-power probes add training work, not inference work.
