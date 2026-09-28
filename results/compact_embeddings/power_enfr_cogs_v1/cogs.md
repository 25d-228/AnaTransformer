# COGS generalization: compact embeddings and power-based training

Each corpus retains its own training recipe. The two compact rows have identical model structure and parameters; only the training objective differs. The power row uses two dropout passes, four positive probability terms, fixed p = 0.5, and the original all-vocabulary-pair consistency penalty. Power is training-only.

| Model | Parameters | Exact match (%) |
|---|---:|---:|
| Full Transformer | 8,844,800 | 78.70 ± 0.53 |
| Similar-size Transformer | 5,690,376 | 74.99 ± 0.58 |
| Shared-QKV | 5,702,144 | 80.38 ± 0.50 |
| Compact embeddings, ordinary training | 5,689,236 | 78.88 ± 0.55 |
| Same compact model + power training, p = 0.5 | 5,689,236 | 81.10 ± 0.53 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

## Power-trained model minus controls

| Control | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | +2.23 | [+1.97, +2.44] |
| Similar-size Transformer | +6.12 | [+5.76, +6.48] |
| Shared-QKV | +0.73 | [+0.45, +1.00] |

Completed rows are scored from their saved predictions. Copied reference-run paths and full source records are retained in the JSON report. Pending rows are not scores; no significance symbols are inferred.
