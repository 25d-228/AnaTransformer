# Multi30k English → French: compact embeddings and power-based training

Each corpus retains its own training recipe. The two compact rows have identical model structure and parameters; only the training objective differs. The power row uses two dropout passes, four positive probability terms, fixed p = 0.5, and the original all-vocabulary-pair consistency penalty. Power is training-only.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer | 2,605,568 | 60.80 ± 1.65 |
| Similar-size Transformer | 2,249,936 | 59.52 ± 1.67 |
| Shared-QKV | 2,213,888 | 58.40 ± 1.70 |
| Compact embeddings, ordinary training | 2,248,512 | 60.72 ± 1.72 |
| Same compact model + power training, p = 0.5 | 2,248,512 | 60.37 ± 1.66 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

## Power-trained model minus controls

| Control | Difference | 95% paired interval |
|---|---:|---:|
| Compact embeddings, ordinary training | -0.34 | [-1.09, +0.36] |
| Similar-size Transformer | +0.85 | [+0.08, +1.61] |
| Shared-QKV | +1.97 | [+1.18, +2.76] |

Completed rows are scored from their saved predictions. Copied reference-run paths and full source records are retained in the JSON report. Pending rows are not scores; no significance symbols are inferred.
