# COGS generalization: analogy-preserving specialization, batch 7

Six designs test small-branch placement, two-rail readouts and query specialization. Original embeddings, corpus recipes, effective batches and single-pass cross-entropy are retained. Controllers use learning-rate scale 0.1. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | Exact match (%) |
|---|---:|---:|
| Full Transformer (existing) | 8,844,800 | 78.70 ± 0.53 |
| Shared-QKV (existing) | 5,702,144 | 80.38 ± 0.50 |
| L: D without analogy (existing) | 6,488,576 | 78.11 ± 0.53 |
| D: compact cross-Q/K/V (existing) | 6,522,200 | 82.05 ± 0.50 |
| N1: analogy-specialized encoder branches | 6,522,194 | 80.30 ± 0.53 |
| N2: analogy-specialized encoder and cross branches | 6,555,812 | 78.49 ± 0.55 |
| N3: model D with signed and magnitude rail changes | 6,525,272 | 77.69 ± 0.54 |
| N4: specialized branches with magnitude rail changes | 6,525,266 | 79.81 ± 0.51 |
| N5: model D plus decoder self-attention query analogy | 6,539,256 | 78.70 ± 0.55 |
| N6: query-only encoder and decoder self-attention analogy | 6,522,688 | 81.20 ± 0.49 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
