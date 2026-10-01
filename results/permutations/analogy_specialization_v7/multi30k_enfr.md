# Multi30k English → French: analogy-preserving specialization, batch 7

Six designs test small-branch placement, two-rail readouts and query specialization. Original embeddings, corpus recipes, effective batches and single-pass cross-entropy are retained. Controllers use learning-rate scale 0.1. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer (existing) | 2,605,568 | 60.80 ± 1.65 |
| Shared-QKV (existing) | 2,213,888 | 58.40 ± 1.70 |
| L: D without analogy (existing) | 2,312,192 | 59.15 ± 1.65 |
| D: compact cross-Q/K/V (existing) | 2,328,464 | 59.66 ± 1.66 |
| N1: analogy-specialized encoder branches | 2,328,452 | 59.76 ± 1.74 |
| N2: analogy-specialized encoder and cross branches | 2,344,712 | 59.35 ± 1.65 |
| N3: model D with signed and magnitude rail changes | 2,330,000 | 59.45 ± 1.66 |
| N4: specialized branches with magnitude rail changes | 2,329,988 | 59.05 ± 1.71 |
| N5: model D plus decoder self-attention query analogy | 2,336,176 | 59.23 ± 1.70 |
| N6: query-only encoder and decoder self-attention analogy | 2,327,616 | 59.40 ± 1.68 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
