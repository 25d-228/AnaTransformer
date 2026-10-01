# Multi30k English → German: analogy-preserving specialization, batch 7

Six designs test small-branch placement, two-rail readouts and query specialization. Original embeddings, corpus recipes, effective batches and single-pass cross-entropy are retained. Controllers use learning-rate scale 0.1. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer (existing) | 2,605,568 | 40.78 ± 1.74 |
| Shared-QKV (existing) | 2,213,888 | 38.65 ± 1.65 |
| L: D without analogy (existing) | 2,312,192 | 39.76 ± 1.73 |
| D: compact cross-Q/K/V (existing) | 2,328,464 | 39.92 ± 1.68 |
| N1: analogy-specialized encoder branches | 2,328,452 | 39.26 ± 1.68 |
| N2: analogy-specialized encoder and cross branches | 2,344,712 | 39.59 ± 1.69 |
| N3: model D with signed and magnitude rail changes | 2,330,000 | 40.13 ± 1.66 |
| N4: specialized branches with magnitude rail changes | 2,329,988 | 40.22 ± 1.74 |
| N5: model D plus decoder self-attention query analogy | 2,336,176 | 40.10 ± 1.67 |
| N6: query-only encoder and decoder self-attention analogy | 2,327,616 | 40.02 ± 1.70 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
