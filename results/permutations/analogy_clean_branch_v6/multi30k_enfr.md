# Multi30k English → French: model D with unchanged-input small branches, batch 6

Same model D weights at initialization, parameter count, branch ranks, corpus recipes and controller learning-rate scale 0.1. Only the small encoder Q/K/V branches receive the original attention input; the shared projection still receives the analogy-modified input. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer (existing) | 2,605,568 | 60.80 ± 1.65 |
| Shared-QKV (existing) | 2,213,888 | 58.40 ± 1.70 |
| L: D without analogy (existing) | 2,312,192 | 59.15 ± 1.65 |
| D: compact cross-Q/K/V (existing) | 2,328,464 | 59.66 ± 1.66 |
| D-clean: unchanged-input small branches | 2,328,464 | 59.52 ± 1.65 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
