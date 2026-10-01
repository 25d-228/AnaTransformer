# COGS generalization: model D with unchanged-input small branches, batch 6

Same model D weights at initialization, parameter count, branch ranks, corpus recipes and controller learning-rate scale 0.1. Only the small encoder Q/K/V branches receive the original attention input; the shared projection still receives the analogy-modified input. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | Exact match (%) |
|---|---:|---:|
| Full Transformer (existing) | 8,844,800 | 78.70 ± 0.53 |
| Shared-QKV (existing) | 5,702,144 | 80.38 ± 0.50 |
| L: D without analogy (existing) | 6,488,576 | 78.11 ± 0.53 |
| D: compact cross-Q/K/V (existing) | 6,522,200 | 82.05 ± 0.50 |
| D-clean: unchanged-input small branches | 6,522,200 | 79.68 ± 0.52 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
