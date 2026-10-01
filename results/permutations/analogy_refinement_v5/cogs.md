# COGS generalization: same-size model D refinements, batch 5

Original embeddings, corpus recipes and single-pass cross-entropy. D1/D2 use controller learning-rate scales 0.3/1.0. D3 retains 0.1 and redistributes the encoder/cross-attention branch budget. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | Exact match (%) |
|---|---:|---:|
| Full Transformer (existing) | 8,844,800 | 78.70 ± 0.53 |
| Shared-QKV (existing) | 5,702,144 | 80.38 ± 0.50 |
| L: D without analogy (existing) | 6,488,576 | 78.11 ± 0.53 |
| D: compact cross-Q/K/V (existing) | 6,522,200 | 82.05 ± 0.50 |
| D1: model D with controller learning rate 0.3x | 6,522,200 | 77.60 ± 0.56 |
| D2: model D with controller learning rate 1.0x | 6,522,200 | 77.72 ± 0.55 |
| D3: model D with cross-attention-focused branches | 6,522,200 | 77.59 ± 0.54 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
