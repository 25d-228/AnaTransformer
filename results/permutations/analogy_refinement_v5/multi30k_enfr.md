# Multi30k English → French: same-size model D refinements, batch 5

Original embeddings, corpus recipes and single-pass cross-entropy. D1/D2 use controller learning-rate scales 0.3/1.0. D3 retains 0.1 and redistributes the encoder/cross-attention branch budget. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer (existing) | 2,605,568 | 60.80 ± 1.65 |
| Shared-QKV (existing) | 2,213,888 | 58.40 ± 1.70 |
| L: D without analogy (existing) | 2,312,192 | 59.15 ± 1.65 |
| D: compact cross-Q/K/V (existing) | 2,328,464 | 59.66 ± 1.66 |
| D1: model D with controller learning rate 0.3x | 2,328,464 | 59.48 ± 1.65 |
| D2: model D with controller learning rate 1.0x | 2,328,464 | 59.64 ± 1.64 |
| D3: model D with cross-attention-focused branches | 2,328,464 | 59.40 ± 1.63 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
