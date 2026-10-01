# Multi30k English → German: same-size model D refinements, batch 5

Original embeddings, corpus recipes and single-pass cross-entropy. D1/D2 use controller learning-rate scales 0.3/1.0. D3 retains 0.1 and redistributes the encoder/cross-attention branch budget. COGS scores final weights. IWSLT14 is excluded.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer (existing) | 2,605,568 | 40.78 ± 1.74 |
| Shared-QKV (existing) | 2,213,888 | 38.65 ± 1.65 |
| L: D without analogy (existing) | 2,312,192 | 39.76 ± 1.73 |
| D: compact cross-Q/K/V (existing) | 2,328,464 | 39.92 ± 1.68 |
| D1: model D with controller learning rate 0.3x | 2,328,464 | 40.04 ± 1.64 |
| D2: model D with controller learning rate 1.0x | 2,328,464 | 39.49 ± 1.65 |
| D3: model D with cross-attention-focused branches | 2,328,464 | 39.36 ± 1.65 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
