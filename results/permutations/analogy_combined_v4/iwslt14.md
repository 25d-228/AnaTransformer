# IWSLT14 German → English: combined analogy-preserving projections, batch 4

Original standard embeddings and corpus recipes; ordinary single-pass cross-entropy training. Routers use one tenth of the base learning rate. COGS still scores final weights; its best-dev checkpoint is diagnostic only.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer, standard embeddings (existing) | 36,665,344 | 33.45 ± 0.51 |
| Shared QKV (existing) | 27,237,376 | 31.58 ± 0.49 |
| A: encoder role mixing + independent cross-attention Q | 30,093,832 | 32.92 ± 0.49 |
| B: wider encoder role mixing + independent cross-attention Q | 31,273,480 | 32.88 ± 0.49 |
| C: encoder role mixing + compact cross-attention Q | 28,911,112 | 32.81 ± 0.49 |
| D: encoder role mixing + compact cross-attention Q/K/V | 29,697,544 | 32.95 ± 0.51 |
| E: combined model + compact cross-attention K/V | 30,880,264 | 32.96 ± 0.50 |
| F: combined model + compact decoder self-attention Q/K | 30,880,264 | 33.11 ± 0.51 |
| G: redistributed encoder and cross-attention Q/K/V mixing | 28,517,896 | 32.64 ± 0.50 |
| H: token-controlled compact cross-attention Q/K/V | 29,706,778 | 32.86 ± 0.51 |
| I: redistributed branches with token controls | 28,527,130 | 32.75 ± 0.50 |
| J: redistributed branches with shared bottlenecks | 28,124,680 | 32.80 ± 0.48 |
| K: analogy inputs with diagonal shortcuts | 27,356,680 | 31.70 ± 0.49 |
| L: model D without the analogy module | 29,596,672 | 32.64 ± 0.49 |
| M: model G without the analogy module | 28,417,024 | 32.77 ± 0.51 |
| Preprojection analogy + independent cross-attention Q | 28,914,184 | 31.57 ± 0.46 |
| Preprojection analogy + small independent role residuals | 28,517,896 | 32.99 ± 0.49 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
