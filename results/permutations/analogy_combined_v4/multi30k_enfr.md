# Multi30k English → French: combined analogy-preserving projections, batch 4

Original standard embeddings and corpus recipes; ordinary single-pass cross-entropy training. Routers use one tenth of the base learning rate. COGS still scores final weights; its best-dev checkpoint is diagnostic only.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer, standard embeddings (existing) | 2,605,568 | 60.80 ± 1.65 |
| Shared QKV (existing) | 2,213,888 | 58.40 ± 1.70 |
| Design 2: before projection + separate cross-attention Q (existing) | 2,296,208 | 58.70 ± 1.65 |
| Design 4: before projection + small role-specific mixing (existing) | 2,279,312 | 59.51 ± 1.67 |
| A: encoder role mixing + independent cross-attention Q | 2,345,360 | 59.01 ± 1.63 |
| B: wider encoder role mixing + independent cross-attention Q | 2,394,512 | 59.52 ± 1.62 |
| C: encoder role mixing + compact cross-attention Q | 2,295,696 | 59.10 ± 1.64 |
| D: encoder role mixing + compact cross-attention Q/K/V | 2,328,464 | 59.66 ± 1.66 |
| E: combined model + compact cross-attention K/V | 2,378,128 | 59.80 ± 1.68 |
| F: combined model + compact decoder self-attention Q/K | 2,378,128 | 59.55 ± 1.68 |
| G: redistributed encoder and cross-attention Q/K/V mixing | 2,279,312 | 59.51 ± 1.68 |
| H: token-controlled compact cross-attention Q/K/V | 2,330,012 | 59.96 ± 1.59 |
| I: redistributed branches with token controls | 2,280,860 | 58.93 ± 1.65 |
| J: redistributed branches with shared bottlenecks | 2,262,928 | 58.68 ± 1.72 |
| K: analogy inputs with diagonal shortcuts | 2,233,232 | 58.07 ± 1.66 |
| L: model D without the analogy module | 2,312,192 | 59.15 ± 1.65 |
| M: model G without the analogy module | 2,263,040 | 59.21 ± 1.70 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
