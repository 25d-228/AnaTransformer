# Multi30k English → German: combined analogy-preserving projections, batch 4

Original standard embeddings and corpus recipes; ordinary single-pass cross-entropy training. Routers use one tenth of the base learning rate. COGS still scores final weights; its best-dev checkpoint is diagnostic only.

| Model | Parameters | BLEU |
|---|---:|---:|
| Full Transformer, standard embeddings (existing) | 2,605,568 | 40.78 ± 1.74 |
| Shared QKV (existing) | 2,213,888 | 38.65 ± 1.65 |
| Design 2: before projection + separate cross-attention Q (existing) | 2,296,208 | 39.20 ± 1.66 |
| Design 4: before projection + small role-specific mixing (existing) | 2,279,312 | 38.98 ± 1.69 |
| A: encoder role mixing + independent cross-attention Q | 2,345,360 | 39.69 ± 1.64 |
| B: wider encoder role mixing + independent cross-attention Q | 2,394,512 | 39.80 ± 1.62 |
| C: encoder role mixing + compact cross-attention Q | 2,295,696 | 39.74 ± 1.69 |
| D: encoder role mixing + compact cross-attention Q/K/V | 2,328,464 | 39.92 ± 1.68 |
| E: combined model + compact cross-attention K/V | 2,378,128 | 39.40 ± 1.59 |
| F: combined model + compact decoder self-attention Q/K | 2,378,128 | 39.26 ± 1.56 |
| G: redistributed encoder and cross-attention Q/K/V mixing | 2,279,312 | 39.39 ± 1.67 |
| H: token-controlled compact cross-attention Q/K/V | 2,330,012 | 39.71 ± 1.68 |
| I: redistributed branches with token controls | 2,280,860 | 39.32 ± 1.68 |
| J: redistributed branches with shared bottlenecks | 2,262,928 | 38.56 ± 1.68 |
| K: analogy inputs with diagonal shortcuts | 2,233,232 | 38.84 ± 1.62 |
| L: model D without the analogy module | 2,312,192 | 39.76 ± 1.73 |
| M: model G without the analogy module | 2,263,040 | 39.26 ± 1.71 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
