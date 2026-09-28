# COGS generalization: combined analogy-preserving projections, batch 4

Original standard embeddings and corpus recipes; ordinary single-pass cross-entropy training. Routers use one tenth of the base learning rate. COGS still scores final weights; its best-dev checkpoint is diagnostic only.

| Model | Parameters | Exact match (%) |
|---|---:|---:|
| Full Transformer, standard embeddings (existing) | 8,844,800 | 78.70 ± 0.53 |
| Shared QKV (existing) | 5,702,144 | 80.38 ± 0.50 |
| Design 2: before projection + separate cross-attention Q (existing) | 6,261,080 | 82.06 ± 0.50 |
| Design 4: before projection + small role-specific mixing (existing) | 6,128,984 | 78.72 ± 0.54 |
| A: encoder role mixing + independent cross-attention Q | 6,654,296 | 78.25 ± 0.53 |
| B: wider encoder role mixing + independent cross-attention Q | 7,047,512 | 75.80 ± 0.55 |
| C: encoder role mixing + compact cross-attention Q | 6,260,056 | 76.40 ± 0.55 |
| D: encoder role mixing + compact cross-attention Q/K/V | 6,522,200 | 82.05 ± 0.50 |
| E: combined model + compact cross-attention K/V | 6,916,440 | 78.80 ± 0.53 |
| F: combined model + compact decoder self-attention Q/K | 6,916,440 | 81.02 ± 0.51 |
| G: redistributed encoder and cross-attention Q/K/V mixing | 6,128,984 | 76.35 ± 0.54 |
| H: token-controlled compact cross-attention Q/K/V | 6,525,278 | 80.73 ± 0.52 |
| I: redistributed branches with token controls | 6,132,062 | 74.90 ± 0.57 |
| J: redistributed branches with shared bottlenecks | 5,997,912 | 76.28 ± 0.56 |
| K: analogy inputs with diagonal shortcuts | 5,741,912 | 77.69 ± 0.54 |
| L: model D without the analogy module | 6,488,576 | 78.11 ± 0.53 |
| M: model G without the analogy module | 6,095,360 | 74.90 ± 0.55 |

New-run ± is half the width of a 95% example-bootstrap interval (1,000 resamples). Reference rows are supplied existing results, not new runs. Their provenance is retained in JSON. COGS reports generalization only.

Per-model analogy and routing diagnostics at the scored checkpoint are in JSON. They are measured on development data with dropout off, not on test feedback.
