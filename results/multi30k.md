# multi30k

metric: **BLEU/13a** &nbsp;|&nbsp; published: 41.02 BLEU (Wu et al. 2021, transformer_tiny, beam 5, averaged checkpoints)

1 training run, as the paper for this corpus reports. **±** is a bootstrap over the test set.

| model | params | saved | test |
|---|---|---|---|
| `baseline` | 2.61M | 0.0% | 40.78 ± 1.74 |
| `baseline_matched` | 2.25M | 13.6% | 40.00 ± 1.69 |
| `shared_qkv` | 2.21M | 15.0% | 38.64 ± 1.64 |
| `ana_seq_enc` | 2.23M | 14.5% | 38.76 ± 1.64 |
| `ana_feat_enc` | 2.23M | 14.5% | 39.24 ± 1.66 |
| `ana_feat_1_enc` | 2.21M | 15.0% | 38.43 ± 1.66 |
| `ana_feat_2_enc` | 2.21M | 15.0% | 38.92 ± 1.62 |
| `ana_feat_all` | 2.26M | 13.4% | 39.35 ± 1.71 |

## against `baseline`

The accuracy each model trades away for its parameter saving.

Koehn's paired bootstrap: both models are scored on the same resample of the test set, so the variation they share cancels. `*` marks p < 0.05, uncorrected for multiplicity.

| split | model | difference | 95% interval | p | |
|---|---|---|---|---|---|
| test | `baseline_matched` | -0.77 | [-1.65, +0.01] | 0.027 | * |
| test | `shared_qkv` | -2.13 | [-3.03, -1.23] | 0.000 | * |
| test | `ana_seq_enc` | -2.02 | [-2.85, -1.07] | 0.000 | * |
| test | `ana_feat_enc` | -1.53 | [-2.40, -0.60] | 0.000 | * |
| test | `ana_feat_1_enc` | -2.35 | [-3.21, -1.45] | 0.000 | * |
| test | `ana_feat_2_enc` | -1.86 | [-2.72, -1.03] | 0.000 | * |
| test | `ana_feat_all` | -1.43 | [-2.27, -0.56] | 0.000 | * |

## against `shared_qkv`

The accuracy each ana_* operator recovers over the shared projection it is built on. Every model here is within 1.9% of `shared_qkv`'s parameter count, so this is a comparison at equal size.

Koehn's paired bootstrap: both models are scored on the same resample of the test set, so the variation they share cancels. `†` marks p < 0.05, uncorrected for multiplicity.

| split | model | difference | 95% interval | p | |
|---|---|---|---|---|---|
| test | `ana_seq_enc` | +0.11 | [-0.68, +0.93] | 0.387 |  |
| test | `ana_feat_enc` | +0.60 | [-0.13, +1.33] | 0.056 |  |
| test | `ana_feat_1_enc` | -0.21 | [-1.02, +0.59] | 0.307 |  |
| test | `ana_feat_2_enc` | +0.27 | [-0.56, +1.15] | 0.241 |  |
| test | `ana_feat_all` | +0.70 | [+0.01, +1.44] | 0.023 | † |
