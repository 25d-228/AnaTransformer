# iwslt14

metric: **BLEU/13a** &nbsp;|&nbsp; published: 34.4 BLEU (Wu et al. 2019, transformer_iwslt_de_en, beam 4, averaged checkpoints)

1 training run, as the paper for this corpus reports. **±** is a bootstrap over the test set.

| model | params | saved | test |
|---|---|---|---|
| `baseline` | 36.67M | 0.0% | 33.45 ± 0.51 |
| `baseline_matched` | 27.25M | 25.7% | 32.94 ± 0.50 |
| `shared_qkv` | 27.24M | 25.7% | 31.58 ± 0.49 |
| `ana_seq_enc` | 27.32M | 25.5% | 31.70 ± 0.50 |
| `ana_feat_enc` | 27.32M | 25.5% | 32.19 ± 0.49 |
| `ana_feat_1_enc` | 27.24M | 25.7% | 31.71 ± 0.49 |
| `ana_feat_2_enc` | 27.24M | 25.7% | 31.68 ± 0.49 |
| `ana_feat_all` | 27.49M | 25.0% | 32.19 ± 0.48 |

## against `baseline`

The accuracy each model trades away for its parameter saving.

Koehn's paired bootstrap: both models are scored on the same resample of the test set, so the variation they share cancels. `*` marks p < 0.05, uncorrected for multiplicity.

| split | model | difference | 95% interval | p | |
|---|---|---|---|---|---|
| test | `baseline_matched` | -0.51 | [-0.73, -0.28] | 0.000 | * |
| test | `shared_qkv` | -1.86 | [-2.14, -1.61] | 0.000 | * |
| test | `ana_seq_enc` | -1.75 | [-1.99, -1.49] | 0.000 | * |
| test | `ana_feat_enc` | -1.25 | [-1.50, -1.00] | 0.000 | * |
| test | `ana_feat_1_enc` | -1.73 | [-1.99, -1.48] | 0.000 | * |
| test | `ana_feat_2_enc` | -1.76 | [-2.03, -1.52] | 0.000 | * |
| test | `ana_feat_all` | -1.26 | [-1.50, -1.01] | 0.000 | * |

## against `shared_qkv`

The accuracy each ana_* operator recovers over the shared projection it is built on. Every model here is within 0.9% of `shared_qkv`'s parameter count, so this is a comparison at equal size.

Koehn's paired bootstrap: both models are scored on the same resample of the test set, so the variation they share cancels. `†` marks p < 0.05, uncorrected for multiplicity.

| split | model | difference | 95% interval | p | |
|---|---|---|---|---|---|
| test | `ana_seq_enc` | +0.12 | [-0.10, +0.34] | 0.139 |  |
| test | `ana_feat_enc` | +0.61 | [+0.39, +0.85] | 0.000 | † |
| test | `ana_feat_1_enc` | +0.13 | [-0.08, +0.36] | 0.107 |  |
| test | `ana_feat_2_enc` | +0.10 | [-0.13, +0.32] | 0.181 |  |
| test | `ana_feat_all` | +0.60 | [+0.35, +0.84] | 0.000 | † |
