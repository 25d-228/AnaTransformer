# cogs

metric: **exact match** &nbsp;|&nbsp; published: 96% test (Kim & Linzen 2020); ~80% gen (Csordas et al. 2021, this embedding scaling)

5 training runs (seeds [42, 43, 44, 45, 46]). **±** is the spread across them.

| model | params | saved | gen | test |
|---|---|---|---|---|
| `baseline` | 8.84M | 0.0% | 76.41 ± 4.02 | 99.73 ± 0.08 |
| `baseline_matched` | 5.69M | 35.7% | 79.46 ± 1.25 | 99.76 ± 0.06 |
| `shared_qkv` | 5.70M | 35.5% | 80.76 ± 1.06 | 99.79 ± 0.09 |
| `ana_seq_enc` | 5.73M | 35.2% | 79.07 ± 1.53 | 99.75 ± 0.09 |
| `ana_feat_enc` | 5.73M | 35.2% | 79.08 ± 1.47 | 99.75 ± 0.08 |
| `ana_feat_1_enc` | 5.70M | 35.5% | 80.59 ± 2.16 | 99.68 ± 0.02 |
| `ana_feat_2_enc` | 5.70M | 35.5% | 78.84 ± 2.96 | 99.71 ± 0.10 |
| `ana_feat_all` | 5.79M | 34.6% | 80.01 ± 0.91 | 99.63 ± 0.06 |

## against `baseline`

The accuracy each model trades away for its parameter saving.

Paired across seeds. `*` marks p < 0.05, uncorrected for multiplicity.

| split | model | difference | 95% interval | p | |
|---|---|---|---|---|---|
| gen | `baseline_matched` | +3.04 | [-2.35, +8.44] | 0.192 |  |
| gen | `shared_qkv` | +4.35 | [+0.05, +8.65] | 0.048 | * |
| gen | `ana_seq_enc` | +2.65 | [-2.37, +7.67] | 0.216 |  |
| gen | `ana_feat_enc` | +2.67 | [-2.10, +7.44] | 0.195 |  |
| gen | `ana_feat_1_enc` | +4.17 | [-0.24, +8.59] | 0.059 |  |
| gen | `ana_feat_2_enc` | +2.43 | [-3.20, +8.06] | 0.297 |  |
| gen | `ana_feat_all` | +3.60 | [-1.57, +8.76] | 0.125 |  |
| test | `baseline_matched` | +0.03 | [-0.09, +0.15] | 0.486 |  |
| test | `shared_qkv` | +0.07 | [-0.01, +0.14] | 0.075 |  |
| test | `ana_seq_enc` | +0.03 | [-0.13, +0.19] | 0.670 |  |
| test | `ana_feat_enc` | +0.02 | [-0.10, +0.14] | 0.656 |  |
| test | `ana_feat_1_enc` | -0.05 | [-0.17, +0.08] | 0.351 |  |
| test | `ana_feat_2_enc` | -0.02 | [-0.23, +0.19] | 0.803 |  |
| test | `ana_feat_all` | -0.09 | [-0.20, +0.02] | 0.080 |  |

## against `shared_qkv`

The accuracy each ana_* operator recovers over the shared projection it is built on. Every model here is within 1.5% of `shared_qkv`'s parameter count, so this is a comparison at equal size.

Paired across seeds. `†` marks p < 0.05, uncorrected for multiplicity.

| split | model | difference | 95% interval | p | |
|---|---|---|---|---|---|
| gen | `ana_seq_enc` | -1.70 | [-3.49, +0.10] | 0.059 |  |
| gen | `ana_feat_enc` | -1.68 | [-4.32, +0.97] | 0.153 |  |
| gen | `ana_feat_1_enc` | -0.18 | [-2.63, +2.28] | 0.853 |  |
| gen | `ana_feat_2_enc` | -1.92 | [-5.22, +1.38] | 0.181 |  |
| gen | `ana_feat_all` | -0.75 | [-3.00, +1.49] | 0.404 |  |
| test | `ana_seq_enc` | -0.04 | [-0.20, +0.12] | 0.535 |  |
| test | `ana_feat_enc` | -0.05 | [-0.14, +0.05] | 0.245 |  |
| test | `ana_feat_1_enc` | -0.11 | [-0.25, +0.02] | 0.077 |  |
| test | `ana_feat_2_enc` | -0.09 | [-0.32, +0.14] | 0.354 |  |
| test | `ana_feat_all` | -0.16 | [-0.29, -0.03] | 0.026 | † |
