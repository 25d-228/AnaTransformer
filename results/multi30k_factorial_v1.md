# multi30k_factorial_v1

Multi30k magnitude × routed-D4 factorial screen. Development BLEU is the primary screening signal; test BLEU is secondary descriptive evidence.

## Individual runs

| model | seed | dev loss | selected step | dev BLEU | test BLEU | params | saved | hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_matched` | 42 | 2.8142 | 20,000 | 40.86 | 40.10 | 2,249,936 | 13.6% | 0.32 |
| `baseline_matched` | 43 | 2.8289 | 20,000 | 39.93 | 39.21 | 2,249,936 | 13.6% | 0.33 |
| `baseline_matched` | 44 | 2.8221 | 18,000 | 40.38 | 38.85 | 2,249,936 | 13.6% | 0.31 |
| `shared_qkv` | 42 | 2.8868 | 20,000 | 39.07 | 39.03 | 2,213,888 | 15.0% | 0.31 |
| `shared_qkv` | 43 | 2.8900 | 19,000 | 38.91 | 38.24 | 2,213,888 | 15.0% | 0.32 |
| `shared_qkv` | 44 | 2.8852 | 20,000 | 39.15 | 39.24 | 2,213,888 | 15.0% | 0.31 |
| `ana_mag_enc` | 42 | 2.8823 | 20,000 | 39.54 | 38.66 | 2,215,448 | 15.0% | 0.39 |
| `ana_mag_enc` | 43 | 2.8959 | 19,000 | 38.88 | 38.73 | 2,215,448 | 15.0% | 0.35 |
| `ana_mag_enc` | 44 | 2.8968 | 19,000 | 39.05 | 38.83 | 2,215,448 | 15.0% | 0.36 |
| `ana_d4_enc` | 42 | 2.8708 | 20,000 | 39.20 | 39.01 | 2,226,284 | 14.6% | 0.39 |
| `ana_d4_enc` | 43 | 2.8954 | 19,000 | 38.89 | 38.26 | 2,226,284 | 14.6% | 0.39 |
| `ana_d4_enc` | 44 | 2.8857 | 20,000 | 40.00 | 38.56 | 2,226,284 | 14.6% | 0.38 |
| `ana_feat_enc` | 42 | 2.8806 | 20,000 | 39.80 | 38.96 | 2,227,832 | 14.5% | 0.39 |
| `ana_feat_enc` | 43 | 2.8780 | 20,000 | 38.89 | 38.90 | 2,227,832 | 14.5% | 0.39 |
| `ana_feat_enc` | 44 | 2.8794 | 19,000 | 39.52 | 38.98 | 2,227,832 | 14.5% | 0.40 |

## Mean and sample standard deviation

| model | dev BLEU | test BLEU |
|---|---:|---:|
| `baseline_matched` | 40.39 ± 0.46 | 39.39 ± 0.65 |
| `shared_qkv` | 39.04 ± 0.13 | 38.84 ± 0.53 |
| `ana_mag_enc` | 39.16 ± 0.34 | 38.74 ± 0.09 |
| `ana_d4_enc` | 39.36 ± 0.57 | 38.61 ± 0.38 |
| `ana_feat_enc` | 39.41 ± 0.47 | 38.95 ± 0.04 |

## Paired per-seed differences against `shared_qkv`

| model | seed | dev BLEU Δ | test BLEU Δ |
|---|---:|---:|---:|
| `baseline_matched` | 42 | +1.78 | +1.07 |
| `baseline_matched` | 43 | +1.02 | +0.98 |
| `baseline_matched` | 44 | +1.23 | -0.40 |
| `ana_mag_enc` | 42 | +0.46 | -0.37 |
| `ana_mag_enc` | 43 | -0.02 | +0.50 |
| `ana_mag_enc` | 44 | -0.10 | -0.41 |
| `ana_d4_enc` | 42 | +0.12 | -0.02 |
| `ana_d4_enc` | 43 | -0.01 | +0.02 |
| `ana_d4_enc` | 44 | +0.84 | -0.68 |
| `ana_feat_enc` | 42 | +0.73 | -0.07 |
| `ana_feat_enc` | 43 | -0.02 | +0.67 |
| `ana_feat_enc` | 44 | +0.37 | -0.27 |

## Paired per-seed differences against `baseline_matched`

| model | seed | dev BLEU Δ | test BLEU Δ |
|---|---:|---:|---:|
| `shared_qkv` | 42 | -1.78 | -1.07 |
| `shared_qkv` | 43 | -1.02 | -0.98 |
| `shared_qkv` | 44 | -1.23 | +0.40 |
| `ana_mag_enc` | 42 | -1.32 | -1.44 |
| `ana_mag_enc` | 43 | -1.05 | -0.48 |
| `ana_mag_enc` | 44 | -1.33 | -0.01 |
| `ana_d4_enc` | 42 | -1.66 | -1.10 |
| `ana_d4_enc` | 43 | -1.04 | -0.96 |
| `ana_d4_enc` | 44 | -0.39 | -0.28 |
| `ana_feat_enc` | 42 | -1.05 | -1.15 |
| `ana_feat_enc` | 43 | -1.04 | -0.31 |
| `ana_feat_enc` | 44 | -0.86 | +0.13 |

## Factorial contrasts

| seed | dev magnitude | dev D4 | dev interaction | test magnitude | test D4 | test interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.53 | +0.19 | +0.15 | -0.21 | +0.14 | +0.32 |
| 43 | -0.01 | -0.00 | +0.02 | +0.57 | +0.10 | +0.15 |
| 44 | -0.29 | +0.66 | -0.37 | +0.00 | -0.27 | +0.82 |
| **mean** | **+0.08** | **+0.28** | **-0.07** | **+0.12** | **-0.01** | **+0.43** |

## Screening questions

- Magnitude alone reproduces 31% of the combined model's mean development gain over `shared_qkv` (+0.11 of +0.36).
- D4 mixing without the magnifier changes development BLEU by +0.32 versus `shared_qkv`; the paired difference is positive on 2/3 seeds: +0.12, -0.01, +0.84.
- The combined model averages +0.25 development BLEU versus magnitude alone and +0.05 versus D4 alone. Against the better single-component model on each seed, it averages -0.07 and is positive on 1/3 seeds.
- The combined gain over `shared_qkv` is positive on 2/3 seeds: +0.73, -0.02, +0.37.
- The best shared-QKV variant is `ana_feat_enc` at 39.41 mean development BLEU, versus 40.39 for `baseline_matched` (-0.98).

No p-values or bootstrap significance tests are reported for this screening study.
