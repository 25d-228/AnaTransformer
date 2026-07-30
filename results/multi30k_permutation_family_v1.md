# multi30k_permutation_family_v1

Multi30k development-only screening comparison of the analogy-equivalent D4 family against three preregistered, cycle-type-matched, non-closed permutation families. The test split was not loaded, decoded, or reported.

## Fixed families and matching checks

| family | SHA-256 | closed | generated closure | D4 overlap | Hamming histogram |
|---|---|---:|---:|---:|---|
| `ana_d4_enc` | `90c21558836e96f94a693e5a285569c1be24b3ef9802f262b44ada79a50fad28` | true | 8 | 8 | d=2: 8, d=4: 20 |
| `perm_ctrl_a_enc` | `e92ecfb9092b8ab19d22efe68d04df351388fbfa2dd50d0797af43cb914f235e` | false | 24 | 4 | d=2: 8, d=3: 4, d=4: 16 |
| `perm_ctrl_b_enc` | `e7284cebb2ade50c8b501e33b2164e50240a457607473d1fe60eb814c09a9081` | false | 24 | 4 | d=2: 8, d=3: 4, d=4: 16 |
| `perm_ctrl_c_enc` | `cb43c7026c67afb484b08bd14b605552b922e73cd32b6a5bb702896a53155ec3` | false | 24 | 4 | d=2: 8, d=3: 4, d=4: 16 |

Every family has one identity, two transpositions, three double transpositions, two four-cycles, and no three-cycles. Each control generates all 24 elements of S4 and shares exactly identity plus the three double transpositions with D4. The controls have four distance-3 pairs; D4 instead has twenty distance-4 pairs rather than sixteen. This residual geometry means the screen tests the complete D4 family, not group closure in isolation.

## Compatibility and initialization preflight

All six prior D4 diagnostic checkpoints strictly loaded and reproduced within 0.05 BLEU before any control training began.

| seed | common trainable initialization SHA-256 |
|---:|---|
| 42 | `2e84edeecd3b29566f5b8bac483422c10591b7f04fedd43e7b5ed1793a6e04e0` |
| 43 | `e72034658ed6da4463af31d9509d2b988196bc0d0a00cc4f3948d8854e3ea9ea` |
| 44 | `db1c38fc38a1584691a0207efe07fcf272ff59b34954415fbf70ca1e61430e2f` |

## Individual development results

| origin | model | seed | soft BLEU | hard BLEU | hard - soft | params | hours |
|---|---|---:|---:|---:|---:|---:|---:|
| reused reference | `ana_d4_enc` | 42 | 39.20 | 39.18 | -0.02 | 2,226,284 | 0.39 |
| reused reference | `ana_d4_enc` | 43 | 38.89 | 38.77 | -0.12 | 2,226,284 | 0.39 |
| reused reference | `ana_d4_enc` | 44 | 40.00 | 39.89 | -0.10 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_a_enc` | 42 | 39.58 | 39.55 | -0.03 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_a_enc` | 43 | 39.18 | 39.01 | -0.17 | 2,226,284 | 0.39 |
| new run | `perm_ctrl_a_enc` | 44 | 38.95 | 39.02 | +0.06 | 2,226,284 | 0.39 |
| new run | `perm_ctrl_b_enc` | 42 | 39.69 | 39.59 | -0.10 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_b_enc` | 43 | 38.91 | 39.09 | +0.18 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_b_enc` | 44 | 39.68 | 39.60 | -0.08 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_c_enc` | 42 | 39.52 | 39.37 | -0.15 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_c_enc` | 43 | 38.60 | 38.75 | +0.15 | 2,226,284 | 0.38 |
| new run | `perm_ctrl_c_enc` | 44 | 39.43 | 39.59 | +0.16 | 2,226,284 | 0.38 |

## Family means

| model | soft BLEU mean ± sample SD | hard BLEU mean ± sample SD | mean hard - soft | mean hours |
|---|---:|---:|---:|---:|
| `ana_d4_enc` | 39.36 ± 0.57 | 39.28 ± 0.57 | -0.08 | 0.39 |
| `perm_ctrl_a_enc` | 39.24 ± 0.32 | 39.19 ± 0.31 | -0.05 | 0.39 |
| `perm_ctrl_b_enc` | 39.43 ± 0.45 | 39.43 ± 0.30 | +0.00 | 0.38 |
| `perm_ctrl_c_enc` | 39.18 ± 0.51 | 39.23 ± 0.44 | +0.05 | 0.38 |

## D4-specific contrast

| seed | D4 | control A | control B | control C | control average | D4 - control average |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 39.20 | 39.58 | 39.69 | 39.52 | 39.60 | -0.40 |
| 43 | 38.89 | 39.18 | 38.91 | 38.60 | 38.90 | -0.00 |
| 44 | 40.00 | 38.95 | 39.68 | 39.43 | 39.36 | +0.64 |
| **mean** | | | | | | **+0.08** |

## Paired contextual comparisons

| model | comparison | seed 42 | seed 43 | seed 44 | mean |
|---|---|---:|---:|---:|---:|
| `ana_d4_enc` | minus `perm_ctrl_a_enc` | -0.39 | -0.29 | +1.04 | +0.12 |
| `ana_d4_enc` | minus `perm_ctrl_b_enc` | -0.49 | -0.01 | +0.31 | -0.07 |
| `ana_d4_enc` | minus `perm_ctrl_c_enc` | -0.32 | +0.29 | +0.56 | +0.18 |
| `ana_d4_enc` | minus `shared_qkv` | +0.12 | -0.01 | +0.84 | +0.32 |
| `ana_d4_enc` | minus `baseline_matched` | -1.66 | -1.04 | -0.39 | -1.03 |
| `perm_ctrl_a_enc` | minus `shared_qkv` | +0.51 | +0.28 | -0.20 | +0.20 |
| `perm_ctrl_a_enc` | minus `baseline_matched` | -1.27 | -0.75 | -1.43 | -1.15 |
| `perm_ctrl_b_enc` | minus `shared_qkv` | +0.62 | -0.00 | +0.53 | +0.38 |
| `perm_ctrl_b_enc` | minus `baseline_matched` | -1.17 | -1.02 | -0.70 | -0.96 |
| `perm_ctrl_c_enc` | minus `shared_qkv` | +0.44 | -0.31 | +0.28 | +0.14 |
| `perm_ctrl_c_enc` | minus `baseline_matched` | -1.34 | -1.33 | -0.95 | -1.21 |

## Router statistics across seeds

| family | layer | role | gate | H/log8 | max p | effective | identity | nearest family | token JS | QKV JS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 1 | Q | 0.452 | 0.168 | 0.865 | 1.54 | 0.316 | 0.101 | 0.6206 | 0.8063 |
| `ana_d4_enc` | 1 | K | 0.279 | 0.198 | 0.838 | 1.62 | 0.210 | 0.134 | 0.6327 | 0.8063 |
| `ana_d4_enc` | 1 | V | 0.744 | 0.141 | 0.897 | 1.51 | 0.198 | 0.077 | 0.6618 | 0.8063 |
| `ana_d4_enc` | 2 | Q | 0.521 | 0.164 | 0.863 | 1.51 | 0.053 | 0.119 | 0.7075 | 0.8100 |
| `ana_d4_enc` | 2 | K | 0.262 | 0.257 | 0.781 | 1.81 | 0.199 | 0.188 | 0.6256 | 0.8100 |
| `ana_d4_enc` | 2 | V | 0.549 | 0.147 | 0.890 | 1.48 | 0.074 | 0.083 | 0.6882 | 0.8100 |
| `ana_d4_enc` | 3 | Q | 0.348 | 0.101 | 0.916 | 1.33 | 0.081 | 0.060 | 0.7100 | 0.8682 |
| `ana_d4_enc` | 3 | K | 0.222 | 0.118 | 0.902 | 1.35 | 0.234 | 0.086 | 0.6878 | 0.8682 |
| `ana_d4_enc` | 3 | V | 0.517 | 0.156 | 0.875 | 1.48 | 0.040 | 0.099 | 0.7461 | 0.8682 |
| `ana_d4_enc` | 4 | Q | 0.305 | 0.043 | 0.967 | 1.12 | 0.091 | 0.028 | 0.6080 | 0.9054 |
| `ana_d4_enc` | 4 | K | 0.268 | 0.056 | 0.953 | 1.16 | 0.100 | 0.043 | 0.6509 | 0.9054 |
| `ana_d4_enc` | 4 | V | 0.487 | 0.109 | 0.920 | 1.33 | 0.029 | 0.061 | 0.7265 | 0.9054 |
| `perm_ctrl_a_enc` | 1 | Q | 0.467 | 0.154 | 0.879 | 1.47 | 0.249 | 0.100 | 0.6404 | 0.7755 |
| `perm_ctrl_a_enc` | 1 | K | 0.235 | 0.188 | 0.847 | 1.60 | 0.193 | 0.124 | 0.6555 | 0.7755 |
| `perm_ctrl_a_enc` | 1 | V | 0.738 | 0.180 | 0.861 | 1.63 | 0.200 | 0.114 | 0.6539 | 0.7755 |
| `perm_ctrl_a_enc` | 2 | Q | 0.535 | 0.121 | 0.905 | 1.36 | 0.077 | 0.076 | 0.6818 | 0.8041 |
| `perm_ctrl_a_enc` | 2 | K | 0.226 | 0.245 | 0.793 | 1.76 | 0.273 | 0.178 | 0.6221 | 0.8041 |
| `perm_ctrl_a_enc` | 2 | V | 0.505 | 0.164 | 0.878 | 1.58 | 0.085 | 0.092 | 0.7010 | 0.8041 |
| `perm_ctrl_a_enc` | 3 | Q | 0.359 | 0.118 | 0.909 | 1.35 | 0.132 | 0.070 | 0.6532 | 0.8738 |
| `perm_ctrl_a_enc` | 3 | K | 0.222 | 0.092 | 0.927 | 1.28 | 0.054 | 0.060 | 0.6957 | 0.8738 |
| `perm_ctrl_a_enc` | 3 | V | 0.491 | 0.122 | 0.897 | 1.37 | 0.051 | 0.085 | 0.7554 | 0.8738 |
| `perm_ctrl_a_enc` | 4 | Q | 0.201 | 0.089 | 0.926 | 1.25 | 0.265 | 0.060 | 0.7548 | 0.8693 |
| `perm_ctrl_a_enc` | 4 | K | 0.302 | 0.058 | 0.954 | 1.17 | 0.120 | 0.039 | 0.7526 | 0.8693 |
| `perm_ctrl_a_enc` | 4 | V | 0.519 | 0.082 | 0.944 | 1.24 | 0.022 | 0.041 | 0.7175 | 0.8693 |
| `perm_ctrl_b_enc` | 1 | Q | 0.393 | 0.115 | 0.909 | 1.34 | 0.430 | 0.074 | 0.5609 | 0.8019 |
| `perm_ctrl_b_enc` | 1 | K | 0.213 | 0.173 | 0.861 | 1.55 | 0.165 | 0.111 | 0.7199 | 0.8019 |
| `perm_ctrl_b_enc` | 1 | V | 0.738 | 0.161 | 0.880 | 1.57 | 0.213 | 0.090 | 0.6289 | 0.8019 |
| `perm_ctrl_b_enc` | 2 | Q | 0.486 | 0.157 | 0.870 | 1.47 | 0.130 | 0.109 | 0.7597 | 0.8475 |
| `perm_ctrl_b_enc` | 2 | K | 0.300 | 0.227 | 0.808 | 1.73 | 0.181 | 0.159 | 0.6492 | 0.8475 |
| `perm_ctrl_b_enc` | 2 | V | 0.523 | 0.119 | 0.912 | 1.38 | 0.055 | 0.066 | 0.7119 | 0.8475 |
| `perm_ctrl_b_enc` | 3 | Q | 0.374 | 0.108 | 0.917 | 1.33 | 0.103 | 0.064 | 0.6264 | 0.8959 |
| `perm_ctrl_b_enc` | 3 | K | 0.200 | 0.156 | 0.871 | 1.46 | 0.284 | 0.110 | 0.6755 | 0.8959 |
| `perm_ctrl_b_enc` | 3 | V | 0.492 | 0.082 | 0.934 | 1.24 | 0.014 | 0.055 | 0.7358 | 0.8959 |
| `perm_ctrl_b_enc` | 4 | Q | 0.261 | 0.052 | 0.959 | 1.15 | 0.168 | 0.034 | 0.7875 | 0.8923 |
| `perm_ctrl_b_enc` | 4 | K | 0.348 | 0.040 | 0.969 | 1.11 | 0.221 | 0.025 | 0.7652 | 0.8923 |
| `perm_ctrl_b_enc` | 4 | V | 0.537 | 0.090 | 0.937 | 1.27 | 0.049 | 0.049 | 0.7420 | 0.8923 |
| `perm_ctrl_c_enc` | 1 | Q | 0.458 | 0.185 | 0.857 | 1.61 | 0.310 | 0.108 | 0.5985 | 0.7779 |
| `perm_ctrl_c_enc` | 1 | K | 0.230 | 0.246 | 0.803 | 1.81 | 0.231 | 0.156 | 0.6768 | 0.7779 |
| `perm_ctrl_c_enc` | 1 | V | 0.747 | 0.140 | 0.896 | 1.50 | 0.219 | 0.082 | 0.6899 | 0.7779 |
| `perm_ctrl_c_enc` | 2 | Q | 0.478 | 0.105 | 0.915 | 1.31 | 0.058 | 0.072 | 0.6563 | 0.8610 |
| `perm_ctrl_c_enc` | 2 | K | 0.292 | 0.215 | 0.816 | 1.68 | 0.158 | 0.145 | 0.6681 | 0.8610 |
| `perm_ctrl_c_enc` | 2 | V | 0.487 | 0.139 | 0.896 | 1.45 | 0.033 | 0.082 | 0.6909 | 0.8610 |
| `perm_ctrl_c_enc` | 3 | Q | 0.364 | 0.078 | 0.933 | 1.24 | 0.171 | 0.057 | 0.5459 | 0.8498 |
| `perm_ctrl_c_enc` | 3 | K | 0.247 | 0.088 | 0.930 | 1.25 | 0.195 | 0.058 | 0.7181 | 0.8498 |
| `perm_ctrl_c_enc` | 3 | V | 0.459 | 0.124 | 0.903 | 1.38 | 0.047 | 0.079 | 0.7338 | 0.8498 |
| `perm_ctrl_c_enc` | 4 | Q | 0.290 | 0.096 | 0.926 | 1.29 | 0.015 | 0.060 | 0.7560 | 0.8768 |
| `perm_ctrl_c_enc` | 4 | K | 0.226 | 0.063 | 0.951 | 1.18 | 0.105 | 0.040 | 0.8230 | 0.8768 |
| `perm_ctrl_c_enc` | 4 | V | 0.540 | 0.115 | 0.907 | 1.35 | 0.024 | 0.069 | 0.7150 | 0.8768 |

## Decision questions

- **Does D4 beat the average matched non-closed family?** Its mean paired effect is +0.08 BLEU and is positive on 1/3 seeds.
- **Does D4 beat A, B, and C separately?** Mean paired differences are +0.12, -0.07, and +0.18 BLEU.
- **Is any D4 advantage consistent across seeds?** 1/3 seed-level effects against the control average are positive.
- **Do arbitrary families improve over `shared_qkv`?** Mean paired effects are +0.20, +0.38, and +0.14 BLEU.
- **Do arbitrary families become near-discrete and retain BLEU under hard argmax?** Mean maximum route probability / nearest-family distance / hard-minus-soft BLEU is 0.893 / 0.087 / -0.05 for A, 0.902 / 0.079 / +0.00 for B, and 0.894 / 0.084 / +0.05 for C. The best-soft control is `perm_ctrl_b_enc` at 39.43 BLEU.
- **Does any permutation family approach `baseline_matched`?** D4 is -1.03 BLEU relative to it; the control differences are -1.15, -0.96, and -1.21.

**Screening interpretation:** The screen supports generic token- and role-conditioned permutation routing, not D4 specifically.

D4 soft-to-hard mean change is -0.08 BLEU. These are three-seed screening effects; no p-values, bootstrap tests, or significance claims are reported.
