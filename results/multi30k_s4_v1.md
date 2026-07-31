# multi30k_s4_v1

Three-seed, development-only Multi30k screen of a soft router over all 24 permutations of four channels. Differences are `s4_enc` minus the named reference.

| seed | S4 dev BLEU | shared_qkv (Δ) | ana_d4_enc (Δ) | perm_ctrl_b_enc (Δ) | baseline_matched (Δ) | step | dev loss | params | hours |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 39.36 | 39.07 (+0.29) | 39.20 (+0.17) | 39.69 (-0.33) | 40.86 (-1.49) | 20,000 | 2.8701 | 2,251,052 | 0.39 |
| 43 | 39.41 | 38.91 (+0.51) | 38.89 (+0.52) | 38.91 (+0.51) | 39.93 (-0.51) | 20,000 | 2.8960 | 2,251,052 | 0.39 |
| 44 | 39.22 | 39.15 (+0.06) | 40.00 (-0.78) | 39.68 (-0.47) | 40.38 (-1.16) | 20,000 | 2.8787 | 2,251,052 | 0.38 |
| **mean ± sample SD** | **39.33 ± 0.10** | 39.04 (+0.29) | 39.36 (-0.03) | 39.43 (-0.09) | 40.39 (-1.06) | — | — | 2,251,052 | 0.39 |

## Decision

**stop fixed permutation expansion.** Mean gain over `perm_ctrl_b_enc` is -0.09 BLEU (1/3 paired seed wins); 0/3 seeds reach 39.90. Mean gap to `baseline_matched` is -1.06 BLEU, and runtime is 1.01× the eight-route control.

For the preregistered “not one exceptional seed” criterion, this compact screen uses at least two individual S4 seeds reaching 39.90. “Not grossly worse” uses at most 1.5× the mean `perm_ctrl_b_enc` runtime. No test split, hard routing, router statistics, p-values, or bootstrap tests are included.
