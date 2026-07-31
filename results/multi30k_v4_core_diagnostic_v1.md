# multi30k_v4_core_diagnostic_v1

Inference-only diagnostic of whether the four permutations shared by D4 and all three matched control families carry their near-discrete routing behavior. All 72 decodes used the full Multi30k development set under evaluation mode and no-gradient inference. The test split was not loaded, decoded, or reported.

## Fixed V4 core and family partitions

V4 core SHA-256: `3cc4ea231875cb1d9579cff27efb1056fe3b79e7db3ca9bb328f76d9a59d8249`

| model | core router indices in V4 order | non-core indices | non-core tuple order |
|---|---|---|---|
| `ana_d4_enc` | [0, 4, 6, 2] | [1, 3, 5, 7] | `[[0, 2, 1, 3], [2, 0, 3, 1], [1, 3, 0, 2], [3, 1, 2, 0]]` |
| `perm_ctrl_a_enc` | [0, 3, 5, 7] | [1, 2, 4, 6] | `[[0, 1, 3, 2], [0, 3, 2, 1], [1, 2, 3, 0], [2, 3, 1, 0]]` |
| `perm_ctrl_b_enc` | [0, 3, 4, 7] | [1, 2, 5, 6] | `[[0, 1, 3, 2], [0, 3, 2, 1], [3, 0, 1, 2], [3, 2, 0, 1]]` |
| `perm_ctrl_c_enc` | [0, 2, 5, 7] | [1, 3, 4, 6] | `[[1, 0, 2, 3], [1, 2, 3, 0], [2, 1, 0, 3], [3, 2, 0, 1]]` |

Every mask is derived by matching tuples against the active module's nonpersistent permutation buffer. Each family contains all four core members and exactly four family-specific non-core members.

## Initial-function caveat

Issue #6 matched trainable initialization, not the initial function: uniform routing averages each fixed family into its own centroid. Matrices below use row-major order.

| model | centroid | distance from D4 centroid | singular values | initial effective residual |
|---|---|---:|---|---|
| `ana_d4_enc` | `[0.250, 0.250, 0.250, 0.250; 0.250, 0.250, 0.250, 0.250; 0.250, 0.250, 0.250, 0.250; 0.250, 0.250, 0.250, 0.250]` | 0.0000 | 1.0000, 0.0000, 0.0000, 0.0000 | `[0.911, 0.030, 0.030, 0.030; 0.030, 0.911, 0.030, 0.030; 0.030, 0.030, 0.911, 0.030; 0.030, 0.030, 0.030, 0.911]` |
| `perm_ctrl_a_enc` | `[0.375, 0.250, 0.250, 0.125; 0.125, 0.250, 0.250, 0.375; 0.125, 0.250, 0.250, 0.375; 0.375, 0.250, 0.250, 0.125]` | 0.3536 | 1.0000, 0.3536, 0.0000, 0.0000 | `[0.925, 0.030, 0.030, 0.015; 0.015, 0.911, 0.030, 0.045; 0.015, 0.030, 0.911, 0.045; 0.045, 0.030, 0.030, 0.896]` |
| `perm_ctrl_b_enc` | `[0.375, 0.125, 0.125, 0.375; 0.250, 0.250, 0.250, 0.250; 0.250, 0.250, 0.250, 0.250; 0.125, 0.375, 0.375, 0.125]` | 0.3536 | 1.0000, 0.3536, 0.0000, 0.0000 | `[0.925, 0.015, 0.015, 0.045; 0.030, 0.911, 0.030, 0.030; 0.030, 0.030, 0.911, 0.030; 0.015, 0.045, 0.045, 0.896]` |
| `perm_ctrl_c_enc` | `[0.125, 0.375, 0.250, 0.250; 0.250, 0.250, 0.375, 0.125; 0.375, 0.125, 0.250, 0.250; 0.250, 0.250, 0.125, 0.375]` | 0.3536 | 1.0000, 0.2500, 0.2500, 0.0000 | `[0.896, 0.045, 0.030, 0.030; 0.030, 0.911, 0.045, 0.015; 0.045, 0.015, 0.911, 0.030; 0.030, 0.030, 0.015, 0.925]` |

The initial residual gate is `sigmoid(-2) = 0.119203`; learned diagonal scaling is excluded from the displayed effective matrices.

## Original-score reproduction guard

All 12 original reproductions completed before any subset intervention began. Existing all-family hard-argmax scores were loaded from committed artifacts and not rerun.

| model | seed | selected step | stored dev BLEU | reproduced | difference |
|---|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 42 | 20,000 | 39.1970 | 39.1970 | +0.0000 |
| `ana_d4_enc` | 43 | 19,000 | 38.8934 | 38.8934 | +0.0000 |
| `ana_d4_enc` | 44 | 20,000 | 39.9955 | 39.9955 | +0.0000 |
| `perm_ctrl_a_enc` | 42 | 20,000 | 39.5836 | 39.5836 | +0.0000 |
| `perm_ctrl_a_enc` | 43 | 19,000 | 39.1822 | 39.1822 | +0.0000 |
| `perm_ctrl_a_enc` | 44 | 20,000 | 38.9544 | 38.9544 | +0.0000 |
| `perm_ctrl_b_enc` | 42 | 20,000 | 39.6916 | 39.6916 | +0.0000 |
| `perm_ctrl_b_enc` | 43 | 20,000 | 38.9053 | 38.9053 | +0.0000 |
| `perm_ctrl_b_enc` | 44 | 20,000 | 39.6841 | 39.6841 | +0.0000 |
| `perm_ctrl_c_enc` | 42 | 20,000 | 39.5191 | 39.5191 | +0.0000 |
| `perm_ctrl_c_enc` | 43 | 19,000 | 38.5994 | 38.5994 | +0.0000 |
| `perm_ctrl_c_enc` | 44 | 20,000 | 39.4334 | 39.4334 | +0.0000 |

## Core usage in trained routers

Subset-conditional distributions and entropies exclude token-subset pairs with mass at most 1e-12; zero is reported only if no valid pair remains.

| family | core mass | p05 | p50 | p95 | argmax in core | H(core)/log4 | H(non-core)/log4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 0.545 | 0.015 | 0.580 | 0.999 | 0.552 | 0.150 | 0.172 |
| `perm_ctrl_a_enc` | 0.596 | 0.013 | 0.724 | 1.000 | 0.600 | 0.152 | 0.194 |
| `perm_ctrl_b_enc` | 0.608 | 0.005 | 0.743 | 1.000 | 0.608 | 0.152 | 0.197 |
| `perm_ctrl_c_enc` | 0.602 | 0.004 | 0.727 | 1.000 | 0.608 | 0.156 | 0.195 |
| **all 12** | **0.588** | **0.009** | **0.693** | **1.000** | **0.592** | **0.153** | **0.190** |

### Family, layer, and role means

| family | layer | role | core mass | argmax in core | core distribution (V4 order) | non-core distribution |
|---|---:|---|---:|---:|---|---|
| `ana_d4_enc` | 1 | Q | 0.624 | 0.634 | 0.537, 0.163, 0.192, 0.108 | 0.329, 0.132, 0.185, 0.354 |
| `ana_d4_enc` | 1 | K | 0.431 | 0.438 | 0.553, 0.070, 0.260, 0.117 | 0.313, 0.167, 0.293, 0.228 |
| `ana_d4_enc` | 1 | V | 0.668 | 0.683 | 0.306, 0.119, 0.448, 0.127 | 0.284, 0.316, 0.264, 0.136 |
| `ana_d4_enc` | 2 | Q | 0.543 | 0.553 | 0.224, 0.270, 0.224, 0.283 | 0.347, 0.139, 0.153, 0.361 |
| `ana_d4_enc` | 2 | K | 0.591 | 0.591 | 0.505, 0.099, 0.293, 0.103 | 0.093, 0.336, 0.355, 0.216 |
| `ana_d4_enc` | 2 | V | 0.693 | 0.703 | 0.183, 0.216, 0.302, 0.299 | 0.247, 0.217, 0.375, 0.161 |
| `ana_d4_enc` | 3 | Q | 0.438 | 0.454 | 0.276, 0.313, 0.148, 0.262 | 0.302, 0.195, 0.327, 0.176 |
| `ana_d4_enc` | 3 | K | 0.509 | 0.514 | 0.480, 0.120, 0.150, 0.250 | 0.189, 0.399, 0.225, 0.187 |
| `ana_d4_enc` | 3 | V | 0.589 | 0.602 | 0.140, 0.265, 0.370, 0.224 | 0.164, 0.311, 0.340, 0.185 |
| `ana_d4_enc` | 4 | Q | 0.648 | 0.650 | 0.257, 0.355, 0.208, 0.180 | 0.333, 0.171, 0.360, 0.135 |
| `ana_d4_enc` | 4 | K | 0.352 | 0.359 | 0.343, 0.064, 0.179, 0.413 | 0.286, 0.426, 0.103, 0.184 |
| `ana_d4_enc` | 4 | V | 0.455 | 0.441 | 0.097, 0.384, 0.177, 0.342 | 0.108, 0.280, 0.472, 0.140 |
| `perm_ctrl_a_enc` | 1 | Q | 0.488 | 0.481 | 0.594, 0.104, 0.229, 0.073 | 0.067, 0.362, 0.348, 0.223 |
| `perm_ctrl_a_enc` | 1 | K | 0.658 | 0.667 | 0.391, 0.137, 0.334, 0.137 | 0.164, 0.384, 0.184, 0.267 |
| `perm_ctrl_a_enc` | 1 | V | 0.740 | 0.752 | 0.302, 0.274, 0.268, 0.157 | 0.161, 0.184, 0.263, 0.392 |
| `perm_ctrl_a_enc` | 2 | Q | 0.396 | 0.400 | 0.294, 0.196, 0.325, 0.185 | 0.148, 0.288, 0.329, 0.236 |
| `perm_ctrl_a_enc` | 2 | K | 0.676 | 0.682 | 0.543, 0.098, 0.208, 0.152 | 0.247, 0.335, 0.182, 0.236 |
| `perm_ctrl_a_enc` | 2 | V | 0.692 | 0.702 | 0.223, 0.220, 0.249, 0.309 | 0.199, 0.211, 0.264, 0.326 |
| `perm_ctrl_a_enc` | 3 | Q | 0.721 | 0.734 | 0.203, 0.428, 0.166, 0.204 | 0.385, 0.230, 0.266, 0.119 |
| `perm_ctrl_a_enc` | 3 | K | 0.509 | 0.508 | 0.243, 0.188, 0.149, 0.420 | 0.276, 0.103, 0.288, 0.333 |
| `perm_ctrl_a_enc` | 3 | V | 0.666 | 0.663 | 0.131, 0.303, 0.317, 0.249 | 0.295, 0.137, 0.304, 0.264 |
| `perm_ctrl_a_enc` | 4 | Q | 0.570 | 0.560 | 0.457, 0.220, 0.178, 0.145 | 0.292, 0.198, 0.274, 0.236 |
| `perm_ctrl_a_enc` | 4 | K | 0.456 | 0.455 | 0.250, 0.245, 0.312, 0.193 | 0.227, 0.057, 0.373, 0.343 |
| `perm_ctrl_a_enc` | 4 | V | 0.586 | 0.594 | 0.072, 0.375, 0.173, 0.380 | 0.077, 0.143, 0.531, 0.249 |
| `perm_ctrl_b_enc` | 1 | Q | 0.704 | 0.712 | 0.591, 0.138, 0.190, 0.080 | 0.359, 0.220, 0.295, 0.125 |
| `perm_ctrl_b_enc` | 1 | K | 0.636 | 0.635 | 0.343, 0.118, 0.269, 0.270 | 0.132, 0.392, 0.200, 0.277 |
| `perm_ctrl_b_enc` | 1 | V | 0.690 | 0.697 | 0.314, 0.093, 0.341, 0.251 | 0.208, 0.148, 0.213, 0.430 |
| `perm_ctrl_b_enc` | 2 | Q | 0.554 | 0.554 | 0.333, 0.258, 0.163, 0.246 | 0.345, 0.201, 0.186, 0.268 |
| `perm_ctrl_b_enc` | 2 | K | 0.590 | 0.576 | 0.476, 0.128, 0.259, 0.137 | 0.232, 0.282, 0.171, 0.315 |
| `perm_ctrl_b_enc` | 2 | V | 0.674 | 0.682 | 0.156, 0.278, 0.245, 0.321 | 0.204, 0.261, 0.296, 0.239 |
| `perm_ctrl_b_enc` | 3 | Q | 0.396 | 0.392 | 0.290, 0.291, 0.264, 0.156 | 0.156, 0.231, 0.239, 0.374 |
| `perm_ctrl_b_enc` | 3 | K | 0.656 | 0.654 | 0.467, 0.086, 0.230, 0.217 | 0.256, 0.372, 0.201, 0.171 |
| `perm_ctrl_b_enc` | 3 | V | 0.676 | 0.676 | 0.081, 0.290, 0.396, 0.234 | 0.226, 0.198, 0.279, 0.297 |
| `perm_ctrl_b_enc` | 4 | Q | 0.589 | 0.589 | 0.230, 0.467, 0.149, 0.154 | 0.304, 0.182, 0.252, 0.262 |
| `perm_ctrl_b_enc` | 4 | K | 0.505 | 0.506 | 0.300, 0.169, 0.218, 0.314 | 0.050, 0.231, 0.266, 0.453 |
| `perm_ctrl_b_enc` | 4 | V | 0.625 | 0.629 | 0.121, 0.403, 0.173, 0.303 | 0.085, 0.212, 0.379, 0.324 |
| `perm_ctrl_c_enc` | 1 | Q | 0.680 | 0.702 | 0.466, 0.160, 0.248, 0.126 | 0.226, 0.178, 0.444, 0.152 |
| `perm_ctrl_c_enc` | 1 | K | 0.598 | 0.607 | 0.497, 0.099, 0.219, 0.184 | 0.259, 0.197, 0.335, 0.209 |
| `perm_ctrl_c_enc` | 1 | V | 0.713 | 0.722 | 0.343, 0.160, 0.232, 0.265 | 0.173, 0.296, 0.153, 0.378 |
| `perm_ctrl_c_enc` | 2 | Q | 0.506 | 0.504 | 0.259, 0.281, 0.180, 0.280 | 0.288, 0.305, 0.173, 0.234 |
| `perm_ctrl_c_enc` | 2 | K | 0.510 | 0.511 | 0.430, 0.173, 0.310, 0.087 | 0.168, 0.247, 0.360, 0.225 |
| `perm_ctrl_c_enc` | 2 | V | 0.656 | 0.662 | 0.181, 0.206, 0.341, 0.272 | 0.319, 0.265, 0.162, 0.254 |
| `perm_ctrl_c_enc` | 3 | Q | 0.529 | 0.531 | 0.395, 0.188, 0.167, 0.250 | 0.251, 0.245, 0.240, 0.264 |
| `perm_ctrl_c_enc` | 3 | K | 0.687 | 0.684 | 0.343, 0.222, 0.310, 0.126 | 0.320, 0.182, 0.314, 0.183 |
| `perm_ctrl_c_enc` | 3 | V | 0.688 | 0.693 | 0.133, 0.275, 0.344, 0.248 | 0.258, 0.367, 0.151, 0.223 |
| `perm_ctrl_c_enc` | 4 | Q | 0.566 | 0.563 | 0.127, 0.356, 0.214, 0.304 | 0.403, 0.164, 0.149, 0.285 |
| `perm_ctrl_c_enc` | 4 | K | 0.514 | 0.515 | 0.241, 0.276, 0.300, 0.183 | 0.099, 0.333, 0.252, 0.316 |
| `perm_ctrl_c_enc` | 4 | V | 0.579 | 0.602 | 0.090, 0.237, 0.272, 0.400 | 0.102, 0.445, 0.221, 0.232 |

## Development BLEU by checkpoint

Original and all-family hard are reused references. The five subset columns are new inference-only decodes from unchanged checkpoints.

| model | seed | original | all hard | core soft | core hard | core uniform | non-core soft | non-core hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 42 | 39.20 | 39.18 | 38.87 | 38.94 | 37.69 | 38.54 | 38.66 |
| `ana_d4_enc` | 43 | 38.89 | 38.77 | 38.41 | 38.32 | 37.74 | 38.32 | 38.31 |
| `ana_d4_enc` | 44 | 40.00 | 39.89 | 39.24 | 39.15 | 38.64 | 39.46 | 39.27 |
| `perm_ctrl_a_enc` | 42 | 39.58 | 39.55 | 38.81 | 38.67 | 38.45 | 38.88 | 38.63 |
| `perm_ctrl_a_enc` | 43 | 39.18 | 39.01 | 37.96 | 37.65 | 36.94 | 38.19 | 38.21 |
| `perm_ctrl_a_enc` | 44 | 38.95 | 39.02 | 38.90 | 38.88 | 37.64 | 38.69 | 38.70 |
| `perm_ctrl_b_enc` | 42 | 39.69 | 39.59 | 39.49 | 39.16 | 38.04 | 38.87 | 38.91 |
| `perm_ctrl_b_enc` | 43 | 38.91 | 39.09 | 38.94 | 38.63 | 37.31 | 38.78 | 38.36 |
| `perm_ctrl_b_enc` | 44 | 39.68 | 39.60 | 38.82 | 38.69 | 38.33 | 39.28 | 39.17 |
| `perm_ctrl_c_enc` | 42 | 39.52 | 39.37 | 38.74 | 38.56 | 38.52 | 38.94 | 38.58 |
| `perm_ctrl_c_enc` | 43 | 38.60 | 38.75 | 37.97 | 37.71 | 37.56 | 37.46 | 37.42 |
| `perm_ctrl_c_enc` | 44 | 39.43 | 39.59 | 38.84 | 38.82 | 38.38 | 39.27 | 39.10 |

## Primary paired contrasts

| model | seed | core hard − all hard | core soft − original | core hard − non-core hard | core soft − non-core soft | core soft − core uniform |
|---|---:|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 42 | -0.24 | -0.33 | +0.28 | +0.33 | +1.18 |
| `ana_d4_enc` | 43 | -0.46 | -0.49 | +0.01 | +0.08 | +0.67 |
| `ana_d4_enc` | 44 | -0.74 | -0.75 | -0.12 | -0.22 | +0.60 |
| `perm_ctrl_a_enc` | 42 | -0.88 | -0.77 | +0.04 | -0.07 | +0.36 |
| `perm_ctrl_a_enc` | 43 | -1.37 | -1.22 | -0.56 | -0.23 | +1.02 |
| `perm_ctrl_a_enc` | 44 | -0.14 | -0.05 | +0.18 | +0.22 | +1.27 |
| `perm_ctrl_b_enc` | 42 | -0.43 | -0.20 | +0.26 | +0.61 | +1.45 |
| `perm_ctrl_b_enc` | 43 | -0.45 | +0.04 | +0.27 | +0.17 | +1.63 |
| `perm_ctrl_b_enc` | 44 | -0.91 | -0.86 | -0.48 | -0.45 | +0.49 |
| `perm_ctrl_c_enc` | 42 | -0.81 | -0.78 | -0.03 | -0.21 | +0.22 |
| `perm_ctrl_c_enc` | 43 | -1.04 | -0.63 | +0.28 | +0.51 | +0.41 |
| `perm_ctrl_c_enc` | 44 | -0.77 | -0.59 | -0.27 | -0.43 | +0.46 |
| **all 12 mean** | | **-0.69** | **-0.55** | **-0.01** | **+0.03** | **+0.81** |

### Family means

| family | core hard − all hard | core soft − original | core hard − non-core hard | core soft − non-core soft | core soft − core uniform |
|---|---:|---:|---:|---:|---:|
| `ana_d4_enc` | -0.48 | -0.52 | +0.06 | +0.07 | +0.81 |
| `perm_ctrl_a_enc` | -0.79 | -0.68 | -0.11 | -0.03 | +0.88 |
| `perm_ctrl_b_enc` | -0.60 | -0.34 | +0.02 | +0.11 | +1.19 |
| `perm_ctrl_c_enc` | -0.87 | -0.67 | -0.01 | -0.04 | +0.36 |

## Decision questions

- **Do trained routers favor the shared core?** Across all checkpoints and modules, mean core probability mass is 0.588 and all-family argmax selects the core on 0.592 of real source tokens.
- **Does core soft retain original BLEU?** The aggregate paired difference is -0.55 BLEU.
- **Does core hard retain all-family hard BLEU?** The aggregate paired difference is -0.69 BLEU.
- **Does learned core selection beat core uniform?** The aggregate paired difference is +0.81 BLEU.
- **Does the core beat the non-core complement?** Core minus non-core is -0.01 BLEU under hard selection and +0.03 BLEU under soft routing.
- **Is the comparison consistent across families?** Core hard exceeds non-core hard in 2/4 family means.

## Screening interpretation

Decision: **`full_family_coverage_is_necessary`**.

Both four-member subsets lose more than 0.50 BLEU while all-family hard selection retains performance. The next comparison should test full S4 coverage against a learned basis rather than train a V4-only model.

These are descriptive 12-checkpoint screening contrasts. No p-values or bootstrap significance claims are reported.
