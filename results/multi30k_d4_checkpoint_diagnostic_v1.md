# multi30k_d4_checkpoint_diagnostic_v1

Inference-only analysis of six trained Multi30k D4 checkpoints. All statistics and decodes use the full development set; the test set is not loaded or decoded.

## Checkpoint reproduction

| model | seed | selected step | stored dev BLEU | reproduced | difference |
|---|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 42 | 20,000 | 39.1970 | 39.1970 | +0.0000 |
| `ana_d4_enc` | 43 | 19,000 | 38.8934 | 38.8934 | +0.0000 |
| `ana_d4_enc` | 44 | 20,000 | 39.9955 | 39.9955 | +0.0000 |
| `ana_feat_enc` | 42 | 20,000 | 39.8047 | 39.8047 | +0.0000 |
| `ana_feat_enc` | 43 | 20,000 | 38.8890 | 38.8890 | +0.0000 |
| `ana_feat_enc` | 44 | 19,000 | 39.5246 | 39.5246 | +0.0000 |

## Development-set interventions

| model | seed | condition | dev BLEU | Δ original |
|---|---:|---|---:|---:|
| `ana_d4_enc` | 42 | `original` | 39.20 | +0.00 |
| `ana_d4_enc` | 42 | `gate_zero` | 36.11 | -3.09 |
| `ana_d4_enc` | 42 | `uniform_router` | 37.69 | -1.50 |
| `ana_d4_enc` | 42 | `identity_router` | 36.11 | -3.09 |
| `ana_d4_enc` | 42 | `hard_argmax` | 39.18 | -0.02 |
| `ana_d4_enc` | 43 | `original` | 38.89 | +0.00 |
| `ana_d4_enc` | 43 | `gate_zero` | 37.75 | -1.15 |
| `ana_d4_enc` | 43 | `uniform_router` | 37.74 | -1.15 |
| `ana_d4_enc` | 43 | `identity_router` | 37.75 | -1.15 |
| `ana_d4_enc` | 43 | `hard_argmax` | 38.77 | -0.12 |
| `ana_d4_enc` | 44 | `original` | 40.00 | +0.00 |
| `ana_d4_enc` | 44 | `gate_zero` | 37.74 | -2.25 |
| `ana_d4_enc` | 44 | `uniform_router` | 38.64 | -1.35 |
| `ana_d4_enc` | 44 | `identity_router` | 37.74 | -2.25 |
| `ana_d4_enc` | 44 | `hard_argmax` | 39.89 | -0.10 |
| `ana_feat_enc` | 42 | `original` | 39.80 | +0.00 |
| `ana_feat_enc` | 42 | `gate_zero` | 26.89 | -12.91 |
| `ana_feat_enc` | 42 | `uniform_router` | 37.27 | -2.54 |
| `ana_feat_enc` | 42 | `identity_router` | 34.71 | -5.09 |
| `ana_feat_enc` | 42 | `hard_argmax` | 39.69 | -0.12 |
| `ana_feat_enc` | 42 | `magnitude_one` | 32.48 | -7.32 |
| `ana_feat_enc` | 43 | `original` | 38.89 | +0.00 |
| `ana_feat_enc` | 43 | `gate_zero` | 25.08 | -13.81 |
| `ana_feat_enc` | 43 | `uniform_router` | 37.16 | -1.73 |
| `ana_feat_enc` | 43 | `identity_router` | 35.69 | -3.20 |
| `ana_feat_enc` | 43 | `hard_argmax` | 38.91 | +0.02 |
| `ana_feat_enc` | 43 | `magnitude_one` | 33.57 | -5.32 |
| `ana_feat_enc` | 44 | `original` | 39.52 | +0.00 |
| `ana_feat_enc` | 44 | `gate_zero` | 31.48 | -8.04 |
| `ana_feat_enc` | 44 | `uniform_router` | 37.24 | -2.28 |
| `ana_feat_enc` | 44 | `identity_router` | 36.69 | -2.83 |
| `ana_feat_enc` | 44 | `hard_argmax` | 39.41 | -0.11 |
| `ana_feat_enc` | 44 | `magnitude_one` | 35.34 | -4.18 |

## Intervention means across seeds

| model | condition | dev BLEU mean ± sample SD | mean Δ | paired Δ by seed |
|---|---|---:|---:|---|
| `ana_d4_enc` | `original` | 39.36 ± 0.57 | +0.00 | 42: +0.00, 43: +0.00, 44: +0.00 |
| `ana_d4_enc` | `gate_zero` | 37.20 ± 0.94 | -2.16 | 42: -3.09, 43: -1.15, 44: -2.25 |
| `ana_d4_enc` | `uniform_router` | 38.03 ± 0.53 | -1.34 | 42: -1.50, 43: -1.15, 44: -1.35 |
| `ana_d4_enc` | `identity_router` | 37.20 ± 0.94 | -2.16 | 42: -3.09, 43: -1.15, 44: -2.25 |
| `ana_d4_enc` | `hard_argmax` | 39.28 ± 0.57 | -0.08 | 42: -0.02, 43: -0.12, 44: -0.10 |
| `ana_feat_enc` | `original` | 39.41 ± 0.47 | +0.00 | 42: +0.00, 43: +0.00, 44: +0.00 |
| `ana_feat_enc` | `gate_zero` | 27.82 ± 3.30 | -11.59 | 42: -12.91, 43: -13.81, 44: -8.04 |
| `ana_feat_enc` | `uniform_router` | 37.22 ± 0.06 | -2.18 | 42: -2.54, 43: -1.73, 44: -2.28 |
| `ana_feat_enc` | `identity_router` | 35.70 ± 0.99 | -3.71 | 42: -5.09, 43: -3.20, 44: -2.83 |
| `ana_feat_enc` | `hard_argmax` | 39.34 ± 0.39 | -0.07 | 42: -0.12, 43: +0.02, 44: -0.11 |
| `ana_feat_enc` | `magnitude_one` | 33.80 ± 1.44 | -5.61 | 42: -7.32, 43: -5.32, 44: -4.18 |

## Router statistics across seeds

Each row is the mean of the same labeled module over seeds 42, 43, and 44.

| model | layer | role | gate | H/log8 | max p | effective | identity | nearest D4 | token JS | QKV JS | magnitude mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ana_d4_enc` | 1 | Q | 0.452 | 0.168 | 0.865 | 1.54 | 0.316 | 0.101 | 0.6206 | 0.8063 | — |
| `ana_d4_enc` | 1 | K | 0.279 | 0.198 | 0.838 | 1.62 | 0.210 | 0.134 | 0.6327 | 0.8063 | — |
| `ana_d4_enc` | 1 | V | 0.744 | 0.141 | 0.897 | 1.51 | 0.198 | 0.077 | 0.6618 | 0.8063 | — |
| `ana_d4_enc` | 2 | Q | 0.521 | 0.164 | 0.863 | 1.51 | 0.053 | 0.119 | 0.7075 | 0.8100 | — |
| `ana_d4_enc` | 2 | K | 0.262 | 0.257 | 0.781 | 1.81 | 0.199 | 0.188 | 0.6256 | 0.8100 | — |
| `ana_d4_enc` | 2 | V | 0.549 | 0.147 | 0.890 | 1.48 | 0.074 | 0.083 | 0.6882 | 0.8100 | — |
| `ana_d4_enc` | 3 | Q | 0.348 | 0.101 | 0.916 | 1.33 | 0.081 | 0.060 | 0.7100 | 0.8682 | — |
| `ana_d4_enc` | 3 | K | 0.222 | 0.118 | 0.902 | 1.35 | 0.234 | 0.086 | 0.6878 | 0.8682 | — |
| `ana_d4_enc` | 3 | V | 0.517 | 0.156 | 0.875 | 1.48 | 0.040 | 0.099 | 0.7461 | 0.8682 | — |
| `ana_d4_enc` | 4 | Q | 0.305 | 0.043 | 0.967 | 1.12 | 0.091 | 0.028 | 0.6080 | 0.9054 | — |
| `ana_d4_enc` | 4 | K | 0.268 | 0.056 | 0.953 | 1.16 | 0.100 | 0.043 | 0.6509 | 0.9054 | — |
| `ana_d4_enc` | 4 | V | 0.487 | 0.109 | 0.920 | 1.33 | 0.029 | 0.061 | 0.7265 | 0.9054 | — |
| `ana_feat_enc` | 1 | Q | 0.450 | 0.080 | 0.933 | 1.23 | 0.033 | 0.054 | 0.6278 | 0.8731 | 3.624 |
| `ana_feat_enc` | 1 | K | 0.097 | 0.156 | 0.866 | 1.50 | 0.178 | 0.109 | 0.6435 | 0.8731 | 2.160 |
| `ana_feat_enc` | 1 | V | 0.889 | 0.042 | 0.967 | 1.12 | 0.242 | 0.026 | 0.7723 | 0.8731 | 0.131 |
| `ana_feat_enc` | 2 | Q | 0.339 | 0.101 | 0.918 | 1.29 | 0.116 | 0.065 | 0.6331 | 0.8829 | 3.432 |
| `ana_feat_enc` | 2 | K | 0.150 | 0.199 | 0.841 | 1.62 | 0.227 | 0.124 | 0.6336 | 0.8829 | 2.807 |
| `ana_feat_enc` | 2 | V | 0.767 | 0.132 | 0.901 | 1.41 | 0.214 | 0.082 | 0.7392 | 0.8829 | 0.248 |
| `ana_feat_enc` | 3 | Q | 0.187 | 0.043 | 0.966 | 1.12 | 0.295 | 0.027 | 0.6518 | 0.8793 | 3.614 |
| `ana_feat_enc` | 3 | K | 0.150 | 0.087 | 0.930 | 1.25 | 0.164 | 0.059 | 0.6769 | 0.8793 | 3.307 |
| `ana_feat_enc` | 3 | V | 0.692 | 0.103 | 0.918 | 1.32 | 0.108 | 0.062 | 0.7707 | 0.8793 | 0.397 |
| `ana_feat_enc` | 4 | Q | 0.377 | 0.124 | 0.905 | 1.40 | 0.137 | 0.078 | 0.6932 | 0.8684 | 3.612 |
| `ana_feat_enc` | 4 | K | 0.157 | 0.143 | 0.883 | 1.46 | 0.229 | 0.093 | 0.6450 | 0.8684 | 2.658 |
| `ana_feat_enc` | 4 | V | 0.680 | 0.154 | 0.886 | 1.49 | 0.073 | 0.088 | 0.7442 | 0.8684 | 0.369 |

## Decision questions

- **Did the gates open enough for the branch to matter?** Mean gate strength is 0.413 for `ana_d4_enc` and 0.411 for `ana_feat_enc` (initial value 0.119); the causal check is the gate-zero intervention below.
- **Does disabling the routed branch reduce BLEU?** Gate zero changes mean BLEU by -2.16 and -11.59, respectively.
- **Did routing move away from uniform?** Mean normalized entropy / maximum probability is 0.138 / 0.889 for `ana_d4_enc` and 0.114 / 0.910 for `ana_feat_enc`; uniform routing is 1.000 / 0.125.
- **Are routes token-conditioned?** Generalized token JS averages 0.6721 and 0.6860; zero is token-constant routing.
- **Did Q, K, and V differentiate?** Normalized role JS averages 0.8475 and 0.8759; zero means identical role distributions.
- **Are soft matrices close to exact D4 permutations?** Mean normalized nearest-form distance is 0.090 and 0.072; zero is an exact form.
- **Does hard argmax retain BLEU?** Its mean difference is -0.08 for `ana_d4_enc` and -0.07 for `ana_feat_enc`. Hard argmax makes only the routed matrix `P` an exact D4 permutation; the learned gate, diagonal scale, and (for `ana_feat_enc`) magnitude remain active.
- **Does learned routing beat uniform and identity?** Relative to original, uniform / identity change mean BLEU by -1.34 / -2.16 for `ana_d4_enc`, and -2.18 / -3.71 for `ana_feat_enc`.
- **Are gate zero and identity routing independent checks?** Not for `ana_d4_enc`: magnitude is fixed to one, so identity mixing gives `mixed = z`, exactly as gate zero does. Their identical scores are one effective ablation, not independent evidence.
- **Does the combined model rely on magnification?** Setting magnitude to one changes mean BLEU by -5.61.

## Interpretation for the next discussion

Descriptively, this matches the first decision pattern: hard argmax retains nearly all BLEU, routing is strongly token- and role-differentiated, the soft matrices lie near individual D4 forms, and effective gate/router interventions matter on every seed. This is evidence for a near-discrete, token- and role-conditioned D4 routing component. It does not make the whole role transform exactly D4: under hard argmax the D4-only role is `diag(scale)[(1-g)I + gP]z`, while the combined role also contains a learned magnitude. The full transform is therefore not shown to lie exactly in an analogy-preserving `D4 × R+` orbit. Exact D4 routing remains a live direction for a later comparison with random permutation families and generic local mixers. Those comparisons are not implemented here.

The combined checkpoints also rely heavily on their learned magnitudes, but the factorial screen did not show a useful combined-model advantage. Checkpoint reliance therefore demonstrates co-adaptation, not that the magnifier improves the architecture.

These three-seed effects are descriptive screening evidence. No p-values or bootstrap significance tests are reported.
