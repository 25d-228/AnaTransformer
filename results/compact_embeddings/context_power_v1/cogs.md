# COGS generalization: context-conditioned power

Four completed reference rows are reused. Five new variants share the same compact backbone and two-dropout-pass training recipe. A learns one prediction power per position; B learns one power per feature pair and position. Their heads are active at inference and are learned through the supervised prediction loss. The fixed B control has no power head. A and B use different analogy penalties; coefficient 1 does not make their penalty strengths equivalent.

| Model and training | Parameters | Exact match (%) |
|---|---:|---:|
| Compact embeddings, ordinary training | 5,689,236 | 78.88 ± 0.55 |
| Same compact model, two dropout passes without analogy | 5,689,236 | 81.52 ± 0.50 |
| Same compact model + analogy training, fixed p = 0.5 | 5,689,236 | 81.10 ± 0.53 |
| Compact model + previous lookahead power training | 5,689,236 | 81.74 ± 0.51 |
| A: context-dependent prediction power, without analogy | 5,689,637 | 81.42 ± 0.51 |
| A+: context-dependent prediction power + analogy | 5,689,637 | 81.73 ± 0.52 |
| B: context-dependent feature power, without analogy | 5,769,436 | 80.04 ± 0.54 |
| B+: context-dependent feature power + analogy | 5,769,436 | 81.12 ± 0.53 |
| B fixed: p = 1 feature analogy, without a power head | 5,689,236 | 78.81 ± 0.54 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

## Context-dependent p at the scored checkpoint

These summaries use valid development positions with dropout off; they are not a single global p or test-set feedback.

| Model | Mean p | Minimum p | Maximum p |
|---|---:|---:|---:|
| A: context-dependent prediction power, without analogy | 0.7472 | 0.3953 | 0.7500 |
| A+: context-dependent prediction power + analogy | 0.6327 | 0.3031 | 0.7321 |
| B: context-dependent feature power, without analogy | 1.3181 | 0.5000 | 1.5000 |
| B+: context-dependent feature power + analogy | 1.1212 | 0.5000 | 1.5000 |
| B fixed: p = 1 feature analogy, without a power head | 1.0000 | 1.0000 | 1.0000 |

## Matched comparisons

| First model minus second model | Difference and 95% paired interval |
|---|---:|
| A+: context-dependent prediction power + analogy minus A: context-dependent prediction power, without analogy | +0.31 [+0.15, +0.49] |
| B+: context-dependent feature power + analogy minus B: context-dependent feature power, without analogy | +1.08 [+0.84, +1.30] |
| B+: context-dependent feature power + analogy minus B fixed: p = 1 feature analogy, without a power head | +2.31 [+2.03, +2.60] |

## New models minus completed references

Each cell gives the difference and its 95% paired bootstrap interval.

| New model | Ordinary | Two-pass, no analogy | Fixed p = 0.5 | Lookahead |
|---|---:|---:|---:|---:|
| A: context-dependent prediction power, without analogy | +2.55 [+2.31, +2.77] | -0.10 [-0.28, +0.07] | +0.32 [+0.16, +0.47] | -0.31 [-0.49, -0.15] |
| A+: context-dependent prediction power + analogy | +2.86 [+2.61, +3.10] | +0.21 [+0.05, +0.39] | +0.63 [+0.46, +0.80] | -0.00 [-0.19, +0.18] |
| B: context-dependent feature power, without analogy | +1.17 [+0.89, +1.46] | -1.48 [-1.68, -1.25] | -1.06 [-1.30, -0.82] | -1.70 [-1.93, -1.46] |
| B+: context-dependent feature power + analogy | +2.25 [+2.00, +2.50] | -0.40 [-0.59, -0.21] | +0.02 [-0.14, +0.19] | -0.61 [-0.80, -0.42] |
| B fixed: p = 1 feature analogy, without a power head | -0.06 [-0.32, +0.18] | -2.70 [-2.99, -2.43] | -2.29 [-2.58, -2.00] | -2.92 [-3.22, -2.63] |

Completed scores are computed from saved predictions. Exact interval endpoints, source records, model counts, and paired comparisons remain in JSON. Pending entries have no inferred scores. Each dataset retains its own base training recipe and checkpoint rule; COGS reports generalization only.
