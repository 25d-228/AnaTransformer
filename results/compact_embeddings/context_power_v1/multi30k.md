# Multi30k English → German: context-conditioned power

Four completed reference rows are reused. Five new variants share the same compact backbone and two-dropout-pass training recipe. A learns one prediction power per position; B learns one power per feature pair and position. Their heads are active at inference and are learned through the supervised prediction loss. The fixed B control has no power head. A and B use different analogy penalties; coefficient 1 does not make their penalty strengths equivalent.

| Model and training | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 40.31 ± 1.64 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 40.12 ± 1.67 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 40.90 ± 1.64 |
| Compact model + previous lookahead power training | 2,248,512 | 40.82 ± 1.62 |
| A: context-dependent prediction power, without analogy | 2,248,641 | 39.61 ± 1.68 |
| A+: context-dependent prediction power + analogy | 2,248,641 | 41.13 ± 1.65 |
| B: context-dependent feature power, without analogy | 2,256,768 | 40.61 ± 1.68 |
| B+: context-dependent feature power + analogy | 2,256,768 | 40.08 ± 1.62 |
| B fixed: p = 1 feature analogy, without a power head | 2,248,512 | 40.53 ± 1.67 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

## Context-dependent p at the scored checkpoint

These summaries use valid development positions with dropout off; they are not a single global p or test-set feedback.

| Model | Mean p | Minimum p | Maximum p |
|---|---:|---:|---:|
| A: context-dependent prediction power, without analogy | 0.3734 | 0.2534 | 0.7016 |
| A+: context-dependent prediction power + analogy | 0.7500 | 0.7500 | 0.7500 |
| B: context-dependent feature power, without analogy | 0.8529 | 0.5000 | 1.5000 |
| B+: context-dependent feature power + analogy | 0.9915 | 0.5000 | 1.5000 |
| B fixed: p = 1 feature analogy, without a power head | 1.0000 | 1.0000 | 1.0000 |

## Matched comparisons

| First model minus second model | Difference and 95% paired interval |
|---|---:|
| A+: context-dependent prediction power + analogy minus A: context-dependent prediction power, without analogy | +1.52 [+0.71, +2.26] |
| B+: context-dependent feature power + analogy minus B: context-dependent feature power, without analogy | -0.53 [-1.37, +0.21] |
| B+: context-dependent feature power + analogy minus B fixed: p = 1 feature analogy, without a power head | -0.44 [-1.22, +0.36] |

## New models minus completed references

Each cell gives the difference and its 95% paired bootstrap interval.

| New model | Ordinary | Two-pass, no analogy | Fixed p = 0.5 | Lookahead |
|---|---:|---:|---:|---:|
| A: context-dependent prediction power, without analogy | -0.70 [-1.64, +0.22] | -0.51 [-1.47, +0.40] | -1.29 [-2.08, -0.42] | -1.21 [-2.07, -0.38] |
| A+: context-dependent prediction power + analogy | +0.81 [-0.04, +1.63] | +1.00 [+0.14, +1.76] | +0.23 [-0.40, +0.96] | +0.31 [-0.47, +1.04] |
| B: context-dependent feature power, without analogy | +0.30 [-0.49, +1.21] | +0.49 [-0.34, +1.37] | -0.28 [-0.98, +0.58] | -0.20 [-0.92, +0.64] |
| B+: context-dependent feature power + analogy | -0.23 [-1.06, +0.58] | -0.04 [-0.94, +0.81] | -0.81 [-1.55, -0.00] | -0.73 [-1.52, +0.12] |
| B fixed: p = 1 feature analogy, without a power head | +0.21 [-0.64, +1.19] | +0.40 [-0.46, +1.27] | -0.37 [-1.10, +0.41] | -0.29 [-1.11, +0.56] |

Completed scores are computed from saved predictions. Exact interval endpoints, source records, model counts, and paired comparisons remain in JSON. Pending entries have no inferred scores. Each dataset retains its own base training recipe and checkpoint rule; COGS reports generalization only.
