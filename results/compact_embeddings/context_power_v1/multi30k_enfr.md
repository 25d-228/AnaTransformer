# Multi30k English → French: context-conditioned power

Four completed reference rows are reused. Five new variants share the same compact backbone and two-dropout-pass training recipe. A learns one prediction power per position; B learns one power per feature pair and position. Their heads are active at inference and are learned through the supervised prediction loss. The fixed B control has no power head. A and B use different analogy penalties; coefficient 1 does not make their penalty strengths equivalent.

| Model and training | Parameters | BLEU |
|---|---:|---:|
| Compact embeddings, ordinary training | 2,248,512 | 60.72 ± 1.72 |
| Same compact model, two dropout passes without analogy | 2,248,512 | 60.66 ± 1.65 |
| Same compact model + analogy training, fixed p = 0.5 | 2,248,512 | 60.37 ± 1.66 |
| Compact model + previous lookahead power training | 2,248,512 | 60.03 ± 1.62 |
| A: context-dependent prediction power, without analogy | 2,248,641 | 60.41 ± 1.69 |
| A+: context-dependent prediction power + analogy | 2,248,641 | 60.63 ± 1.65 |
| B: context-dependent feature power, without analogy | 2,256,768 | 60.53 ± 1.62 |
| B+: context-dependent feature power + analogy | 2,256,768 | 60.60 ± 1.66 |
| B fixed: p = 1 feature analogy, without a power head | 2,248,512 | 60.82 ± 1.67 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON. COGS reports generalization only.

## Context-dependent p at the scored checkpoint

These summaries use valid development positions with dropout off; they are not a single global p or test-set feedback.

| Model | Mean p | Minimum p | Maximum p |
|---|---:|---:|---:|
| A: context-dependent prediction power, without analogy | 0.3461 | 0.2532 | 0.7343 |
| A+: context-dependent prediction power + analogy | 0.7500 | 0.7500 | 0.7500 |
| B: context-dependent feature power, without analogy | 0.7749 | 0.5000 | 1.5000 |
| B+: context-dependent feature power + analogy | 1.0107 | 0.5000 | 1.5000 |
| B fixed: p = 1 feature analogy, without a power head | 1.0000 | 1.0000 | 1.0000 |

## Matched comparisons

| First model minus second model | Difference and 95% paired interval |
|---|---:|
| A+: context-dependent prediction power + analogy minus A: context-dependent prediction power, without analogy | +0.22 [-0.62, +1.03] |
| B+: context-dependent feature power + analogy minus B: context-dependent feature power, without analogy | +0.07 [-0.59, +0.78] |
| B+: context-dependent feature power + analogy minus B fixed: p = 1 feature analogy, without a power head | -0.22 [-1.00, +0.55] |

## New models minus completed references

Each cell gives the difference and its 95% paired bootstrap interval.

| New model | Ordinary | Two-pass, no analogy | Fixed p = 0.5 | Lookahead |
|---|---:|---:|---:|---:|
| A: context-dependent prediction power, without analogy | -0.30 [-0.99, +0.45] | -0.25 [-0.98, +0.53] | +0.04 [-0.70, +0.85] | +0.39 [-0.34, +1.17] |
| A+: context-dependent prediction power + analogy | -0.08 [-0.82, +0.67] | -0.03 [-0.77, +0.79] | +0.26 [-0.47, +1.07] | +0.61 [-0.10, +1.42] |
| B: context-dependent feature power, without analogy | -0.19 [-0.95, +0.55] | -0.13 [-0.86, +0.60] | +0.16 [-0.58, +0.92] | +0.50 [-0.22, +1.28] |
| B+: context-dependent feature power + analogy | -0.12 [-0.81, +0.59] | -0.06 [-0.78, +0.68] | +0.22 [-0.47, +0.95] | +0.57 [-0.16, +1.34] |
| B fixed: p = 1 feature analogy, without a power head | +0.10 [-0.74, +0.92] | +0.16 [-0.61, +0.95] | +0.45 [-0.27, +1.20] | +0.79 [+0.11, +1.54] |

Completed scores are computed from saved predictions. Exact interval endpoints, source records, model counts, and paired comparisons remain in JSON. Pending entries have no inferred scores. Each dataset retains its own base training recipe and checkpoint rule; COGS reports generalization only.
