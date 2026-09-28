# Compact embeddings: current results

The context-power batch has completed all 15 runs. Its results are recorded
for [English–German](context_power_v1/multi30k.md),
[English–French](context_power_v1/multi30k_enfr.md), and
[COGS](context_power_v1/cogs.md). Its small prediction heads change the inference
network; the completed training-only comparisons below remain unchanged.

The [lookahead p-selection batch](adaptive_power_lookahead_v1/README.md)
completed on 2026-09-16 for both Multi30k directions and COGS. Its results are
included below; all earlier rows are unchanged.

The [rescaled adaptive-power follow-up](adaptive_power_balanced_v1/README.md)
completed on 2026-09-16 for both Multi30k directions and COGS. Its results are
included below; all earlier rows are unchanged.

All rows below use the same compact architecture for a given dataset, with
independent Q/K/V projections and factorized tied input/output embeddings.
Only training differs. Power and its adaptation add no inference parameters.

| Training method | Multi30k EN→DE BLEU | Multi30k EN→FR BLEU | IWSLT14 DE→EN BLEU | COGS generalization EM (%) |
|---|---:|---:|---:|---:|
| Ordinary training | 40.31 ± 1.64 | 60.72 ± 1.72 | 32.82 ± 0.51 | 78.88 ± 0.55 |
| Two dropout passes, without analogy | 40.12 ± 1.67 | 60.66 ± 1.65 | — | 81.52 ± 0.50 |
| Analogy training, fixed p = 0.5 | 40.90 ± 1.64 | 60.37 ± 1.66 | 33.60 ± 0.50 | 81.10 ± 0.53 |
| Analogy training, adaptive p | 39.59 ± 1.66 | 56.83 ± 1.63 | — | 76.05 ± 0.55 |
| Analogy training, rescaled adaptive p | 40.44 ± 1.63 | 59.85 ± 1.64 | — | 80.99 ± 0.54 |
| Analogy training, lookahead adaptive p | 40.82 ± 1.62 | 60.03 ± 1.62 | — | 81.74 ± 0.51 |

± is half the width of a 95% example-bootstrap interval with 1,000 resamples.
Exact endpoints and paired comparisons remain in the dataset reports. A dash
means that the IWSLT14 two-pass or adaptive-power experiment was not run.
COGS shows generalization only.

Ordinary compact training is the main control. The two-pass row checks the
effect of the analogy penalty beyond using two supervised dropout passes.
Each corpus retains its own base recipe, checkpoint rule, and decoding setup.

The lookahead method has mixed translation results: EN→DE is close to fixed p
and above ordinary compact training in point score; EN→FR remains below both.
COGS improves over ordinary, two-pass, and fixed-p training. Its exact paired
comparisons are in the [completed lookahead report](adaptive_power_lookahead_v1/README.md).

| Dataset | Compact parameters | Reduction from full Transformer |
|---|---:|---:|
| Multi30k, either direction | 2,248,512 | 13.70% |
| IWSLT14 | 27,241,504 | 25.70% |
| COGS | 5,689,236 | 35.68% |

## Completed comparisons

- Context-dependent power batch: [English–German](context_power_v1/multi30k.md), [English–French](context_power_v1/multi30k_enfr.md), and [COGS](context_power_v1/cogs.md).
- Lookahead adaptive-power batch: [English–German](adaptive_power_lookahead_v1/multi30k.md), [English–French](adaptive_power_lookahead_v1/multi30k_enfr.md), and [COGS](adaptive_power_lookahead_v1/cogs.md).
- Rescaled adaptive-power batch: [English–German](adaptive_power_balanced_v1/multi30k.md), [English–French](adaptive_power_balanced_v1/multi30k_enfr.md), and [COGS](adaptive_power_balanced_v1/cogs.md).
- Adaptive-power batch: [English–German](adaptive_power_v1/multi30k.md), [English–French](adaptive_power_v1/multi30k_enfr.md), and [COGS](adaptive_power_v1/cogs.md).
- Fixed-power IWSLT14: [ordinary versus power training](power_consistency_iwslt_v1/iwslt14.md).
- Original fixed-power reports with model-size references: [English–German](power_consistency_v1/multi30k.md), [English–French](power_enfr_cogs_v1/multi30k_enfr.md), and [COGS](power_enfr_cogs_v1/cogs.md).
- [Method overview and completed alternatives](overview.md).

Each dataset report has a same-named JSON file containing its full-precision
results and original provenance. All three adaptive batches and the IWSLT14 pair are
complete; this table does not infer scores for unrun cells.

## Earlier embedding designs

- [Direct power-analogy embeddings](embedding_analogy_v1/multi30k.md).
- [Unit-scale embeddings](embedding_unit_v1/multi30k.md).
- [Residual embeddings](embedding_residual_v1/multi30k.md).
- [Mixed embeddings](embedding_mix_v1/multi30k.md).
- [Frequency-aware compression](frequency_lexical_v1/multi30k.md).
- [One-gradient consistency training](power_consistency_singlegrad_v1/multi30k.md).
- [Causal translation-example analogy](causal_translation_v1/multi30k.md).

Protocols and runnable scripts: [compact-embedding experiments](../../experiments/compact_embeddings/README.md).
