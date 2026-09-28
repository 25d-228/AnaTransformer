# Context-dependent power: completed results

All 15 new runs are complete. They were launched in eight detached GPU queues
on exp14–18 on 2026-09-16 at 22:21 JST. Final scores, parameter counts, learned
power summaries, and paired comparisons are in the completed reports for
[English–German](multi30k.md), [English–French](multi30k_enfr.md), and
[COGS](cogs.md). Their accompanying JSON files retain exact results and provenance.

The four completed comparison rows below are reused unchanged.

| Model and training | Multi30k EN→DE BLEU | Multi30k EN→FR BLEU | COGS generalization EM (%) |
|---|---:|---:|---:|
| Compact embeddings, ordinary training | 40.31 ± 1.64 | 60.72 ± 1.72 | 78.88 ± 0.55 |
| Compact embeddings, two passes without analogy | 40.12 ± 1.67 | 60.66 ± 1.65 | 81.52 ± 0.50 |
| Compact embeddings, fixed p = 0.5 analogy | 40.90 ± 1.64 | 60.37 ± 1.66 | 81.10 ± 0.53 |
| Compact embeddings, lookahead p analogy | 40.82 ± 1.62 | 60.03 ± 1.62 | 81.74 ± 0.51 |

The five new variants completed on all three datasets are A (context-dependent
prediction power), A+ (prediction power + analogy), B (context-dependent feature
power), B+ (feature power + analogy), and B fixed (p = 1 feature analogy).

A adjusts prediction confidence with one learned power per position. B learns
one power per feature pair and changes the decoder features before prediction.
The new power heads learn through the prediction loss and operate at inference;
the fixed-B control has no head and an unchanged prediction path.

The key matched comparisons are A+ versus A, B+ versus B, and B+ versus fixed B.
All five new variants use two supervised dropout passes and the same compact
backbone, with different small head counts. Analogy penalties differ between A
and B; coefficient 1 does not imply equal strength.

± is half the width of a 95% example-bootstrap interval with 1,000 resamples.
COGS shows generalization only.

Completed reference reports: [English–German](../adaptive_power_lookahead_v1/multi30k.md),
[English–French](../adaptive_power_lookahead_v1/multi30k_enfr.md), and
[COGS](../adaptive_power_lookahead_v1/cogs.md). Their accompanying JSON files
retain exact scores, intervals, and original provenance.

Method, parameter counts, recipes, and execution:
[experiment protocol](../../../experiments/compact_embeddings/context_power_v1/README.md).
