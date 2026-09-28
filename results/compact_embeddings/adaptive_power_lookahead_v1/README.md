# Lookahead adaptive power: completed results

Status: all three runs completed on 2026-09-16, including decoding and bootstrap
reporting. They keep the exact balanced analogy loss
and compact model, but choose p through three-step temporary trials followed by
three training-batch checks. The five completed reference rows are unchanged.

| Compact-model training | Multi30k EN→DE BLEU | Multi30k EN→FR BLEU | COGS generalization EM (%) |
|---|---:|---:|---:|
| Ordinary training | 40.31 ± 1.64 | 60.72 ± 1.72 | 78.88 ± 0.55 |
| Two dropout passes, without analogy | 40.12 ± 1.67 | 60.66 ± 1.65 | 81.52 ± 0.50 |
| Analogy training, fixed p = 0.5 | 40.90 ± 1.64 | 60.37 ± 1.66 | 81.10 ± 0.53 |
| Previous adaptive-power training | 39.59 ± 1.66 | 56.83 ± 1.63 | 76.05 ± 0.55 |
| Scale-balanced adaptive-power training | 40.44 ± 1.63 | 59.85 ± 1.64 | 80.99 ± 0.54 |
| Lookahead adaptive-power training | 40.82 ± 1.62 | 60.03 ± 1.62 | 81.74 ± 0.51 |

± is half the width of a 95% example-bootstrap interval with 1,000 resamples.
Each dataset retains the same compact architecture
and parameter count across rows. COGS shows generalization only.

Translation results remain mixed: EN→DE is close to the fixed-power score and
above ordinary compact training in point score; EN→FR remains below both.
COGS improves over ordinary, two-pass, and fixed-power training.

## Lookahead minus the main comparison rows

Each cell gives the score difference and its 95% paired bootstrap interval.
Translation differences are BLEU; COGS differences are percentage points.
Three decimals retain the small positive upper endpoint for EN→FR versus
ordinary training; full precision remains in JSON.

| Comparison row | EN→DE | EN→FR | COGS generalization |
|---|---:|---:|---:|
| Ordinary training | +0.505 [−0.326, +1.397] | −0.690 [−1.447, +0.004] | +2.862 [+2.619, +3.114] |
| Two dropout passes, without analogy | +0.693 [−0.189, +1.568] | −0.634 [−1.363, +0.140] | +0.219 [+0.033, +0.405] |
| Fixed p = 0.5 | −0.080 [−0.791, +0.716] | −0.346 [−0.991, +0.299] | +0.633 [+0.476, +0.810] |

Both translation runs completed and scored step 20,000; COGS completed and
scored step 50,000. Final p was 0.71 for EN→DE, 0.74 for EN→FR, and 0.35 for
COGS. The power-selection probes used training examples only.

Completed reports: [English–German](multi30k.md), [English–French](multi30k_enfr.md),
and [COGS](cogs.md). Their same-named JSON files include exact scores, all five
paired comparisons, controller history, and training times.

Reference reports: [English–German](../adaptive_power_balanced_v1/multi30k.md),
[English–French](../adaptive_power_balanced_v1/multi30k_enfr.md), and
[COGS](../adaptive_power_balanced_v1/cogs.md). Their same-named JSON files retain
full-precision scores, intervals, and original provenance.

Method and execution: [experiment protocol](../../../experiments/compact_embeddings/adaptive_power_lookahead_v1/README.md).
