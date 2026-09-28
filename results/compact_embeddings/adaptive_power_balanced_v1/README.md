# Scale-balanced adaptive power: completed results

Status: all three runs completed on 2026-09-16, including decoding and bootstrap
reporting. They keep the same compact model and adaptive p rule, with a detached
per-batch scale correction for the analogy penalty.
The completed comparison rows below are reused unchanged.

| Compact-model training | Multi30k EN→DE BLEU | Multi30k EN→FR BLEU | COGS generalization EM (%) |
|---|---:|---:|---:|
| Ordinary training | 40.31 ± 1.64 | 60.72 ± 1.72 | 78.88 ± 0.55 |
| Two dropout passes, without analogy | 40.12 ± 1.67 | 60.66 ± 1.65 | 81.52 ± 0.50 |
| Analogy training, fixed p = 0.5 | 40.90 ± 1.64 | 60.37 ± 1.66 | 81.10 ± 0.53 |
| Previous adaptive-power training | 39.59 ± 1.66 | 56.83 ± 1.63 | 76.05 ± 0.55 |
| Scale-balanced adaptive-power training | 40.44 ± 1.63 | 59.85 ± 1.64 | 80.99 ± 0.54 |

± is half the width of the 95% example-bootstrap interval with 1,000 resamples.
Models within each dataset share the same
architecture and parameter count; inference does not use p.

The rescaled variant scores higher than the previous adaptive variant on all
three datasets. It is close to ordinary training on EN→DE, below it on EN→FR,
and above it on COGS. It does not exceed the fixed-p row on any dataset.

Completed new reports: [English–German](multi30k.md),
[English–French](multi30k_enfr.md), and [COGS](cogs.md). The corresponding JSON
files retain full precision and paired comparisons. At the end of training,
p was 0.64, 0.75, and 0.43, respectively. Translation completed 20,000 updates
per direction; COGS completed 50,000.

Completed reference sources: [English–German](../adaptive_power_v1/multi30k.md),
[English–French](../adaptive_power_v1/multi30k_enfr.md), and
[COGS](../adaptive_power_v1/cogs.md). Exact values, intervals, and provenance
remain in their accompanying JSON files.

Method and execution: [experiment protocol](../../../experiments/compact_embeddings/adaptive_power_balanced_v1/README.md).
