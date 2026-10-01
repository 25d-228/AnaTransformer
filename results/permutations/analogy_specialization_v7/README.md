# Analogy-preserving specialization: batch 7

All eighteen runs completed successfully. The last queue ended on September 30, 2026, at 04:40 JST. No runs remain queued or active.

Six designs cover Multi30k EN→DE, Multi30k EN→FR and COGS. IWSLT14 was
excluded. This table contains completed results, not the launch snapshot.

Sizes below are verified from instantiated models. Percentages are increases
over Shared-QKV, not over the full Transformer. COGS reports generalization only.
Existing scores use the recorded symmetric 95% bootstrap half-width.

| Model | Multi30k size | EN→DE BLEU | EN→FR BLEU | COGS size | COGS (%) |
|---|---:|---:|---:|---:|---:|
| Full Transformer (existing) | 2.606M (+17.7%) | 40.78 ± 1.74 | 60.80 ± 1.65 | 8.845M (+55.1%) | 78.70 ± 0.53 |
| Shared-QKV (existing) | 2.214M (+0.0%) | 38.65 ± 1.65 | 58.40 ± 1.70 | 5.702M (+0.0%) | 80.38 ± 0.50 |
| L: ordinary small branches (existing) | 2.312M (+4.4%) | 39.76 ± 1.73 | 59.15 ± 1.65 | 6.489M (+13.8%) | 78.11 ± 0.53 |
| D (existing) | 2.328M (+5.2%) | 39.92 ± 1.68 | 59.66 ± 1.66 | 6.522M (+14.4%) | 82.05 ± 0.50 |
| N1: analogy in small branches | 2.328M (+5.2%) | 39.26 ± 1.68 | 59.76 ± 1.74 | 6.522M (+14.4%) | 80.30 ± 0.53 |
| N2: encoder + cross small branches | 2.345M (+5.9%) | 39.59 ± 1.69 | 59.35 ± 1.65 | 6.556M (+15.0%) | 78.49 ± 0.55 |
| N3: D + richer rail readout | 2.330M (+5.2%) | 40.13 ± 1.66 | 59.45 ± 1.66 | 6.525M (+14.4%) | 77.69 ± 0.54 |
| N4: small branches + richer readout | 2.330M (+5.2%) | 40.22 ± 1.74 | 59.05 ± 1.71 | 6.525M (+14.4%) | 79.81 ± 0.51 |
| N5: D + decoder-query analogy | 2.336M (+5.5%) | 40.10 ± 1.67 | 59.23 ± 1.70 | 6.539M (+14.7%) | 78.70 ± 0.55 |
| N6: query-only analogy | 2.328M (+5.1%) | 40.02 ± 1.70 | 59.40 ± 1.68 | 6.523M (+14.4%) | 81.20 ± 0.49 |

## Completed queue assignments

Arrows mean sequential jobs. All assignments are within the approved batch.

| Server / GPU | Corpus | Ordered models | Microbatch | Effective batch | GPU allocator cap (MiB) |
|---|---|---|---:|---:|---:|
| exp14 / 0 | EN→DE | N1 → N4 | 128 | 256 | 9,216 |
| exp14 / 1 | EN→DE | N3 → N6 | 128 | 256 | 9,216 |
| exp14 / 2 | EN→DE | N2 → N5 | 128 | 256 | 9,216 |
| exp15 / 0 | EN→FR | N1 → N3 | 64 | 256 | 6,144 |
| exp16 / 0 | COGS | N1 → N4 | 64 | 128 | 9,216 |
| exp17 / 0 | COGS | N3 → N6 | 64 | 128 | 9,216 |
| exp18 / 0 | EN→FR | N2 → N4 → N5 → N6 | 256 | 256 | 12,288 |
| exp18 / 1 | COGS | N2 → N5 | 128 | 128 | 16,384 |

Each queue checks free RAM and VRAM before its next cell, saves resumable
checkpoints every 1,000 updates, and writes bootstrap reports after scoring.
Closing the client session does not stop the detached processes.

Live reports and saved predictions are under
`/mango/homes/YUE_Ziran/workspace/ana-analogy-specialization-v7/`.
Server logs are under the matching `/home/Yue_Ziran/workspace/` task folder.
These assignments are retained as execution history. All queues completed
with zero failures; exact scores and diagnostics are in the reports below.

[Designs and recipes](../../../experiments/permutations/analogy_specialization_v7/README.md) · [Exact reference records](../../../experiments/permutations/analogy_specialization_v7/reference_results.json)

## Completed reports

Retrieved from the saved automatic reports on October 1, 2026. The JSON
files retain full-precision intervals, training recipes, parameter counts,
diagnostics, and paths to the saved predictions and source run records.

- EN→DE: [table](multi30k.md) and [exact record](multi30k.json).
- EN→FR: [table](multi30k_enfr.md) and [exact record](multi30k_enfr.json).
- COGS generalization: [table](cogs.md) and [exact record](cogs.json).

## Reading this batch

N3, N5 and N6 have higher point scores than L on both translation tasks.
N3 exceeds L by 0.37 BLEU on EN→DE and 0.31 on EN→FR; N6 exceeds L by
0.26 in each direction and reaches 81.20% on COGS. No new candidate
improves both translation directions over D.

The richer readout improves German but lowers French in both matched
architectural comparisons: N3 versus D and N4 versus N1. This is a
direction-dependent result, not evidence that the added signal is always
helpful. The larger cross-attention placement in N2 also does not improve
either translation direction over D. Further IWSLT14 runs are not queued.
