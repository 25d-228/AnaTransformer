# Combined analogy-preserving projection designs

Final reports checked on **September 28, 2026**:
**all 54 runs completed**, including the 20-run I-M weekend extension.
J's two failed-start runs completed after relaunch on exp16 and exp17.

Sizes are total trainable parameters in millions (M). Percentages show
**how much larger each model is than shared-QKV**, not savings relative to
the full Transformer. Both Multi30k directions use the same model size.
All counts, including I-M, were measured from instantiated models.

| Model | Multi30k size (+% vs shared) | EN-to-DE BLEU | EN-to-FR BLEU | COGS size (+% vs shared) | COGS generalization EM (%) | IWSLT14 size (+% vs shared) | IWSLT14 DE-to-EN BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Transformer | 2.606M (+17.7%) | 40.78 ± 1.74 | 60.80 ± 1.65 | 8.845M (+55.1%) | 78.70 ± 0.53 | 36.665M (+34.6%) | 33.45 ± 0.51 |
| Shared-QKV | 2.214M (0%) | 38.65 ± 1.65 | 58.40 ± 1.70 | 5.702M (0%) | 80.38 ± 0.50 | 27.237M (0%) | 31.58 ± 0.49 |
| Earlier 2: separate cross-Q | 2.296M (+3.7%) | 39.20 ± 1.66 | 58.70 ± 1.65 | 6.261M (+9.8%) | 82.06 ± 0.50 | 28.914M (+6.2%) | 31.57 ± 0.46 |
| Earlier 4: encoder mixing | 2.279M (+3.0%) | 38.98 ± 1.69 | 59.51 ± 1.67 | 6.129M (+7.5%) | 78.72 ± 0.54 | 28.518M (+4.7%) | 32.99 ± 0.49 |
| A: combine 2 + 4 | 2.345M (+5.9%) | 39.69 ± 1.64 | 59.01 ± 1.63 | 6.654M (+16.7%) | 78.25 ± 0.53 | 30.094M (+10.5%) | 32.92 ± 0.49 |
| B: wider encoder mixing | 2.395M (+8.2%) | 39.80 ± 1.62 | 59.52 ± 1.62 | 7.048M (+23.6%) | 75.80 ± 0.55 | 31.273M (+14.8%) | 32.88 ± 0.49 |
| C: compact cross-Q | 2.296M (+3.7%) | 39.74 ± 1.69 | 59.10 ± 1.64 | 6.260M (+9.8%) | 76.40 ± 0.55 | 28.911M (+6.1%) | 32.81 ± 0.49 |
| D: compact cross-Q/K/V | 2.328M (+5.2%) | 39.92 ± 1.68 | 59.66 ± 1.66 | 6.522M (+14.4%) | 82.05 ± 0.50 | 29.698M (+9.0%) | 32.95 ± 0.51 |
| E: A + cross-K/V mixing | 2.378M (+7.4%) | 39.40 ± 1.59 | 59.80 ± 1.68 | 6.916M (+21.3%) | 78.80 ± 0.53 | 30.880M (+13.4%) | 32.96 ± 0.50 |
| F: A + decoder self-Q/K mixing | 2.378M (+7.4%) | 39.26 ± 1.56 | 59.55 ± 1.68 | 6.916M (+21.3%) | 81.02 ± 0.51 | 30.880M (+13.4%) | 33.11 ± 0.51 |
| G: redistributed branches | 2.279M (+3.0%) | 39.39 ± 1.67 | 59.51 ± 1.68 | 6.129M (+7.5%) | 76.35 ± 0.54 | 28.518M (+4.7%) | 32.64 ± 0.50 |
| H: token-controlled branches | 2.330M (+5.2%) | 39.71 ± 1.68 | 59.96 ± 1.59 | 6.525M (+14.4%) | 80.73 ± 0.52 | 29.707M (+9.1%) | 32.86 ± 0.51 |
| I: gated G | 2.281M (+3.0%) | 39.32 ± 1.68 | 58.93 ± 1.65 | 6.132M (+7.5%) | 74.90 ± 0.57 | 28.527M (+4.7%) | 32.75 ± 0.50 |
| J: shared bottleneck | 2.263M (+2.2%) | 38.56 ± 1.68 | 58.68 ± 1.72 | 5.998M (+5.2%) | 76.28 ± 0.56 | 28.125M (+3.3%) | 32.80 ± 0.48 |
| K: tiny diagonal shortcuts | 2.233M (+0.9%) | 38.84 ± 1.62 | 58.07 ± 1.66 | 5.742M (+0.7%) | 77.69 ± 0.54 | 27.357M (+0.4%) | 31.70 ± 0.49 |
| L: D without analogy | 2.312M (+4.4%) | 39.76 ± 1.73 | 59.15 ± 1.65 | 6.489M (+13.8%) | 78.11 ± 0.53 | 29.597M (+8.7%) | 32.64 ± 0.49 |
| M: G without analogy | 2.263M (+2.2%) | 39.26 ± 1.71 | 59.21 ± 1.70 | 6.095M (+6.9%) | 74.90 ± 0.55 | 28.417M (+4.3%) | 32.77 ± 0.51 |

The full-Transformer and shared-QKV scores, and earlier designs 2/4 on
Multi30k and COGS, are reused references. Earlier designs 2/4 on IWSLT14
and all A-M scores come from this batch. COGS reports generalization only.

New-run ± is half the width of a 95% example-bootstrap interval with 1,000
resamples. Exact scores, intervals, parameter counts, configurations and
prediction paths are stored in the canonical reports, copied here as
[Multi30k EN-to-DE](multi30k.json), [Multi30k EN-to-FR](multi30k_enfr.json),
[COGS](cogs.json) and [IWSLT14](iwslt14.json).

## Weekend extension

I adds token controls to G. J shares each small branch's input bottleneck
across Q/K/V. K replaces low-rank branches with zero-initialized, channel-wise
input-to-output shortcuts. I/J/K retain the protected input-dependent full
permutation and positive ordered-quartet operation with eight equivalent
forms and common positive scaling. No power is calculated or learned.
The analogy-preservation claim is local to that protected operation.

L and M remove D/G's analogy modules but retain their ordinary mixing
branches and exact initial retained weights. They are comparison models,
not analogy-preserving candidates. Decoder self-attention stays shared.

Five new IWSLT14 runs started at 14:06:05 JST on September 25:
J/K/L on exp14 GPUs 0/1/2, M on exp16 GPU 0, and I on exp18 GPU 1.
H's restarted COGS run completed normally, and exp17 started K Multi30k
at 14:07:11. Its queue covered K/L/M on both Multi30k directions and COGS.
A second short-task queue covered I/J on both Multi30k directions and
COGS on exp15 GPU 0.

Those six I/J short tasks initially failed exp15's RAM safety guard before
training. No checkpoints were created there; the queue was retired and
replaced on exp18 at 14:07:49. After the user freed exp15 RAM, these six
still-unstarted tasks moved back to exp15 at 14:44:40. The waiting exp18
queue was retired before that launch, avoiding duplicate runs. I's Multi30k
EN-to-DE run completed its first update with finite loss. Exp15 used a
4,096 MiB GPU cap, Multi30k/COGS microbatches of 32/16 and decoding batches
of 8; effective training batches and recipes remain unchanged. Other users'
processes and all already-running experiments were left untouched.

By September 28, 52 runs had finished, but J's French and COGS runs had
failed the startup resource guard before training. They restarted on exp16
and exp17 respectively, then completed successfully: **58.68 ± 1.72 BLEU**
and **76.28 ± 0.56% generalization exact match**. All 54 final reports are
now complete. The original corpus recipes, effective batches and scoring
rules were unchanged.

Canonical reports:
`/mango/homes/YUE_Ziran/workspace/ana-analogy-combined-v4/reports/`.

This file records the completed batch, not a live progress feed.

[Design and execution notes](../../../experiments/permutations/analogy_combined_v4/README.md).
