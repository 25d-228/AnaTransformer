# D-clean: results

All three runs completed successfully. D-clean gained 0.14 BLEU on EN→DE relative to D, but lost 0.14 on EN→FR and 2.37 percentage points on COGS.

One design was evaluated on three tasks. Reference rows reuse completed
experiments. No IWSLT14 runs are included.

M means million parameters; percentages show extra parameters relative to
shared-QKV. Translation uses BLEU. COGS reports generalization exact match.
The ± value is half the width of the 95% example-bootstrap interval.

| Model | Multi30k size | EN→DE BLEU | EN→FR BLEU | COGS size | COGS (%) |
|---|---:|---:|---:|---:|---:|
| Full Transformer, existing | 2.606M (+17.7%) | 40.78 ± 1.74 | 60.80 ± 1.65 | 8.845M (+55.1%) | 78.70 ± 0.53 |
| Shared-QKV, existing | 2.214M (0%) | 38.65 ± 1.65 | 58.40 ± 1.70 | 5.702M (0%) | 80.38 ± 0.50 |
| L: D without analogy, existing | 2.312M (+4.4%) | 39.76 ± 1.73 | 59.15 ± 1.65 | 6.489M (+13.8%) | 78.11 ± 0.53 |
| D: original, existing | 2.328M (+5.2%) | 39.92 ± 1.68 | 59.66 ± 1.66 | 6.522M (+14.4%) | 82.05 ± 0.50 |
| D-clean: unchanged-input small branches | 2.328M (+5.2%) | 40.06 ± 1.67 | 59.52 ± 1.65 | 6.522M (+14.4%) | 79.68 ± 0.52 |

The only change from D is that its small encoder Q/K/V branches receive
the original input instead of the analogy-modified input. D's large shared
projection still receives the analogy-modified input. No parameters are added.

The change did not improve both translation directions over D. It remains
an informative placement comparison, not a replacement for the parent.

## Records

- [Design and execution](../../../experiments/permutations/analogy_clean_branch_v6/README.md)
- [Exact existing scores and provenance](../../../experiments/permutations/analogy_clean_branch_v6/reference_results.json)
- Remote predictions and automatic bootstrap reports:
  `/mango/homes/YUE_Ziran/workspace/ana-analogy-clean-branch-v6`.

## Completed reports

Retrieved from the saved automatic reports on October 1, 2026. The JSON
files retain full-precision intervals, training recipes, parameter counts,
diagnostics, and paths to the saved predictions and source run records.

- EN→DE: [table](multi30k.md) and [exact record](multi30k.json).
- EN→FR: [table](multi30k_enfr.md) and [exact record](multi30k_enfr.json).
- COGS generalization: [table](cogs.md) and [exact record](cogs.json).
