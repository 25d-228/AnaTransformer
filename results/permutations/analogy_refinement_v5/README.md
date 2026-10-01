# Same-size refinements of model D: results

All nine runs completed successfully. None improved both translation directions over D, and all three COGS scores were lower than D.

The three models each cover Multi30k English to German, Multi30k English
to French and COGS generalization. No IWSLT14 runs are included.

On September 28 at 17:05:43 JST, the two exp18 runs restarted from saved
checkpoints with larger microbatches: FR D2 from step 3,000 (32 to 256),
and COGS D3 from step 5,000 (16 to 128). Effective training batches are
unchanged. Other runs were not interrupted.

M means million parameters. Parentheses give extra parameters compared with
shared-QKV. Translation scores are BLEU; COGS reports generalization exact
match as a percentage. The `±` value is half the width of a 95%
example-bootstrap interval with 1,000 resamples. Reference rows are existing
results, not new runs.

| Model | Multi30k size | EN→DE BLEU | EN→FR BLEU | COGS size | COGS (%) |
|---|---:|---:|---:|---:|---:|
| Full Transformer, existing | 2.606M (+17.7%) | 40.78 ± 1.74 | 60.80 ± 1.65 | 8.845M (+55.1%) | 78.70 ± 0.53 |
| Shared-QKV, existing | 2.214M (0%) | 38.65 ± 1.65 | 58.40 ± 1.70 | 5.702M (0%) | 80.38 ± 0.50 |
| L: D without analogy, existing | 2.312M (+4.4%) | 39.76 ± 1.73 | 59.15 ± 1.65 | 6.489M (+13.8%) | 78.11 ± 0.53 |
| D: compact cross-Q/K/V, existing | 2.328M (+5.2%) | 39.92 ± 1.68 | 59.66 ± 1.66 | 6.522M (+14.4%) | 82.05 ± 0.50 |
| D1: controller learning rate 0.3× | 2.328M (+5.2%) | 40.04 ± 1.64 | 59.48 ± 1.65 | 6.522M (+14.4%) | 77.60 ± 0.56 |
| D2: controller learning rate 1× | 2.328M (+5.2%) | 39.49 ± 1.65 | 59.64 ± 1.64 | 6.522M (+14.4%) | 77.72 ± 0.55 |
| D3: more capacity in cross-attention | 2.328M (+5.2%) | 39.36 ± 1.65 | 59.40 ± 1.63 | 6.522M (+14.4%) | 77.59 ± 0.54 |

All nine new parameter counts were verified from instantiated models:
2,328,464 for either Multi30k direction and 6,522,200 for COGS.

D1 and D2 keep D's architecture and change only the learning-rate multiplier
for its permutation and scaling controllers, from the existing 0.1× to
0.3× or 1×. D3 retains 0.1× and redistributes its small Q/K/V branches:
encoder rank `d_model / 16`, cross-attention rank `3 * d_model / 16`.
The ordinary parameters retain a learning-rate multiplier of 1× throughout.

All candidates keep input-dependent full Beneš grouping, positive ordered
quadruples, eight equivalent forms and common positive scaling. Ordinary
embeddings and dataset-specific recipes are unchanged. L is the existing
no-analogy comparison for D's branch allocation; it does not match D3's
redistributed branches.

## Records

- [Experiment design and recipes](../../../experiments/permutations/analogy_refinement_v5/README.md)
- [Exact reused scores and provenance](../../../experiments/permutations/analogy_refinement_v5/reference_results.json)
- [Completed parent batch](../analogy_combined_v4/README.md)

Predictions, run records and bootstrap reports are saved under
`/mango/homes/YUE_Ziran/workspace/ana-analogy-refinement-v5`.

## Completed reports

Retrieved from the saved automatic reports on October 1, 2026. The JSON
files retain full-precision intervals, training recipes, parameter counts,
diagnostics, and paths to the saved predictions and source run records.

- EN→DE: [table](multi30k.md) and [exact record](multi30k.json).
- EN→FR: [table](multi30k_enfr.md) and [exact record](multi30k_enfr.json).
- COGS generalization: [table](cogs.md) and [exact record](cogs.json).
