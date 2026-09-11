# Beneš + D4 — all three datasets

Model: `ana_shuffle_benes_enc`. Study: `benes_all_corpora_v1`.
All three datasets are complete; results recorded September 9, 2026.
This is the global Beneš permutation followed by the encoder D4 roles,
not a power/analogy add-on.

| Dataset and split | Metric | Score ± bootstrap half-width | 95% bootstrap interval | Examples |
|---|---|---:|---:|---:|
| Multi30k test | BLEU/13a | 38.87 ± 1.60 | [37.26, 40.46] | 1,000 |
| IWSLT14 test | BLEU/13a | 32.21 ± 0.50 | [31.71, 32.71] | 6,750 |
| COGS gen (primary) | Exact match (%) | 76.04 ± 0.55 | [75.50, 76.60] | 21,000 |
| COGS IID test | Exact match (%) | 99.73 ± 0.20 | [99.53, 99.93] | 3,000 |

The ± column follows the existing translation-table convention:
`half_width = (high - low) / 2`, computed from full-precision bounds before
rounding. It summarizes the interval's width. Because the score need not
be at the interval's midpoint, `score ± half_width` does not necessarily
reproduce its endpoints. The exact 95% interval is recorded separately.

All intervals use the project's percentile bootstrap with 1,000 resamples
and bootstrap seed 12345. They resample evaluation examples at one fixed
checkpoint; they do not measure training-seed variation. In particular,
the original COGS table's five-seed standard deviations are a different
quantity. No paired significance tests or improvement claims are made here.

## Comparison status

As of September 11, the comparisons against the original table's `baseline`
and `shared_qkv` are **not computed** for all three datasets. The combined
table labels these cells `NT`; an absent star or dagger must not be read as
a completed test finding no significant difference.

The saved Beneš predictions are available, but the corresponding original
reference predictions were not located in the local repository or the
checked exp14–18 task directories/shared storage. Later studies' baseline
scores differ from the original table and are not substitutes for these
missing references. Paired tests require aligned predictions and references;
scores and individual ± intervals are insufficient.

Once those artifacts are available, comparisons can be computed without
retraining. COGS also requires an explicitly compatible comparison with the
displayed aggregate reference scores; its historical statistical test cannot
be applied directly to Beneš's example-bootstrap summary.

## Training and checkpoint protocol

| Dataset | Training seed | Scored checkpoint | Parameters | Ordinary Transformer parameters | Parameter saving |
|---|---:|---|---:|---:|---:|
| Multi30k | 44 | Best dev loss, step 19,000 of 20,000 | 2,232,940 | 2,605,568 | 14.30% |
| IWSLT14 | 42 | Best dev loss, step 50,000 of 50,000 | 27,363,490 | 36,665,344 | 25.37% |
| COGS | 42 | Final step 50,000 | 5,744,182 | 8,844,800 | 35.06% |

Each corpus retains its original training recipe and beam-5 decoding.
Multi30k reuses the earlier seed-44 run, chosen by development BLEU among
seeds 42, 43 and 44 before test inspection; no new Multi30k training was
performed for this report. IWSLT14 and COGS each use one seed-42 run.
COGS's best-dev step 31,000 is diagnostic only: FINAL weights were scored.

## Saved evidence

The [machine-readable report](benes_all_corpora_v1.json) retains the scores,
full-precision interval bounds, symmetric half-widths, sample counts,
checkpoint rules and source paths.

Multi30k and COGS use their existing saved bootstrap reports. IWSLT14's
interval was computed once from its completed run's saved hypotheses and
references using `ana.stats.bootstrap_score` and `ana.metrics.Bleu`; the
point score matches its saved `results.json`. This was CPU-only on exp17
through shared NAS, without accessing the exp15 host. No retraining or
decoding was needed to record these results.
