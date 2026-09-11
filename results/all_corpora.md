# All three datasets — original models and Beneš + D4

Results recorded September 9, 2026. This combines the eight original models
with the completed Beneš + D4 experiment. COGS reports generalization only;
IWSLT14 and Multi30k report test BLEU/13a. Higher scores are better.
“Saved” is the percentage of parameters removed relative to the ordinary
Transformer for that dataset.

| Model | COGS saved | COGS gen EM (%) ↑ | IWSLT14 saved | IWSLT14 BLEU ↑ | Multi30k saved | Multi30k BLEU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.0% | 76.41 ± 4.02 | 0.0% | 33.45 ± 0.51 | 0.0% | 40.78 ± 1.74 |
| `baseline_matched` | 35.7% | 79.46 ± 1.25 | 25.7% | 32.94 ± 0.50 | 13.6% | 40.00 ± 1.69 |
| `shared_qkv` | 35.5% | 80.76 ± 1.06 | 25.7% | 31.58 ± 0.49 | 15.0% | 38.64 ± 1.64 |
| `ana_seq_enc` | 35.2% | 79.07 ± 1.53 | 25.5% | 31.70 ± 0.50 | 14.5% | 38.76 ± 1.64 |
| `ana_feat_enc` | 35.2% | 79.08 ± 1.47 | 25.5% | 32.19 ± 0.49 | 14.5% | 39.24 ± 1.66 |
| `ana_feat_1_enc` | 35.5% | 80.59 ± 2.16 | 25.7% | 31.71 ± 0.49 | 15.0% | 38.43 ± 1.66 |
| `ana_feat_2_enc` | 35.5% | 78.84 ± 2.96 | 25.7% | 31.68 ± 0.49 | 15.0% | 38.92 ± 1.62 |
| `ana_feat_all` | 34.6% | 80.01 ± 0.91 | 25.0% | 32.19 ± 0.48 | 13.4% | 39.35 ± 1.71 |
| Beneš + D4 (`ana_shuffle_benes_enc`) | 35.1% | 76.04 ± 0.55 | 25.4% | 32.21 ± 0.50 | 14.3% | 38.87 ± 1.60 |

**± convention:** translation and Beneš values use half-widths of 95%
example-bootstrap intervals; the earlier COGS rows retain their reported
standard deviations. These are different uncertainty summaries, not a
common significance test. Original scores and uncertainty values are
preserved; this table does not assign new significance marks.

Sources: [original COGS](cogs.md), [original IWSLT14](iwslt14.md),
[original Multi30k](multi30k.md), and [Beneš + D4](benes_all_corpora_v1.md).
The [Beneš data](benes_all_corpora_v1.json) retains exact interval bounds
and parameter counts; its displayed half-widths and savings are rounded
from full precision.
