# All three datasets — original models and Beneš + D4

Scores recorded September 9, 2026; comparison labels updated September 11.
This combines the eight original models
with the completed Beneš + D4 experiment. COGS reports generalization only;
IWSLT14 and Multi30k report test BLEU/13a. Higher scores are better.
“Saved” is the percentage of parameters removed relative to the ordinary
Transformer for that dataset.

| Model | COGS saved | COGS gen EM (%) ↑ | IWSLT14 saved | IWSLT14 BLEU ↑ | Multi30k saved | Multi30k BLEU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.0% | 76.41 ± 4.02 | 0.0% | 33.45 ± 0.51 | 0.0% | 40.78 ± 1.74 |
| `baseline_matched` | 35.7% | 79.46 ± 1.25 | 25.7% | 32.94 ± 0.50<sup>*</sup> | 13.6% | 40.00 ± 1.69<sup>*</sup> |
| `shared_qkv` | 35.5% | 80.76 ± 1.06<sup>*</sup> | 25.7% | 31.58 ± 0.49<sup>*</sup> | 15.0% | 38.64 ± 1.64<sup>*</sup> |
| `ana_seq_enc` | 35.2% | 79.07 ± 1.53 | 25.5% | 31.70 ± 0.50<sup>*</sup> | 14.5% | 38.76 ± 1.64<sup>*</sup> |
| `ana_feat_enc` | 35.2% | 79.08 ± 1.47 | 25.5% | 32.19 ± 0.49<sup>*†</sup> | 14.5% | 39.24 ± 1.66<sup>*</sup> |
| `ana_feat_1_enc` | 35.5% | 80.59 ± 2.16 | 25.7% | 31.71 ± 0.49<sup>*</sup> | 15.0% | 38.43 ± 1.66<sup>*</sup> |
| `ana_feat_2_enc` | 35.5% | 78.84 ± 2.96 | 25.7% | 31.68 ± 0.49<sup>*</sup> | 15.0% | 38.92 ± 1.62<sup>*</sup> |
| `ana_feat_all` | 34.6% | 80.01 ± 0.91 | 25.0% | 32.19 ± 0.48<sup>*†</sup> | 13.4% | 39.35 ± 1.71<sup>*†</sup> |
| Beneš + D4 (`ana_shuffle_benes_enc`) | 35.1% | 76.04 ± 0.55<sup>NT</sup> | 25.4% | 32.21 ± 0.50<sup>NT</sup> | 14.3% | 38.87 ± 1.60<sup>NT</sup> |

**Comparison marks:** `*` means p < 0.05 against `baseline`; `†` means
p < 0.05 against `shared_qkv`, using the tests already reported in the
original corpus tables, without correction for multiple comparisons.
Marks indicate a difference in either direction, not necessarily an improvement.
`NT` means the Beneš comparisons have **not been computed**, not that they
found no significant difference. Its score intervals alone cannot determine
these marks. See the [Beneš comparison status](benes_all_corpora_v1.md#comparison-status).

**± convention:** translation and Beneš values use half-widths of 95%
example-bootstrap intervals; the earlier COGS rows retain their reported
standard deviations. These are different uncertainty summaries, not a
common significance test. Original scores, uncertainty values and existing
comparison decisions are preserved; no new significance tests are claimed.

Sources: [original COGS](cogs.md), [original IWSLT14](iwslt14.md),
[original Multi30k](multi30k.md), and [Beneš + D4](benes_all_corpora_v1.md).
The [Beneš data](benes_all_corpora_v1.json) retains exact interval bounds
and parameter counts; its displayed half-widths and savings are rounded
from full precision.
