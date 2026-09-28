# IWSLT14: power-analogy dropout consistency

Test · BLEU/13a

| Model | Parameters | Result |
|---|---:|---:|
| Full Transformer | 36.67M | 33.45 ± 0.51 |
| Similar-size Transformer | 27.25M | 32.94 ± 0.50 |
| Shared-QKV | 27.24M | 31.58 ± 0.49 |
| Linear embeddings, ordinary training | 27,241,504 | 32.82 ± 0.51 |
| Linear embeddings + power-analogy dropout consistency (p = 0.5) | 27,241,504 | 33.60 ± 0.50 |

The three controls are existing reported references from results/iwslt14.md, preserved at their reported table precision. Their original predictions are unavailable for this report. Completed compact rows are scored from saved test predictions; their ± is half the width of a 95% example-bootstrap interval (1,000 resamples), with exact endpoints retained in JSON.

Candidate minus rounded reported BLEU: Full Transformer +0.15; Similar-size Transformer +0.66; Shared-QKV +2.02. These are rounded-point comparisons, not paired tests.

Power-trained minus ordinary compact model: +0.79 BLEU; 95% paired-bootstrap interval [+0.60, +1.00].

Both compact models use width 448, FFN width 856, tied 336→448 linear embeddings and independent full Q/K/V. The original 50,000-update IWSLT14 optimizer schedule is retained. The ordinary row uses ordinary translation training. The power row averages two dropout-pass translation losses and adds the all-pairs power-consistency penalty with fixed p = 0.5 and coefficient 1. The power affects training only, with no additional inference operations. Two gradient-bearing passes increase training work; this is not an equal-compute comparison.
