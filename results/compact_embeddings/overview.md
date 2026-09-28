# Translation with compact embeddings and power analogy

Multi30k completed on 2026-09-15 with the original 20,000-update schedule.
The [current cross-dataset table](README.md) also includes the completed
English–French, IWSLT14, COGS, and adaptive-power comparisons.

| Model | Parameters | Multi30k test BLEU |
|---|---:|---:|
| Similar-size Transformer | 2,249,936 | 40.12 ± 1.71 |
| Shared-QKV | 2,213,888 | 38.65 ± 1.65 |
| Linear compressed embeddings | 2,248,512 | 40.31 ± 1.64 |
| Linear embeddings + power-analogy dropout consistency, two gradient passes | 2,248,512 | 40.90 ± 1.64 |
| Frequency-aware embeddings + power-analogy dropout consistency, two gradient passes | 2,248,512 | 40.69 ± 1.67 |
| Linear embeddings + power-analogy dropout consistency, one gradient pass | 2,248,512 | 40.00 ± 1.64 |
| Linear embeddings + causal translation-example analogy, p=0.5 | 2,249,545 | 39.93 ± 1.67 |

Symmetric ± is half the width of a 95% test-example bootstrap interval
(1,000 resamples). Both two-gradient compact designs exceed both historical size controls
in point score; the one-gradient version exceeds shared-QKV but not the
similar-size Transformer.

The compact-embedding models keep independent full Q/K/V projections.
The uniform versions store 96 coordinates per token and a shared 96→128
linear map, with tied input/output embeddings: 13.70% fewer parameters than
the full Transformer. The frequency-aware version uses the same parameter
budget but gives full 128-coordinate vectors to 3,376 entries and 80-coordinate
codes with an 80→128 map to the other 6,624. Special tokens receive full vectors;
remaining assignments use source/target training counts only. Neither design
is a shared-QKV or D4 variant.

Two independent dropout passes produce next-word distributions P and Q.
A tiny uniform mixture keeps probabilities positive. For any vocabulary
pair i,j, the four quantities are A=P_i, B=P_j, C=Q_i, D=Q_j. Fixed p=0.5
encourages A^p+D^p=B^p+C^p. Compute delta=sqrt(P)−sqrt(Q), center it across
vocabulary entries, and sum its squares. This covers every pair in O(V).
Training averages the two existing translation losses and adds this penalty
with coefficient 1, averaged over valid target positions.

Power is training-only; inference has no added power computation or
parameters. Two gradient-bearing passes increase training work.

Sources: [Lepage and Couceiro's four-term condition](https://arxiv.org/abs/2407.18770)
and [R-Drop's two-view consistency approach](https://arxiv.org/abs/2106.14448),
which uses a different, KL-based penalty.

Evidence: [completed report](power_consistency_v1/multi30k.md)
and [exact results](power_consistency_v1/multi30k.json).

The frequency-aware redesign preserves that original two-gradient p=0.5
objective unchanged. It completed and scored 20,000 updates, taking 2,964.02
training seconds including periodic development evaluation. Its 40.69 ± 1.67
is +0.58 BLEU against the matched Transformer, +2.04 against shared-QKV and
-0.20 against the original uniform power model. Retain **40.90 ± 1.64** as the
preferred candidate; the frequency design is a completed alternative, not an
upgrade. See the [frequency report](frequency_lexical_v1/multi30k.md)
and [exact results](frequency_lexical_v1/multi30k.json) for
bootstrap endpoints and genuine paired intervals. No further task is queued.

The one-gradient follow-up keeps two independent training-mode dropout
predictions but uses one as a no-gradient reference. Its objective is
CE(P)+2S(P,stop(Q)), with the same fixed p=0.5 and compact architecture.
It took 2,081.40 training seconds versus 2,933.36 for two gradient passes:
29.04% less measured time on the same host/GPU model. Final BLEU was lower,
so the original two-gradient version remains preferred. Timing includes
periodic development evaluation and can depend on machine load.
See the [follow-up report](power_consistency_singlegrad_v1/multi30k.md)
and [exact follow-up results](power_consistency_singlegrad_v1/multi30k.json).

The causal translation-example model adds a small inference-time score using
earlier source-context/target-token pairs and the current source context with
each candidate next token. It uses fixed p=0.5 in the four-term mismatch,
two positive feature maps and one learned gain, adding 1,033 parameters.
Ordinary one-pass translation training took 1,426.35 seconds; the selected
19,000-update checkpoint scored 39.93 ± 1.67 after 20,000 training updates.
It remains below the matched-size control and the 40.90 two-gradient model,
so it is not promoted. See its [completed report](causal_translation_v1/multi30k.md)
and [exact results](causal_translation_v1/multi30k.json).

IWSLT14 (27,241,504 parameters) resumed from its saved 16,000-update checkpoint
and completed the original 50,000-update recipe on 2026-09-16. Fixed-power
training scored **33.60 ± 0.50** BLEU; ordinary compact training scored
**32.82 ± 0.51**. The paired difference was +0.79 BLEU, with a 95% interval
of [+0.60, +1.00]. Both use the same compact model, 25.70% smaller than the
full Transformer. See the [completed comparison](power_consistency_iwslt_v1/iwslt14.md)
and [exact results](power_consistency_iwslt_v1/iwslt14.json).
