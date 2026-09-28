# Multi30k: causal translation-example analogy

Test · BLEU

The candidate keeps independent full Q/K/V, ordinary feed-forward layers and tied linear compressed embeddings. Its width is 128, FFN width 232, with four heads and four encoder/four decoder layers. A small vocabulary-scoring branch adds 1,033 parameters, for 2,249,545 total: below the similar-size Transformer.

The branch uses earlier translation steps in the same sentence. A and C describe earlier and current source contexts; B and D describe the earlier translated token and a candidate next token. Two learned maps produce four positive features for each role. With fixed p = 0.5, a candidate receives a higher score when A^p + D^p is close to B^p + C^p, the four-term condition from [Lepage and Couceiro](https://arxiv.org/abs/2407.18770).

All earlier valid steps contribute, weighted by source-context similarity. Their average powered relation gives the exact weighted squared-mismatch score without comparing every vocabulary item separately with every reference. No future target tokens are read. The first position has no earlier example and uses ordinary logits. Power affects training and inference; training uses one forward pass and the ordinary translation cross-entropy.

The references are reused completed runs. The power-consistency reference is a different method: two dropout passes and a training-only p = 0.5 penalty. All rows use the original 20,000-update Multi30k base recipe: batch 256, peak learning rate 0.005, 2,000 warmup updates and inverse-square-root decay. Lowest development loss selects the checkpoint; test decoding uses beam 5. Parameter savings use the full Transformer's 2,605,568 parameters.

| Model | Parameters | Saved vs full | Result |
|---|---:|---:|---:|
| Similar-size Transformer | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + linear compressed embeddings | 2,248,512 | 13.70% | 40.31 ± 1.64 |
| Linear embeddings + power consistency, p = 0.5 | 2,248,512 | 13.70% | 40.90 ± 1.64 |
| Linear embeddings + causal translation analogy, p = 0.5 | 2,249,545 | 13.66% | 39.93 ± 1.67 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

## Candidate minus saved references

| Reference | BLEU difference | 95% paired interval |
|---|---:|---:|
| Similar-size Transformer | -0.19 | [-1.12, +0.79] |
| Shared-QKV | +1.28 | [+0.34, +2.20] |
| Transformer + linear compressed embeddings | -0.38 | [-1.27, +0.62] |
| Linear embeddings + power consistency, p = 0.5 | -0.97 | [-1.80, -0.04] |
