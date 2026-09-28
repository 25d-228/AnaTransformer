# Multi30k: frequency-aware lexical compression and power consistency

Test · BLEU

The candidate changes how the lexical table uses its parameter budget. It stores full 128-dimensional vectors for 3,376 vocabulary entries, and 80-dimensional codes plus a shared 80×128 basis for the other 6,624. Special tokens receive full vectors; the remaining full-vector entries are chosen by combined source/target training frequency, with token-ID tie breaking. Development and test frequencies are not used. Token IDs, tied lookup/output weights and the global vocabulary softmax are retained.

This costs 972,288 lexical parameters, exactly the same as the saved uniform 10,000×96 table and 96×128 basis. Both models have 2,248,512 total parameters: width 128, FFN width 232, four heads, four encoder/four decoder layers and independent full Q/K/V. The allocation follows the frequency-dependent capacity idea in [Baevski and Auli](https://arxiv.org/abs/1809.10853).

Training retains the original successful two-gradient power-consistency objective. Two independent dropout passes produce distributions P and Q. A small uniform mixture makes them positive: P_tilde = (1-epsilon)P + epsilon/V, and likewise for Q, with epsilon = 0.000001. For vocabulary entries i,j, the four terms are A = P_tilde_i, B = P_tilde_j, C = Q_tilde_i, D = Q_tilde_j. With fixed p = 0.5, the penalty encourages A^p + D^p = B^p + C^p from [Lepage and Couceiro](https://arxiv.org/abs/2407.18770).

Subtract the distributions' square roots, center that difference across vocabulary entries, and sum its squares. This exactly covers every vocabulary pair in O(V), averaged over valid target positions. Add the penalty with coefficient 1 to the average of the two ordinary translation losses; both passes receive gradients. There is no target/rest term. Power remains training-only and adds no inference operation. [R-Drop](https://arxiv.org/abs/2106.14448) provides related two-view consistency with a different, KL-based penalty.

The four references are reused completed runs. All rows use the original 20,000-update Multi30k base recipe: batch 256, peak learning rate 0.005, 2,000 warmup updates and inverse-square-root decay. Ordinary single-pass development loss selects the checkpoint; test decoding uses beam 5. Savings use the full Transformer's 2,605,568 parameters.

| Model | Parameters | Saved vs full | Result |
|---|---:|---:|---:|
| Similar-size Transformer | 2,249,936 | 13.65% | 40.12 ± 1.71 |
| Shared-QKV | 2,213,888 | 15.03% | 38.65 ± 1.65 |
| Transformer + linear compressed embeddings | 2,248,512 | 13.70% | 40.31 ± 1.64 |
| Linear embeddings + power consistency, p = 0.5 | 2,248,512 | 13.70% | 40.90 ± 1.64 |
| Frequency-aware embeddings + power consistency, p = 0.5 | 2,248,512 | 13.70% | 40.69 ± 1.67 |

± is half the width of a 95% example-bootstrap interval (1,000 resamples). Exact endpoints are retained in JSON.

## Candidate minus saved references

| Reference | BLEU difference | 95% paired interval |
|---|---:|---:|
| Similar-size Transformer | +0.58 | [-0.21, +1.45] |
| Shared-QKV | +2.04 | [+1.12, +2.93] |
| Transformer + linear compressed embeddings | +0.38 | [-0.45, +1.35] |
| Linear embeddings + power consistency, p = 0.5 | -0.20 | [-0.92, +0.62] |
