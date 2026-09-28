# Multi30k: power-analogy dropout consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/power_consistency_v1](../../../runs/power_consistency_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Completed on **2026-09-15 at 03:06:21 JST**: **40.90 ± 1.64 test BLEU**.
All 20,000 updates finished, and the lowest-development-loss checkpoint
was update 20,000. Best development loss was 2.7633225781442614;
development BLEU was 41.2331305401031. Recorded training time was
2,933.36 seconds. Saved predictions and the completed bootstrap report
are retained in [reports/multi30k.json](../../../results/compact_embeddings/power_consistency_v1/multi30k.json).

The point differences are **+0.78** against the similar-size Transformer,
**+2.24** against shared-QKV, and **+0.58** against the saved linear model.
The two primary point-score targets are exceeded. A fresh IWSLT14 follow-up
launched on exp18 GPU1 at 03:09:28 JST under ../power_consistency_iwslt_v1
and passed its first 100 updates. It uses the same fixed-p training method.
Earlier completed results remain unchanged, and no COGS job is queued.

The run launched at 02:17:13 JST on exp15 GPU0 (wrapper 1943596, worker
1943602), detached with wrapper parent PID 1. It reached 1,000 updates
in 145.45 seconds and 10,000 in 1,465.80 seconds, retaining resumable
checkpoints throughout. Both process identifiers were verified absent at
03:06:41 JST. This remained the original two-gradient-pass experiment;
the analyzed one-gradient-view alternative was not implemented or used.

This changes the training objective, not the inference architecture. It keeps
the completed linear compressed-embedding model, including independent full
Q, K and V projections. Power is fixed at p = 0.5 and used only in the loss:
there is no learned power and no power operation added to decoding.

## Candidate and saved references

| Model | Width / FFN | Parameters | Multi30k test BLEU |
|---|---:|---:|---:|
| Saved similar-size Transformer | 116 / 232 | 2,249,936 | 40.12 ± 1.71 |
| Saved shared-QKV | 128 / 256 | 2,213,888 | 38.65 ± 1.65 |
| Saved linear compressed embeddings | 128 / 232 | 2,248,512 | 40.31 ± 1.64 |
| Linear embeddings + power-analogy dropout consistency | 128 / 232 | 2,248,512 | 40.90 ± 1.64 |

The primary targets remain the similar-size Transformer and shared-QKV.
The saved linear model is an additional structural comparison. No reference
model is retrained. The candidate stores 96 coordinates per token and maps
them through a learned 96×128 matrix; input and output embeddings are tied.
Its 2,248,512 parameters are 13.70% fewer than the original full Transformer's
2,605,568. There are four heads and four encoder/four decoder layers, with
ordinary attention and GELU FFNs. No D4 or shared-QKV module is added.

## Four positive quantities and power

Run each training batch twice with independent dropout masks. At one target
position, the two passes give next-word probability distributions P and Q.
Before comparing them, use P_tilde = (1-epsilon)P + epsilon/V, and likewise
for Q, with epsilon = 0.000001. This tiny uniform mixture keeps every term
positive and is separate from the existing label smoothing.

For any two vocabulary entries i and j, set A = P_tilde_i, B = P_tilde_j,
C = Q_tilde_i and D = Q_tilde_j. Encourage A^p + D^p = B^p + C^p, with
p = 0.5. These are four prediction probabilities, not four identified words
that we claim form a semantic analogy.

Compute delta = sqrt(P_tilde) - sqrt(Q_tilde), then S = sum_v(delta_v -
mean(delta))². This is exactly the sum of squared four-term defects over
all ordered vocabulary pairs, divided by 2V. It takes O(V) work instead
of constructing a V×V matrix. Average S over valid, non-padding target
positions. Both dropout passes receive gradients.

The training loss is 0.5 × (CE_1 + CE_2) + 1.0 × mean(S), using the existing
CE loss for each pass. The power and penalty coefficient are fixed; neither
can learn to turn the penalty off. Translation CE still trains the
predictions against the target words.

The four-term condition follows [Lepage and Couceiro](https://arxiv.org/abs/2407.18770).
[R-Drop](https://arxiv.org/abs/2106.14448) provides the related two-dropout-view
consistency approach, but uses bidirectional KL instead of this centered
power-gap penalty. The completed pilot evaluates this different penalty
on the compact linear-embedding model.

## Training and reporting

Use the original 20,000 optimizer updates, batch 256 distinct examples,
learning rate 0.005, 2,000 warmup updates, inverse-square-root decay,
dropout 0.3 and label smoothing 0.1. Select by ordinary single-pass
development translation loss; decode with beam 5. Reuse existing data and
tokenizer, but start model weights fresh. There is no teacher, pretrained
checkpoint, continuation or parameter sweep.

Two gradient-bearing model passes roughly double training work per update.
This is not an equal-compute comparison, despite retaining the base optimizer
and update settings. Inference parameters and operations are unchanged from
the saved linear model.

The report reuses primary predictions from ana-compact-power-v1 and linear
predictions from ana-embedding-analogy-v1. It checks alignment with the same
test references and reuses cached bootstrap intervals. Symmetric ± is half
the width of a 95% example-bootstrap interval, using 1,000 resamples. Exact
endpoints and paired candidate-minus-reference intervals for all three
comparisons are saved in JSON. No p-values or significance marks are added.

Before launch, one focused CPU check passed in 1.21 seconds on exp15.
The configuration check confirmed 2,248,512 parameters, and the initial
report verified all three saved reference results and predictions. No broad
test suite or training smoke was run.

## Files

Run ID: `embedding_power_consistency`; registry: `embedding_linear`.

SERVER: /home/Yue_Ziran/workspace/ana-power-consistency-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-power-consistency-v1

The task-local trainer contains the changed loss. The runner records its
exact configuration in checkpoints and result manifests; the report checks
the candidate's fixed power, coefficient, probability smoothing, two-pass
training and unchanged inference scope. Launch status belongs in launch.json.
