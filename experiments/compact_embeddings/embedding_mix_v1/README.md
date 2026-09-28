# Multi30k: mixing power-completion residual features

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/embedding_mix_v1](../../../runs/embedding_mix_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Completed **2026-09-15 at 02:04:58 JST** on exp15 GPU0 after 20,000
updates. The scored checkpoint is step 20,000: **39.86 ± 1.65 BLEU**.
The run had already finished when the user's stop-and-redesign request was
checked at 02:05:34 JST. Wrapper 1925521 and worker 1925526 were absent,
the completion marker was present, and the GPU had no compute processes.
No process needed termination. Checkpoints and predictions are preserved;
the final reports have been copied locally. No replacement or IWSLT14 run
was launched, and no stopped experiment is queued for resumption.

The preceding residual model reached 40.20 ± 1.62 BLEU, compared with
40.12 ± 1.71 for the similar-size Transformer and 40.31 ± 1.64 for linear
compressed embeddings. This follow-up retains that model and adds one
learned matrix to combine its power-completion features before using them.

## Candidate and four saved references

| Model | Width / FFN | Parameters | Multi30k test BLEU |
|---|---:|---:|---:|
| Linear embeddings + mixed power-completion residual | 128 / 232 | 2,249,568 | 39.86 ± 1.65 |
| Saved linear embeddings + power-completion residual | 128 / 232 | 2,248,544 | 40.20 ± 1.62 |
| Saved linear compressed embeddings | 128 / 232 | 2,248,512 | 40.31 ± 1.64 |
| Saved similar-size Transformer | 116 / 232 | 2,249,936 | 40.12 ± 1.71 |
| Saved shared-QKV | 128 / 256 | 2,213,888 | 38.65 ± 1.65 |

The original similar-size Transformer and shared-QKV remain the primary
performance targets. The linear and unmixed residual models are additional
structural references, not replacement targets. All four use their saved
predictions and bootstrap results; none is retrained by this task.

## What changes

The previous effective embedding was E = X L + sigma h_p N. The new one is
E = X L + sigma h_p H N. H is a learned 32×32 matrix initialized to identity.
It combines the 32 power-residual features before mapping them through N.
This adds 1,024 trainable values, without changing any model width or FFN.

Each residual feature previously entered one fixed direction of N. H lets
each feature contribute to a learned combination of these directions. It
does not enlarge N's fixed span or create another set of stored token codes.
The matrix has 32 rows because there are 32 residual features; each feature
still comes from an actual four-term numerical-analogy completion.

## Unchanged numerical analogy

X is the learned 10,000×96 token-code table; L is an unrestricted learned
96×128 linear map. Let sigma = 1/sqrt(128). Split each row X/sigma into
32 triples (x,y,z), using epsilon = 0.0001:

- A = softplus(x) + epsilon.
- Delta = softplus(y) + epsilon; B = A + Delta.
- C = softplus(z) + epsilon.
- D_p = (B^p + C^p - A^p)^(1/p).

B>A and C>0 keep the radicand positive for every allowed p. The completed
positive terms obey A^p + D_p^p = B^p + C^p, the condition from
[Lepage and Couceiro](https://arxiv.org/abs/2407.18770).
These terms are learned numerical features, not four identified words or
attention cells. The positive identity does not describe the final signed
embedding coordinates.

The residual is h_p = D_p - D_1, where D_1 = C + Delta. N is the fixed
32×128 orthonormal complement of L's initial row space. It is constructed
without additional random draws and is not updated or recomputed. L stays
freely trainable, so permanent orthogonality is not guaranteed.

There are 32 learned powers shared across vocabulary rows, initialized at
exactly 1 and bounded between 0.75 and 2. At p=1, h_p is exactly zero, so
the initial model function is the complete ordinary linear embedding model.
The derivative with respect to p is generally nonzero there. H starts as
identity: its gradient is initially zero because h_p is zero, but it can
learn after the powers move and generate nonzero residual features. No
separate zero-initialized output gate blocks the first power update.

The completion retains Delta separately and uses stable log/expm1 arithmetic.
Its p=1 reference uses the same differentiable kernel, without detaching the
reference or adding a special-case zero branch. The new H uses identity
initialization and consumes no additional random draws.

The same effective E serves input lookup and tied output prediction during
training and decoding. There is no row normalization, tanh bottleneck,
whitening, scalar correction gate, or second learned lexical table. The
learned mixer can amplify, reduce, or suppress the residual; its usefulness
must come from the completed translation result. Retaining the linear path
does not guarantee that a fresh training run reproduces its old score.

## Budget and architecture

Trainable parameters are 960,000 codes, 12,288 entries in L, 32 powers,
1,024 entries in H, and 1,276,224 backbone parameters: 2,249,568 total.
This is 368 below the similar-size control and 356,000 below the original
2,605,568-parameter full Transformer. N contains 4,096 fixed buffer entries
(16 KiB in float32), not learned weights.

Width 128, FFN width 232, four heads, four encoder layers and four decoder
layers are unchanged from the saved linear and residual models. All 12
attention sites keep independent full Q/K/V, with ordinary GELU FFNs.
No D4, shared QKV, teacher or pretrained checkpoint is added. Compression
comes from lexical storage, not the thesis's earlier projection sharing.

## Training and reporting

One fresh Multi30k run uses the original normal recipe: 20,000 updates,
batch 256, learning rate 0.005, 2,000 warmup updates, inverse-square-root
decay, dropout 0.3, label smoothing 0.1, lowest-development-loss checkpoint
selection and beam 5. Existing data, tokenizer and trainer are reused.
No COGS, IWSLT14, checkpoint continuation or hyperparameter grid is queued.

The report reads primary controls from ana-compact-power-v1, the linear
model from ana-embedding-analogy-v1, and the unmixed residual from
ana-embedding-residual-v1. It checks their saved predictions against the
same references and reuses cached bootstrap intervals. Symmetric ± means
half the width of a 95% example-bootstrap interval with 1,000 resamples.
Exact endpoints and candidate-minus-reference paired intervals for all
four comparisons are saved in JSON. No p-values or significance marks
are added. Only one candidate needs training.

## Files and execution

SERVER: /home/Yue_Ziran/workspace/ana-embedding-mix-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-embedding-mix-v1

Run ID: `embedding_mix_learned`.
Registry name: `ana_embedding_mix_learned`.
Core: `ResidualPowerAnalogyEmbedding(..., learn_mixing=True)`.

Launch details are recorded in launch.json. The worker is detached,
uses the existing environment read-only, checks available resources, and
leaves other users' processes untouched. The reused runner retains resumable
checkpoints every 1,000 updates, selected weights, predictions and reports.
The single focused check passed on exp15 CPU in 1.62 seconds before launch.
It verified the actual count, unchanged initial function and randomness,
first-step power gradient, mixing gradients after power changes, and tied
cached decoding. The report verified all four saved references before launch.
No broad suite, extra training smoke or environment change was performed.
