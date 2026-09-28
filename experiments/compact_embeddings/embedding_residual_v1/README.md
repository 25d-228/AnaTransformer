# Multi30k: linear embeddings with a power-completion residual

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/embedding_residual_v1](../../../runs/embedding_residual_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Completed on **2026-09-15 at 01:41:39 JST**: **40.20 ± 1.62 test BLEU**,
after 20,000 updates; the lowest-development-loss checkpoint was update
19,000. Development BLEU was 40.9253 and best development loss 2.83723.
Training took 1,042.53 seconds. The run launched at 01:24:00 JST on exp15
GPU0 (wrapper 1913293, worker 1913298); both processes are now verified gone.
Weights, predictions and the completed bootstrap report are retained.
Earlier stopped models are not resumed.

The point difference is +0.08 against the similar-size control, +1.55 against
shared QKV, and -0.11 against the linear reference. This meets both primary
point-score targets, but the margin over the similar-size model is small.
The next bounded Multi30k trial, embedding_mix_v1, will test learned mixing
of the same extra features before considering the expensive IWSLT14 run.
This does not redefine the primary controls or discard this completed result.

This model keeps the completed linear embedding design intact and adds
nonlinear features from four-term power analogy. It does not force the
ordinary lexical coordinates through a positive quartet representation.
The change follows the [design proposal](../analogy_embedding_redesign/linear_residual_proposal.md).

## Candidate and saved references

| Model | Width / FFN | Parameters | Multi30k test BLEU |
|---|---:|---:|---:|
| Linear embeddings + power-completion residual | 128 / 232 | 2,248,544 | 40.20 ± 1.62 |
| Saved linear compressed embeddings | 128 / 232 | 2,248,512 | 40.31 ± 1.64 |
| Saved similar-size Transformer | 116 / 232 | 2,249,936 | 40.12 ± 1.71 |
| Saved shared-QKV | 128 / 256 | 2,213,888 | 38.65 ± 1.65 |

The two primary targets remain the similar-size Transformer and shared-QKV.
The completed linear model supplies an additional structural comparison;
its predictions and bootstrap interval are reused, not retrained or invented.
The earlier fully transformed power-embedding model scored 39.19 ± 1.67;
the linear result motivates retaining its complete representation path.

## Computation

Keep X, the 10,000×96 table of learned token codes, and L, its unrestricted
learned 96×128 linear map. Set sigma = 1/sqrt(128). For each token, split
X/sigma into 32 triples (x,y,z). With epsilon = 0.0001, each triple gives
positive quantities:

- A = softplus(x) + epsilon.
- Delta = softplus(y) + epsilon, with B = A + Delta.
- C = softplus(z) + epsilon.
- D_p = (B^p + C^p - A^p)^(1/p).

B is greater than A, so the completion radicand is positive for every
allowed p. The four terms satisfy A^p + D_p^p = B^p + C^p. This is the
positive numerical-analogy condition from
[Lepage and Couceiro](https://arxiv.org/abs/2407.18770). The terms are latent
numerical features, not four labeled words or selected attention cells.
Their number is four because the equation has four terms; this does not
introduce an arbitrary 8×8 mixing block.

Compute the additional feature h_p = D_p - D_1, where D_1 = C + Delta.
The effective embedding is E = X L + sigma h_p N. N is a fixed 32×128
orthonormal complement of L's initial row space. It is obtained without
additional random draws and is not recomputed as L trains. The same E is
used for input lookup and tied output prediction, including during decoding.

Only 32 learned powers are added, shared across vocabulary rows. Each starts
at exactly 1 and stays between 0.75 and 2. At p = 1, h_p is exactly zero:
the model contains the complete ordinary linear embedding model as a special
case. This is equality of model functions, not a promise that training a
different model will reproduce the old BLEU score.

The derivative with respect to p is generally nonzero at this starting point.
For example, A=1, B=2, C=3 gives D_1=4 and D_2=sqrt(12), so h_2=-0.5359.
The derivative at p=1 is -0.8630. A power change can therefore activate the
nonlinear path immediately; no separate zero-initialized gate blocks it.

The numerical implementation retains Delta rather than recovering B-A,
uses stable log/expm1 completion, and computes the p=1 reference through
the same differentiable kernel. The reference is not detached and there
is no special-case zero branch at p=1. The positive identity applies to
the completion terms, not the final signed embedding coordinates.

No row normalization, tanh bottleneck, whitening, learned correction gain,
or extra learned feature projector is added. L remains free to rotate, so
N need not stay orthogonal to it and full effective rank is not guaranteed.
Extra nonlinear features can still hurt optimization or prove irrelevant;
the linear route removes a mandatory representation restriction, not risk.

## Budget and architecture

A read-only measurement of the update-8,000 checkpoint found a mean ordinary
embedding-row norm of 4.8767 and mean extra-row norm of 0.01951. The extra
table's Frobenius norm was 0.004229 times the ordinary table's norm. Powers
ranged from 0.9300 to 1.0655 then. No live or saved weights were changed.
This establishes a small contribution, not that a larger contribution helps.

Trainable parameters are 960,000 token codes, 12,288 linear-map entries,
32 powers, and 1,276,224 backbone parameters: 2,248,544 in total. This is
1,392 fewer than the similar-size control and 13.70% below the original
2,605,568-parameter full Transformer. N adds 4,096 fixed buffer entries
(16 KiB in float32), not trainable weights.

The width is 128 and FFN width 232, exactly as in the completed linear model.
All 12 attention sites keep independent full Q, K, and V projections.
There are four heads, four encoder layers, four decoder layers, and ordinary
GELU FFNs. No D4, shared QKV, teacher, or pretrained checkpoint is used.
Compression comes from lexical storage; this is not the same architecture
as the thesis's shared-QKV/local-permutation model.

## Training and reporting

One fresh Multi30k run uses the original normal recipe: 20,000 updates,
batch 256, learning rate 0.005, 2,000 warmup updates, inverse-square-root
decay, dropout 0.3, label smoothing 0.1, lowest-development-loss checkpoint
selection, and beam 5. The existing tokenizer, data and trainer are reused.
No COGS, IWSLT14, continuation, or hyperparameter grid is queued.

The report reads the two primary controls from ana-compact-power-v1 and
the saved linear model from ana-embedding-analogy-v1. It checks saved
predictions against the same test references and reuses cached bootstrap
intervals. Symmetric ± is half the width of a 95% example-bootstrap interval
with 1,000 resamples. Exact endpoints and the candidate's paired differences
against all three references are retained in JSON. No p-values or significance
marks are added.

Only the single new candidate needs training. A fixed-p=1 version is the
linear model function, not another queued architecture. The single focused
check passed on exp15 CPU in 1.64 seconds before launch. It verified exact
initial model identity and code gradients, nonzero first translation-loss
power gradients, finite values, tied incremental decoding, saved state and
the actual 2,248,544 parameter count. The report verified all three saved
controls and their predictions before launch. No broad suite or extra
training smoke was run.

## Files and execution

SERVER: /home/Yue_Ziran/workspace/ana-embedding-residual-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-embedding-residual-v1

Run ID: `embedding_residual_learned`.
Registry name: `ana_embedding_residual_learned`.
Saved structural run ID: `embedding_linear` in ana-embedding-analogy-v1.

Launch status, host, GPU and process identifiers are recorded in launch.json.
The worker is detached, uses existing environments read-only, checks
available resources, and leaves other users' processes untouched. Resumable
checkpoints are saved every 1,000 updates; selected weights, predictions
and reports are retained. No stopped earlier experiment is resumed.
