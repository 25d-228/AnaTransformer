# Multi30k: power-analogy lexical compression

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/embedding_analogy_v1](../../../runs/embedding_analogy_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


**Completed on 2026-09-15 at 01:01:42–01:01:44 JST.** Both models trained
for 20,000 updates and scored their update-20,000 best-dev-loss checkpoints.
The powered model scored **39.19 +/- 1.67 BLEU**; linear compression scored
**40.31 +/- 1.64 BLEU**. The original similar-size and shared-QKV controls
remain **40.12 +/- 1.71** and **38.65 +/- 1.65**. Exact intervals and paired
comparisons are in reports/multi30k.json and reports/multi30k.md.

Both workers and wrappers were verified absent at 01:02:36 JST. Weights,
predictions, checkpoints and completed reports are retained. The powered
candidate did not beat the similar-size control. A separate fresh unit-row
normalization study, embedding_unit_v1, was launched at 01:03:16 JST;
this completed study was neither modified during training nor resumed.

At **00:46:14 JST**, both live workers had logged update **2,000/20,000**,
and both resumable checkpoint files existed on shared storage. Development
loss was 3.4612 for the powered model and 3.2456 for the linear control;
these are intermediate losses, not final BLEU results.

A read-only CPU diagnostic of the update-7,000 checkpoints found effective
embedding RMS 1.5792 for the powered model versus 0.4329 for the linear model
(initial target: 0.0884). Mean token-vector norms were 17.0178 versus 4.7478.
The powered model's mean p was 3.9858, and 42.4% of its normalized contrast
codes had absolute tanh output above 0.98. With its other checkpoint weights
held fixed, substituting p=2 in memory raised embedding RMS to 2.1136. No
checkpoint, live model, training recipe or evaluation setting was modified.
These observations suggest feature scale/saturation as a possible next issue
to address; they do not establish a cause or a final translation result.

The one focused correctness check passed in 2.36 seconds. It verified actual
parameter counts, the positive equation, initial feature scale, first-update
power gradients, tied input/output use, saved state and incremental decoding.
No extra training smoke run or full suite was run. The report has loaded and
checked both saved primary controls and the final predictions of both new models.

This redesign compresses the word-piece embedding table instead of sharing
attention projections. Both new models retain independent, full Q, K, and V
at every attention site, with ordinary GELU feed-forward blocks. The source
embedding, target embedding, and output prediction weights remain tied.

## Two models

| Model | Width / FFN | Parameters | Role |
|---|---:|---:|---|
| Power-analogy embeddings, learned p | 128 / 230 | 2,248,528 | New candidate |
| Linear compressed embeddings | 128 / 232 | 2,248,512 | New structural control |
| Similar-size Transformer | 116 / 232 | 2,249,936 | Existing primary control |
| Shared-QKV | 128 / 256 | 2,213,888 | Existing primary control |
| Original full Transformer | 128 / 256 | 2,605,568 | Full-size parameter reference |

The new linear control does not replace either existing performance target.
The saved normal-recipe scores to beat remain 40.11867272699844 BLEU for the
similar-size Transformer and 38.651869943809345 for shared-QKV.

The linear control stores 96 values per token and uses one learned 96×128
matrix to obtain the 128-dimensional embedding. The candidate also stores
96 values per token, but treats them as 32 groups of three values. Each group
generates four positive numerical-analogy terms, giving 128 latent features.
A learned shared 128×128 basis then maps these to the model's coordinates.

## Four-term construction

Three stored values produce a positive radius R through softplus plus a
fixed 0.0001 floor, and two
bounded values, u = 0.95 tanh(alpha) and v = 0.95 tanh(beta). Set
A = R(1+u)^(1/p), D = R(1-u)^(1/p), B = R(1+v)^(1/p), and
C = R(1-v)^(1/p). These positive terms satisfy
A^p + D^p = B^p + C^p = 2R^p without a radicand clamp.

The four-term power condition comes from the project's
[numerical-analogy paper](https://arxiv.org/abs/2407.18770).
The model stores three coordinates that describe a valid four-term numerical
analogy; it does not independently store literal A, B, C and then another D.
These are learned latent features, not labeled word analogies or attention cells.

There are 32 powers, one per group, shared across the vocabulary. They start
at 2 and remain between 0.5 and 4. Changing p changes the relative values
within a quartet, not just a common multiplier. At p = 1 the quartet has an
exact linear dependence; initial p = 2 avoids that starting degeneracy.

Fixed common centering and scaling convert the positive features to signed
features. A shared learned basis, initialized with covariance whitening,
handles their initial unequal variances. Centering is not computed separately
for each token or quartet. The power identity holds before this signed
conversion and basis map; the final rotated embedding coordinates are not
claimed to satisfy the same identity.

Both models store their codes at the ordinary embedding scale,
sigma = 1/sqrt(d). The candidate divides its codes by sigma before the
positive construction and includes sigma in the initial basis. This keeps
the raw code scale comparable under the unchanged learning rate. Its fixed
centering, scale and blockwise covariance whitening come from 65,536 private
synthetic samples at p = 2, not from corpus or test examples. Whitening gain
is capped at 8. The basis remains freely learned after initialization.
The linear model starts with Gaussian codes at scale sigma and an orthogonal
row basis scaled by sqrt(128/96), giving the same initial output variance.

Input lookup and output prediction use the same effective table. If G is the
generated signed table and M is the basis, the effective table is G M.
Lookup uses its token row; output scores use hidden M-transpose G-transpose.
There is no D4, copied head, reconstructed FFN column, correction gain, teacher,
or pretrained checkpoint. Standard attention and the FFN computation are unchanged.

## Parameter budget

For vocabulary V and model width d, linear lexical compression stores
3Vd/4 + 3d²/4 parameters. The power-analogy version stores
3Vd/4 + d² + d/4: the last term is the learned powers.
At V = 10,000 and d = 128, these embedding modules contain 972,288 and
976,416 parameters, respectively, instead of 1,280,000 in the original table.

Reducing the candidate FFN from 232 to 230 offsets 4,112 of its 4,128 extra
lexical parameters. The complete new models differ by only 16 parameters;
both remain below the similar-size Transformer. No improvement is assumed:
compression reduces independently stored lexical information, and the
analogy manifold may constrain useful token representations.

## Training and reporting

Multi30k only: four heads, four encoder/four decoder layers, and 12 independent
Q/K/V attention sites. Train from scratch for 20,000 updates with batch 256,
LR 0.005, 2,000 warmup updates, inverse-square-root decay, dropout 0.3,
label smoothing 0.1, lowest-development-loss checkpoint selection, and beam 5.
The existing tokenizer, data, trainer, and normal recipe are reused.

The report reuses the saved normal-rate controls from ana-compact-power-v1,
checking their predictions against the same test references. Results use
symmetric ±: half the width of a 95% example-bootstrap interval with 1,000
resamples. Exact endpoints and paired difference intervals are retained in
JSON. The report includes each new model against both primary controls and
the power-analogy model against the linear structural control. Existing
bootstrap results are cached. No p-values or significance marks are added.

COGS and IWSLT14 are not queued by this task. Testing is limited to one focused
numerical/gradient/tied-decoding check and an actual-model parameter count;
no full test suite, extra training smoke, package update, or hash pass.

## Files and execution

SERVER: /home/Yue_Ziran/workspace/ana-embedding-analogy-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-embedding-analogy-v1

Run IDs are `embedding_linear` and `embedding_learned`. Registry names are
`embedding_linear` and `ana_embedding_learned`. Implementation lives in
`ana.nn.embeddings.analogy_embedding.LinearCompressedEmbedding` and
`ana.nn.embeddings.analogy_embedding.PowerAnalogyEmbedding`.

Launch status, host, GPU and process identifiers are recorded in launch.json.
Workers must be detached, check available resources, and leave other users'
processes untouched. Existing environments remain read-only. Exp14 must not
receive new work while its GPU hardware fault remains unresolved.

The existing runner saves resumable checkpoints every 1,000 updates and
retains selected weights, predictions, and reports. No stopped earlier
experiment is resumed by this task.
