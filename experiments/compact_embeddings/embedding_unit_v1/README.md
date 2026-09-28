# Multi30k: unit-normalized power-analogy embeddings

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/embedding_unit_v1](../../../runs/embedding_unit_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Stopped at the user's request on **2026-09-15 at 01:11:30 JST** for redesign.
Both wrappers and training workers were verified gone by 01:12:01 JST.
Both models reached 9,000 logged updates; their resumable checkpoints remain
on NAS. Neither has a final test score. No automatic resume or follow-up is
queued. The historical launch details below are not current running status.

Launched detached on **2026-09-15 at 01:03:16 JST**. The powered model runs
on exp15 GPU0 (wrapper 1901093, worker 1901100), and the linear control runs
on exp16 GPU0 (wrapper 1805709, worker 1805716). Both were verified live at
01:04:02 JST, past 100 real updates with finite losses. Initial rates were
17.82 and 18.22 updates/second, respectively. Initial result ETA:
approximately **01:22–01:27 JST**, subject to progress.

The preceding unnormalized study finished first: power analogy scored
39.19 +/- 1.67 BLEU and linear compression scored 40.31 +/- 1.64.
The powered model missed the similar-size control's 40.12 +/- 1.71.
This separate study tests the measured scale issue. No preceding checkpoint
is resumed or modified; both new models start from fresh initialization.
No final result exists for either unit-normalized model yet.

The one focused correctness check passed on exp15 in 1.77 seconds. It checked
actual counts, unit row lengths, the padding exception, unchanged old defaults,
tied lookup/output behavior, power gradients through translation loss, and
incremental decoding. No extra training smoke run or full suite was run.

The motivation is a read-only diagnostic of the earlier update-7,000
checkpoints, recorded in the [earlier embedding study](../embedding_analogy_v1/README.md). Effective
embedding RMS was 1.5792 for the powered model versus 0.4329 for the linear
model, compared with an initial target of 0.0884. Mean token-vector norms
were 17.0178 versus 4.7478. The powered model's mean p was 3.9858, and 42.4%
of its normalized contrast codes had absolute tanh output above 0.98.
These measurements motivate controlling vector scale; they do not establish
that scale caused any performance difference. No old checkpoint is loaded
or altered by this fresh-training design.

This redesign compresses the word-piece embedding table instead of sharing
attention projections. Both new models retain independent, full Q, K, and V
at every attention site, with ordinary GELU feed-forward blocks. The source
embedding, target embedding, and output prediction weights remain tied.

## Two models

| Model | Width / FFN | Parameters | Role |
|---|---:|---:|---|
| Unit-normalized power-analogy embeddings, learned p | 128 / 230 | 2,248,528 | New candidate |
| Unit-normalized linear compressed embeddings | 128 / 232 | 2,248,512 | New structural control |
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
Both models apply the same parameter-free normalization after this map.

## The change: control each token vector's length

For every non-padding token, divide the effective signed embedding row by
its L2 norm, using a denominator floor of 0.000001. Above this floor, each
row has norm 1 and coordinate RMS 1/sqrt(128), matching the ordinary
embedding's intended initial scale. The padding row remains unnormalized
so that a near-zero padding vector is not amplified. Apply this identical
rule in both input lookup and output prediction; the table remains tied.

The usual sqrt(d_model) multiplier on input embeddings is unchanged. There
is no added learned scale or parameter. The existing final decoder LayerNorm
stays trainable, including its gamma, so the network can still adjust output
logit scale. This change controls row magnitude but does not prevent contrast
codes from saturating or guarantee that the analogy representation is useful.
It also removes non-padding token-specific norm as a learned degree of freedom.

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

Input lookup and output prediction use the same effective table. First form
the signed table G M, where G contains the generated latent features and M
is the learned basis. Normalize each non-padding row as described above to
obtain E, leaving the padding row unchanged. Lookup uses a row of E;
output scores use hidden E-transpose. Normalizing vector length is not
per-token or per-quartet mean subtraction, and adds no zero-sum constraint.
There is no D4, copied head, reconstructed FFN column, correction gain, teacher,
or pretrained checkpoint. Standard attention and the FFN computation are unchanged.

## Parameter budget

For vocabulary V and model width d, linear lexical compression stores
3Vd/4 + 3d²/4 parameters. The power-analogy version stores
3Vd/4 + d² + d/4: the last term is the learned powers.
At V = 10,000 and d = 128, these embedding modules contain 972,288 and
976,416 parameters, respectively, instead of 1,280,000 in the original table.
Row normalization has no trainable parameters, so these counts are unchanged.

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

SERVER: /home/Yue_Ziran/workspace/ana-embedding-unit-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-embedding-unit-v1

Run IDs are `embedding_unit_linear` and `embedding_unit_learned`. Registry names are
`embedding_unit_linear` and `ana_embedding_unit_learned`. Implementation lives in
`ana.nn.embeddings.analogy_embedding.LinearCompressedEmbedding` and
`ana.nn.embeddings.analogy_embedding.PowerAnalogyEmbedding`.

If launched, status, host, GPU and process identifiers will be recorded separately.
Workers must be detached, check available resources, and leave other users'
processes untouched. Existing environments remain read-only. Exp14 must not
receive new work while its GPU hardware fault remains unresolved.

The existing runner saves resumable checkpoints every 1,000 updates and
retains selected weights, predictions, and reports. No stopped earlier
experiment is resumed by this task.
