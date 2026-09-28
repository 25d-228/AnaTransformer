# Redesign: frequency-aware lexical compression with power consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/frequency_lexical_redesign](../../../runs/frequency_lexical_redesign/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Proposed 2026-09-15 after the user requested stopping the current experiment
and rethinking the design. Implementation subsequently started under the
standing implementation/iteration goal as ../frequency_lexical_v1 and completed
on exp15 at 09:38:28 JST. No further task is queued.
This design note is not itself permission to change packages or resume tasks.
The target/rest consistency run on exp15 was stopped at 08:35:04 JST;
its 7,000-update checkpoint is retained. Do not resume stopped tasks.

## Recommendation and evidence

Preserve the original successful two-gradient p=0.5 full-vocabulary consistency
objective. Redesign how the compact lexical table spends its parameters,
not the four-term equation or an attention-cell interaction.

Completed Multi30k results with authentic saved predictions:

| Model | Parameters | Test BLEU |
|---|---:|---:|
| Similar-size Transformer | 2,249,936 | 40.12 ± 1.71 |
| Shared-QKV | 2,213,888 | 38.65 ± 1.65 |
| Uniform linear compressed embeddings | 2,248,512 | 40.31 ± 1.64 |
| Uniform embeddings + original power consistency | 2,248,512 | 40.90 ± 1.64 |
| Frequency-aware embeddings + original power consistency | 2,248,512 | 40.69 ± 1.67 |

The completed weight/head/FFN reconstruction proposals did not surpass the
matched control. That does not establish a failure mechanism, and stopped
experiments must not be treated as completed negative results. The 40.90 model
remains the strongest completed candidate. The frequency redesign finished
20,000 updates and scored its 20,000-update checkpoint: +0.58 BLEU against
the matched control, +2.04 against shared-QKV and -0.20 against the original
power model. Retain 40.90 as preferred; this redesign is a completed alternative,
not an improvement over that model. See the [completed report](../../../results/compact_embeddings/frequency_lexical_v1/multi30k.md)
for actual predictions-based bootstrap and paired comparisons.

## One budget-neutral architecture change

Keep width 128, FFN 232, four heads, four encoder/four decoder layers, independent
full Q/K/V, ordinary GELU FFNs, and tied input/output representations.

Replace the uniform 10,000×96 table and 96×128 basis with:

- Full 128-dimensional vectors for 3,376 vocabulary entries.
- Eighty-dimensional codes for the other 6,624 entries.
- One shared 80×128 basis for the compressed entries.

The parameter count is exactly 3376×128 + 6624×80 + 80×128 = 972,288,
the same lexical budget as 10000×96 + 96×128. The instantiated model confirmed
2,248,512 total parameters, 13.70% below the full 2,605,568-parameter Transformer.

Reserve special tokens within the full-vector allocation. Choose the remaining
entries by combined source/target token counts in the training split only,
with token-ID tie breaking. Do not use dev/test frequencies. Keep original
token IDs through fixed mappings; do not change the tokenizer or output order.
Tie lookup and output weights, and retain the current global softmax rather
than introducing adaptive-softmax clusters or a new decoding rule.

The existing uniform factorization limits the effective lexical matrix to
rank 96. Full vectors for frequent entries permit overall rank 128 while placing
more compression on less frequent entries. This is a capacity-allocation
hypothesis, not evidence that rank caused earlier errors. Rare content words
may suffer, and frequent words may not need the additional capacity. Eighty
is a proposed storage width, and 3,376 follows from the fixed parameter budget;
neither number represents analogy roles. No width/cutoff grid is proposed.

This follows the frequency-dependent capacity idea in
[Baevski and Auli](https://arxiv.org/abs/1809.10853), not a claim that adaptive
embeddings are new or that their language-model results establish a gain here.

## Unchanged four-term power mechanism

Two independent dropout predictions give positive smoothed probabilities P,Q.
For vocabulary entries i,j, use A=P_i, B=P_j, C=Q_i, D=Q_j. The original loss
penalizes the defect A^p+D^p-B^p-C^p with fixed p=0.5. Centering and squaring
sqrt(P)-sqrt(Q) computes the all-pairs penalty without a vocabulary-pair matrix.
Average the two ordinary translation losses, then add the original penalty
with coefficient 1. Both views receive gradients. Preserve the existing tiny
uniform probability mixture and padding exclusion exactly.

This is the positive numerical condition of
[Lepage and Couceiro](https://arxiv.org/abs/2407.18770), used as a training
constraint. The two-dropout setup is related to
[R-Drop](https://arxiv.org/abs/2106.14448), with a different penalty.
Power remains training-only; there is no additional analogy computation at
inference and no claim of identifying semantic word analogies. Do not add the
stopped target/rest term, an exponent sweep, or another inference branch.

## Thesis scope and completed pilot

This preserves the broader compact-Transformer plus numerical-analogy goal,
but it does not preserve the thesis's current specific shared-QKV architecture.
The existing thesis explicitly frames its main objective as replacing three
dense projections with one shared projection. Embedding compression must be
introduced as a distinct extension, not silently described as a D4 variant.
The thesis, paper, and presentation files remain unchanged/read-only.

Implemented one small tied-embedding class and fixed vocabulary mapping,
reused the original power-consistency trainer, and completed one fresh
Multi30k pilot. It retained the original 20,000-update recipe, ordinary
single-pass best-dev-loss selection and beam 5 decoding. Training with periodic
development evaluation took 2,964.02 seconds; no training-speed gain is claimed.
One focused check passed, covering parameter count, lookup/output equivalence,
padding and cached decoding. No broad suite, training smoke, checksum campaign,
or package changes were used.

Compare the completed candidate with the saved matched/shared controls and
the 40.90 model, reporting actual BLEU and bootstrap ±. Retain 40.90 if this does
not improve on it. Do not automatically start a cutoff sweep, COGS, IWSLT14,
or any previously stopped task. Review the outcome before any next launch.
