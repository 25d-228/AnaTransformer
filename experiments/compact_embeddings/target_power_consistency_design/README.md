# Next pilot: full-vocabulary plus target/rest power consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/target_power_consistency_design](../../../runs/target_power_consistency_design/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Proposed 2026-09-15 after the causal translation-example model completed at
39.93 +/- 1.67 Multi30k BLEU, below matched-size 40.12 and the earlier
two-gradient power-consistency result 40.90. Implemented under
runs/target_power_consistency_v1 in the subsequent goal continuation.
One focused check passed in 1.73 seconds; actual model count is 2,248,512.
Launched fresh on exp15 GPU0 at 08:17:08 JST; no stopped task was resumed.
Do not resume the stopped quarter-power or IWSLT14 experiments.

Keep the 2,248,512-parameter linear compressed embedding backbone, independent
full Q/K/V, ordinary FFNs, and both gradient-bearing dropout passes from
power_consistency_v1. Retain its full-vocabulary centered square-root penalty.
Do not include the unsuccessful causal vocabulary-scoring branch.

Add one target-informed quartet at each valid prediction position. For the
known training target y and the two raw softmax predictions P and Q, set
A=(1-epsilon)*P_y+epsilon/2, B=1-A,
C=(1-epsilon)*Q_y+epsilon/2, D=1-C, with epsilon=1e-6.
These are strictly positive correct-target/rest probabilities. Use fixed
p=0.5 and S_target=0.5*(sqrt(A)+sqrt(D)-sqrt(B)-sqrt(C))^2.
The total objective is mean(CE(P),CE(Q))+S_full+S_target, averaging the
penalties over non-padding target positions. Each defined penalty has
coefficient 1; no coefficient or exponent grid is proposed.

The four-term condition A^p+D^p=B^p+C^p comes from
[Lepage and Couceiro](https://arxiv.org/html/2407.18770v1).
Retaining S_full preserves agreement among individual alternative words.
The new quartet separately preserves agreement about the correct word's
probability versus their aggregate mass. Centering vocabulary root gaps can
hide a common change over many weak alternatives; the aggregate quartet can
detect it. That is an algebraic motivation, not an observed cause of earlier
BLEU differences. Extra consistency can also overregularize confidence.

Reuse the existing two softmax tensors and gather their target probabilities;
no new parameters, teacher, model pass or inference operations are needed.
Power remains training-only. The two-pass cost remains; no speed gain is
assumed. The architecture and inference are exactly the completed linear model.

One fresh 20,000-update Multi30k trial, same original data, batch, optimizer,
selection and decoding recipe. Reuse saved matched/shared/linear/full-consistency
references and symmetric bootstrap half-widths. No other dataset is queued.
One focused identity/gradient/padding check is sufficient before launch;
no training smoke, broad suite, package changes or hashes.
