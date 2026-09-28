# Context-dependent power in compact Transformers

Status: launched on 2026-09-16 at 22:21 JST in eight detached GPU queues.
This batch contains 15 fresh runs: five variants on
Multi30k English–German, Multi30k English–French, and COGS. Four completed
reference models are reused. See the [table to fill](../../../results/compact_embeddings/context_power_v1/README.md).

The new p heads learn through the ordinary prediction loss and stay active at
inference. They replace neither the compact embedding table nor the independent
Q/K/V projections. Unlike the earlier lookahead controller, they predict p from
the current decoder context and need no temporary optimizer trials.

## A: prediction confidence

A small linear head predicts one p per target position:
`p = 0.25 + 0.5 * sigmoid(head(context))`. A zero-initialized head starts at
p = 0.5. Multiply the ordinary output logits by `p / 0.5` before prediction.
This is a context-dependent confidence adjustment: it changes the probability
distribution but preserves vocabulary ranking at that position. The supervised
loss trains the head. A and A+ share exactly this predictor.

A has no analogy penalty. A+ adds the earlier centered all-vocabulary-pair
penalty, using the two views' uncalibrated softmax probabilities with the
uniform positive mixture of mass 1e-6. For vocabulary entries i and j, let
A = P1(i), B = P1(j), C = P2(i), and D = P2(j); encourage
`A^p + D^p = B^p + C^p`. The quartet uses one common p: the mean of the two
views' predicted powers, detached from the penalty's gradient.

The probability difference is scaled by `V^(p-0.5)/(2*p)` before centering.
One detached scalar per microbatch balances its raw squared penalty against
the p = 0.5 square-root reference from the same predictions. Tiny denominators
are guarded. This balances penalty values, not gradient norms.

## B: feature transformation

B predicts a separate power for each pair of decoder features: 64 pairs for
Multi30k and 200 for COGS. The head uses `p = 1 + 0.5 * tanh(head(context))`;
zero initialization gives p = 1. Define positive features
`u = softplus(h) + 1e-6`, then use `h' = h + u^p - u` with the existing compact
output projection. The stable implementation uses
`h + u * expm1((p-1) * log(u))`. At p = 1 this is the identity transformation.
The supervised loss learns the power head and can change relative word scores.

B has no analogy penalty. B+ compares each feature pair across the two dropout
views: A = u1(a), B = u1(b), C = u2(a), and D = u2(b). Its common p is the
detached mean of that pair's two predicted powers. Square the residual
`((u1(a)^p-u2(a)^p) - (u1(b)^p-u2(b)^p))/p` and average across feature pairs
and valid target positions. There is no batch balancing for this B penalty.

The fixed-B control uses p = 1, no power head, the identity prediction path,
and the same feature analogy penalty. It distinguishes the learned feature
transform from this simpler training penalty.

All five new variants average two gradient-bearing supervised dropout passes.
Analogy variants add their penalty with coefficient 1; non-analogy variants do
not. A and B have different penalty definitions, so the same coefficient does
not imply equal penalty strengths. The matched comparisons are A+ versus A,
B+ versus B, and B+ versus fixed B.

## Models and recipes

| New variant | Model identifier | Multi30k parameters | COGS parameters |
|---|---|---:|---:|
| A, prediction power | `context_power_prediction` | 2,248,641 | 5,689,637 |
| A+, prediction power + analogy | `context_power_prediction_analogy` | 2,248,641 | 5,689,637 |
| B, feature power | `context_power_features` | 2,256,768 | 5,769,436 |
| B+, feature power + analogy | `context_power_features_analogy` | 2,256,768 | 5,769,436 |
| B fixed p = 1 + analogy | `context_power_features_fixed` | 2,248,512 | 5,689,236 |

The prediction head adds 129 parameters for Multi30k and 401 for COGS. The
feature head adds 8,256 and 80,200, respectively. The fixed-B control adds none.

| Dataset | Updates | Effective batch | Microbatch | Checkpoint |
|---|---:|---:|---:|---|
| Multi30k English–German | 20,000 | 256 | 128 | Best ordinary development loss |
| Multi30k English–French | 20,000 | 256 | 128 | Best ordinary development loss |
| COGS | 50,000 | 128 | 64 | Final |

All 15 runs use this same two-piece gradient-accumulation setup for their
dataset. Effective batch sizes and main-update counts are unchanged. A's loss
balancing is applied separately to each microbatch; this is not claimed to
give exactly the gradient of one full-batch balancing calculation.

Translation retains width 128, FFN 232, rank-96 tied embeddings, four
encoder/four decoder layers, dropout 0.3, label smoothing 0.1, peak learning
rate 0.005, 2,000 warmup updates, inverse-square-root decay, and each direction's
existing train-only joint 10k tokenizer. COGS retains width 400, FFN 509,
rank 160, two encoder/two decoder layers, dropout 0.1, no label smoothing, and
constant learning rate 0.0001 without warmup. Existing Adam settings, clipping,
evaluation intervals, and beam-5 decoding remain unchanged. COGS reports
generalization only. No previous checkpoint is continued.

## Execution and results

Task name: `ana-context-power-v1`. Code, logs, reports, and temporary files use
`/home/Yue_Ziran/workspace/ana-context-power-v1`. Shared datasets, predictions,
and checkpoints use `/mango/homes/YUE_Ziran/workspace/ana-context-power-v1`.
`ANA_SERVER` and `ANA_NAS` can override these roots. Other users' processes
remain untouched; no runtime environment was changed.

| Server / GPU | First cell | Next cell | Allocator cap |
|---|---|---|---:|
| exp18 / 0 | EN→DE A+ | EN→DE A | 15 GiB |
| exp18 / 1 | EN→FR A+ | EN→FR A | 15 GiB |
| exp15 / 0 | COGS B+ | COGS B | 16 GiB |
| exp16 / 0 | EN→DE B+ | EN→DE B | 9 GiB |
| exp17 / 0 | EN→FR B+ | EN→FR B | 9 GiB |
| exp14 / 0 | COGS A+ | COGS A | 9 GiB |
| exp14 / 1 | COGS fixed B | EN→DE fixed B | 9 GiB |
| exp14 / 2 | EN→FR fixed B | — | 9 GiB |

Each queue starts with `nohup env ANA_MAX_GPU_MIB=CAP bash run.sh GPU
queues/HOST_gpuGPU.txt`, with standard input disconnected and output saved
to `logs/worker_gpuGPU.log`. Queue files are explicit and contain all 15 cells
exactly once. A queue waits for sufficient free GPU memory and RAM before each
cell, never stops other jobs, and continues to its next cell if a cell fails,
while recording the failure. Successful cells decode and report automatically;
each queue also refreshes all three reports when it finishes.

Checkpoints are saved every 1,000 updates. Closing the chat does not stop the
queues. A server power loss does stop them; preserved checkpoints allow a later
explicit restart. There is no reboot service or automatic IWSLT14 follow-up.
Process IDs and cells are recorded in the
[launch record](../../../runs/context_power_v1/launch.json).

New records use `runs/<corpus>_<model>_seed42`. Copy four existing models into
`reference_runs/<corpus>_<model>_seed42`: ordinary compact, two-pass compact,
fixed p = 0.5, and lookahead p. Their sources are the completed
`ana-adaptive-power-lookahead-v1` task's `reference_runs` for the first three
and `runs` for lookahead. Preserve their original records and predictions.

`report.py CORPUS` reports the nine rows from saved predictions; `report.py all`
covers all three datasets. Per-corpus locks serialize concurrent report writes.
Reports keep bootstrap ±, exact endpoints, source records, each new model minus
all four references, and the three matched within-family contrasts. Power
summaries at the final and scored checkpoints describe valid development
positions with dropout off, not a single global p. Missing results remain
pending; original completed results are unchanged.
