# Combined analogy-preserving projection designs

**Completed September 28, 2026: all 54 runs have final results.** This
includes J's Multi30k EN-to-FR and COGS runs, which completed after their
startup failures were resolved by relaunching on exp16 and exp17. The
execution sections below retain the launch and recovery history.

The original 26 runs launched on 2026-09-22 at 17:56 JST in eight detached
GPU queues on exp14-18. On September 24, two follow-up designs, G and H,
were approved for all four tasks, bringing the batch to **34 runs**.
The eight additional runs launched in four detached queues on
2026-09-24 at 12:29:55 JST. Minimal GPU forward/backward and cached-decoding
checks passed, and the IWSLT14 memory check fit the configured limit.

The weekend extension adds five designs, I-M, on all four tasks:
**20 additional runs, bringing the batch total to 54**. Its seven detached
queue wrappers first launched on **2026-09-25 at 14:06:05 JST**. After exp15
failed its available-RAM check before training, its six cells were requeued
on exp18 GPU 0 at **14:07:49 JST**. After the user freed exp15 RAM, these
six still-unstarted cells moved back to exp15 at **14:44:40 JST**, with a
4,096 MiB GPU cap and smaller microbatches. The exp18 replacement queue was
retired before the exp15 launch. H's COGS run finished normally; the earlier
A-H settings remain unchanged.

The first 34 runs cover eight designs, A-H, on Multi30k EN-to-DE, Multi30k
EN-to-FR, COGS and IWSLT14 DE-to-EN, plus the existing `pre_crossq` and
`pre_lowrank` designs on IWSLT14 only. Their completed Multi30k and COGS
results are reused. At the September 24 progress check, 22 original runs
were complete; IWSLT14 C-F were still running.

Translation performance against the full Transformer and shared-QKV is the
main decision criterion. COGS generalization is supplementary. Ordinary
embeddings stay in use; the compact-embedding branch is not part of this batch.
IWSLT14 is included specifically for this batch, including the approved
G/H and I-M extensions. For the weekend extension, the decision also considers
extra parameters compared with shared-QKV, rather than only savings compared
with the full Transformer.

## Designs

The analogy-preserving candidates extend the earlier `analogy_projection_v3`
encoder before-projection design. The implementation is self-contained here;
its protected operation is described below.

| ID | Identifier | Change |
|---|---|---|
| A | `combo` | Encoder role-specific low-rank mixing plus separate decoder cross-attention Q; cross-attention K/V stay shared |
| B | `combo_wide` | A with twice the encoder mixing width |
| C | `compact_q` | Encoder role-specific mixing plus a small cross-attention Q mixing branch instead of a separate full Q projection |
| D | `compact_qkv` | C with small cross-attention K and V mixing branches as well |
| E | `combo_crosskv` | A with small cross-attention K and V mixing branches |
| F | `combo_selfqk` | A with small decoder self-attention Q and K mixing branches |
| G | `balanced_qkv` | Split existing design 4's mixing budget equally between encoder and cross-attention Q/K/V branches |
| H | `gated_qkv` | D with a token-dependent scalar gate on each cross-attention Q/K/V mixing branch |
| Existing 2 | `pre_crossq` | Encoder analogy transformation plus separate decoder cross-attention Q; no encoder low-rank mixing |
| Existing 4 | `pre_lowrank` | Encoder analogy transformation plus role-specific low-rank mixing; decoder projections stay shared |

Default low-rank width is `d_model / 8`; B uses `d_model / 4` for its encoder
mixing branches. G uses `d_model / 16` in both the encoder and cross-attention,
keeping exactly the parameter budget of existing design 4 for these recipes.
H keeps D's branch widths and adds three token gates per decoder layer.
These widths control parameter cost. Analogy groups always
have **four numbers** and use exactly **eight equivalent forms**.
Small mixing branches and separate dense projections are ordinary learned
readouts; no new preservation claim applies to them.

G tests whether distributing a fixed parameter budget between encoder and
cross-attention works better than encoder-only mixing. H tests whether
token-dependent adjustment helps D. Each H gate is `2 * sigmoid(linear(x))`,
one scalar per role and token, initially one. Q gates use the query input;
K/V gates use the source input. Non-gate initial weights match D, and the
low-rank output matrices still start at zero. The protected analogy block
is unchanged in both follow-ups.

Measured parameter counts from instantiated models, including G/H:

| Model | Either Multi30k direction | COGS | IWSLT14 |
|---|---:|---:|---:|
| Full Transformer | 2,605,568 | 8,844,800 | 36,665,344 |
| A | 2,345,360 | 6,654,296 | 30,093,832 |
| B | 2,394,512 | 7,047,512 | 31,273,480 |
| C | 2,295,696 | 6,260,056 | 28,911,112 |
| D | 2,328,464 | 6,522,200 | 29,697,544 |
| E | 2,378,128 | 6,916,440 | 30,880,264 |
| F | 2,378,128 | 6,916,440 | 30,880,264 |
| G | 2,279,312 | 6,128,984 | 28,517,896 |
| H | 2,330,012 | 6,525,278 | 29,706,778 |

Every candidate remains smaller than its dataset's full Transformer.

### Weekend extension: I-M

I-K are new candidates. L and M are matched comparisons that deliberately
remove the analogy block; they are not analogy-preserving candidates.

| ID | Identifier | Change |
|---|---|---|
| I | `balanced_gated` | G with H's token-dependent scalar controls on cross-attention Q/K/V mixing branches |
| J | `shared_bottleneck` | G with one shared input factor per attention site for its Q/K/V low-rank branches, while keeping separate output factors |
| K | `diagonal_shortcuts` | Keep the protected encoder analogy block, but replace encoder and cross-attention low-rank branches with zero-initialized, role-specific diagonal input-to-output shortcuts |
| L | `compact_qkv_no_analogy` | D with the analogy block removed; keep the same ordinary projection and mixing branches |
| M | `balanced_qkv_no_analogy` | G with the analogy block removed; keep the same ordinary projection and mixing branches |

I keeps G's `d_model / 16` branch widths. Its gates follow H: initially one,
query-dependent for Q and source-dependent for K/V. J also uses
`d_model / 16`. Its shared input factor is shared within each encoder or
cross-attention site, not across different layers or attention sites.
Separate Q/K/V output factors retain role-specific adjustments.

K gives each role a learned channel-by-channel scaled copy of that role's
input as an output shortcut. This is an input-to-output path, not another
scale applied to the existing shared projection output. The shortcut weights
start at zero, and no additional dense projection is introduced.

I-K retain the existing input-dependent full permutation, positive ordered
quartets, hard choice among eight equivalent forms and common positive
scaling. These new gates, shared factors and shortcuts remain ordinary
network operations outside the protected analogy-preservation claim.

L and M initialize their respective parent model first and then remove the
analogy parameters. Their retained ordinary weights therefore match D and G
at initialization. These comparisons isolate the addition of the protected
module while retaining each parent's ordinary mixing branches.

The parameter counts below were verified from instantiated models and match
the architecture calculations. Parenthesized percentages give
**extra parameters compared with shared-QKV**, not savings compared with the
full Transformer.

| Model | Either Multi30k direction | COGS | IWSLT14 |
|---|---:|---:|---:|
| Shared-QKV | 2,213,888 (0%) | 5,702,144 (0%) | 27,237,376 (0%) |
| I | 2,280,860 (+3.0%) | 6,132,062 (+7.5%) | 28,527,130 (+4.7%) |
| J | 2,262,928 (+2.2%) | 5,997,912 (+5.2%) | 28,124,680 (+3.3%) |
| K | 2,233,232 (+0.9%) | 5,741,912 (+0.7%) | 27,356,680 (+0.4%) |
| L | 2,312,192 (+4.4%) | 6,488,576 (+13.8%) | 29,596,672 (+8.7%) |
| M | 2,263,040 (+2.2%) | 6,095,360 (+6.9%) | 28,417,024 (+4.3%) |

Each model completed Multi30k EN-to-DE, Multi30k EN-to-FR, COGS
generalization and IWSLT14 DE-to-EN, with the unchanged recipes and bootstrap
reporting below. Results are reported separately from parameter counts.

## Protected positive-quartet operation

For the analogy-preserving candidates, the encoder block is unchanged from
v3. Controls L and M omit it. A token-dependent hard full Beneš
permutation groups channels. Q/K/V reuse that grouping calculation but choose
their own local equivalent forms and common positive multipliers.

Matched positive rails `softplus(x) + epsilon` and
`softplus(-x) + epsilon` provide strictly positive features. Each quartet is
sorted and accepted only when both rails satisfy `0 < a < b <= c < d`.
Invalid quartets bypass the local change. Each accepted quartet receives one
hard choice among the eight equivalent forms and a common positive lambda.
The negative rail uses reversal-conjugate routing to keep channel choices
consistent across rails.

The operation preserves the quartet's existing, unspecified analogy power.
No power is calculated or learned. Global regrouping and sorting establish
the positive ordered groups; they are not claimed to preserve a previous
grouping. The preservation claim concerns only the eight equivalent forms
and common positive scaling inside an accepted quartet.

After undoing the local order, taking the signed rail difference and restoring
global channel positions, a learned fraction of the change modifies each
role's input to the shared projection. Signed recombination, residual mixing,
low-rank branches and dense projections are outside the preservation claim.
The protected analogy operation remains in the encoder; the extra decoder
branches in C through K do not introduce another analogy block. H's and I's gates
adjust only these ordinary cross-attention branches, not the protected
positive-quartet operation.

## Training and reporting

- Keep each dataset's existing architecture and recipe. Ordinary tied
  embeddings, single-pass cross-entropy and existing data/tokenizers remain.
- Multi30k: 20,000 updates, effective batch 256, best development-loss
  checkpoint and beam five.
- COGS: 50,000 updates, effective batch 128, final checkpoint and beam five.
  Report generalization only; best-dev weights are retained for diagnosis.
- IWSLT14: 50,000 updates, effective batch 160, best development-loss
  checkpoint and beam five. Use `recipes/iwslt14.json`: peak learning rate
  0.0005, 4,000 warmup updates, inverse-square-root schedule, Adam betas
  (0.9, 0.98) and weight decay 0.0001.
- Analogy controllers use 0.1 times the ordinary learning rate. Other
  parameters follow the dataset's existing schedule.
- Save resume state every 1,000 updates, predictions, scores, parameter
  counts and configuration. Symmetric +/- is half the width of a 95%
  example-bootstrap interval with 1,000 resamples.
- Reuse the existing training, decoding and reporting code. No new
  orchestration framework is needed. Parameter sharing does not imply
  faster computation.

## Execution

SERVER: `/home/Yue_Ziran/workspace/ana-analogy-combined-v4`

NAS: `/mango/homes/YUE_Ziran/workspace/ana-analogy-combined-v4`

Code, logs and temporary files stay on SERVER. Copied datasets, checkpoints,
predictions and caches stay on NAS. Previous task directories are read-only.
Other users' processes are left untouched.

The queue files in [`queues/`](queues/) recorded this execution order. DE and FR
below refer to the two Multi30k directions; an arrow means the next queued
run, not concurrent training on that GPU.

| Server / GPU | Ordered runs | Memory cap (MiB) | Microbatch size |
|---|---|---:|---|
| exp14 / 0 | DE A → FR A → DE B → FR B → IWSLT14 D | 9,216 | Multi30k 128; IWSLT14 32 |
| exp14 / 1 | DE C → FR C → DE D → FR D → IWSLT14 E | 9,216 | Multi30k 128; IWSLT14 32 |
| exp14 / 2 | DE E → FR E → DE F → FR F → IWSLT14 F | 9,216 | Multi30k 128; IWSLT14 32 |
| exp15 / 0 | COGS A → COGS B | 6,144 | COGS 32 |
| exp15 / 0, added September 23 | IWSLT14 C | 7,680 | IWSLT14 32 |
| exp16 / 0 | COGS C → COGS D → IWSLT14 existing 2 | 9,216 | COGS 64; IWSLT14 32 |
| exp17 / 0 | COGS E → COGS F → IWSLT14 existing 4 | 9,216 | COGS 64; IWSLT14 32 |
| exp18 / 0 | IWSLT14 A | 16,384 | IWSLT14 64 |
| exp18 / 1 | IWSLT14 B | 24,576 | IWSLT14 80 |

At the user's request, IWSLT14 C moved from exp18's waiting queue to exp15
GPU 0 on 2026-09-23 at 10:47 JST, after exp15's COGS queue had finished.
Its detached queue is `queues/exp15_gpu0_iwslt_c.txt`; its wrapper log is
`logs/queue_gpu0_iwslt_c.log`. C had not started on exp18. Only its pending
queue entry was removed; A and other existing processes were left running.
The first training update on exp15 completed with finite loss.

### G/H extension, launched September 24

The extension used four new detached queues. Existing IWSLT14 C-F jobs on
exp14/15 remained unchanged. On exp18, the new runs shared GPUs with other
users' processes only when sufficient memory was available.

| Server / GPU | Ordered additional runs | Memory cap (MiB) | Microbatch size |
|---|---|---:|---|
| exp16 / 0 | DE G → FR G → COGS G | 9,216 | Multi30k 128; COGS 64 |
| exp17 / 0 | DE H → FR H → COGS H | 9,216 | Multi30k 128; COGS 64 |
| exp18 / 0 | IWSLT14 G | 12,288 | IWSLT14 40 |
| exp18 / 1 | IWSLT14 H | 12,288 | IWSLT14 40 |

The new queue files are `queues/exp16_gpu0_gh.txt`,
`queues/exp17_gpu0_gh.txt`, `queues/exp18_gpu0_gh.txt` and
`queues/exp18_gpu1_gh.txt`. The extension wrapper is deployed as
`run_extra.sh`, a snapshot of the updated `run.sh`, so the original running
wrappers are not replaced. All four queues launched at 12:29:55 JST.
All four initial jobs logged finite training loss. These live progress
checks followed the minimal forward/backward, cached-decoding and memory checks.

### I-M weekend extension, launched September 25

The first seven detached queue wrappers started on 2026-09-25 at
14:06:05 JST. All five new IWSLT14 jobs completed their first actual update
with finite loss, confirmed at 14:06:37 JST. The shorter tasks use two serial
queues. Existing G/H jobs and other users' processes were left untouched.
Both short-task queues use `ANA_WAIT_GPU_LOCK=1`. Exp17 initially waited for
H's existing COGS wrapper and acquired the lock at 14:07:11 JST after H
finished normally. The initial replacement queue on exp18 GPU 0 waited for
G's existing IWSLT14 wrapper.

Exp15's initial queue attempted all six I/J short tasks between 14:06:11
and 14:06:19 JST, but each failed the available-RAM guard before its first
training update. Free RAM was initially 8.18 GiB but fell below the 8 GiB
guard after Python loaded. No checkpoints were created. That wrapper ended
at 14:06:47 JST; its queue file was retired with comments and its failure
logs were preserved. A replacement queue started on exp18 GPU 0 at
14:07:49 JST, with the six cells in the same order.

After the user freed exp15 RAM, approximately 22 GiB was available.
A bounded GPU memory check passed with a 4,096 MiB cap: peak reserved
memory was 1,186 MiB for a Multi30k training update at microbatch 32 and
1,672 MiB for COGS at microbatch 16. The six cells had not started on exp18;
its queue was retired before a new exp15 wrapper launched at 14:44:40 JST.
Exp18's waiting wrapper was left to exit naturally after G finished. Neither G nor
any other running experiment was interrupted. Earlier failure logs remain.
The new exp15 wrapper log is
`logs/queue_gpu0_weekend_rescheduled_20260925.log`.
Effective batches and recipes remain unchanged; smaller microbatches can
change dropout RNG trajectories, but these six cells had not trained yet.

The resulting seven queues at that time were:

| Server / GPU | Ordered additional runs | Memory cap (MiB) | Microbatch size |
|---|---|---:|---|
| exp14 / 0 | IWSLT14 J | 9,216 | IWSLT14 32 |
| exp14 / 1 | IWSLT14 K | 9,216 | IWSLT14 32 |
| exp14 / 2 | IWSLT14 L | 9,216 | IWSLT14 32 |
| exp16 / 0 | IWSLT14 M | 9,216 | IWSLT14 32 |
| exp17 / 0 | DE K → FR K → COGS K → DE L → FR L → COGS L → DE M → FR M → COGS M | 9,216 | Multi30k 128; COGS 64 |
| exp15 / 0 | DE I → FR I → COGS I → DE J → FR J → COGS J | 4,096 | Multi30k 32; COGS 16 |
| exp18 / 1 | IWSLT14 I | 12,288 | IWSLT14 40 |

Queue files follow `queues/expNN_gpuN_weekend.txt` for these seven
server/GPU pairs. The retired `queues/exp18_gpu0_weekend.txt` contains only
comments; it no longer schedules any cells.
At 14:09 JST, K's Multi30k EN-to-DE run on exp17 had reached step 400
(worker PID 3041427), leaving eight short tasks queued there and six on
exp18 GPU 0. H's completed COGS generalization score is 80.73 +/- 0.52;
it finished naturally and was not interrupted. G's existing IWSLT14 run was
at step 39,600. It no longer holds up any extension training after the
exp15 move.
The updated wrapper is deployed as `run_weekend.sh`;
existing running wrappers were not modified. It permits one automatic
checkpoint retry for native exit codes 139 or 134 only. Other failures are
not automatically retried.

Minimal verification passed two GPU training updates, cached and causal
decoding checks, shared-initialization comparisons, and confirmation that
old model metadata stayed unchanged. All listed parameter counts were
verified. The five IWSLT14 jobs passed their initial live update checks.
At 14:09 JST, those five jobs and K's Multi30k EN-to-DE job were training,
with the remaining 14 extension cells queued.
No packages were installed for this extension. Decoding batch size is 32,
except for the rescheduled exp15 queue, which uses 8. Activation
checkpointing is disabled on every queue.

Gradient accumulation retains the recipe's effective batch size. Activation
checkpointing is disabled. These execution settings do not change the
number of training updates.

Detached queues automatically start each next cell after training and
scoring. They wait when the required memory is unavailable. Closing the local session
does not stop detached queues; a server reboot may require manual resume.
Resume checkpoints are saved every 1,000 updates.
Canonical reports update under `NAS/reports/` as scoring finishes.

### J's two failed-start runs, relaunched September 28

At the Monday check, 52 of 54 runs had completed and all queues had ended.
J's Multi30k EN-to-FR and COGS cells had failed exp15's startup resource
guard before training, leaving no checkpoints. The failure messages did
not identify which resource triggered the guard. Their failure logs remain.

At the user's request, these two cells restarted in parallel at 11:08 JST
on September 28: French on exp16 GPU 0 and COGS on exp17 GPU 0. Both GPUs
were idle and both hosts had over 29 GiB available RAM. Each detached
wrapper uses the existing `run_weekend.sh` with a 9,216 MiB GPU cap,
microbatch 128 for Multi30k or 64 for COGS, and decoding batch 32.
Effective batches and all model, training and scoring recipes are unchanged.
Both first training updates had finite loss. No packages or code were changed.

Queues: `queues/exp16_gpu0_j_fr_retry_20260928.txt` and
`queues/exp17_gpu0_j_cogs_retry_20260928.txt`.
Both runs completed successfully. The final results are 58.68 +/- 1.72 BLEU
for Multi30k EN-to-FR and 76.28 +/- 0.56% generalization exact match for
COGS. Canonical reports, including the run configuration and prediction
paths, are copied into the results directory linked below. All 54 runs in
this batch are complete.

[Results table](../../../results/permutations/analogy_combined_v4/README.md).
