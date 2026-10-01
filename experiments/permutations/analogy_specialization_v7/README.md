# Analogy-preserving specialization: batch 7

All eighteen runs are complete. The last queue finished on September 30,
2026, at 04:40 JST with zero failures. The [completed table](../../../results/permutations/analogy_specialization_v7/README.md)
contains model sizes, bootstrap intervals and links to exact result records.

Approved scope: six designs on Multi30k English to German, Multi30k English
to French, and COGS generalization, for eighteen training runs. IWSLT14 is
excluded. Existing full Transformer, shared-QKV, L and D results
are references, not new training runs.

## Designs

All designs retain ordinary embeddings and D's small ordinary projection
branches unless the table explicitly changes their input. The rank of each
ordinary encoder and cross-attention branch remains `d_model / 8`.

| Label | Model identifier | Change from D |
|---|---|---|
| N1 | `small_branch` | The encoder's large shared projection receives the original input; its small Q/K/V branches receive the full analogy-transformed input. |
| N2 | `small_branch_cross` | Use N1's small-branch treatment in both encoder self-attention and decoder cross-attention. |
| N3 | `rail_readout` | Keep D's placement, but retain both the difference and a learned contribution from the sum of the positive-rail changes. |
| N4 | `small_branch_rail` | Combine N1's encoder small-branch placement with N3's two-rail readout. |
| N5 | `decoder_query` | Keep D's encoder Q/K/V analogy modules and add analogy only to the query input of decoder self-attention. |
| N6 | `query_only` | Use analogy on encoder queries and decoder self-attention queries only; encoder keys and values use ordinary inputs. |

N5 and N6 do **not** add analogy to decoder cross-attention. Their ordinary
cross-attention Q/K/V branches remain D's. Decoder self-attention keys and
values remain ordinary shared projections. N2 is the only new design that
extends the protected operation to cross-attention in this batch.

N1 and N2 leave the large shared path unchanged and use the full transformed
view in their small branches. This changes the placement and strength
strategy together; it is not merely a change to D's gain. N3 versus D and
N4 versus N1 each isolate the addition of the two-rail readout.

## Protected operation and readout

The operation retains token-dependent hard Beneš feature routing, positive
ordered quadruples satisfying `0 < a < b <= c < d`, a hard choice among the
eight equivalent forms, and a common positive multiplier. Feature routing
acts within a token, not across sequence positions. Invalid quadruples
bypass the local operation. No analogy power is calculated or learned in
this permutation branch.

The two positive rails are `softplus(x) + epsilon` and
`softplus(-x) + epsilon`. Their matched permutations and common multiplier
mean that their signed difference cancels the softplus transformation in
real arithmetic. N3 and N4 also retain the **sum of the two rail changes**,
which carries magnitude information absent from that signed difference.

The additional contribution is multiplied by a zero-initialized,
`tanh`-bounded coefficient after restoration of the original feature
positions. It uses rail changes, not raw rail values, so it vanishes when
the protected operation is the identity with unit multiplier. Setting the
new coefficient to zero recovers the corresponding signed-only readout.

The numerical-analogy preservation claim applies inside each accepted
positive quadruple. Sorting, signed readouts, their learned combinations,
and ordinary neural projections remain outside that claim. The new
readout is a hypothesis about useful features, not an assertion that its
output is itself a numerical analogy.

## Unchanged recipes and reporting

- Both Multi30k directions: 20,000 updates, effective batch 256, learning
  rate 0.005, 2,000 warmup updates, inverse-square-root schedule, best
  development-loss checkpoint, and beam five.
- COGS: 50,000 updates, effective batch 128, constant learning rate 0.0001,
  final checkpoint, and beam five. Report generalization only.
- Controllers use 0.1 times the backbone learning rate, as in D. Ordinary
  branches and readout parameters use the backbone rate.
- Reuse the original data, tokenizers, embeddings, and single-pass
  cross-entropy objective. No new tokenization or recipe sweep.
- Development evaluation and checkpoints occur every 1,000 updates.
  Existing diagnostics, checkpointing, decoding, and scoring are reused.
- Report score plus or minus half the width of the 95% example-bootstrap
  interval, using 1,000 resamples. Parameter counts come from the actual
  instantiated models, not estimates from model names.

## Execution and records

SERVER: `/home/Yue_Ziran/workspace/ana-analogy-specialization-v7`

NAS: `/mango/homes/YUE_Ziran/workspace/ana-analogy-specialization-v7`

Code, queue logs and small reports stay on SERVER. Data, checkpoints and
saved predictions stay on NAS. Existing remote experiment directories are
not modified. Exp14 through exp18 may be used when there is sufficient
free memory; other users' processes must remain untouched.

`run_cell.py` and `report.py` reuse the batch-4 trainer and bootstrap
reporter. `run.sh` validates the approved model and corpus names, then
reuses the existing detached queue loop. There is no new scheduler or test
framework. Each queue continues to its next entry after a run finishes
and updates the report after scoring.

The deployment must include the existing batch-4 package and trainer,
`src`, the three original files under `recipes`, and this study's model
module. Place `reference_results.json` at the SERVER task root for the
reporter's default reference lookup. Per-corpus JSON and Markdown reports
are written under `reports` on both SERVER and NAS. The JSON retains
prediction paths, exact intervals, model sizes, and recipe provenance.

Queue files under `queues` and the launch record describe the actual
server/GPU assignment, microbatches and memory limits. Resource-specific
microbatches do not change the prescribed effective batch sizes.

## Completed execution

All eight detached queues started on 2026-09-29 at 15:19 JST across exp14–18.
Every initial worker passed 100 updates with finite losses and no memory
error. All eighteen cells subsequently finished, including automatic
decoding and bootstrap reports. Other users' processes and the existing
environments were left unchanged.

The short check before launch verified all eighteen parameter counts,
initial forward equivalence with D, six tiny CUDA backward passes, and
full-versus-cached decoding for the new decoder paths. No test framework
or scheduler was added.

See the [table and queue assignments](../../../results/permutations/analogy_specialization_v7/README.md).
The exact launch record is `runs/analogy_specialization_v7/launch.json`
locally and `launch.json` in each SERVER task directory. This paragraph is
an execution record; no runs remain active or queued.
