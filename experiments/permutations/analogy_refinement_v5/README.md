# Same-size refinements of model D

All nine experiments are complete. The [completed results](../../../results/permutations/analogy_refinement_v5/README.md)
include the exact bootstrap reports and parameter counts, retrieved on
October 1, 2026. The execution notes below retain the original launch and
authorized microbatch-restart history.

Eight detached queue wrappers launched on September 28, 2026, at
15:57:33–34 JST for the approved nine-run batch. By 15:59 JST, all eight
active workers had passed 100 updates with finite losses; the ninth run
is queued automatically. The aim is to improve translation quality without
increasing model D's parameter count, on Multi30k English to German,
Multi30k English to French and COGS generalization.

**IWSLT14 is excluded and must not be automatically queued.** It will be
reconsidered only if these results are exceptionally promising for the
IJCAI goal and the user approves a further run.

## Designs

The implementation reuses the parent
[`analogy_combined_v4`](../analogy_combined_v4/) model, protected numerical
analogy operation, training, decoding and reporting code through imports.
Ordinary embeddings and shared Q/K/V projections remain unchanged.

| ID | Identifier | Controller learning-rate multiplier | Encoder branch rank | Cross-attention branch rank |
|---|---|---:|---:|---:|
| D, existing | `compact_qkv` | 0.1 | `d_model / 8` | `d_model / 8` |
| D1 | `d_router_03` | 0.3 | `d_model / 8` | `d_model / 8` |
| D2 | `d_router_10` | 1.0 | `d_model / 8` | `d_model / 8` |
| D3 | `d_cross_focus` | 0.1 | `d_model / 16` | `3 * d_model / 16` |

D1 and D2 change the learning-rate multiplier for the existing permutation
and scaling controllers only. All other parameters, including correction
gains and ordinary branches, retain the ordinary learning-rate multiplier
of 1.0. D3 moves capacity from the encoder's small Q/K/V branches to the
cross-attention Q/K/V branches. Equal encoder and decoder layer counts in
these recipes make its parameter budget identical to D.

Each candidate will run on all three tasks: **three designs times three
tasks, nine new runs**. Existing full Transformer, shared-QKV, D and L
results are references, not extra training jobs. The exact scores and
their provenance are in [`reference_results.json`](reference_results.json).

Parameter counts verified from all nine instantiated models:

| Model | Either Multi30k direction | COGS |
|---|---:|---:|
| D1 | 2,328,464 | 6,522,200 |
| D2 | 2,328,464 | 6,522,200 |
| D3 | 2,328,464 | 6,522,200 |

## Numerical-analogy operation

All three candidates retain D's encoder operation: an input-dependent hard
full Beneš permutation forms groups of features, shared by the Q/K/V
roles. Matched positive rails use `softplus(x) + epsilon` and
`softplus(-x) + epsilon`. Each quadruple is sorted and accepted only when
both rails satisfy `0 < a < b <= c < d`; invalid groups bypass the local
change. Each role chooses one of the eight equivalent forms and a common
positive multiplier for the accepted quadruple.

The preservation claim concerns those equivalent forms and common scaling
inside each accepted positive quadruple. The existing power is neither
calculated nor learned. Global grouping, sorting, signed recombination,
residual mixing and ordinary projection branches are outside that claim.
No analogy block is added to the decoder.

## Training and decisions

- Keep the established dataset recipes, tokenizers, data splits and scoring.
- Multi30k: 20,000 updates, effective batch 256, best development-loss
  checkpoint and beam five.
- COGS: 50,000 updates, effective batch 128, final checkpoint and beam five.
  Report generalization only.
- Keep ordinary single-pass cross-entropy training. Save resume state every
  1,000 updates and retain predictions, model counts and run configuration.
- Reuse 95% example-bootstrap intervals with 1,000 resamples. Report the
  symmetric half-width as `score ± half-width`.
- Translation improvement is the main decision criterion. Use development
  results to choose the next candidate; COGS is supplementary.
- L remains D's no-analogy comparison for D1/D2's architecture. If D3 is
  pursued, its no-analogy comparison must use D3's redistributed branches.

## Execution locations

SERVER: `/home/Yue_Ziran/workspace/ana-analogy-refinement-v5`

NAS: `/mango/homes/YUE_Ziran/workspace/ana-analogy-refinement-v5`

Code and logs stay on SERVER; data, tokenizers, checkpoints, predictions
and caches stay on NAS. Parent task directories remain read-only. Exp14-18
are allowed, including GPUs with existing processes when sufficient VRAM
is available. Do not interrupt or modify other users' tasks.

Eight workers were assigned first runs, with one follow-on run queued on
exp14 GPU 0. An arrow below means sequential runs, not concurrent training.
DE and FR are the two Multi30k directions.

| Server / GPU | Ordered runs | GPU cap (MiB) | Microbatch | Decode batch |
|---|---|---:|---:|---:|
| exp14 / 0 | DE D1 → FR D1 | 9,216 | 128 | 32 |
| exp14 / 1 | DE D2 | 9,216 | 128 | 32 |
| exp14 / 2 | DE D3 | 9,216 | 128 | 32 |
| exp15 / 0 | FR D3 | 6,144 | 64 | 16 |
| exp16 / 0 | COGS D1 | 9,216 | 64 | 32 |
| exp17 / 0 | COGS D2 | 9,216 | 64 | 32 |
| exp18 / 0 | FR D2 | 12,288 | 256 | 32 |
| exp18 / 1 | COGS D3 | 16,384 | 128 | 32 |

At the user's request, only the two exp18 workers and their queue wrappers
were stopped at 17:03:20 JST on September 28. Both restarted detached at
17:05:43 after more VRAM became available. FR D2 resumes its saved step
3,000 with microbatch 256 (previously 32), and COGS D3 resumes step 5,000
with microbatch 128 (previously 16). Decode batches increase from 8 to 32.
The table above shows the new exp18 settings; other hosts are unchanged.

Effective batches remain 256 and 128, respectively. Model, optimizer, RNG
state and training history were retained; only checkpoint execution-batch
metadata was adjusted for the authorized resume. Unsaved updates are
recomputed. Original checkpoints are backed up in each run's
`before_microbatch_restart_20260928/`, and `execution_changes.json` records
both configurations. Changing microbatches can change dropout draws.
Other users' processes and existing environments were left untouched.
Both restarted workers passed 100 resumed updates with finite losses.
Initial throughput was about 3.8 updates/second for FR D2 and 9.6 for
COGS D3, versus about 1.0 and 1.5 before the restart.

Four tiny CUDA backward checks produced finite results, and initialization
comparisons passed before launch. Existing environments were reused without
package changes: PyTorch 2.5.1/cu118 on exp14/16/17, 2.5.1/cu121 on exp15,
and 2.11/cu128 on exp18.

The [results table](../../../results/permutations/analogy_refinement_v5/README.md)
contains the reused references and all nine completed scores.
