# D-clean: an unchanged-input branch beside the analogy path

All three experiments are complete. The [results table](../../../results/permutations/analogy_clean_branch_v6/README.md)
and per-corpus reports record their bootstrap intervals and unchanged model
sizes. D-clean improves EN→DE over D but not EN→FR or COGS.

Approved batch: one new model on Multi30k English to German, Multi30k
English to French, and COGS generalization. Existing full Transformer,
shared-QKV, D and L scores are references, not new training runs.
IWSLT14 is excluded.

## The one change

Model D sends the analogy-modified encoder input to both its large shared
projection and its small ordinary Q/K/V branches. D-clean keeps the shared
projection's analogy-modified input but sends the original input to the
small branches. Cross-attention and decoder self-attention are unchanged.

The hypothesis is that the ordinary branches can learn from the original
information alongside the analogy-modified path. This is not an established
explanation of the previous batch's results.

The model reuses D's exact initialization and parameter budget. It retains
input-dependent full Beneš routing, positive ordered quadruples, hard
selection among eight equivalent forms, and common positive scaling.
The preservation claim remains local to each accepted positive quadruple;
the ordinary projections and signed readout are outside that claim.
No power is calculated or learned in this permutation design.

## Unchanged training

- Model identifier: `compact_qkv_clean`.
- Encoder and cross-attention ordinary branch ranks: `d_model / 8`.
- Controller learning rate: 0.1 times the backbone rate, as in D.
- Multi30k: 20,000 updates, effective batch 256, original learning-rate
  schedule, best development-loss checkpoint, beam five.
- COGS: 50,000 updates, effective batch 128, original constant learning
  rate, final checkpoint, beam five. Report generalization only.
- Reuse the original data and tokenizers; no new tokenization or tuning.
- Existing checkpointing, development diagnostics and bootstrap reporting
  are reused. Report score ± half the 95% example-bootstrap interval width
  from 1,000 resamples.

Verified parameters: 2,328,464 for either Multi30k direction and 6,522,200
for COGS, equal to D. All other models retain their previous behavior;
the new parent projection flag defaults to off.

## Execution

SERVER: `/home/Yue_Ziran/workspace/ana-analogy-clean-branch-v6`

NAS: `/mango/homes/YUE_Ziran/workspace/ana-analogy-clean-branch-v6`

Code and logs stay on SERVER; copied data, checkpoints and predictions stay
on NAS. Previous remote tasks remain unchanged. Existing environments are
reused without package changes.

| Run | Server / GPU | Microbatch | Effective batch | GPU cap (MiB) |
|---|---|---:|---:|---:|
| Multi30k EN→DE | exp14 / 0 | 128 | 256 | 9,216 |
| Multi30k EN→FR | exp17 / 0 | 128 | 256 | 9,216 |
| COGS | exp16 / 0 | 64 | 128 | 9,216 |

Decode batch is 32 throughout. One detached queue wrapper per GPU handles
training, decoding and the bootstrap report. Checkpoints are saved every
1,000 updates. At preparation time these GPUs were idle; exp15 and exp18
were not needed for the three-run batch.

All three detached workers started on September 29, 2026, at 12:03:31 JST.
Training, decoding and bootstrap reporting are complete. No jobs remain
running or waiting in this batch.

A brief check confirmed identical initial D/D-clean weights and parameter
counts, unchanged optimizer groups, matching initial forward outputs,
finite CUDA backward passes at both model widths, and original-input
delivery to the small branches when the analogy change is nonzero.
No new test framework or environment changes were needed.

See the [results table](../../../results/permutations/analogy_clean_branch_v6/README.md)
and [reference provenance](reference_results.json).
