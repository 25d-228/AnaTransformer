# Lookahead adaptive-power training

Status: all three runs completed training, decoding, and bootstrap reporting
on 2026-09-16. Multi30k English–German scored 40.82 ± 1.62 BLEU,
English–French scored 60.03 ± 1.62 BLEU, and COGS generalization scored
81.74 ± 0.51% exact match. Five completed comparison rows are reused,
not retrained. See the [results table](../../../results/compact_embeddings/adaptive_power_lookahead_v1/README.md).

## Change only how p is chosen

Keep the compact model and the exact scale-balanced analogy loss from the
[previous study](../adaptive_power_balanced_v1/README.md). The same four-number
condition, A^p + D^p = B^p + C^p, compares vocabulary entries i and j in two
dropout predictions: A = P1(i), B = P1(j), C = P2(i), and D = P2(j).
Positive probabilities, the O(V) all-pairs reduction, and the detached
reference(p = 0.5)/current-penalty scale are unchanged.

The new controller checks whether a different p helps over several temporary
updates, rather than trusting one short trial:

1. Every 200 main updates after warmup, try the distinct values among p minus
   0.05, the current p, and p plus 0.05, clipped to 0.25–0.75.
2. For each candidate, start from the same model and optimizer state and make
   three sequential temporary AdamW updates on the same three small training
   batches. Use that candidate's p and the unchanged balanced penalty.
3. Turn dropout off and measure ordinary prediction loss on three further
   training batches, disjoint from the update batches. Compare each candidate
   with the current-p trial, not with an untrained starting model.
4. Move p by at most 0.01 only if its mean loss improves by at least
   `max(1e-7, abs(current-p mean loss) * 1e-4)` and it wins on at least two of
   the three scoring batches. Otherwise keep p unchanged.

The trials restore model, optimizer, random-generator, and training-mode state;
temporary updates do not become main training updates. Only the accepted p
change is kept. P starts at 0.5. No development or test examples guide these
decisions. The controller and power loss are absent from inference.

## Unchanged recipes

| Dataset | Main updates | Effective batch | Compact parameters | Checkpoint |
|---|---:|---:|---:|---|
| Multi30k English–German | 20,000 | 256 | 2,248,512 | Best ordinary development loss |
| Multi30k English–French | 20,000 | 256 | 2,248,512 | Best ordinary development loss |
| COGS | 50,000 | 128 | 5,689,236 | Final |

Translation retains width 128, FFN 232, rank-96 tied embeddings, four
encoder/four decoder layers, dropout 0.3, label smoothing 0.1, peak learning
rate 0.005, 2,000 warmup updates, inverse-square-root decay, and each direction's
existing train-only joint 10k tokenizer. COGS retains width 400, FFN 509,
rank 160, two encoder/two decoder layers, dropout 0.1, no label smoothing,
and constant learning rate 0.0001 without warmup. Existing Adam settings,
clipping, evaluation interval, and beam-5 decoding remain unchanged.

All three new runs start fresh with full batches. The model keeps independent
Q/K/V projections. Compare with ordinary compact training, two-pass training
without analogy, fixed p = 0.5, the original adaptive controller, and the
scale-balanced controller. COGS reports generalization only. Additional probe
steps cost training time but add no inference parameters or operations.

## Execution and reports

Task name: `ana-adaptive-power-lookahead-v1`. Server code, logs, temporary files,
and reports use `/home/Yue_Ziran/workspace/ana-adaptive-power-lookahead-v1`.
Data, predictions, and checkpoints use
`/mango/homes/YUE_Ziran/workspace/ana-adaptive-power-lookahead-v1`.
`ANA_SERVER` and `ANA_NAS` can override these roots. Other users' processes
remain untouched; no environment was modified.

| Dataset | Server | GPU | Full batch | Allocator cap |
|---|---|---:|---:|---:|
| Multi30k English–German | exp18 | 0 | 256 | 15 GiB |
| Multi30k English–French | exp18 | 1 | 256 | 15 GiB |
| COGS | exp15 | 0 | 128 | 16 GiB |

Workers run with `nohup` and do not depend on this chat session. They checkpoint
every 1,000 main updates and then decode and generate bootstrap reports
automatically after training. Logs are
`logs/<corpus>_embedding_adaptive_power_lookahead.log`. The
[launch record](../../../runs/adaptive_power_lookahead_v1/launch.json) records
process IDs and source paths.

The English–German worker completed at 21:02:24 JST, English–French at
20:44:17 JST, and COGS at 21:50:15 JST on 2026-09-16. All three workers have
exited. Final p values were 0.71, 0.74, and 0.35, respectively. No further
experiments were launched when collecting these results.

The new model is `embedding_adaptive_power_lookahead`; its run folders are
`runs/<corpus>_embedding_adaptive_power_lookahead_seed42`. Copy five completed
reference rows into `reference_runs/<corpus>_<model>_seed42` from the previous
`ana-adaptive-power-balanced-v1` task: four earlier rows from `reference_runs`
and the balanced row from `runs`. Preserve their original records and provenance.

Run `report.py CORPUS` after each cell completes, or `report.py all` for all three.
Reports retain six rows, symmetric 95% bootstrap ±, exact endpoints, final p,
training time, and new-minus-each-reference paired intervals. Missing scores
remain pending. Existing reference scores are unchanged.
