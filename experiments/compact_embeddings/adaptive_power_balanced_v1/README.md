# Scale-balanced adaptive-power training

Status: all three runs completed on 2026-09-16, including decoding and bootstrap
reports. EN→DE finished at 18:23 JST, EN→FR at 18:23 JST, and COGS at 19:20 JST.
The four completed comparison rows are reused, not retrained. See the
[completed table](../../../results/compact_embeddings/adaptive_power_balanced_v1/README.md).

## One change from the previous adaptive method

Keep the same compact embeddings, independent Q/K/V, two dropout predictions,
and adaptive-power rule. Change only the scale of the analogy penalty. On each
training batch, compute the penalty at the current p and its reference value at
p = 0.5 from those same predictions. Multiply the current-p penalty by the
reference/current ratio, calculated without gradients. Guard tiny or zero
denominators. The reference computation needs no extra model forward pass.

This balances the numerical penalty magnitude against the fixed-power reference
recipe while keeping the current p's gradient direction. It does not make the
two gradients identical or guarantee equal gradient norms. The four-number
condition is unchanged: A^p + D^p = B^p + C^p, with A = P1(i), B = P1(j),
C = P2(i), and D = P2(j). Probabilities retain the uniform mixture of mass 1e-6;
the centered power differences cover all vocabulary pairs in O(V).

As before, one training-only p starts at 0.5 and stays within 0.25–0.75.
After warmup, every 200 main updates, temporary updates test nearby powers
against ordinary prediction loss on a separate small training batch. Each
adjustment is limited to 0.01. Temporary model updates are discarded; development
and test data do not guide p. The same balanced penalty is used for those trial
updates. Power adds no inference parameters or inference operations.

## Recipes and comparisons

| Dataset | Main updates | Effective batch | Compact parameters | Checkpoint |
|---|---:|---:|---:|---|
| Multi30k English–German | 20,000 | 256 | 2,248,512 | Best ordinary development loss |
| Multi30k English–French | 20,000 | 256 | 2,248,512 | Best ordinary development loss |
| COGS | 50,000 | 128 | 5,689,236 | Final |

Both translation directions retain width 128, FFN 232, rank-96 embeddings,
four encoder/four decoder layers, dropout 0.3, label smoothing 0.1, peak learning
rate 0.005, 2,000 warmup updates, inverse-square-root decay, and their existing
train-only joint 10k tokenizers. COGS retains width 400, FFN 509, rank 160,
two encoder/two decoder layers, dropout 0.1, no label smoothing, and constant
learning rate 0.0001 without warmup. Existing Adam settings, gradient clipping,
evaluation interval, and beam-5 decoding remain unchanged.

New runs start fresh with the original full batches intended. Historical
adaptive translation references used microbatch accumulation, as recorded in
their manifests. Existing checkpoints are not continued. Compare the new method
against ordinary compact training, two-pass training without analogy, fixed
p = 0.5, and the previous adaptive method. COGS reports generalization only.

## Execution and reports

Task name: `ana-adaptive-power-balanced-v1`. Server code, logs, temporary files,
and reports use `/home/Yue_Ziran/workspace/ana-adaptive-power-balanced-v1`.
Data, predictions, and checkpoints use
`/mango/homes/YUE_Ziran/workspace/ana-adaptive-power-balanced-v1`.
`ANA_SERVER` and `ANA_NAS` can override these roots. Placement on exp14–18 is
recorded below; other users' processes remain untouched.

| Dataset | Server | GPU | Batch | Allocator cap |
|---|---|---:|---:|---:|
| Multi30k English–German | exp18 | 0 | 256 | 16 GiB |
| Multi30k English–French | exp18 | 1 | 256 | 16 GiB |
| COGS | exp15 | 0 | 128 | 16 GiB |

Each worker runs with `nohup`, independently of this chat session. Checkpoints
are saved every 1,000 updates; decoding and bootstrap reporting follow training
automatically. Logs are `logs/<corpus>_embedding_adaptive_power_balanced.log`.
The local [launch record](../../../runs/adaptive_power_balanced_v1/launch.json)
contains process IDs and storage paths. No environment was modified.

The new model is `embedding_adaptive_power_balanced`; its runs use
`runs/<corpus>_embedding_adaptive_power_balanced_seed42`. Copy the four prior
rows' records and predictions into `reference_runs/<corpus>_<model>_seed42`.
All references come from the completed `ana-adaptive-power-v1` task: ordinary
and fixed-power references under `reference_runs`, and two-pass and previous
adaptive rows under `runs`. Preserve their original records and provenance.

Run `report.py CORPUS` after a cell completes, or `report.py all` for all three
datasets. Reports retain all five rows, use symmetric 95% bootstrap ±, and keep
exact interval endpoints and new-minus-reference paired intervals in JSON.
Missing results stay pending; no score is inferred for an unfinished run.
