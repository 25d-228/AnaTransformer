# Multi30k: full-vocabulary plus target/rest power consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/target_power_consistency_v1](../../../runs/target_power_consistency_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Launched detached on exp15 GPU0 at 08:17:08 JST, 2026-09-15.
Stopped at the user's request at 08:35:04 JST for a redesign. Wrapper 2151566
and worker 2151571 were verified absent at 08:35:42. The last completed
evaluation was at 7,000 updates: dev loss 2.83142, session time 1,064.11 seconds.
The 36,291,490-byte resume checkpoint saved at 08:34:54 is preserved.
There is no final test BLEU, no active ETA, and no automatic resume is allowed.
This is one fresh trial, not a restart of any stopped experiment.

Keep the completed 40.90-BLEU two-gradient power-consistency model and add
the correct-target/rest quartet described in
[the design](../target_power_consistency_design/README.md). Architecture and
inference remain the ordinary linear compressed embedding Transformer:
d128, FFN232, four encoder/four decoder layers, full separate Q/K/V, ordinary
FFNs, and 2,248,512 parameters. No causal translation head is included.

Fixed p=0.5 is used only in training. Both dropout predictions receive
gradients. Average their original translation losses, then add the existing
centered full-vocabulary root-gap penalty and the new target/rest penalty,
each with coefficient 1. For raw target probabilities P_y and Q_y, use
A=(1-epsilon)*P_y+epsilon/2, B=1-A,
C=(1-epsilon)*Q_y+epsilon/2, D=1-C, epsilon=1e-6.
The added term is 0.5*(sqrt(A)+sqrt(D)-sqrt(B)-sqrt(C))^2.
Compute it as differences of roots to retain exact zero for identical views.
Padding contributes to neither consistency term. No extra model forward,
parameter, teacher or inference computation is introduced by this addition.

The original fresh 20,000-update recipe, batch 256, LR 0.005, warmup 2,000,
dropout 0.3, label smoothing 0.1, best-dev-loss checkpoint selection and beam 5
are retained. The two-pass training cost remains; no speed improvement is
assumed. No other dataset, exponent grid or automatic follow-up is queued.
Reuse completed matched/shared/linear/full-consistency references and their
actual predictions. Report final BLEU with symmetric 95% bootstrap half-widths.

One focused CPU check passed in 1.73 seconds: explicit all-pairs plus target
quartet value/gradients, gradients to both views, identical-view zero,
padding/all-padding, and finite low-precision/extreme inputs. Actual model
construction confirmed 2,248,512 parameters, 12 independent-QKV sites and
8 ordinary FFNs. No training smoke, broad suite, package changes or hashes.

Exp15 was idle at 08:12:19 JST with 20,156 MiB GPU memory free and 30,789 MiB
available RAM. Shared storage was read/write; both new task roots were absent
before creation at 08:15:36. Existing environments and other tasks are unchanged.

SERVER: /home/Yue_Ziran/workspace/ana-target-power-consistency-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-target-power-consistency-v1

Run: bash run.sh embedding_power_consistency_target 0

Actual process/checkpoint status is in launch.json. Final reports are written
only after genuine completed predictions exist, under SERVER/reports.
