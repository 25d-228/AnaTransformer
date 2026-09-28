# Multi30k: causal translation-example analogy

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/causal_translation_v1](../../../runs/causal_translation_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Completed on exp15 GPU0 at 08:06:56 JST, 2026-09-15, after starting at 07:42:51.
Final Multi30k test BLEU is **39.93 +/- 1.67**, using 20,000 training updates
and the selected 19,000-update checkpoint. Training took 1,426.35 seconds;
dev BLEU is 40.63609 and the scored analogy gain is 0.17662.
The exact 95% bootstrap interval is [38.26932, 41.60615].
Wrapper 2135218 and worker 2135223 were both absent at 08:07:34, with no
remaining GPU compute process. Checkpoints and predictions are preserved.
No follow-up is queued by this task.

This candidate exceeds shared-QKV but not the matched-size control or the
earlier 40.90 power-consistency result. It is not promoted. See
[the completed report](../../../results/compact_embeddings/causal_translation_v1/multi30k.md) and [exact values](../../../results/compact_embeddings/causal_translation_v1/multi30k.json).
This is a fresh task, not a restart of any user-stopped experiment.

The candidate keeps the completed linear compressed embedding Transformer:
d128, FFN232, four heads, four encoder/four decoder layers, full independent
Q/K/V, and ordinary FFNs. A small vocabulary-scoring branch uses earlier
source-context/target-token pairs as references for the next prediction.
The [design](../causal_translation_analogy_design/README.md) defines all four
positive terms, the fixed p=0.5 condition, causal masks and exact reduction.

Verified count: 2,249,545 parameters. It includes two 128-to-4 biased maps
and a learned positive gain initialized to 0.1. Parameter-free normalization
precedes each map; softplus/log(2)+1e-6 supplies positive features.
Every feature coordinate has four analogy roles. The number of features is
a budget choice, not an eight-permutation or block-size argument.

Train only `causal_translation_p05`, fresh for the original 20,000 updates,
batch 256, LR 0.005, warmup 2,000, inverse-square-root schedule, dropout 0.3,
label smoothing 0.1, best-dev-loss selection, and beam 5. This uses one ordinary
translation-loss pass per update. No teacher, warm start, two-pass consistency,
extra dataset, exponent grid, package changes, or automatic follow-up.

Reuse authentic saved predictions for matched-size Transformer 40.12,
shared-QKV 38.65, linear embeddings 40.31, and two-pass power consistency 40.90.
The completed candidate uses the existing symmetric 95% bootstrap half-width.

Host: exp15, GPU0, existing environment read-only. At 07:36:47 JST the GPU
had 20,156 MiB free and no compute processes; RAM had 30,803 MiB available.
The shared-storage mount was read/write. Both new task roots were absent
before creation at 07:38:41. Only this task's data copy and files were created.

SERVER: /home/Yue_Ziran/workspace/ana-causal-translation-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-causal-translation-v1

One focused CPU check passed in 1.65 seconds. It covers the exact four-term score and gradients,
unchanged backbone initialization, parameter count, causal prefix access,
and full-vs-cached decoding after beam-parent reordering. No training smoke
or broad test suite. Actual verification and launch state are in launch.json.
