# Multi30k: frequency-aware lexical compression plus power consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/frequency_lexical_v1](../../../runs/frequency_lexical_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Completed on exp15 GPU0 at 09:38:28 JST, 2026-09-15, after launching detached
at 08:48:48. Wrapper 2169786 (parent PID 1) and worker 2169792 were verified
absent at 09:45:50. No training or follow-up remains queued.

Final Multi30k test BLEU is **40.69 ± 1.67**. All 20,000 updates completed;
the 20,000-update checkpoint was scored. Training with periodic development
evaluation took 2,964.02 seconds. Development BLEU is 41.08448 and best
development loss is 2.72243. Final resume and model checkpoints are retained
on NAS; actual predictions, results and reports are also archived locally.

This is +0.58 BLEU against the matched Transformer and +2.04 against shared-QKV,
but -0.20 against the original uniform-embedding power model. Keep the original
**40.90 ± 1.64** model as the preferred candidate. These are point-score
comparisons; exact paired intervals are in the [completed report](../../../results/compact_embeddings/frequency_lexical_v1/multi30k.md).
This is a fresh trial of [the redesign](../frequency_lexical_redesign/README.md),
not a resume of any stopped experiment. No other dataset or variant is queued.

Store full 128-dimensional vectors for 3,376 vocabulary entries and
80-dimensional codes for 6,624 entries, with one shared 80×128 basis for the
compressed group. Reserve special tokens in the full group and select the rest
by combined source/target training frequency, breaking ties by token ID.
Token IDs, tokenizer, input/output tying and global vocabulary softmax stay
unchanged. Selection does not read dev or test frequencies.

Keep d128, FFN232, four heads, four encoder/four decoder layers, independent
Q/K/V and ordinary FFNs. The verified parameter total is 2,248,512, exactly the
same as the uniform linear embedding model and 13.70% below the full model.

The original successful two-gradient p=0.5 trainer is reused unchanged.
Average two translation losses and add the centered squared gap between
the two square-root probability vectors, coefficient 1. This covers all
four-term vocabulary-pair defects in linear vocabulary work. Power is
training-only. The target/rest penalty and one-gradient shortcut are absent.

Fresh 20,000 updates retain batch 256, LR 0.005, warmup 2,000, dropout 0.3,
label smoothing 0.1, original optimizer settings, lowest-dev-loss selection,
beam 5 and decode batch 32. Two gradient passes still increase training work;
no speed improvement is assumed. Use actual predictions for BLEU and symmetric
95% bootstrap half-widths, with real paired comparisons against matched 40.12,
shared 38.65, uniform linear 40.31 and original power-consistency 40.90.

One focused CPU check passed in 1.41 seconds. Actual model construction confirms
2,248,512 parameters, with 12 independent-QKV sites and 8 ordinary FFNs.
The full-vector band covers 92.36% of training tokens, or 91.53% excluding
special tokens; the boundary frequency is 23 occurrences with token-ID tie
breaking. Only training counts are used.
No broad suite, separate training smoke,
checksums, environment changes, thesis edits, or general training infrastructure.
Existing stopped checkpoints and other users' processes remain untouched.

Exp15 GPU0 was idle at 08:43:31 JST, with 20,156 MiB free GPU memory and
30,793 MiB available RAM. Home storage had 322 GB free; NAS had 3.4 TB free and
was mounted read/write. Both new task roots were absent before creation at
08:44:07. The task reads the existing Python environment without modifying it.

SERVER: /home/Yue_Ziran/workspace/ana-frequency-lexical-v1

NAS: /mango/homes/YUE_Ziran/workspace/ana-frequency-lexical-v1

Model: embedding_frequency_power_consistency

Detached command: bash run.sh embedding_frequency_power_consistency 0

See launch.json for completion and process status, the [report](../../../results/compact_embeddings/frequency_lexical_v1/multi30k.md)
and [exact results](../../../results/compact_embeddings/frequency_lexical_v1/multi30k.json) for actual bootstrap comparisons,
and artifacts/ for saved predictions and run results. No stopped task was resumed.
