# Compact embeddings with adaptive-power training

Current status: all six runs completed. See the [EN→DE](../../../results/compact_embeddings/adaptive_power_v1/multi30k.md), [EN→FR](../../../results/compact_embeddings/adaptive_power_v1/multi30k_enfr.md) and [COGS](../../../results/compact_embeddings/adaptive_power_v1/cogs.md) reports. No job is launched by this repository reorganization.

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/adaptive_power_v1](../../../runs/adaptive_power_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Six new runs: two training methods on Multi30k English–German, Multi30k English–French, and COGS. Both methods retain the completed compact embedding architecture, independent Q/K/V, ordinary attention and feed-forward layers, and tied input/output embeddings. There is no inference-time power module or new inference parameter.

| Compact-model training | English–German | English–French | COGS generalization |
|---|---|---|---|
| Ordinary training | Reuse completed run | Reuse completed run | Reuse completed run |
| Two dropout passes without analogy | New run | New run | New run |
| Analogy training, fixed p = 0.5 | Reuse completed run | Reuse completed run | Reuse completed run |
| Analogy training, adaptive p | New run | New run | New run |

The two-pass control averages the two ordinary supervised losses without a consistency penalty. The adaptive method adds the all-vocabulary-pair power-analogy consistency penalty. For entries i and j in two positive next-word distributions P1 and P2, the four terms are A = P1(i), B = P1(j), C = P2(i), and D = P2(j); the target relation remains A^p + D^p = B^p + C^p. Centering the vector of power differences covers all pairs in O(V), without a vocabulary-squared tensor. A uniform mixture of mass 1e-6 keeps the probabilities positive.

One shared training-only p starts at 0.5 and stays between 0.25 and 0.75. Every 200 main updates after warmup, temporary AdamW updates test nearby powers, p minus/plus 0.05 within the allowed bounds. Their ordinary prediction loss on a separate small training batch guides p, with each adjustment limited to 0.01; temporary model updates are discarded. Development and test data are not used for these probes. The penalty scale matches the existing p = 0.5 loss exactly at p = 0.5 and matches its local derivative at uniform predictions for other powers. This local normalization does not imply identical loss magnitudes for all p. The run manifest records the exact probe configuration and adaptation history is saved with the run.

## Base recipes

Both Multi30k directions retain 20,000 main optimizer updates, batch 256, dropout 0.3, label smoothing 0.1, Adam betas (0.9, 0.98), peak learning rate 0.005, 2,000 warmup updates, inverse-square-root decay, no weight decay, clipping at 1, best ordinary development-loss checkpoint, and beam 5. Each direction uses its existing train-only joint 10k tokenizer. The compact architecture is width 128, FFN 232, four heads, four encoder/four decoder layers, and embedding rank 96: 2,248,512 parameters.

COGS retains 50,000 main updates, batch 128, constant learning rate 0.0001, no warmup, dropout 0.1, no label smoothing, Adam betas (0.9, 0.999), no weight decay, clipping at 1, final checkpoint, and beam 5. The compact architecture is width 400, FFN 509, eight heads, two encoder/two decoder layers, and embedding rank 160: 5,689,236 parameters. Report generalization only, not IID results.

Both new rows use two gradient-bearing dropout passes per main update; adaptive probes add training work. Main update counts and inference stay unchanged. The primary comparison is against the same compact model with ordinary training; the two-pass row separates the consistency penalty from the extra supervised pass, and the fixed-power row measures the value of adaptation.

The adaptive translation workers on exp16/17 use `ANA_MICRO_BATCH_SIZE=128`: two pieces of each original 256-example batch, weighted by valid target-token counts, followed by one gradient clip and one optimizer update. This reduces peak VRAM without changing the effective batch or main-update count. Full-batch startup attempts exhausted their allocator budgets before the first checkpoint; their logs are retained under `initial-memory-failure`. Other workers use their original full batches. Exp16's older driver obscured the allocation error with an NVML-symbol error; no driver or environment was changed.

## Launch placement

| Dataset | Two-pass control | Adaptive power |
|---|---|---|
| Multi30k English–German | exp14 GPU 0, 9 GiB cap | exp16 GPU 0, 9 GiB cap, microbatch 128 |
| Multi30k English–French | exp14 GPU 1, 9 GiB cap | exp17 GPU 0, 9 GiB cap, microbatch 128 |
| COGS | exp15 GPU 0, 12 GiB cap | exp18 GPU 1, 10 GiB cap |

Workers run detached with `nohup`; each writes an evaluation/resume checkpoint every 1,000 main updates and scores predictions after training. `launch.json` stores the initial process IDs and launch times; live logs and completed results determine current status. Exp18 shares its GPU with the existing Ollama process, which is unchanged. To resume an adaptive translation worker, retain both `ANA_MICRO_BATCH_SIZE=128` and `ANA_MAX_GPU_MIB=9216` when calling `bash run.sh 0 CORPUS embedding_adaptive_power` from its server task directory.

## Storage and results

Task name: `ana-adaptive-power-v1`. Server code, logs, reports, and temporary files live under `/home/Yue_Ziran/workspace/ana-adaptive-power-v1`. Shared data, checkpoints, and predictions live under `/mango/homes/YUE_Ziran/workspace/ana-adaptive-power-v1`. `ANA_SERVER` and `ANA_NAS` can override these roots. Detached workers may use exp14–18 where enough VRAM is available, without changing other users' processes.

New run folders are `runs/<corpus>_<model>_seed42`, with corpus `multi30k`, `multi30k_enfr`, or `cogs`, and model `embedding_two_pass` or `embedding_adaptive_power`. Completed ordinary and fixed-power result records and predictions are copied into matching `reference_runs` folders, keeping their originals unchanged. English–German sources are `ana-embedding-analogy-v1` for ordinary training and `ana-power-consistency-v1` for fixed-power training. English–French and COGS sources are `ana-power-enfr-cogs-v1`.

Run `report.py CORPUS` after completed cells, or `report.py all` for all three. Reports retain all four rows and use saved predictions to display symmetric ±, defined as half the width of a 95% example-bootstrap interval with 1,000 resamples. JSON retains exact interval endpoints, source records, training times, and adaptive-minus-ordinary/two-pass/fixed paired comparisons as each becomes available. Unfinished rows remain pending. No additional full training run is needed for reporting.
