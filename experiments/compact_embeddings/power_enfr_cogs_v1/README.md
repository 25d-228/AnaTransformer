# Compact embeddings with power-consistency training: English–French and COGS

Current status: the English→French and COGS comparison is complete. See the [EN→FR](../../../results/compact_embeddings/power_enfr_cogs_v1/multi30k_enfr.md) and [COGS](../../../results/compact_embeddings/power_enfr_cogs_v1/cogs.md) reports. Launch details below are retained as history.

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/power_enfr_cogs_v1](../../../runs/power_enfr_cogs_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Continue the successful Multi30k English–German method without changing its power objective. The powered model uses two independently dropped-out prediction distributions, mean supervised cross-entropy, and the exact all-vocabulary-pairs consistency penalty at fixed p = 0.5 and weight 1.0. A uniform probability mixture of mass 1e-6 keeps all four terms positive. Inference is unchanged. The model has independent Q/K/V projections, ordinary attention and feed-forward layers, and a factorized shared input/output word embedding; it does not use D4 or Beneš.

For vocabulary entries i and j, the four numbers are A = P1(i), B = P1(j), C = P2(i), D = P2(j). The penalty encourages A^p + D^p = B^p + C^p across all vocabulary pairs, without constructing a vocabulary-squared tensor.

## Comparison to fill

| Model | EN–FR parameters | EN–FR test BLEU ± | COGS parameters | COGS generalization EM (%) ± |
|---|---:|---|---:|---|
| Full reference Transformer | 2,605,568 | Pending | 8,844,800 | Existing compatible control |
| Similar-size Transformer | 2,249,936 | Pending | 5,690,376 | Existing compatible control |
| Shared-QKV | 2,213,888 | Pending | 5,702,144 | Existing compatible control |
| Compact embeddings, ordinary training | 2,248,512 | Pending | 5,689,236 | Pending |
| Compact embeddings + power training | 2,248,512 | Pending | 5,689,236 | Pending |

The two compact rows have identical shapes and initialization. EN–FR retains width 128, FFN 232, embedding rank 96, and four encoder/four decoder layers. COGS uses width 400, rank 160, two encoder/two decoder layers, and FFN 509: the closest integer FFN width to the similar-size control's parameter count at vocabulary size 835. This size choice used parameter counts, not performance.

EN–FR uses a new train-only joint 10k SentencePiece tokenizer and the official 29,000/1,014/1,000 train/dev/test2016 pairs. Its recipe retains the repository's Multi30k adaptation: 20,000 updates, batch 256, dropout 0.3, label smoothing 0.1, inverse-square-root schedule with peak 0.005 and 2,000 warmup updates, best development-loss checkpoint, and beam 5. The full shape follows Wu et al. (2021), but sentence batching, fixed updates, and checkpoint selection are the repository adaptation, not an exact reproduction of the paper. No German calibration threshold is reused.

COGS retains the existing corpus recipe: 50,000 updates, batch 128, constant learning rate 0.0001, dropout 0.1, no label smoothing, final checkpoint and beam 5. Only generalization results are reported. Existing controls are reused only after checking raw data, tokenizer, configurations, ordered references, and prediction scores. Original result records remain unchanged; reference_provenance.json identifies their sources.

## Execution

Task name: `ana-power-enfr-cogs-v1`. Code, logs, small reports and temporary files live under `/home/Yue_Ziran/workspace/ana-power-enfr-cogs-v1` on each host. Data, checkpoints and predictions live under `/mango/homes/YUE_Ziran/workspace/ana-power-enfr-cogs-v1`.

Planned placement: EN–FR full/matched/shared controls on exp14 GPUs 0/1/2; EN–FR power on exp15 GPU 0; EN–FR compact ordinary on exp16 GPU 0; COGS compact ordinary on exp17 GPU 0; COGS power on exp18 GPU 1. Existing users' jobs are left untouched, with a per-process allocation limit and free-memory headroom.

Each detached worker runs a finite queue and stops on failure. Checkpoints are saved every 1,000 updates; this study does not resume earlier stopped experiments. Run `bash run.sh CORPUS GPU MODEL...` from the server task directory with its existing runtime. `report.py CORPUS` produces bootstrap reports automatically after completed cells. Displayed ± is the symmetric half-width of the percentile 95% bootstrap interval, matching the existing table convention; JSON retains the actual interval endpoints.

Verification is limited to one focused CPU model check and actual configurations/data preparation before full training. No separate training smoke run or environment update was needed.
