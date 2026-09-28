# IWSLT14 power consistency

Current status: both the ordinary compact control and fixed-power training run completed after the recorded restart. The [final IWSLT14 report](../../../results/compact_embeddings/power_consistency_iwslt_v1/iwslt14.md) supersedes the earlier interruption and launch status below.

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/power_consistency_iwslt_v1](../../../runs/power_consistency_iwslt_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


## Resumed comparison — 2026-09-15, 19:36 JST

The user authorized completing the missing IWSLT14 cells on exp14–18.
The original power-trained run resumes its own 16,000-update checkpoint;
an ordinary-training compact control starts fresh alongside it. The older
stop record below is retained as history, not the current status.

| Model | Host / GPU | Starting update | Wrapper / worker PID | Log |
|---|---|---:|---|---|
| Compact embeddings + power consistency, p=0.5 | exp18 / 1 | 16,000 | 1947755 / 1947772 | logs/embedding_power_consistency_resume_20260915.out |
| Identical compact embeddings, ordinary CE | exp18 / 0 | 0 | 1947775 / 1947780 | logs/embedding_linear.out |

Both use 27,241,504 parameters, width 448, FFN 856, embedding rank 336,
and the unchanged 50,000-update IWSLT14 recipe. One CPU-only check confirmed
that the original powered checkpoint context and model schema still match,
its weights load, and both model names have identical initialized tensors.
There was no additional training smoke or environment update. The deployed
model source and powered trainer were left unchanged; the ordinary row
uses the existing one-pass trainer.

Exp18 had 38,968 MiB free on GPU1 and 19,989 MiB on GPU0 before launch.
The power allocator cap is 30 GiB, ordinary cap 16 GiB, with at least 2 GiB
free headroom required before each process starts. Other GPU processes were
left untouched. The two detached workers run independently; no additional
training follows them. Checkpoints remain on the same NAS task, with a new
ordinary run folder. Reports are written under the server task's reports/.

The report has five rows: three existing reported Transformer/Shared-QKV
references, ordinary compact training, and powered compact training. The
new rows get bootstrap intervals from their saved predictions; their direct
comparison gets a paired interval. Old controls are preserved at their
reported precision; no missing reference predictions are invented.

Run commands: `ANA_MAX_GPU_MIB=30720 bash run.sh 1` for power and
`ANA_MAX_GPU_MIB=16384 bash run.sh 0 embedding_linear` for ordinary.
These commands are already running; do not start duplicate workers.

## Original launch and interruption

Launched fresh on **2026-09-15 at 03:09:28 JST**, exp18 GPU1, wrapper
945911 and worker 945917. The completed Multi30k method scored 40.90 ± 1.64,
above the similar-size and shared-QKV controls. This task tests the same
two-gradient-pass, fixed-p training method on IWSLT14; no Multi30k weights
are loaded. No IWSLT14 score exists yet.

Stopped at the user's explicit request on 2026-09-15 at 06:17:05 JST.
Wrapper 945911 and worker 945917 were both confirmed absent at 06:17:29;
the worker's GPU allocation was released. Other users' processes were left
untouched. The last completed evaluation was update 16,000 of 50,000,
with development loss 2.8808311851702597. Its 436,327,627-byte resumable
checkpoint, saved at 06:13:26 JST, is preserved. No final test BLEU was
produced. There is no result ETA, automatic restart, or queued follow-up.
Do not resume without an explicit user request.

Before launch, the focused loss check passed in 1.59 seconds on exp18 CPU,
and an instantiated model confirmed 27,241,504 parameters. The GPU allocation
cap is 30 GiB, with 36,613 MiB free before launch. Existing Ollama PID 3079393
was left untouched. Existing environments were not modified; no training
smoke or broad test suite was run. Launch details are in launch.json.

## Candidate and parameter budget

`embedding_power_consistency` uses the existing `embedding_linear` registry
model: width 448, FFN width 856, four heads, six encoder and six decoder layers, and
the existing joint 10,000-token vocabulary. All 18 attention sites retain full
independent Q/K/V projections; all 12 FFNs remain ordinary GELU blocks.
The tied embedding stores a 10,000 by 336 code table and a 336 by 448 basis.
It has no power parameters, nonlinear completion, residual mixer, or row
normalization. Inference is the ordinary linear-compressed Transformer.

The registry's linear-embedding count is `3*V*d/4 + 3*d*d/4`, with no `d/4`
power term. At d448/FF856:

| Component | Parameters |
|---|---:|
| Tied linear embedding | 3,510,528 |
| Independent attention projections | 14,482,944 |
| Ordinary FFNs | 9,219,360 |
| Layer norms | 28,672 |
| Total | 27,241,504 |

This arithmetic count is 5,008 below the similar-size Transformer's 27,246,512
and 112 below the earlier, conditional residual-embedding shape. The count
was verified on the instantiated model before launch. The full d512/FF1024
Transformer has 36,665,344 parameters.

## Training-only numerical analogy

The trainer is an unchanged copy of `../power_consistency_v1/trainer.py`.
Each update makes two gradient-bearing passes with independent dropout on the
same batch. It averages their original label-smoothed cross-entropies and adds
the same fixed-weight consistency term: p=0.5, coefficient 1, and uniform
probability floor mass 1e-6.

For each valid target position, smooth each predicted distribution as
`P=(1-epsilon)*softmax(logits)+epsilon/V`. The four terms for vocabulary items
i and j are `P1(i):P1(j)::P2(i):P2(j)`. Set `delta=sqrt(P1)-sqrt(P2)` and penalize
`sum((delta-mean(delta))^2)`, averaged over valid target positions. This equals
the sum of all ordered-pair squared power-analogy defects divided by `2*V`,
computed exactly in O(V) rather than materializing a V by V matrix. Padding
targets are ignored; both predictions receive gradients. Power is used only
in training, not in the saved inference model.

## Preserved IWSLT14 recipe and execution

The existing `recipes/iwslt14.json` was copied unchanged. Training is fresh,
with no Multi30k or other checkpoint loaded: 50,000 updates, batch 160,
learning rate 0.0005, warmup 4,000, inverse-square-root schedule, weight decay
0.0001, AdamW (0.9, 0.98), epsilon 1e-9, clipping 1, dropout 0.3, smoothing 0.1,
and development evaluation every 1,000 updates. Select by ordinary development
cross-entropy, then decode with beam 5 and batch 32. Source, target, and decode
caps are 96; maximum positions are 256. Existing checkpoint/resume and saved
prediction handling are retained.

Code path:
`/home/Yue_Ziran/workspace/ana-power-consistency-iwslt-v1`.
Data/checkpoint path:
`/mango/homes/YUE_Ziran/workspace/ana-power-consistency-iwslt-v1`.
The runner requires the existing IWSLT14 tokenizer and text files under the
new data path; it does not create a new tokenizer.

Two passes increase activation memory and training work. Training uses an
explicit `ANA_MAX_GPU_MIB=30720`, requires at least 2 GiB free headroom above
that allocation, and checks RAM/storage before CUDA training. There is no
CPU fallback, activation checkpointing, automatic smaller batch, or smoke
run. The first 100 real updates fit within the cap; longer-run memory and
throughput remain subject to the actual batches and other GPU activity.

The report will compute the candidate's bootstrap interval from its actual
saved predictions. It retains the original three control scores from
the [original IWSLT14 table](../../../results/permutations/iwslt14.md) at their reported precision. Those original prediction
files are not available here, so their comparisons are rounded point-score
differences, not newly computed paired tests. No significance marks are added.
