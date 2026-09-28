# Multi30k: one-gradient-view power consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/power_consistency_singlegrad_v1](../../../runs/power_consistency_singlegrad_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Fresh experiment, not a resume of the user-stopped IWSLT14 run. Selected
server: exp15 GPU0. No additional dataset or automatic follow-up is queued.

Completed on 2026-09-15 at 07:01:29 JST after a 06:26:31 launch. Wrapper
2098695 and worker 2098701 were confirmed absent at 07:01:55. All 20,000
updates completed; ordinary development loss selected update 20,000.
Test BLEU is 40.00 ± 1.64, dev BLEU 40.23367, and training time 2,081.40
seconds. This saved 29.04% of the two-gradient run's training time but lost
0.90 BLEU. It exceeds shared-QKV's point score but not the matched control,
so it is not promoted. The two-gradient method remains the stronger result.
Reports and a small local archive of actual predictions/results are retained;
checkpoints remain on NAS. See reports/multi30k.json and artifacts/.
The focused CPU check passed in 1.27 seconds. No package changes.

Keep the existing linear-compressed embedding Transformer: width128, FFN232,
four heads, four encoder and four decoder layers, full independent Q/K/V,
2,248,512 parameters. Input/output embeddings remain tied. Power is fixed
at0.5 and used only in training; there is no new inference module.

Each batch gets two independent training-mode dropout predictions from the
same current model. The first runs without a gradient graph as a temporary
reference Q; the second P supplies CE(P)+2*S(P,Q). S is the existing exact
all-vocabulary-pairs four-term power penalty, using positive smoothed
probabilities with uniform mass1e-6. Only P receives gradients. The factor2
matches the completed symmetric method's expected raw gradient under IID
dropout, not its gradient variance, clipping, Adam trajectory, or BLEU.
No frozen/EMA teacher, pretrained weights, or evaluation-mode reference.

Retain the original 20,000-update Multi30k recipe and lowest ordinary dev-loss
checkpoint selection. Measure actual training time and final test BLEU with
95% example-bootstrap half-widths. Reuse original matched/shared controls,
the linear embedding reference, and the completed two-gradient-view result
40.90±1.64 (2,933.36 training seconds on exp15). Timing depends on load.

One small CPU check covers the factor2 gradient identity, the no-gradient
reference in training mode, and padding. No training smoke, broad tests,
package updates, baseline retraining, or new general infrastructure.

Code/logs: /home/Yue_Ziran/workspace/ana-power-consistency-singlegrad-v1
Data/checkpoints: /mango/homes/YUE_Ziran/workspace/ana-power-consistency-singlegrad-v1
The detached wrapper trains, decodes, and produces reports from saved real
predictions. Actual launch status and process IDs belong in launch.json.
