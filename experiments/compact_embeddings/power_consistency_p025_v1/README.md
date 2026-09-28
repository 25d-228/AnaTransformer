# Multi30k: normalized quarter-power consistency

Repository layout: this folder holds authored experiment code and notes. Saved outputs remain under [runs/power_consistency_p025_v1](../../../runs/power_consistency_p025_v1/); curated result tables are indexed in the [family results](../../../results/compact_embeddings/README.md). Dated launch details below are historical, not a statement that a job is running.


Fresh fixed-p=0.25 follow-up to the completed two-gradient p=0.5 method.
Use exp15 GPU0. This is not a restart of the user-stopped IWSLT14 task;
no additional dataset or automatic follow-up is queued.

Stopped at the user's request on 2026-09-15 at 07:29:04 JST for redesign.
Wrapper 2118939 and worker 2118945 were both absent at 07:29:45, with no
remaining GPU compute process. The last completed evaluation was at 6,000
updates: dev loss 2.86932 and 913.50 training seconds. Its 36,290,978-byte
resume checkpoint, saved at 07:26:34, and all logs are preserved. There is no
final test BLEU or result ETA. Do not resume this task automatically.

Originally launched detached at 07:11:19 JST. One focused CPU check passed
in 2.61 seconds, and model construction confirmed 2,248,512 parameters.

Keep the successful compact architecture: full independent Q/K/V, ordinary
FFNs, tied linear-compressed embeddings, d_model=128, d_ff=232, four heads,
four encoder/four decoder layers, and 2,248,512 trainable parameters.
The original 20,000-update Multi30k optimizer, batch, selection, and decoding
recipe is unchanged. Both independent training-mode dropout predictions
receive gradients. No pretrained checkpoint or teacher is used.

For two vocabulary choices i,j, use A=P(i), B=P(j), C=Q(i), D=Q(j) from
the two predictions. P and Q are smoothed with uniform probability mass
1e-6 to keep all terms positive. The four-term power condition remains
A^p+D^p=B^p+C^p, now with fixed p=0.25.

Compute delta=c*(P^p-Q^p), where c=V^(p-1/2)/(2p) and V is vocabulary size.
Center delta across vocabulary, sum its squares, then average over valid
target positions. This equals c^2/V times the sum of squared four-term
defects over unordered vocabulary pairs, calculated exactly in O(V).
The total training loss is mean(CE(P),CE(Q))+S, with coefficient 1.

The scaling makes the transformed-probability derivative at the uniform
probability 1/V equal to sqrt(V)/2 for every p. At p=0.5 the loss reduces
to the completed centered square-root loss. This does not equalize loss
magnitudes on arbitrary trained predictions. The smaller power gives
relatively more sensitivity to weaker alternatives. Power is training-only;
there are no new inference operations or parameters.

Reuse completed matched/shared controls, the ordinary linear embedding
reference, and two-gradient p=0.5 result 40.90 ± 1.64. The cheaper one-gradient
attempt scored 40.00 ± 1.64 and is not the parent method for this trial.
Report actual final predictions with symmetric 95% bootstrap half-widths.
No claim that this new power improves BLEU is made before its result.

One small CPU check covers the all-pairs identity, p=0.5 value/gradient
recovery, gradient flow to both predictions, padding, and finite low-precision
inputs. No training smoke, broad test suite, package changes, or new general
infrastructure. Actual launch status is in launch.json.

Code/logs/reports: /home/Yue_Ziran/workspace/ana-power-consistency-p025-v1
Data/checkpoints/predictions: /mango/homes/YUE_Ziran/workspace/ana-power-consistency-p025-v1
Method condition: https://arxiv.org/html/2407.18770v1
Related dropout consistency: https://arxiv.org/abs/2106.14448
