# Compact embeddings and power-based training

This branch is on hold as of 2026-09-17. It keeps independent Q, K and V projections and compresses the tied token-embedding/output table. Earlier power-analogy variants change training only; the completed context-power batch also learns small prediction components that remain active at inference.

[Result tables](../../results/compact_embeddings/README.md)

## Completed recent work

- [Context-dependent prediction and feature powers](context_power_v1/README.md): 15 completed runs; see the collected result reports rather than historical launch notes.
- [Multi-batch lookahead p selection](adaptive_power_lookahead_v1/README.md): three completed runs on both Multi30k directions and COGS, reported on 2026-09-16.
- [Rescaled adaptive-power penalty](adaptive_power_balanced_v1/README.md): three completed runs on Multi30k EN→DE, EN→FR and COGS, reported on 2026-09-16.
- [Fixed-power Multi30k EN→DE](power_consistency_v1/README.md).
- [Fixed-power EN→FR and COGS](power_enfr_cogs_v1/README.md).
- [Fixed-power IWSLT14 and ordinary compact control](power_consistency_iwslt_v1/README.md).
- [Adaptive power and two-pass controls](adaptive_power_v1/README.md): six completed runs across Multi30k EN→DE, EN→FR and COGS.

The fixed-power comparison, all three global-adaptive batches, and context-power batch are complete. The context-power batch learns powers through the prediction loss instead of temporary trial updates. COGS was included alongside translation.

## Study folders

Each folder contains its authored runner, analysis helpers or design notes. Model implementations live under [`src/ana`](../../src/ana/). Historical launch records, snapshots and large outputs remain in the local, untracked `runs/` directory; remote SERVER/NAS task paths are unchanged.

| Study | Purpose |
|---|---|
| [context_power_v1](context_power_v1/README.md) | Prediction-trained local powers, with and without four-term analogy penalties |
| [adaptive_power_lookahead_v1](adaptive_power_lookahead_v1/README.md) | Three-step, multi-batch power selection with a stay-put option |
| [adaptive_power_balanced_v1](adaptive_power_balanced_v1/README.md) | Adaptive power with a detached fixed-power reference scale |
| [adaptive_power_v1](adaptive_power_v1/README.md) | Compact embeddings with adaptive-power training |
| [analogy_embedding_redesign](analogy_embedding_redesign/README.md) | Redesign: compact lexical representations through power analogy |
| [causal_translation_analogy_design](causal_translation_analogy_design/README.md) | Proposed redesign: causal translation-example analogy |
| [causal_translation_v1](causal_translation_v1/README.md) | Multi30k: causal translation-example analogy |
| [embedding_analogy_v1](embedding_analogy_v1/README.md) | Multi30k: power-analogy lexical compression |
| [embedding_iwslt_v1](embedding_iwslt_v1/README.md) | Historical IWSLT14 compact-embedding runner |
| [embedding_mix_v1](embedding_mix_v1/README.md) | Multi30k: mixing power-completion residual features |
| [embedding_residual_v1](embedding_residual_v1/README.md) | Multi30k: linear embeddings with a power-completion residual |
| [embedding_unit_v1](embedding_unit_v1/README.md) | Multi30k: unit-normalized power-analogy embeddings |
| [frequency_lexical_redesign](frequency_lexical_redesign/README.md) | Redesign: frequency-aware lexical compression with power consistency |
| [frequency_lexical_v1](frequency_lexical_v1/README.md) | Multi30k: frequency-aware lexical compression plus power consistency |
| [power_consistency_iwslt_v1](power_consistency_iwslt_v1/README.md) | IWSLT14 power consistency |
| [power_consistency_p025_v1](power_consistency_p025_v1/README.md) | Multi30k: normalized quarter-power consistency |
| [power_consistency_singlegrad_v1](power_consistency_singlegrad_v1/README.md) | Multi30k: one-gradient-view power consistency |
| [power_consistency_v1](power_consistency_v1/README.md) | Multi30k: power-analogy dropout consistency |
| [power_enfr_cogs_v1](power_enfr_cogs_v1/README.md) | Compact embeddings with power-consistency training: English–French and COGS |
| [target_power_consistency_design](target_power_consistency_design/README.md) | Next pilot: full-vocabulary plus target/rest power consistency |
| [target_power_consistency_v1](target_power_consistency_v1/README.md) | Multi30k: full-vocabulary plus target/rest power consistency |

A dated launch paragraph is not a live status report. Use completed result records for outcomes; stopped runs are not resumed automatically.
