"""Use the existing bootstrap reporter for the nine D-refinement runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(1, str(ROOT))
sys.path.append(str(STUDY.parent / "analogy_combined_v4"))
os.environ.setdefault(
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-refinement-v5"
)
os.environ.setdefault(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-refinement-v5"
)

from experiments.permutations.analogy_combined_v4 import report as base  # noqa: E402


def main():
    base.STUDY_ID = "analogy_refinement_v5"
    base.STUDY_TITLE = "same-size model D refinements, batch 5"
    base.TRAINING_NOTE = (
        "Original embeddings, corpus recipes and single-pass cross-entropy. "
        "D1/D2 use controller learning-rate scales 0.3/1.0. D3 retains 0.1 "
        "and redistributes the encoder/cross-attention branch budget. "
        "COGS scores final weights. IWSLT14 is excluded."
    )
    base.SPLITS = {key: base.SPLITS[key] for key in ("multi30k", "multi30k_enfr", "cogs")}
    base.REFERENCE_LABELS = {
        "baseline": "Full Transformer",
        "shared_qkv": "Shared-QKV",
        "compact_qkv_no_analogy": "L: D without analogy",
        "compact_qkv": "D: original",
    }
    base.main()


if __name__ == "__main__":
    main()
