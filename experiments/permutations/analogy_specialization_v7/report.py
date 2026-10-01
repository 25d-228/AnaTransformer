"""Report the eighteen specialization runs with example-bootstrap intervals."""

from __future__ import annotations

import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(1, str(ROOT))
sys.path.append(str(STUDY.parent / "analogy_combined_v4"))
os.environ.setdefault(
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-specialization-v7"
)
os.environ.setdefault(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-specialization-v7"
)

from models import CORPUS_NAMES  # noqa: E402
from experiments.permutations.analogy_combined_v4 import (  # noqa: E402
    report as base,
)


def main() -> None:
    base.STUDY_ID = "analogy_specialization_v7"
    base.STUDY_TITLE = "analogy-preserving specialization, batch 7"
    base.TRAINING_NOTE = (
        "Six designs test small-branch placement, two-rail readouts and "
        "query specialization. Original embeddings, corpus recipes, "
        "effective batches and single-pass cross-entropy are retained. "
        "Controllers use learning-rate scale 0.1. COGS scores final "
        "weights. IWSLT14 is excluded."
    )
    base.SPLITS = {key: base.SPLITS[key] for key in CORPUS_NAMES}
    base.REFERENCE_LABELS = {
        "baseline": "Full Transformer",
        "shared_qkv": "Shared-QKV",
        "compact_qkv_no_analogy": "L: D without analogy",
        "compact_qkv": "D: original",
    }
    base.main()


if __name__ == "__main__":
    main()
