"""Report D-clean with the existing example-bootstrap procedure."""

from __future__ import annotations

import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(1, str(ROOT))
sys.path.append(str(STUDY.parent / "analogy_combined_v4"))
os.environ.setdefault(
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-clean-branch-v6"
)
os.environ.setdefault(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-clean-branch-v6"
)

from models import CORPUS_NAMES  # noqa: E402
from experiments.permutations.analogy_combined_v4 import report as base  # noqa: E402


def main():
    base.STUDY_ID = "analogy_clean_branch_v6"
    base.STUDY_TITLE = "model D with unchanged-input small branches, batch 6"
    base.TRAINING_NOTE = (
        "Same model D weights at initialization, parameter count, branch ranks, "
        "corpus recipes and controller learning-rate scale 0.1. Only the small "
        "encoder Q/K/V branches receive the original attention input; the "
        "shared projection still receives the analogy-modified input. "
        "COGS scores final weights. IWSLT14 is excluded."
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
