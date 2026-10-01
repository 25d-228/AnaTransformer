"""Run one approved D refinement with the existing training/scoring code."""

from __future__ import annotations

import argparse
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

from models import MODEL_NAMES  # noqa: E402
from experiments.permutations.analogy_combined_v4.run_cell import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("multi30k", "multi30k_enfr", "cogs"))
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run(args.corpus, args.model, args.describe, study_id="analogy_refinement_v5")


if __name__ == "__main__":
    main()
