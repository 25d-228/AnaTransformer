"""Train D-clean with the existing corpus recipes and scoring code."""

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
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-clean-branch-v6"
)
os.environ.setdefault(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-clean-branch-v6"
)

from models import CORPUS_NAMES, MODEL_NAMES  # noqa: E402
from experiments.permutations.analogy_combined_v4.run_cell import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=CORPUS_NAMES)
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run(args.corpus, args.model, args.describe, study_id="analogy_clean_branch_v6")


if __name__ == "__main__":
    main()
