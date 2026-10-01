"""Train one specialization design with the existing recipes and scorer."""

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
    "ANA_SERVER", "/home/Yue_Ziran/workspace/ana-analogy-specialization-v7"
)
os.environ.setdefault(
    "ANA_NAS", "/mango/homes/YUE_Ziran/workspace/ana-analogy-specialization-v7"
)

from models import CORPUS_NAMES, MODEL_NAMES  # noqa: E402
from experiments.permutations.analogy_combined_v4 import (  # noqa: E402
    run_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=CORPUS_NAMES)
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    run_cell.run(
        args.corpus, args.model, args.describe,
        study_id="analogy_specialization_v7",
    )


if __name__ == "__main__":
    main()
