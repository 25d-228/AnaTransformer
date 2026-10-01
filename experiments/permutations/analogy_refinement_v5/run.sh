#!/usr/bin/env bash
set -euo pipefail
export ANA_SERVER=${ANA_SERVER:-/home/Yue_Ziran/workspace/ana-analogy-refinement-v5}
export ANA_NAS=${ANA_NAS:-/mango/homes/YUE_Ziran/workspace/ana-analogy-refinement-v5}
cd "$ANA_SERVER"
queue_file=${2:?queue file required}
while read -r corpus model; do
    [[ -z $corpus || $corpus == \#* ]] && continue
    case "$corpus" in
        multi30k|multi30k_enfr|cogs) ;;
        *) echo "Corpus not approved for this batch: $corpus"; exit 2 ;;
    esac
    case "$model" in
        d_router_03|d_router_10|d_cross_focus) ;;
        *) echo "Model not approved for this batch: $model"; exit 2 ;;
    esac
done < "$queue_file"
export ANA_RUN_CELL=experiments/permutations/analogy_refinement_v5/run_cell.py
export ANA_REPORT=experiments/permutations/analogy_refinement_v5/report.py
exec bash experiments/permutations/analogy_combined_v4/run.sh "$@"
